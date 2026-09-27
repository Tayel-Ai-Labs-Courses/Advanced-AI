import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import joblib

from reliable_ai import (
    KEYS,
    CLASS_NAMES,
    LABEL_TO_ID,
    MODEL_FEATURES,
    MONITOR_FEATURES,
    PROHIBITED_FEATURES,
    VALID_RANGES,
    RAW_FEATURES,
    ENGINEERED_FEATURES,
    RANDOM_STATE,
    generate_raw_sources,
    load_sources,
    audit_sources,
    aggregate_duplicates,
    invalidate_ranges,
    temporal_impute,
    fuse_sources,
    chronological_split,
    encode_target,
    logistic_pipeline,
    advanced_model_builder,
    compare_candidates,
    select_legitimate_model,
    LEGITIMATE_MODEL_NAMES,
    TemperatureScaler,
    multiclass_brier,
    expected_calibration_error,
    choose_review_threshold,
    validate_measurements,
    make_safe_decision,
    run_training_comparison,
    fit_and_persist_artifacts,
    load_artifacts,
    engineer_features,
    build_feature_row,
)


@pytest.fixture(scope="module")
def data_dir():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "data"
        generate_raw_sources(d)
        yield d


@pytest.fixture(scope="module")
def data(data_dir):
    return fuse_sources(data_dir)


@pytest.fixture(scope="module")
def comparison(data):
    return compare_candidates(data)


def test_fusion_contract(data):
    assert len(data) == 840
    assert not data.duplicated(subset=KEYS).any()
    assert not data[MODEL_FEATURES].isna().any().any()
    prohibited_in_model = set(PROHIBITED_FEATURES).intersection(MODEL_FEATURES)
    assert not prohibited_in_model


def test_chronological_split_is_future_aware(data):
    split = chronological_split(data)
    train = split["train"]
    cal = split["calibration"]
    test = split["test"]

    assert train["year"].max() < cal["year"].min()
    assert cal["year"].max() < test["year"].min()
    assert test["year"].max() == 2025


def test_leaky_model_is_not_eligible_for_selection(comparison):
    assert comparison["selected_model_name"] != "advanced_leaky"

    leaky_row = comparison["metrics"][comparison["metrics"]["model"] == "advanced_leaky"].iloc[0]
    correct_row = comparison["metrics"][comparison["metrics"]["model"] == "advanced_correct"].iloc[0]
    assert leaky_row["macro_f1"] >= correct_row["macro_f1"]


def test_evidence_selects_best_legitimate_model(comparison):
    metrics = comparison["metrics"]
    legitimate = metrics[metrics["model"].isin(LEGITIMATE_MODEL_NAMES)].copy()
    sorted_legit = legitimate.sort_values(
        by=["macro_f1", "log_loss", "model"],
        ascending=[False, True, True],
    )
    expected_name = sorted_legit.iloc[0]["model"]
    assert comparison["selected_model_name"] == expected_name


def test_temperature_scaling_returns_probability_distribution():
    rng = np.random.default_rng(42)
    n = 50
    raw = rng.dirichlet(np.ones(3), size=n)
    y = rng.integers(0, 3, size=n)

    scaler = TemperatureScaler()
    scaler.fit(raw, y)
    calibrated = scaler.transform(raw)

    assert (calibrated >= 0).all()
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_persisted_model_matches_selected_model_and_runs(data_dir):
    comparison = run_training_comparison(data_dir)
    selected_name = comparison["selected_model_name"]

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        fit_and_persist_artifacts(comparison, data_dir, model_dir)
        artifacts = load_artifacts(model_dir)

    meta_name = artifacts["metadata"]["selected_model_name"]
    assert meta_name == selected_name

    sample = pd.DataFrame({
        "month": [6, 3, 9],
        "temperature_c": [28.0, 30.0, 25.0],
        "rainfall_mm": [40.0, 20.0, 60.0],
        "humidity_pct": [55.0, 40.0, 70.0],
        "soil_moisture": [0.5, 0.3, 0.6],
        "soil_ph": [7.0, 6.5, 7.5],
        "vegetation_index": [0.6, 0.4, 0.7],
        "disease_cases": [30.0, 50.0, 20.0],
        "pest_reports": [10.0, 25.0, 5.0],
        "irrigation_capacity": [0.5, 0.3, 0.7],
        "vulnerability_index": [0.4, 0.6, 0.3],
    })
    sample = engineer_features(sample)
    proba = artifacts["model"].predict_proba(sample[MODEL_FEATURES])
    assert proba.shape == (3, 3)


def test_invalid_input_is_rejected(data_dir):
    comparison = run_training_comparison(data_dir)
    split = comparison["split"]
    train = split["train"]
    calibration = split["calibration"]
    test = split["test"]

    selected_model = comparison["selected_model"]

    from reliable_ai import fit_reliability_components
    temp_scaler, anomaly_pipeline, anomaly_threshold, ref_medians = \
        fit_reliability_components(selected_model, train, calibration)

    from reliable_ai import choose_review_threshold
    X_cal = calibration[MODEL_FEATURES]
    y_cal = encode_target(calibration["risk_label"])
    raw_cal = selected_model.predict_proba(X_cal)
    cal_proba = temp_scaler.transform(raw_cal)
    review_threshold, _ = choose_review_threshold(cal_proba, y_cal)

    impossible = {
        "month": 13, "temperature_c": 150, "rainfall_mm": -1, "humidity_pct": 500,
        "soil_moisture": 2, "soil_ph": 20, "vegetation_index": -1,
        "disease_cases": 9000, "pest_reports": 2000,
        "irrigation_capacity": 4, "vulnerability_index": -3,
    }
    result = make_safe_decision(
        impossible, selected_model, temp_scaler, anomaly_pipeline,
        anomaly_threshold, review_threshold, ref_medians
    )
    assert result["status"] == "REJECT_INVALID_INPUT"
    assert len(result["problems"]) > 0


def test_artifact_manifest_and_contract_are_consistent(data_dir):
    from reliable_ai import validate_artifact_contract

    comparison = run_training_comparison(data_dir)
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        fit_and_persist_artifacts(comparison, data_dir, model_dir)
        contract = validate_artifact_contract(model_dir)

    assert contract["ok"]
    assert contract["metadata"]["selected_model_name"] == comparison["selected_model_name"]
    assert "risk_model.joblib" in contract["manifest"]["files"]


def test_artifact_contract_detects_metadata_tampering(data_dir):
    from reliable_ai import validate_artifact_contract

    comparison = run_training_comparison(data_dir)
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        fit_and_persist_artifacts(comparison, data_dir, model_dir)
        metadata_path = model_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["model_features"] = list(reversed(metadata["model_features"]))
        metadata_path.write_text(json.dumps(metadata, indent=2))
        contract = validate_artifact_contract(model_dir)

    assert not contract["ok"]
    assert any("model_features" in problem or "fingerprint mismatch" in problem for problem in contract["problems"])


def test_decision_result_contains_session4_audit_fields(data_dir):
    comparison = run_training_comparison(data_dir)
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        fit_and_persist_artifacts(comparison, data_dir, model_dir)
        artifacts = load_artifacts(model_dir)

        valid = {
            "month": 1, "temperature_c": 23.6, "rainfall_mm": 72.4, "humidity_pct": 68.5,
            "soil_moisture": 0.4599, "soil_ph": 6.48, "vegetation_index": 0.8958,
            "disease_cases": 41.3, "pest_reports": 28.8,
            "irrigation_capacity": 0.6734, "vulnerability_index": 0.352,
        }
        result = make_safe_decision(
            valid,
            artifacts["model"],
            artifacts["temperature_scaler"],
            artifacts["anomaly_model"],
            artifacts["anomaly_threshold"],
            artifacts["review_threshold"],
            artifacts["reference_medians"],
            model_version=artifacts["metadata"]["selected_model_name"],
            policy_version=artifacts["metadata"]["policy_version"],
        )

    required = {
        "status", "reason", "prediction", "confidence", "review_threshold",
        "anomaly_score", "anomaly_threshold", "problems", "explanation",
        "model_version", "policy_version",
    }
    assert required.issubset(result)
