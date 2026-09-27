import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from .constants import (
    RANDOM_STATE,
    CLASS_NAMES,
    MODEL_FEATURES,
    MONITOR_FEATURES,
)
from .data import fuse_sources
from .modeling import encode_target, compare_candidates
from .reliability import fit_reliability_components, choose_review_threshold


ARTIFACT_SCHEMA_VERSION = "1.0"
POLICY_VERSION = "reliability_policy_v2"
REQUIRED_ARTIFACT_FILES = (
    "risk_model.joblib",
    "temperature_scaler.joblib",
    "anomaly_model.joblib",
    "reference_medians.joblib",
    "metadata.json",
    "model_comparison.csv",
    "threshold_analysis.csv",
)


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_artifact_manifest(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    files = {}
    for name in REQUIRED_ARTIFACT_FILES:
        path = model_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "files": files,
    }


def write_artifact_manifest(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    manifest = build_artifact_manifest(model_dir)
    with (model_dir / "artifact_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def validate_artifact_contract(model_dir: str | Path) -> dict:
    """Validate presence, fingerprints, and core metadata identity without training."""
    model_dir = Path(model_dir)
    problems = []

    for name in REQUIRED_ARTIFACT_FILES:
        if not (model_dir / name).exists():
            problems.append(f"missing artifact: {name}")

    manifest_path = model_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        problems.append("missing artifact: artifact_manifest.json")
        manifest = None
    else:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)

    metadata = None
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("model_features") != list(MODEL_FEATURES):
            problems.append("metadata model_features do not match the approved feature order")
        if metadata.get("class_names") != CLASS_NAMES.tolist():
            problems.append("metadata class_names do not match the declared class order")
        if metadata.get("policy_version") != POLICY_VERSION:
            problems.append("metadata policy_version does not match the runtime policy")

    if manifest is not None:
        for name, recorded in manifest.get("files", {}).items():
            path = model_dir / name
            if not path.exists():
                continue
            actual = sha256_file(path)
            if actual != recorded.get("sha256"):
                problems.append(f"fingerprint mismatch: {name}")
            if path.stat().st_size != recorded.get("bytes"):
                problems.append(f"size mismatch: {name}")

    return {
        "ok": not problems,
        "problems": problems,
        "metadata": metadata,
        "manifest": manifest,
    }


def run_training_comparison(data_dir: str | Path) -> dict:
    data = fuse_sources(data_dir)
    return compare_candidates(data, data_dir=data_dir)


def get_selected_model(comparison: dict) -> tuple:
    name = comparison["selected_model_name"]
    model = comparison["selected_model"]
    features = comparison["selected_features"]
    return name, model, features


def fit_and_persist_artifacts(
    comparison: dict,
    data_dir: str | Path,
    model_dir: str | Path,
) -> dict:
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    selected_name = comparison["selected_model_name"]
    selected_model = comparison["selected_model"]
    split = comparison["split"]
    metrics = comparison["metrics"]

    train = split["train"]
    calibration = split["calibration"]

    temp_scaler, anomaly_pipeline, anomaly_threshold, ref_medians = \
        fit_reliability_components(selected_model, train, calibration)

    X_cal = calibration[MODEL_FEATURES]
    y_cal = encode_target(calibration["risk_label"])
    raw_cal = selected_model.predict_proba(X_cal)
    cal_proba = temp_scaler.transform(raw_cal)
    review_threshold, threshold_table = choose_review_threshold(
        cal_proba,
        y_cal,
        target_selective_accuracy=0.82,
        minimum_coverage=0.25,
        maximum_review_rate=0.30,
    )

    joblib.dump(selected_model, model_dir / "risk_model.joblib")
    joblib.dump(temp_scaler, model_dir / "temperature_scaler.joblib")
    joblib.dump(anomaly_pipeline, model_dir / "anomaly_model.joblib")
    joblib.dump(ref_medians, model_dir / "reference_medians.joblib")

    metrics.to_csv(model_dir / "model_comparison.csv", index=False)
    threshold_table.to_csv(model_dir / "threshold_analysis.csv", index=False)

    test_row = metrics[metrics["model"] == selected_name].iloc[0]
    classifier = selected_model.named_steps["classifier"] if hasattr(selected_model, "named_steps") else selected_model
    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "selected_model_name": selected_name,
        "selected_model_class": f"{classifier.__class__.__module__}.{classifier.__class__.__name__}",
        "selected_test_metrics": {
            "accuracy": float(test_row["accuracy"]),
            "balanced_accuracy": float(test_row["balanced_accuracy"]),
            "macro_f1": float(test_row["macro_f1"]),
            "log_loss": float(test_row["log_loss"]),
        },
        "model_features": list(MODEL_FEATURES),
        "monitor_features": list(MONITOR_FEATURES),
        "class_names": CLASS_NAMES.tolist(),
        "temperature": float(temp_scaler.temperature),
        "review_threshold": float(review_threshold),
        "anomaly_threshold": float(anomaly_threshold),
        "training_end_year": 2023,
        "calibration_year": 2024,
        "test_year": 2025,
        "random_state": RANDOM_STATE,
        "selection_rule": "highest legitimate macro_f1, then lowest log_loss",
        "review_policy": {
            "target_selective_accuracy": 0.82,
            "minimum_coverage": 0.25,
            "maximum_review_rate": 0.30,
        },
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    manifest = write_artifact_manifest(model_dir)

    return {
        "metadata": metadata,
        "review_threshold": review_threshold,
        "threshold_table": threshold_table,
        "manifest": manifest,
    }


def load_artifacts(model_dir: str | Path, verify: bool = True) -> dict:
    model_dir = Path(model_dir)
    contract = validate_artifact_contract(model_dir)
    if verify and not contract["ok"]:
        details = "; ".join(contract["problems"])
        raise RuntimeError(f"Artifact contract validation failed: {details}")

    model = joblib.load(model_dir / "risk_model.joblib")
    temp_scaler = joblib.load(model_dir / "temperature_scaler.joblib")
    anomaly_model = joblib.load(model_dir / "anomaly_model.joblib")
    ref_medians = joblib.load(model_dir / "reference_medians.joblib")

    with (model_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    return {
        "model": model,
        "temperature_scaler": temp_scaler,
        "anomaly_model": anomaly_model,
        "reference_medians": ref_medians,
        "metadata": metadata,
        "manifest": contract["manifest"],
        "review_threshold": metadata["review_threshold"],
        "anomaly_threshold": metadata["anomaly_threshold"],
    }
