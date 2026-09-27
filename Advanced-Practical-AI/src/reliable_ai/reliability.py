import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

from .constants import (
    RANDOM_STATE,
    KEYS,
    CLASS_NAMES,
    MODEL_FEATURES,
    MONITOR_FEATURES,
    RAW_FEATURES,
    VALID_RANGES,
)
from .features import build_feature_row
from .explainability import local_occlusion_explanation


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> None:
        probabilities = np.clip(probabilities, 1e-9, 1.0)

        def objective(log_t: float) -> float:
            t = float(np.exp(log_t))
            calibrated = softmax(np.log(probabilities) / t, axis=1)
            return log_loss(y_true, calibrated)

        result = minimize_scalar(
            objective,
            bounds=(-2.5, 2.5),
            method="bounded",
        )
        self.temperature = float(np.exp(result.x))

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(probabilities, 1e-9, 1.0)
        return softmax(np.log(probabilities) / self.temperature, axis=1)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    n = len(y_true)
    n_classes = probabilities.shape[1]
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += np.abs(bin_acc - bin_conf) * in_bin.sum()
    return float(ece / len(y_true))


def choose_review_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    target_selective_accuracy: float = 0.82,
    minimum_coverage: float = 0.25,
    maximum_review_rate: float = 0.30,
) -> tuple[float, pd.DataFrame]:
    thresholds = np.arange(0.50, 0.975, 0.025)
    rows = []
    for threshold in thresholds:
        accepted = probabilities.max(axis=1) >= threshold
        coverage = accepted.mean()
        if coverage > 0:
            selective_acc = (probabilities.argmax(axis=1)[accepted] == y_true[accepted]).mean()
        else:
            selective_acc = 0.0
        review_rate = 1.0 - coverage
        rows.append({
            "threshold": round(threshold, 3),
            "coverage": round(coverage, 4),
            "review_rate": round(review_rate, 4),
            "selective_accuracy": round(selective_acc, 4),
        })
    table = pd.DataFrame(rows)

    feasible = table[(table["coverage"] >= minimum_coverage) &
                     (table["selective_accuracy"] >= target_selective_accuracy) &
                     (table["review_rate"] <= maximum_review_rate)]
    if not feasible.empty:
        chosen = feasible.sort_values(
            by=["coverage", "threshold"],
            ascending=[False, True],
        ).iloc[0]
    else:
        fallback = table[table["coverage"] >= minimum_coverage].copy()
        if not fallback.empty:
            chosen = fallback.sort_values(
                by=["selective_accuracy", "coverage"],
                ascending=[False, False],
            ).iloc[0]
        else:
            chosen = table.sort_values(
                by=["selective_accuracy", "coverage"],
                ascending=[False, False],
            ).iloc[0]
    return float(chosen["threshold"]), table


FIXED_MONTH = 1
FIXED_YEAR = 2025


def validate_measurements(measurements: dict) -> list[str]:
    problems = []
    month = measurements.get("month")
    if month is None:
        problems.append("month is required")
    else:
        try:
            m = int(month)
            if m < 1 or m > 12:
                problems.append(f"month must be 1-12, got {month}")
        except (ValueError, TypeError):
            problems.append(f"month must be numeric, got {month}")

    for feature, (lo, hi) in VALID_RANGES.items():
        val = measurements.get(feature)
        if val is None:
            problems.append(f"{feature} is required")
            continue
        try:
            v = float(val)
        except (ValueError, TypeError):
            problems.append(f"{feature} must be numeric, got {val}")
            continue
        if not np.isfinite(v):
            problems.append(f"{feature} must be finite, got {val}")
            continue
        if v < lo or v > hi:
            problems.append(f"{feature} value {v} outside valid range [{lo}, {hi}]")

    return problems


def make_safe_decision(
    measurements: dict,
    model: object,
    temperature_scaler: TemperatureScaler,
    anomaly_model: Pipeline,
    anomaly_threshold: float,
    review_threshold: float,
    reference_medians: pd.Series | None = None,
    model_version: str | None = None,
    policy_version: str = "reliability_policy_v2",
) -> dict:
    problems = validate_measurements(measurements)
    if problems:
        return {
            "status": "REJECT_INVALID_INPUT",
            "reason": "input_contract_failed",
            "prediction": None,
            "confidence": None,
            "review_threshold": review_threshold,
            "anomaly_score": None,
            "anomaly_threshold": anomaly_threshold,
            "problems": problems,
            "explanation": [],
            "model_version": model_version,
            "policy_version": policy_version,
        }

    feature_row = build_feature_row(measurements)
    monitor_row = feature_row[[c for c in MONITOR_FEATURES if c in feature_row.columns]]

    anomaly_score = float(anomaly_model.decision_function(monitor_row)[0])
    if anomaly_score < anomaly_threshold:
        return {
            "status": "HUMAN_REVIEW_ANOMALY",
            "reason": "valid_but_out_of_distribution",
            "prediction": None,
            "confidence": None,
            "review_threshold": review_threshold,
            "anomaly_score": anomaly_score,
            "anomaly_threshold": anomaly_threshold,
            "problems": [],
            "explanation": ["Anomaly score below threshold"],
            "model_version": model_version,
            "policy_version": policy_version,
        }

    raw_proba = model.predict_proba(feature_row[MODEL_FEATURES])
    calibrated_proba = temperature_scaler.transform(raw_proba)
    confidence = float(calibrated_proba.max())
    prediction = int(calibrated_proba.argmax())

    if confidence < review_threshold:
        return {
            "status": "HUMAN_REVIEW_LOW_CONFIDENCE",
            "reason": "confidence_below_declared_policy",
            "prediction": CLASS_NAMES[prediction],
            "confidence": confidence,
            "review_threshold": review_threshold,
            "anomaly_score": anomaly_score,
            "anomaly_threshold": anomaly_threshold,
            "problems": [],
            "explanation": [f"Confidence {confidence:.4f} below threshold {review_threshold}"],
            "model_version": model_version,
            "policy_version": policy_version,
        }

    # Local explanation
    if reference_medians is not None:
        explanation = local_occlusion_explanation(
            model,
            feature_row,
            lambda x: temperature_scaler.transform(model.predict_proba(x[MODEL_FEATURES])),
            reference_medians,
            prediction,
            top_k=4,
        )
    else:
        explanation = []

    return {
        "status": "ACCEPT_AI_DECISION",
        "reason": "valid_familiar_and_confident",
        "prediction": CLASS_NAMES[prediction],
        "confidence": confidence,
        "review_threshold": review_threshold,
        "anomaly_score": anomaly_score,
        "anomaly_threshold": anomaly_threshold,
        "problems": [],
        "explanation": explanation,
        "model_version": model_version,
        "policy_version": policy_version,
    }


def fit_reliability_components(model, train, calibration):
    from .modeling import encode_target

    X_train = train[MODEL_FEATURES]
    y_train = encode_target(train["risk_label"])
    X_cal = calibration[MODEL_FEATURES]
    y_cal = encode_target(calibration["risk_label"])

    # Temperature scaling
    raw_cal = model.predict_proba(X_cal)
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(raw_cal, y_cal)

    # Anomaly model
    anomaly_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("forest", IsolationForest(
            n_estimators=250,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])
    monitor_cols = [c for c in MONITOR_FEATURES if c in X_train.columns]
    anomaly_pipeline.fit(X_train[monitor_cols])
    cal_scores = anomaly_pipeline.decision_function(X_cal[monitor_cols])
    anomaly_threshold = float(np.quantile(cal_scores, 0.02))

    # Reference medians
    reference_medians = X_train[MODEL_FEATURES].median()

    return temp_scaler, anomaly_pipeline, anomaly_threshold, reference_medians
