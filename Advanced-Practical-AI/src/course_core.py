"""Backward-compatible course API used by the classroom notebooks.

The course was refactored into the :mod:`reliable_ai` package.  Some earlier
notebooks still import a single ``course_core`` module and expect the original
function signatures and pipeline step names.  This facade preserves that
notebook contract while delegating data preparation and shared constants to the
current package.  No existing ``reliable_ai`` behavior is modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reliable_ai import (
    CLASS_NAMES,
    ENGINEERED_FEATURES,
    KEYS,
    LABEL_TO_ID,
    MODEL_FEATURES,
    MONITOR_FEATURES,
    PROHIBITED_FEATURES,
    RANDOM_STATE,
    RAW_FEATURES,
    VALID_RANGES,
)
from reliable_ai.data import (
    audit_sources,
    build_feature_registry,
    fuse_sources as _fuse_sources,
    generate_raw_sources as _generate_raw_sources,
    load_sources,
)
from reliable_ai.modeling import encode_target


LEGITIMATE_MODEL_NAMES = ["dummy", "logistic", "advanced_correct"]


def _resolve_data_dir(path: str | Path) -> Path:
    """Return the common data directory from either ``data`` or ``data/raw``."""
    resolved = Path(path)
    return resolved.parent if resolved.name == "raw" else resolved


def generate_raw_sources(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Generate deterministic sources using the legacy ``raw_dir`` signature.

    The modern function accepts the parent data directory and creates a ``raw``
    child.  Legacy notebooks pass the raw directory itself, so that difference
    is normalized here.
    """
    return _generate_raw_sources(_resolve_data_dir(raw_dir))


def fuse_sources(
    raw_dir: str | Path,
    processed_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuse sources and return ``(modeling_table, source_audit)``.

    This preserves the original two-return-value notebook API while the modern
    :func:`reliable_ai.data.fuse_sources` continues to return only the fused
    dataframe.
    """
    data_dir = _resolve_data_dir(raw_dir)
    processed = Path(processed_dir) if processed_dir is not None else data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    data = _fuse_sources(data_dir)
    sources = load_sources(data_dir)
    audit = audit_sources(sources)

    data.to_csv(processed / "modeling_table.csv", index=False)
    audit.to_csv(processed / "source_audit.csv", index=False)
    build_feature_registry().to_csv(processed / "feature_registry.csv", index=False)
    return data, audit


def _numeric_preprocessor(features: list[str]) -> ColumnTransformer:
    """Create the leakage-safe numeric preprocessing contract."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric, features)],
        remainder="drop",
    )


def _logistic_pipeline(features: list[str]) -> Pipeline:
    """Build the legacy-named linear pipeline (``prepare`` then ``model``)."""
    return Pipeline(
        steps=[
            ("prepare", _numeric_preprocessor(features)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _advanced_pipeline(features: list[str]) -> Pipeline:
    """Build the legacy-named nonlinear pipeline with a CPU-safe fallback."""
    try:
        from xgboost import XGBClassifier

        estimator: Any = XGBClassifier(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.055,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            device="cpu",
        )
        used_fallback = False
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=220,
            max_leaf_nodes=15,
            l2_regularization=0.15,
            random_state=RANDOM_STATE,
        )
        used_fallback = True

    pipeline = Pipeline(
        steps=[
            ("prepare", _numeric_preprocessor(features)),
            ("model", estimator),
        ]
    )
    pipeline.used_fallback = used_fallback
    return pipeline


def _metric_row(
    name: str,
    model: Any,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    features: list[str],
) -> dict[str, float | str]:
    """Calculate the shared comparison metrics for one candidate."""
    predictions = model.predict(frame[features])
    probabilities = model.predict_proba(frame[features])
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 4),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro")), 4),
        "log_loss": round(float(log_loss(y_true, probabilities)), 4),
    }


def train_model_comparison(data: pd.DataFrame) -> dict[str, Any]:
    """Train the four Session-2 candidates using the original return contract."""
    train = data[data["year"] <= 2023].copy()
    calibration = data[data["year"] == 2024].copy()
    test = data[data["year"] == 2025].copy()

    if train.empty or calibration.empty or test.empty:
        raise ValueError("The chronological 2019-2023 / 2024 / 2025 split is incomplete.")

    y_train = encode_target(train["risk_label"])
    y_test = encode_target(test["risk_label"])

    models: dict[str, tuple[Any, list[str]]] = {}

    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(train[MODEL_FEATURES], y_train)
    models["dummy"] = (dummy, list(MODEL_FEATURES))

    logistic = _logistic_pipeline(list(MODEL_FEATURES))
    logistic.fit(train[MODEL_FEATURES], y_train)
    models["logistic"] = (logistic, list(MODEL_FEATURES))

    advanced_correct = _advanced_pipeline(list(MODEL_FEATURES))
    advanced_correct.fit(train[MODEL_FEATURES], y_train)
    models["advanced_correct"] = (advanced_correct, list(MODEL_FEATURES))

    leaky_features = list(MODEL_FEATURES) + ["future_loss_index"]
    advanced_leaky = _advanced_pipeline(leaky_features)
    advanced_leaky.fit(train[leaky_features], y_train)
    models["advanced_leaky"] = (advanced_leaky, leaky_features)

    metrics = pd.DataFrame(
        [
            _metric_row(name, model, test, y_test, features)
            for name, (model, features) in models.items()
        ]
    )

    legitimate = metrics[metrics["model"].isin(LEGITIMATE_MODEL_NAMES)].copy()
    selected_row = legitimate.sort_values(
        ["macro_f1", "log_loss", "model"],
        ascending=[False, True, True],
    ).iloc[0]
    selected_name = str(selected_row["model"])
    selected_model, selected_features = models[selected_name]

    return {
        "splits": {"train": train, "calibration": calibration, "test": test},
        "split": {"train": train, "calibration": calibration, "test": test},
        "metrics": metrics,
        "models": models,
        "selected_model_name": selected_name,
        "selected_model": selected_model,
        "selected_features": selected_features,
        "y_test": y_test,
    }


# Modern naming aliases are also exposed for mixed-version notebooks.
compare_candidates = train_model_comparison
chronological_split = lambda data: {
    "train": data[data["year"] <= 2023].copy(),
    "calibration": data[data["year"] == 2024].copy(),
    "test": data[data["year"] == 2025].copy(),
}


__all__ = [
    "RANDOM_STATE",
    "KEYS",
    "CLASS_NAMES",
    "LABEL_TO_ID",
    "RAW_FEATURES",
    "ENGINEERED_FEATURES",
    "MODEL_FEATURES",
    "MONITOR_FEATURES",
    "PROHIBITED_FEATURES",
    "VALID_RANGES",
    "LEGITIMATE_MODEL_NAMES",
    "encode_target",
    "generate_raw_sources",
    "fuse_sources",
    "train_model_comparison",
    "compare_candidates",
    "chronological_split",
]
