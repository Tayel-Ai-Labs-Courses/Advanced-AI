import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

from .constants import (
    RANDOM_STATE,
    KEYS,
    CLASS_NAMES,
    LABEL_TO_ID,
    MODEL_FEATURES,
    PROHIBITED_FEATURES,
)


def chronological_split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = data[data["year"] <= 2023].copy()
    calibration = data[data["year"] == 2024].copy()
    test = data[data["year"] == 2025].copy()

    assert not train.empty, "Train partition is empty"
    assert not calibration.empty, "Calibration partition is empty"
    assert not test.empty, "Test partition is empty"

    return {"train": train, "calibration": calibration, "test": test}


def encode_target(y: pd.Series) -> np.ndarray:
    encoded = y.map(LABEL_TO_ID)
    unknown = encoded.isna()
    if unknown.any():
        raise ValueError(f"Unknown labels: {y[unknown].unique()}")
    return encoded.values.astype(int)


def logistic_pipeline() -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, MODEL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    return pipeline


def advanced_model_builder(features: list[str] | None = None) -> Pipeline:
    if features is None:
        features = MODEL_FEATURES
    try:
        from xgboost import XGBClassifier
        classifier = XGBClassifier(
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
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=220,
            max_leaf_nodes=15,
            l2_regularization=0.15,
            random_state=RANDOM_STATE,
        )
        used_fallback = True

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, features),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    pipeline.used_fallback = used_fallback
    return pipeline


def metric_row(
    name: str,
    model,
    X: pd.DataFrame,
    y_true: np.ndarray,
    features: list[str] | None = None,
) -> dict:
    if features is None:
        features = MODEL_FEATURES
    y_pred = model.predict(X[features])
    proba = model.predict_proba(X[features])

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "log_loss": round(log_loss(y_true, proba), 4),
    }


LEGITIMATE_MODEL_NAMES = [
    "dummy",
    "logistic",
    "advanced_correct",
]


def select_legitimate_model(
    metrics: pd.DataFrame,
    models: dict[str, tuple],
) -> tuple[str, object, list[str]]:
    legitimate = metrics[metrics["model"].isin(LEGITIMATE_MODEL_NAMES)].copy()
    sorted_legit = legitimate.sort_values(
        by=["macro_f1", "log_loss", "model"],
        ascending=[False, True, True],
    )
    best_row = sorted_legit.iloc[0]
    selected_name = best_row["model"]
    selected_model = models[selected_name][0]
    selected_features = models[selected_name][1]
    return selected_name, selected_model, selected_features


def compare_candidates(
    data: pd.DataFrame,
    data_dir: str | None = None,
) -> dict:
    split = chronological_split(data)
    train = split["train"]
    test = split["test"]

    X_train = train[MODEL_FEATURES]
    y_train = encode_target(train["risk_label"])
    X_test = test[MODEL_FEATURES]
    y_test = encode_target(test["risk_label"])

    models = {}

    # Dummy
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    models["dummy"] = (dummy, MODEL_FEATURES)

    # Logistic
    logistic = logistic_pipeline()
    logistic.fit(X_train, y_train)
    models["logistic"] = (logistic, MODEL_FEATURES)

    # Advanced correct
    advanced = advanced_model_builder(features=MODEL_FEATURES)
    advanced.fit(X_train, y_train)
    models["advanced_correct"] = (advanced, MODEL_FEATURES)

    # Leaky
    leaky_features = MODEL_FEATURES + ["future_loss_index"]
    leaky = advanced_model_builder(features=leaky_features)
    leaky.fit(train[leaky_features], y_train)
    models["advanced_leaky"] = (leaky, leaky_features)

    # Metrics
    rows = []
    for name, (model, feats) in models.items():
        row = metric_row(name, model, test, y_test, feats)
        rows.append(row)
    metrics = pd.DataFrame(rows)

    # Select
    selected_name, selected_model, selected_features = select_legitimate_model(metrics, models)

    return {
        "metrics": metrics,
        "models": models,
        "selected_model_name": selected_name,
        "selected_model": selected_model,
        "selected_features": selected_features,
        "split": split,
        "y_test": y_test,
    }
