import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .constants import RANDOM_STATE, MODEL_FEATURES


def global_permutation_importance(
    model: object,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 10,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X[MODEL_FEATURES],
        y,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    table = table.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return table


def local_occlusion_explanation(
    model: object,
    sample: pd.DataFrame,
    calibrated_predict: callable,
    reference: pd.Series,
    class_id: int,
    top_k: int = 6,
) -> list[dict]:
    base_proba = calibrated_predict(sample[MODEL_FEATURES])[0, class_id]
    effects = []
    for feature in MODEL_FEATURES:
        altered = sample[MODEL_FEATURES].copy()
        altered[feature] = reference.get(feature, 0.0)
        altered_proba = calibrated_predict(altered)[0, class_id]
        effect = float(base_proba - altered_proba)
        effects.append({
            "feature": feature,
            "value": float(sample[feature].iloc[0]),
            "effect": round(effect, 6),
            "abs_effect": round(abs(effect), 6),
        })
    effects.sort(key=lambda x: x["abs_effect"], reverse=True)
    return effects[:top_k]


def tree_shap_explanation(
    tree_model: object,
    X: pd.DataFrame,
    max_samples: int = 30,
):
    import shap
    X_sample = X.head(max_samples)
    explainer = shap.TreeExplainer(tree_model)
    explanation = explainer(X_sample)
    return explanation


def predicted_class_slice(
    explanation,
    sample_index: int,
    class_id: int,
) -> np.ndarray:
    values = explanation.values
    if values.ndim == 3:
        return values[sample_index, :, class_id]
    elif values.ndim == 2:
        n_features = values.shape[1]
        if class_id < values.shape[0]:
            return values[class_id]
        return values[sample_index]
    return values


def explanation_table(
    local_explanation: np.ndarray,
    sample: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    if features is None:
        features = MODEL_FEATURES
    rows = []
    for i, feature in enumerate(features):
        rows.append({
            "feature": feature,
            "value": float(sample[feature].iloc[0]) if feature in sample.columns else 0.0,
            "shap_effect": round(float(local_explanation[i]), 6),
            "abs_effect": round(float(abs(local_explanation[i])), 6),
        })
    rows.sort(key=lambda x: x["abs_effect"], reverse=True)
    return pd.DataFrame(rows)
