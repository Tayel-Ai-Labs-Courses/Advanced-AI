import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reliable_ai import (
    fuse_sources,
    compare_candidates,
    encode_target,
    global_permutation_importance,
    local_occlusion_explanation,
    tree_shap_explanation,
    predicted_class_slice,
    explanation_table,
    TemperatureScaler,
    MODEL_FEATURES,
    CLASS_NAMES,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def main():
    print("=" * 60)
    print("03 - EXPLAINABILITY AND DEBUGGING")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = fuse_sources(DATA_DIR)
    comparison = compare_candidates(data)

    selected_name = comparison["selected_model_name"]
    selected_model = comparison["selected_model"]
    split = comparison["split"]
    metrics = comparison["metrics"]
    y_test = comparison["y_test"]

    train = split["train"]
    test = split["test"]
    calibration = split["calibration"]

    print(f"\nSelected model: {selected_name}")

    # Fit temperature scaler for calibrated predictions
    X_cal = calibration[MODEL_FEATURES]
    y_cal = encode_target(calibration["risk_label"])
    raw_cal = selected_model.predict_proba(X_cal)
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(raw_cal, y_cal)

    def calibrated_predict(X):
        return temp_scaler.transform(selected_model.predict_proba(X[MODEL_FEATURES]))

    # Global permutation importance
    print("\nGlobal permutation importance (selected product model):")
    X_test = test[MODEL_FEATURES]
    importance = global_permutation_importance(selected_model, X_test, y_test)
    importance.to_csv(REPORTS_DIR / "selected_model_permutation_importance.csv", index=False)
    print(importance.head(8).to_string(index=False))

    # Local cases
    print("\nLocal cases:")
    X_test_full = test[MODEL_FEATURES]
    y_pred = selected_model.predict(X_test_full)
    ref_medians = train[MODEL_FEATURES].median()

    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = y_pred == class_id
        if mask.sum() == 0:
            print(f"  {class_name}: no predictions available")
            continue
        idx = mask.argmax()
        sample = X_test_full.iloc[[idx]]
        actual = test["risk_label"].iloc[idx]
        print(f"\n  Case: {class_name} (actual: {actual})")
        effects = local_occlusion_explanation(
            selected_model, sample, calibrated_predict, ref_medians, class_id, top_k=4
        )
        for e in effects:
            print(f"    {e['feature']:25s} value={e['value']:8.4f}  effect={e['effect']:.6f}")

    # Error case
    print("\nError case analysis:")
    wrong_mask = y_pred != y_test
    if wrong_mask.sum() > 0:
        first_wrong = np.where(wrong_mask)[0][0]
        sample = X_test_full.iloc[[first_wrong]]
        actual_class = CLASS_NAMES[y_test[first_wrong]]
        pred_class = CLASS_NAMES[y_pred[first_wrong]]
        print(f"  Row index: {first_wrong}")
        print(f"  Actual: {actual_class}")
        print(f"  Predicted: {pred_class}")

        effects = local_occlusion_explanation(
            selected_model, sample, calibrated_predict, ref_medians,
            int(y_pred[first_wrong]), top_k=6
        )
        print(f"  Top local factors for predicted class {pred_class}:")
        for e in effects:
            print(f"    {e['feature']:25s} value={e['value']:8.4f}  effect={e['effect']:.6f}")

        # Debugging hypothesis
        hypothesis = {
            "case_identifier": f"test_row_{first_wrong}",
            "actual_class": actual_class,
            "predicted_class": pred_class,
            "hypothesis": "The model may be underrepresenting interaction between drought_index and water_balance near the MEDIUM/HIGH boundary",
            "evidence_summary": f"drought_index={sample['drought_index'].values[0]:.4f}, water_balance={sample['water_balance'].values[0]:.4f}",
            "next_test": "Add one interaction candidate and compare future-test macro F1 without changing the 2025 split",
        }
        hyp_df = pd.DataFrame([hypothesis])
        hyp_df.to_csv(REPORTS_DIR / "debugging_hypothesis.csv", index=False)
        print(f"\n  Debugging hypothesis saved")
    else:
        print("  No wrong predictions found on test set")

    # Tree SHAP demonstration
    print("\nTree SHAP demonstration:")
    tree_name = "advanced_correct"
    if tree_name in comparison["models"]:
        tree_model = comparison["models"][tree_name][0]
        tree_impl = tree_model.named_steps["classifier"]
        print(f"  Using shap.TreeExplainer on {tree_name}")

        try:
            explanation = tree_shap_explanation(tree_impl, X_test, max_samples=30)
            print(f"  Explanation shape: {explanation.values.shape}")

            # Save one case
            slice_vals = predicted_class_slice(explanation, 0, 0)
            sample_0 = X_test.iloc[[0]]
            tab = explanation_table(slice_vals, sample_0)
            tab.to_csv(REPORTS_DIR / "tree_shap_case_0.csv", index=False)
            print(f"  Tree SHAP case 0 saved")
            print(f"  Top 3 features:")
            for _, row in tab.head(3).iterrows():
                print(f"    {row['feature']:25s} shap={row['shap_effect']:.6f}")

        except Exception as e:
            print(f"  SHAP demonstration failed: {e}")
    else:
        print(f"  {tree_name} not available")

    print(f"\nExplainability script completed in {time.process_time():.2f}s")


if __name__ == "__main__":
    main()
