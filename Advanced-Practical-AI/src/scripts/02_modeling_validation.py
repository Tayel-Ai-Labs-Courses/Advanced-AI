import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reliable_ai import (
    fuse_sources,
    compare_candidates,
    MODEL_FEATURES,
    PROHIBITED_FEATURES,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def main():
    print("=" * 60)
    print("02 - MODELING AND VALIDATION")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = fuse_sources(DATA_DIR)
    comparison = compare_candidates(data)

    metrics = comparison["metrics"]
    print("\nModel comparison (2025 future test):")
    print(metrics.to_string(index=False))

    # Feature availability audit
    print("\nFeature availability audit:")
    rows = []
    for feat in MODEL_FEATURES:
        rows.append({
            "feature": feat,
            "model_role": "approved",
            "prediction_time_availability": "available",
            "eligibility": "eligible",
            "reason": "current measurement",
        })
    for feat in PROHIBITED_FEATURES:
        rows.append({
            "feature": feat,
            "model_role": "prohibited",
            "prediction_time_availability": "post_outcome_only" if feat == "future_loss_index" else "post_response",
            "eligibility": "prohibited",
            "reason": f"{feat} is not available at prediction time",
        })
    audit_df = __import__("pandas").DataFrame(rows)
    audit_df.to_csv(REPORTS_DIR / "feature_availability_audit.csv", index=False)
    print(audit_df.to_string(index=False))

    # Selection record
    selected_name = comparison["selected_model_name"]
    selected_row = metrics[metrics["model"] == selected_name].iloc[0]
    leaky_row = metrics[metrics["model"] == "advanced_leaky"].iloc[0]

    record = __import__("pandas").DataFrame([{
        "selected_model_name": selected_name,
        "selected_macro_f1": selected_row["macro_f1"],
        "selection_rule": "highest legitimate macro_f1, then lowest log_loss",
        "excluded_candidate": "advanced_leaky",
        "excluded_reason": "uses future_loss_index (post-outcome proxy, unavailable at prediction time)",
        "train_years": "2019-2023",
        "calibration_year": 2024,
        "test_year": 2025,
    }])
    record.to_csv(REPORTS_DIR / "model_selection_record.csv", index=False)
    print(f"\nSelection record:")
    print(record.to_string(index=False))

    print(f"\nSelected model: {selected_name}")
    print(f"Leaky macro F1: {leaky_row['macro_f1']:.4f} (EXCLUDED)")


if __name__ == "__main__":
    main()
