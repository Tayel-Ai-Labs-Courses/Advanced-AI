import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reliable_ai import (
    run_training_comparison,
    fit_and_persist_artifacts,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "model"


def main():
    print("=" * 60)
    print("TRAINING AND ARTIFACT PERSISTENCE")
    print("=" * 60)

    print("\nRunning model comparison...")
    comparison = run_training_comparison(DATA_DIR)

    selected_name = comparison["selected_model_name"]
    print(f"Selected model: {selected_name}")

    print("\nFitting reliability components and persisting artifacts...")
    result = fit_and_persist_artifacts(comparison, DATA_DIR, MODEL_DIR)

    print(f"\nArtifacts saved to: {MODEL_DIR}")
    print(f"  - risk_model.joblib")
    print(f"  - temperature_scaler.joblib")
    print(f"  - anomaly_model.joblib")
    print(f"  - reference_medians.joblib")
    print(f"  - metadata.json")
    print(f"  - model_comparison.csv")
    print(f"  - threshold_analysis.csv")

    meta = result["metadata"]
    print(f"\nMetadata summary:")
    print(f"  Selected model: {meta['selected_model_name']}")
    print(f"  Macro F1: {meta['selected_test_metrics']['macro_f1']}")
    print(f"  Review threshold: {meta['review_threshold']}")
    print(f"  Anomaly threshold: {meta['anomaly_threshold']}")
    print(f"  Selection rule: {meta['selection_rule']}")

    print(f"\nTraining complete.")


if __name__ == "__main__":
    main()
