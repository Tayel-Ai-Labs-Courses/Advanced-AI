import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reliable_ai import (
    fuse_sources,
    compare_candidates,
    encode_target,
    TemperatureScaler,
    multiclass_brier,
    expected_calibration_error,
    choose_review_threshold,
    fit_reliability_components,
    make_safe_decision,
    MODEL_FEATURES,
    CLASS_NAMES,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def main():
    print("=" * 60)
    print("04 - RELIABILITY AND HUMAN REVIEW")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = fuse_sources(DATA_DIR)
    comparison = compare_candidates(data)

    selected_model = comparison["selected_model"]
    selected_name = comparison["selected_model_name"]
    split = comparison["split"]
    metrics = comparison["metrics"]
    y_test = comparison["y_test"]

    train = split["train"]
    calibration = split["calibration"]
    test = split["test"]
    X_cal = calibration[MODEL_FEATURES]
    X_test = test[MODEL_FEATURES]
    y_cal = encode_target(calibration["risk_label"])

    print(f"\nSelected model: {selected_name}")
    print(f"Train: {train['year'].min()}-{train['year'].max()}")
    print(f"Calibration: {calibration['year'].min()}")
    print(f"Test: {test['year'].max()}")

    # Fit reliability components
    temp_scaler, anomaly_pipeline, anomaly_threshold, ref_medians = \
        fit_reliability_components(selected_model, train, calibration)

    # Raw vs calibrated metrics
    raw_test = selected_model.predict_proba(X_test)
    cal_test = temp_scaler.transform(raw_test)

    raw_ece = expected_calibration_error(y_test, raw_test)
    cal_ece = expected_calibration_error(y_test, cal_test)
    raw_brier = multiclass_brier(y_test, raw_test)
    cal_brier = multiclass_brier(y_test, cal_test)

    print(f"\nReliability metrics (2025 future test):")
    print(f"{'Metric':25s} {'Raw':>10s} {'Calibrated':>12s}")
    print(f"{'-'*50}")
    print(f"{'ECE':25s} {raw_ece:>10.6f} {cal_ece:>12.6f}")
    print(f"{'Brier':25s} {raw_brier:>10.6f} {cal_brier:>12.6f}")
    print(f"{'Temperature':25s} {1.0:>10.4f} {temp_scaler.temperature:>12.4f}")
    cal_improved = cal_ece < raw_ece
    print(f"\nCalibration {'improved' if cal_improved else 'did not improve'} 2025 ECE")
    if not cal_improved:
        print("(Negative calibration evidence is preserved)")

    # Threshold selection
    raw_cal = selected_model.predict_proba(X_cal)
    cal_cal = temp_scaler.transform(raw_cal)
    review_threshold, threshold_table = choose_review_threshold(cal_cal, y_cal)
    threshold_table.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)

    print(f"\nReview threshold: {review_threshold:.3f}")

    # Evaluate on 2025
    y_pred_cal = cal_test.argmax(axis=1)
    accepted_2025 = cal_test.max(axis=1) >= review_threshold
    coverage_2025 = accepted_2025.mean()
    selective_acc_2025 = (y_pred_cal[accepted_2025] == y_test[accepted_2025]).mean() if accepted_2025.sum() > 0 else 0.0
    print(f"\n2025 evaluation:")
    print(f"  Coverage: {coverage_2025:.4f}")
    print(f"  Review rate: {1-coverage_2025:.4f}")
    print(f"  Selective accuracy: {selective_acc_2025:.4f}")

    # Test cases
    print(f"\nRouting test cases:")
    print(f"{'='*60}")

    normal = {
        "month": 5, "temperature_c": 28.5, "rainfall_mm": 45.0, "humidity_pct": 55.0,
        "soil_moisture": 0.45, "soil_ph": 7.2, "vegetation_index": 0.65,
        "disease_cases": 40.0, "pest_reports": 15.0,
        "irrigation_capacity": 0.6, "vulnerability_index": 0.3,
    }
    result = make_safe_decision(
        normal, selected_model, temp_scaler, anomaly_pipeline,
        anomaly_threshold, review_threshold, ref_medians
    )
    print(f"\nNORMAL request:")
    print(f"  Status: {result['status']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']}")

    unusual = {
        "month": 7, "temperature_c": 48.0, "rainfall_mm": 5.0, "humidity_pct": 10.0,
        "soil_moisture": 0.05, "soil_ph": 9.5, "vegetation_index": 0.08,
        "disease_cases": 5.0, "pest_reports": 2.0,
        "irrigation_capacity": 0.1, "vulnerability_index": 0.9,
    }
    result2 = make_safe_decision(
        unusual, selected_model, temp_scaler, anomaly_pipeline,
        anomaly_threshold, review_threshold, ref_medians
    )
    print(f"\nUNUSUAL VALID request:")
    print(f"  Status: {result2['status']}")
    print(f"  Prediction: {result2['prediction']}")
    print(f"  Confidence: {result2['confidence']}")
    print(f"  Anomaly score: {result2['anomaly_score']:.4f}")

    impossible = {
        "month": 13, "temperature_c": 150, "rainfall_mm": -1, "humidity_pct": 500,
        "soil_moisture": 2, "soil_ph": 20, "vegetation_index": -1,
        "disease_cases": 9000, "pest_reports": 2000,
        "irrigation_capacity": 4, "vulnerability_index": -3,
    }
    result3 = make_safe_decision(
        impossible, selected_model, temp_scaler, anomaly_pipeline,
        anomaly_threshold, review_threshold, ref_medians
    )
    print(f"\nIMPOSSIBLE request:")
    print(f"  Status: {result3['status']}")
    print(f"  Prediction: {result3['prediction']}")
    print(f"  Confidence: {result3['confidence']}")
    print(f"  Problems ({len(result3['problems'])}):")
    for p in result3['problems']:
        print(f"    - {p}")


if __name__ == "__main__":
    main()
