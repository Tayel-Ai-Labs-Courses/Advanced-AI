"""Cold-start verification for the Session 5 product bundle."""
from pathlib import Path

from app import MODEL_DIR, load_runtime, predict_request
from reliable_ai import validate_artifact_contract

DEMO_CASES = {
    "ACCEPT_CASE": {
        "month": 1, "temperature_c": 23.6, "rainfall_mm": 72.4, "humidity_pct": 68.5,
        "soil_moisture": 0.4599, "soil_ph": 6.48, "vegetation_index": 0.8958,
        "disease_cases": 41.3, "pest_reports": 28.8,
        "irrigation_capacity": 0.6734, "vulnerability_index": 0.352,
    },
    "LOW_CONFIDENCE_CASE": {
        "month": 3, "temperature_c": 26.9, "rainfall_mm": 18.7, "humidity_pct": 60.9,
        "soil_moisture": 0.3002, "soil_ph": 5.71, "vegetation_index": 0.677,
        "disease_cases": 66.7, "pest_reports": 18.6,
        "irrigation_capacity": 0.6734, "vulnerability_index": 0.352,
    },
    "ANOMALY_CASE": {
        "month": 10, "temperature_c": 12.6, "rainfall_mm": 76.9, "humidity_pct": 37.1,
        "soil_moisture": 0.6067, "soil_ph": 6.33, "vegetation_index": 0.4306,
        "disease_cases": 23.0, "pest_reports": 6.5,
        "irrigation_capacity": 0.6734, "vulnerability_index": 0.352,
    },
    "INVALID_CASE": {
        "month": 13, "temperature_c": 150.0, "rainfall_mm": -1.0, "humidity_pct": 500.0,
        "soil_moisture": 2.0, "soil_ph": 20.0, "vegetation_index": -1.0,
        "disease_cases": 9000.0, "pest_reports": 2000.0,
        "irrigation_capacity": 4.0, "vulnerability_index": -3.0,
    },
}
EXPECTED = {
    "ACCEPT_CASE": "ACCEPT_AI_DECISION",
    "LOW_CONFIDENCE_CASE": "HUMAN_REVIEW_LOW_CONFIDENCE",
    "ANOMALY_CASE": "HUMAN_REVIEW_ANOMALY",
    "INVALID_CASE": "REJECT_INVALID_INPUT",
}


def main():
    print("SESSION 5 COLD-START PRODUCT VERIFICATION")
    print("=" * 50)
    contract = validate_artifact_contract(MODEL_DIR)
    if not contract["ok"]:
        raise SystemExit("Artifact contract failed: " + "; ".join(contract["problems"]))
    print("[PASS] artifact presence + fingerprints + metadata identity")

    artifacts = load_runtime()
    print(f"[PASS] selected model loaded: {artifacts['metadata']['selected_model_name']}")

    for name, request in DEMO_CASES.items():
        result = predict_request(request, artifacts)
        expected = EXPECTED[name]
        actual = result["status"]
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
        print(f"[PASS] {name:20s} -> {actual}")

    print("=" * 50)
    print("PRODUCT VERIFICATION PASSED")
    print("Next: python app.py")
    print("Temporary public link: python app.py --share")


if __name__ == "__main__":
    main()
