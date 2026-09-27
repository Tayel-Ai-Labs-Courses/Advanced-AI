import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reliable_ai import load_artifacts, make_safe_decision

MODEL_DIR = Path(__file__).resolve().parent / "model"


def load_runtime():
    """Load and verify the persisted product bundle from disk."""
    return load_artifacts(MODEL_DIR, verify=True)


def predict_request(measurements: dict, artifacts: dict | None = None) -> dict:
    """Single application boundary used by the UI, smoke checks, and deployment tests."""
    if artifacts is None:
        artifacts = load_runtime()
    return make_safe_decision(
        measurements,
        artifacts["model"],
        artifacts["temperature_scaler"],
        artifacts["anomaly_model"],
        artifacts["anomaly_threshold"],
        artifacts["review_threshold"],
        artifacts["reference_medians"],
        model_version=artifacts["metadata"]["selected_model_name"],
        policy_version=artifacts["metadata"].get("policy_version", "reliability_policy_v2"),
    )


def smoke_test(artifacts: dict) -> dict:
    test = {
        "month": 6,
        "temperature_c": 30.0,
        "rainfall_mm": 50.0,
        "humidity_pct": 60.0,
        "soil_moisture": 0.5,
        "soil_ph": 7.0,
        "vegetation_index": 0.6,
        "disease_cases": 30.0,
        "pest_reports": 10.0,
        "irrigation_capacity": 0.5,
        "vulnerability_index": 0.4,
    }
    return predict_request(test, artifacts)


def result_to_markdown(result: dict) -> str:
    status = result["status"]
    lines = [f"## {status}", f"**Reason:** `{result.get('reason', 'n/a')}`"]
    if result.get("prediction") is not None:
        lines.append(f"**Predicted risk:** **{result['prediction']}**")
    if result.get("confidence") is not None:
        lines.append(
            f"**Calibrated confidence:** {result['confidence']:.4f}  "
            f"(automatic-action threshold: {result['review_threshold']:.3f})"
        )
    if result.get("anomaly_score") is not None:
        lines.append(
            f"**Anomaly score:** {result['anomaly_score']:.4f}  "
            f"(review boundary: {result['anomaly_threshold']:.4f})"
        )
    if result.get("problems"):
        lines.append("### Input-contract problems")
        lines.extend(f"- {problem}" for problem in result["problems"])
    if result.get("explanation"):
        lines.append("### Evidence / top factors")
        for item in result["explanation"]:
            if isinstance(item, dict):
                lines.append(f"- `{item['feature']}`: effect {item['effect']:+.4f}")
            else:
                lines.append(f"- {item}")
    lines.append(
        f"**Model:** `{result.get('model_version')}` &nbsp; | &nbsp; "
        f"**Policy:** `{result.get('policy_version')}`"
    )
    return "\n\n".join(lines)


def get_app():
    try:
        import gradio as gr
    except ImportError:
        return None

    artifacts = load_runtime()

    def predict(
        month,
        temp,
        rainfall,
        humidity,
        soil_moisture,
        soil_ph,
        vegetation_index,
        disease_cases,
        pest_reports,
        irrigation_capacity,
        vulnerability_index,
    ):
        measurements = {
            "month": month,
            "temperature_c": temp,
            "rainfall_mm": rainfall,
            "humidity_pct": humidity,
            "soil_moisture": soil_moisture,
            "soil_ph": soil_ph,
            "vegetation_index": vegetation_index,
            "disease_cases": disease_cases,
            "pest_reports": pest_reports,
            "irrigation_capacity": irrigation_capacity,
            "vulnerability_index": vulnerability_index,
        }
        return result_to_markdown(predict_request(measurements, artifacts))

    examples = [
        [1, 23.6, 72.4, 68.5, 0.4599, 6.48, 0.8958, 41.3, 28.8, 0.6734, 0.352],
        [3, 26.9, 18.7, 60.9, 0.3002, 5.71, 0.6770, 66.7, 18.6, 0.6734, 0.352],
        [10, 12.6, 76.9, 37.1, 0.6067, 6.33, 0.4306, 23.0, 6.5, 0.6734, 0.352],
        [13, 150.0, -1.0, 500.0, 2.0, 20.0, -1.0, 9000.0, 2000.0, 4.0, -3.0],
    ]

    interface = gr.Interface(
        fn=predict,
        title="Reliable AI Risk Intelligence",
        description=(
            "The same persisted Session 5 decision contract is used here: input validation -> "
            "anomaly review -> calibrated confidence policy -> accepted AI decision or human review."
        ),
        inputs=[
            gr.Number(value=6, label="Month", minimum=1, maximum=12, step=1),
            gr.Number(value=30.0, label="Temperature (C)"),
            gr.Number(value=50.0, label="Rainfall (mm)"),
            gr.Number(value=60.0, label="Humidity (%)"),
            gr.Number(value=0.5, label="Soil Moisture"),
            gr.Number(value=7.0, label="Soil pH"),
            gr.Number(value=0.6, label="Vegetation Index"),
            gr.Number(value=30.0, label="Disease Cases"),
            gr.Number(value=10.0, label="Pest Reports"),
            gr.Number(value=0.5, label="Irrigation Capacity"),
            gr.Number(value=0.4, label="Vulnerability Index"),
        ],
        outputs=gr.Markdown(label="Decision evidence"),
        examples=examples,
        cache_examples=False,
        flagging_mode="never",
    )
    return interface


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Reliable AI Risk Intelligence application.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Request a temporary public Gradio share link. Internet access is required.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Server bind address. Defaults to 127.0.0.1 locally and 0.0.0.0 on Hugging Face Spaces.",
    )
    parser.add_argument("--port", type=int, default=None, help="Optional server port, e.g. 7860.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a local browser tab.")
    parser.add_argument("--smoke-only", action="store_true", help="Verify loading and one request without launching the UI.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 68)
    print("RELIABLE AI RISK INTELLIGENCE APPLICATION")
    print("=" * 68)

    print("\nLoading and validating the persisted artifact bundle...")
    artifacts = load_runtime()
    meta = artifacts["metadata"]
    print(f"  Model:             {meta['selected_model_name']}")
    print(f"  Macro F1 (2025):   {meta['selected_test_metrics']['macro_f1']:.4f}")
    print(f"  Temperature:       {meta['temperature']:.4f}")
    print(f"  Review threshold:  {artifacts['review_threshold']:.3f}")
    print(f"  Anomaly threshold: {artifacts['anomaly_threshold']:.6f}")
    print("  Artifact contract: PASSED")

    result = smoke_test(artifacts)
    print(f"\nSmoke status: {result['status']}")

    if args.smoke_only:
        return

    interface = get_app()
    if interface is None:
        raise RuntimeError("Gradio is not installed. Run: pip install -r requirements.txt")

    default_host = "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"
    host = args.host or default_host
    print(f"\nLaunching Gradio on {host}:{args.port or 7860} ...")
    if args.share:
        print("A temporary public share link will be requested from Gradio.")

    interface.launch(
        server_name=host,
        server_port=args.port,
        share=args.share,
        inbrowser=(not args.no_browser and host in {"127.0.0.1", "localhost"}),
        show_error=True,
    )


if __name__ == "__main__":
    main()
