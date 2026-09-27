# Session 5 deployment paths

The same `src/` package can be used in three ways. No model logic should be copied into a second UI file.

## 1. Local application - fastest reliable path

```bash
cd src
python -m pip install -r requirements.txt
python train_model.py
python verify_product.py
python app.py
```

The app is served on the local machine. The terminal prints the local URL.

## 2. Temporary public internet link - fastest live demo

```bash
cd src
python app.py --share
```

Gradio requests a temporary public share link. This is appropriate for a short demonstration, not a permanent production endpoint. Internet access is required and the link lifetime is controlled by the Gradio sharing service.

## 3. Hugging Face Spaces - persistent hosted Gradio demo

A Gradio Space can host the application on a free CPU tier when that tier is available within the account/platform quota.

1. Create a new **Gradio** Space.
2. Upload the **contents of `src/`** so `app.py`, `requirements.txt`, `reliable_ai/`, `model/`, and `data/` are at the Space repository root.
3. Commit the files. The Space installs `requirements.txt` and launches `app.py`.
4. Open the public Space URL and run the four reference cases.
5. Keep `verify_product.py` in the repository so the artifact contract can be checked before a release.

For a classroom deployment, the already-generated `model/` artifacts can be committed. If training is intentionally repeated, run `python train_model.py` first and commit the newly matched bundle and `artifact_manifest.json` together.

## Deployment acceptance checks

- `python verify_product.py` passes.
- `python app.py --smoke-only` passes.
- `pytest -q` passes.
- The four routes are reproduced: ACCEPT, low-confidence review, anomaly review, and invalid-input rejection.
- `artifact_manifest.json` fingerprints match the deployed bundle.
