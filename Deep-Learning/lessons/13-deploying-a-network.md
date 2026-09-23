# Lesson 13 — Deploying a Network

**Goal:** a model that runs correctly outside your notebook.

## What you will learn

- Saving and loading properly
- Inference that is fast and correct
- TorchScript and ONNX
- What to monitor

---

## Save the state dict, and the recipe

```python
import torch
import torch.nn as nn
import json

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))

torch.save(model.state_dict(), "model.pt")

metadata = {
    "architecture": "Linear(10,32)-ReLU-Linear(32,2)",
    "input_features": 10,
    "classes": ["negative", "positive"],
    "torch_version": torch.__version__,
    "normalisation": {"mean": [0.0] * 10, "std": [1.0] * 10},
    "threshold": 0.5,
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

loaded = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
loaded.load_state_dict(torch.load("model.pt", map_location="cpu"))
loaded.eval()

x = torch.randn(3, 10)
print("identical outputs:", bool(torch.allclose(model(x), loaded(x))))
print("torch:", torch.__version__)
```

```text
identical outputs: True
torch: 2.12.0
```

`state_dict` saves the weights, not the class. You need the code that builds
the architecture to load them — which is correct, because pickling the whole
object ties the file to your exact file layout and breaks on the next
refactor.

The metadata is not optional. The weights alone do not tell the next person
the input size, the normalisation constants or the class order — and **a wrong
class order silently inverts every prediction.**

For a checkpoint you can resume from, save more:

```python
checkpoint = {
    "epoch": 12,
    "model": model.state_dict(),
    "optimiser": optimiser.state_dict(),
    "best_val_loss": 0.184,
}
torch.save(checkpoint, "checkpoint.pt")
```

---

## Inference

```python
import torch
import torch.nn as nn

class Predictor:
    """Load once, predict many times."""

    def __init__(self, weights_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, features, threshold=0.5):
        x = torch.as_tensor(features, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != 10:
            raise ValueError(f"expected (n, 10), got {tuple(x.shape)}")

        probabilities = self.model(x.to(self.device)).softmax(dim=1)[:, 1]
        return [
            {"label": int(p >= threshold), "probability": round(float(p), 4)}
            for p in probabilities.cpu()
        ]

predictor = Predictor("model.pt")
print(predictor.predict(torch.randn(2, 10)))
```

```text
[{'label': 1, 'probability': 0.6878}, {'label': 0, 'probability': 0.4843}]
```

Four things in there, all of which matter in production:

- **Load once**, in `__init__`. Loading per request is the most common
  performance bug in ML services.
- **`eval()`**, or dropout and batch norm behave as if training.
- **`@torch.inference_mode()`** — like `no_grad()`, slightly faster, and it
  prevents the result being accidentally used in a graph.
- **Validate the shape**, so a bad request gives a clear message instead of a
  traceback from inside a matrix multiply.

---

## Batch, do not loop

```python
import time
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 2)).eval()
data = torch.randn(1_000, 10)

with torch.inference_mode():
    start = time.perf_counter()
    for row in data:
        model(row.unsqueeze(0))
    one_by_one = time.perf_counter() - start

    start = time.perf_counter()
    model(data)
    batched = time.perf_counter() - start

print(f"one at a time: {one_by_one * 1000:.1f} ms")
print(f"batched:       {batched * 1000:.1f} ms")
print(f"speedup:       {one_by_one / batched:.0f}x")
```

```text
one at a time: 7.0 ms
batched:       0.2 ms
speedup:       30x
```

Same arithmetic, same results, 30× faster. A thousand small matrix multiplies
pay the fixed per-call overhead a thousand times; one large one pays it once.
(On a GPU the gap is wider still, because a batch of one leaves almost every
core idle.)

If requests arrive one at a time, collect them for a few milliseconds and
process the group — this is *dynamic batching*, and every serving framework
does it for you.

---

## TorchScript and ONNX

Both turn your model into a portable artefact that does not need your Python
class.

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2)).eval()
example = torch.randn(1, 10)

scripted = torch.jit.trace(model, example)
scripted.save("model_traced.pt")

reloaded = torch.jit.load("model_traced.pt")
print("traced matches:", bool(torch.allclose(model(example), reloaded(example))))
```

```text
traced matches: True
```

`torch.jit.trace` records one forward pass, so branches that depend on the
input are baked in as they ran. If your `forward` has `if` statements over
tensor values, use `torch.jit.script` instead.

ONNX exports to a framework-neutral format that ONNX Runtime, TensorRT and
mobile runtimes can execute:

```python
# pip install onnx onnxruntime
torch.onnx.export(
    model, example, "model.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```

`dynamic_axes` is the line people forget. Without it the exported graph accepts
**only** the batch size you traced with, and production sends a different one.

| Format | Use |
|---|---|
| `state_dict` | Training, checkpointing, Python serving |
| TorchScript | Python-free serving, C++, mobile |
| ONNX | Cross-runtime, TensorRT, ONNX Runtime, edge |

The Optimization course takes this further — quantisation, compilation and
the cost arithmetic of serving.

---

## Making it faster, briefly

```python
import torch

# 1. Half precision on a GPU — roughly 2x, minimal accuracy loss
model_half = model.half().cuda() if torch.cuda.is_available() else model

# 2. torch.compile — fuses operations, PyTorch 2.x
compiled = torch.compile(model)

# 3. Fewer threads can be faster for small CPU models
torch.set_num_threads(4)
```

Measure each one. Compilation costs a slow first call and pays back over many;
for a service that starts, answers once and exits, it is a loss.

---

## What to monitor

A network fails silently. Every test passes, every request returns 200, and
the answers drift.

| Watch | Why | Alert when |
|---|---|---|
| Latency p50 and p99 | p99 is what users feel | p99 doubles |
| Input distributions | Drift — ML lesson 13 | KS test p < 0.01 per feature |
| Prediction distribution | The cheapest early warning | The positive rate shifts by 20% |
| Confidence | A drop means unfamiliar input | Mean max-probability falls |
| Error rate | Obvious failures | Any sustained rise |
| Accuracy on labelled data | The real answer, when labels arrive | Below your retraining threshold |

Log the model version with every prediction. Without it you cannot tell which
model produced a bad answer, and rollback becomes guesswork.

---

## Before you ship

- [ ] `state_dict` saved with metadata: input shape, normalisation, class order
- [ ] The loading code is tested in a fresh process
- [ ] `eval()` and `inference_mode()` in the inference path
- [ ] Input validation with a clear error message
- [ ] Batching, or dynamic batching in the server
- [ ] Latency measured at a realistic batch size
- [ ] Model version logged with every prediction
- [ ] A rollback path to the previous version
- [ ] A model card — ML lesson 13 has the template

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Loading the model per request | Slow, and memory churns |
| No `eval()` | Dropout active; different answer each call |
| Class order not recorded | Predictions silently inverted |
| Looping instead of batching | 30× slower for the same answer |
| ONNX export without `dynamic_axes` | Fails on any other batch size |
| Pickling the whole model | Breaks after a refactor or a version bump |

---

## Exercises

1. Save a model with metadata, reload it in a fresh process, prove the outputs
   match.
2. Benchmark one-at-a-time against batched inference at batch sizes 1, 8, 64.
3. Export to TorchScript and ONNX; confirm both give the same outputs.
4. Write the deployment checklist for your Project 4 model.
