# Lesson 12 — Deploying Vision

**Goal:** ship a model whose accuracy survives leaving your notebook.

## What you will learn

- Preprocessing parity — the failure that eats vision projects
- Batching and latency
- Export and edge deployment
- What to monitor

---

## Preprocessing parity

The training pipeline and the serving pipeline must be **the same code**.

```python
import numpy as np
import cv2
from PIL import Image

rng = np.random.default_rng(0)
original = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

pil_resized = np.array(Image.fromarray(original).resize((224, 224), Image.BILINEAR))
cv2_resized = cv2.resize(original, (224, 224), interpolation=cv2.INTER_LINEAR)

difference = np.abs(pil_resized.astype(float) - cv2_resized.astype(float))
print(f"mean absolute difference: {difference.mean():.2f}")
print(f"max difference:           {difference.max():.0f}")
print(f"identical pixels:         {100 * (difference == 0).mean():.1f}%")
```

```text
mean absolute difference: 31.58
max difference:           143
identical pixels:         0.9%
```

Two libraries, the same "bilinear resize to 224×224", and **99% of the pixels
differ** — by 31 levels on average and up to 143 out of 255. PIL and OpenCV implement bilinear
resampling differently (antialiasing, kernel support, pixel-centre
conventions).

Train with `torchvision` (PIL underneath) and serve with OpenCV, and your
model sees systematically different inputs from the ones it learned on. The
accuracy drop is real, silent, and extremely hard to find.

**The fix is structural: one function, imported by both.**

```python
import torch
from torchvision import transforms
from PIL import Image

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def prepare(image_paths):
    """The ONLY way images enter this model, in training and in serving."""
    tensors = [PREPROCESS(Image.open(path).convert("RGB")) for path in image_paths]
    return torch.stack(tensors)
```

`convert("RGB")` is doing quiet work there: it handles greyscale, CMYK and
RGBA inputs, all of which appear in real uploads and all of which would
otherwise be a shape error or a silent channel bug.

---

## Latency and batching

```python
import time
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from torchvision import models

model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.eval()

def benchmark(batch_size, repeats=10):
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        elapsed = (time.perf_counter() - start) / repeats
    return elapsed * 1000, batch_size / elapsed

print(f"{'batch':>6}{'ms/batch':>11}{'ms/image':>11}{'images/s':>11}")
for batch_size in [1, 4, 16, 32]:
    total, throughput = benchmark(batch_size)
    print(f"{batch_size:>6}{total:>11.1f}{total / batch_size:>11.1f}{throughput:>11.1f}")
```

```text
 batch   ms/batch   ms/image   images/s
     1       17.9       17.9       55.9
     4       67.7       16.9       59.1
    16      414.4       25.9       38.6
    32      714.6       22.3       44.8
```

Read that carefully, because it does **not** say what the usual advice says.

Per-image cost improves slightly from batch 1 to batch 4 (17.9 → 16.9 ms) and
then gets **worse**: batch 16 costs 25.9 ms per image, 45% more than batch 1.

On a GPU, larger batches almost always win. On this CPU the model is already
using every core for a single 224×224 image, so batching adds no parallelism —
it only adds memory pressure and cache misses. **The knee here is batch 4, and
beyond it batching actively hurts.**

That is why Optimization lesson 01 insists on finding the knee on your own
hardware rather than assuming it. Note also what a single image costs:
**17.9 ms for a ResNet-18 at 224×224 on a laptop CPU** — a 150 ms end-to-end
budget leaves room for this model and little else.

Ways to cut it, in order of effort:

| Change | Typical effect |
|---|---|
| Smaller input (224 → 160) | ~2× faster; measure the accuracy cost |
| `mobilenet_v3_small` instead of ResNet-18 | 3–5× faster |
| int8 quantisation | 2–4× on supported hardware (Optimization lesson 08) |
| `torch.compile` | 0–30% on CPU; more on GPU (Optimization lesson 11) |
| Batching | Up to the knee above |
| A GPU | 10–50× |

---

## Export

```python
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from torchvision import models

model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.eval()

example = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example)
traced.save("/tmp/vision_model.pt")

reloaded = torch.jit.load("/tmp/vision_model.pt")
with torch.inference_mode():
    difference = (model(example) - reloaded(example)).abs().max().item()
print("traced output matches:", difference < 1e-5, f"(max diff {difference:.2e})")

batch = torch.randn(8, 3, 224, 224)
with torch.inference_mode():
    print("works at batch 8:", tuple(reloaded(batch).shape))
```

```text
traced output matches: True (max diff 0.00e+00)
works at batch 8: (8, 10)
```

For ONNX, remember `dynamic_axes` (Optimization lesson 11) or the exported
graph accepts only the batch size you traced with.

| Target | Format |
|---|---|
| Python service | `state_dict` + your model code |
| C++ / mobile | TorchScript, CoreML, TFLite |
| Cross-runtime, CPU | ONNX + ONNX Runtime |
| NVIDIA production | TensorRT |
| Browser | ONNX.js, TF.js |

---

## The inference service

```python
import torch
import numpy as np
from PIL import Image

class VisionPredictor:
    """Load once; validate, preprocess, batch, return probabilities."""

    def __init__(self, model_path, class_names, device=None, threshold=0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        self.class_names = class_names
        self.threshold = threshold

    @torch.inference_mode()
    def predict(self, images, batch_size=16):
        results = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([PREPROCESS(image.convert("RGB"))
                                 for image in images[start : start + batch_size]])
            probabilities = self.model(batch.to(self.device)).softmax(1).cpu()
            for row in probabilities:
                index = int(row.argmax())
                confidence = float(row[index])
                results.append({
                    "label": self.class_names[index] if confidence >= self.threshold
                             else "uncertain",
                    "confidence": round(confidence, 4),
                    "all_scores": {name: round(float(score), 4)
                                   for name, score in zip(self.class_names, row)},
                })
        return results
```

Five decisions in that class, each of which comes from a production failure:

- **Load once**, not per request.
- **`convert("RGB")`** on every input.
- **Batch internally**, so a caller passing 200 images does not run 200
  forward passes.
- **Return an `"uncertain"` label** below the threshold, rather than a
  confident guess.
- **Return all scores**, so the caller can apply its own policy without a
  second call.

---

## Monitoring

Vision models fail silently when the camera moves, the lighting changes, or a
supplier changes the packaging.

| Watch | Why | Alert when |
|---|---|---|
| Prediction distribution | The cheapest drift signal | The class mix shifts by 20% |
| Mean confidence | Unfamiliar inputs score lower | It drops week on week |
| "Uncertain" rate | Direct measure of unfamiliarity | It doubles |
| Image statistics | Brightness, size, aspect ratio | Any distribution moves |
| Latency p50 / p99 | Load and regressions | p99 doubles |
| Sampled human review | The only ground truth | Accuracy below the floor |

**Save a random 1% of production images** (with permission and a retention
policy). They are your next training set, your drift evidence, and the only
way to answer "was it wrong, or was the label wrong?" three months from now.

---

## Before you ship

- [ ] One preprocessing function, imported by training and serving
- [ ] Interpolation, normalisation and channel order identical in both
- [ ] `convert("RGB")` handles greyscale, RGBA and CMYK inputs
- [ ] Latency measured at batch 1 and at your production batch size
- [ ] The export verified numerically against PyTorch
- [ ] Class order saved with the weights
- [ ] An abstain threshold, chosen on validation data
- [ ] Image statistics and prediction distribution logged
- [ ] A sample of production images retained for review
- [ ] A model card naming the conditions it was validated under

---

## Common mistakes

| Mistake | What happens |
|---|---|
| PIL in training, OpenCV in serving | 98% of pixels differ; silent accuracy loss |
| BGR/RGB mismatch | Same |
| Loading the model per request | 100× the latency |
| No `convert("RGB")` | Crashes on a greyscale upload |
| Class order not saved | Predictions silently permuted |
| No confidence threshold | Confident nonsense on out-of-distribution images |
| No production images kept | Drift undiagnosable |

---

## Exercises

1. Resize the same image with PIL and OpenCV; measure the difference.
2. Benchmark your model at batch 1, 8 and 32; find the knee.
3. Export to TorchScript and verify the outputs match.
4. Write the monitoring list for a camera that might be moved by a cleaner.
