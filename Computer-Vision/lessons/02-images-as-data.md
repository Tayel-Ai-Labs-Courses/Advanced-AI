# Lesson 02 — Images as Data

**Goal:** know exactly what is in the array before you model it.

## What you will learn

- Shape, dtype and channel order
- RGB, BGR, greyscale, HSV
- Resizing, and what interpolation costs
- Normalisation, and matching it at serving

---

## An image is an array

```python
import numpy as np
from PIL import Image

rng = np.random.default_rng(0)
array = rng.integers(0, 256, size=(120, 200, 3), dtype=np.uint8)
image = Image.fromarray(array)

print("numpy shape:", array.shape, array.dtype)
print("PIL size:   ", image.size, image.mode)
print("value range:", array.min(), "to", array.max())
```

```text
numpy shape: (120, 200, 3) uint8
PIL size:    (200, 120) RGB
value range: 0 to 255
```

Look at those two lines carefully. NumPy says `(120, 200, 3)` — **height,
width, channels**. PIL says `(200, 120)` — **width, height**. They describe
the same image and they order the numbers differently.

| Library | Order | Type |
|---|---|---|
| NumPy / scikit-image | `(H, W, C)` | `uint8` 0–255, or float 0–1 |
| PIL | `(W, H)` via `.size` | `uint8` |
| OpenCV | `(H, W, C)` — **BGR** | `uint8` |
| PyTorch | `(C, H, W)`, batched `(N, C, H, W)` | `float32`, normalised |

Four conventions in one pipeline. Print the shape at every boundary.

---

## Channel order

```python
import numpy as np
import cv2
from PIL import Image

pure_red = np.zeros((4, 4, 3), dtype=np.uint8)
pure_red[:, :, 0] = 255                       # channel 0 = red, in RGB

cv2.imwrite("/tmp/red_via_cv2.png", pure_red)         # cv2 writes BGR
Image.fromarray(pure_red).save("/tmp/red_via_pil.png")  # PIL writes RGB

back_cv2 = cv2.imread("/tmp/red_via_cv2.png")
back_pil = np.array(Image.open("/tmp/red_via_pil.png"))

print("written by cv2, read by cv2:", back_cv2[0, 0])
print("written by PIL, read by PIL:", back_pil[0, 0])
print("written by PIL, read by cv2:", cv2.imread("/tmp/red_via_pil.png")[0, 0])
```

```text
written by cv2, read by cv2: [255   0   0]
written by PIL, read by PIL: [255   0   0]
written by PIL, read by cv2: [  0   0 255]
```

The third line is the bug. The same file, read by a different library, hands
you the channels reversed — and nothing warns you. Your red object becomes
blue, your model was trained on one convention and served with the other, and
the accuracy drop looks mysterious.

**Fix it at the boundary:**

```python
import cv2

bgr = cv2.imread("/tmp/red_via_pil.png")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
print("after conversion:", rgb[0, 0])
```

```text
after conversion: [255   0   0]
```

---

## Colour spaces

```python
import numpy as np
import cv2

image = np.zeros((2, 2, 3), dtype=np.uint8)
image[:, :] = (200, 120, 60)                  # an RGB brown

greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

print("rgb:      ", image[0, 0])
print("greyscale:", greyscale[0, 0], " shape", greyscale.shape)
print("hsv:      ", hsv[0, 0], " (hue 0-179 in OpenCV)")
```

```text
rgb:       [200 120  60]
greyscale: 137  shape (2, 2)
hsv:       [ 13 179 200]  (hue 0-179 in OpenCV)
```

| Space | Use for |
|---|---|
| **RGB** | Everything by default; what models expect |
| **Greyscale** | Shape and texture; a third of the data |
| **HSV** | Colour thresholds that survive lighting changes — hue is separate from brightness |
| **LAB** | Perceptual colour distance |

HSV earns its place in one common situation: "find the red objects" is
brittle in RGB (a shadowed red is a different RGB triple entirely) and robust
in HSV, where the hue stays near the same value.

---

## Resizing

```python
import numpy as np
import cv2

rng = np.random.default_rng(0)
original = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

for name, method in [("nearest", cv2.INTER_NEAREST),
                     ("bilinear", cv2.INTER_LINEAR),
                     ("area", cv2.INTER_AREA),
                     ("cubic", cv2.INTER_CUBIC)]:
    small = cv2.resize(original, (224, 224), interpolation=method)
    restored = cv2.resize(small, (640, 480), interpolation=method)
    error = np.abs(original.astype(float) - restored.astype(float)).mean()
    print(f"{name:<10} shape {small.shape}  mean round-trip error {error:6.2f}")
```

```text
nearest    shape (224, 224, 3)  mean round-trip error  85.08
bilinear   shape (224, 224, 3)  mean round-trip error  59.88
area       shape (224, 224, 3)  mean round-trip error  58.16
cubic      shape (224, 224, 3)  mean round-trip error  64.44
```

On random noise every method loses a lot, and the ranking is the useful part:
`INTER_AREA` wins for **downscaling** (58.16 — it averages the pixels it
discards), and `INTER_NEAREST` is far the worst at 85.08, because it simply
throws away three pixels in four — and yet `INTER_NEAREST` is the only
correct choice for **segmentation masks**, where averaging label values 3 and
7 into 5 invents a class that was never there.

Aspect ratio is the other decision:

```python
import cv2
import numpy as np

rng = np.random.default_rng(0)
wide = rng.integers(0, 256, (200, 600, 3), dtype=np.uint8)

stretched = cv2.resize(wide, (224, 224))

scale = 224 / max(wide.shape[:2])
resized = cv2.resize(wide, (int(wide.shape[1] * scale), int(wide.shape[0] * scale)))
padded = np.zeros((224, 224, 3), dtype=np.uint8)
y = (224 - resized.shape[0]) // 2
padded[y : y + resized.shape[0], : resized.shape[1]] = resized

print("stretched:", stretched.shape, "— aspect ratio destroyed")
print("padded:   ", padded.shape, f"— content {resized.shape}, rest is padding")
```

```text
stretched: (224, 224, 3) — aspect ratio destroyed
padded:    (224, 224, 3) — content (74, 224, 3), rest is padding
```

Stretching distorts shapes, which matters when shape is the signal. Padding
preserves them and wastes compute on black pixels. Pick one and **use the same
one at serving**.

---

## Normalisation

```python
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

rng = np.random.default_rng(0)
array = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(array)

to_tensor = transforms.ToTensor()
normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

tensor = to_tensor(image)
normalised = normalise(tensor)

print("uint8 array:   ", array.shape, array.dtype, array.min(), array.max())
print("after ToTensor:", tuple(tensor.shape), tensor.dtype,
      f"{tensor.min():.3f} to {tensor.max():.3f}")
print("after Normalize:", f"{normalised.min():.3f} to {normalised.max():.3f}",
      f"mean {normalised.mean():.3f}")
```

```text
uint8 array:    (224, 224, 3) uint8 0 255
after ToTensor: (3, 224, 224) torch.float32 0.000 to 1.000
after Normalize: -2.118 to 2.640 mean 0.233
```

`ToTensor` does two things at once: it divides by 255 **and** it moves the
channels to the front, `(H, W, C)` → `(C, H, W)`. Then `Normalize` subtracts
the mean and divides by the standard deviation, per channel.

Those particular numbers — `[0.485, 0.456, 0.406]` — are ImageNet's channel
statistics. **Use the values the pretrained model was trained with**, not your
own dataset's, whenever you use pretrained weights (CV lesson 06).

---

## Look at your data

```python
import numpy as np

def describe(image):
    """The five numbers worth printing for every image you load."""
    return {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "range": (float(image.min()), float(image.max())),
        "mean_per_channel": [round(float(image[..., c].mean()), 1)
                             for c in range(image.shape[-1])] if image.ndim == 3 else None,
        "is_grey": bool(image.ndim == 2 or
                        (image.ndim == 3 and np.allclose(image[..., 0], image[..., 1]))),
    }

rng = np.random.default_rng(0)
print(describe(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)))

grey = rng.integers(0, 256, (64, 64), dtype=np.uint8)
print(describe(np.stack([grey] * 3, axis=-1)))
```

```text
{'shape': (64, 64, 3), 'dtype': 'uint8', 'range': (0.0, 255.0), 'mean_per_channel': [127.9, 124.9, 128.4], 'is_grey': False}
{'shape': (64, 64, 3), 'dtype': 'uint8', 'range': (0.0, 255.0), 'mean_per_channel': [127.1, 127.1, 127.1], 'is_grey': True}
```

The second image has three channels and no colour — a greyscale photograph
saved as RGB. Knowing that saves you from training a three-channel model on
data with a third of the information, and from being surprised when colour
augmentation does nothing.

**Run something like `describe()` on a sample of every dataset you receive**,
and plot a grid of twenty images before you write a model.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| BGR/RGB mismatch between training and serving | Silent accuracy loss |
| `(W, H)` confused with `(H, W)` | Rotated or crashed |
| Bilinear interpolation on segmentation masks | Invented class values |
| Normalising with your own stats on a pretrained model | Worse, for no visible reason |
| Forgetting `ToTensor` transposes the channels | Shape error, or a model trained on nonsense |
| Never plotting the images | The whole class of bugs above |

---

## Exercises

1. Load an image with PIL and with OpenCV; compare the first pixel.
2. Resize an image four ways and view the results side by side.
3. Write `describe()` and run it over 100 images from a real dataset.
4. Take a segmentation mask, resize it bilinearly, and count the class values
   that appear.
