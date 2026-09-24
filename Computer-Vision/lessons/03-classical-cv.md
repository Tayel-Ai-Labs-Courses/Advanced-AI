# Lesson 03 — Classical Computer Vision

**Goal:** solve the problems that do not need a neural network.

## What you will learn

- Convolution as a filter you write yourself
- Blur, edges, thresholds
- Contours and measurement
- Template matching, and where classical CV stops

---

## A filter is a small matrix

```python
import numpy as np
import cv2

image = np.zeros((7, 7), dtype=np.float32)
image[:, 3:] = 255.0                          # a vertical edge down the middle

vertical_edge = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

response = cv2.filter2D(image, -1, vertical_edge)

print("image row 3:   ", image[3].astype(int))
print("response row 3:", response[3].astype(int))
```

```text
image row 3:    [  0   0   0 255 255 255 255]
response row 3: [   0    0 1020 1020    0    0    0]
```

The filter is a 3×3 Sobel kernel. Slid over the image, it produces a large
value exactly where the brightness changes horizontally, and zero everywhere
flat.

**That is what the first layer of a CNN learns** — the difference is that here
you wrote the nine numbers, and a CNN finds them from data. Convolution is
identical in both.

---

## Blur

```python
import numpy as np
import cv2

rng = np.random.default_rng(0)
clean = np.zeros((100, 100), dtype=np.uint8)
clean[30:70, 30:70] = 200
noisy = np.clip(clean + rng.normal(0, 25, clean.shape), 0, 255).astype(np.uint8)

for name, filtered in [
    ("noisy", noisy),
    ("gaussian 5x5", cv2.GaussianBlur(noisy, (5, 5), 0)),
    ("median 5", cv2.medianBlur(noisy, 5)),
    ("bilateral", cv2.bilateralFilter(noisy, 9, 75, 75)),
]:
    difference = np.abs(filtered.astype(float) - clean.astype(float)).mean()
    print(f"{name:<14} mean error vs the clean image {difference:6.2f}")
```

```text
noisy          mean error vs the clean image  11.44
gaussian 5x5   mean error vs the clean image  11.19
median 5       mean error vs the clean image   3.80
bilateral      mean error vs the clean image   8.85
```

The spread is larger than the names suggest:

| Filter | Removes | Edges | Error here |
|---|---|---|---|
| Gaussian | General noise | Blurred too | 11.19 |
| Median | Spikes, and moderate noise | Preserved | **3.80** |
| Bilateral | Noise | Preserved — it refuses to average across an edge | 8.85 |

**Median wins by a wide margin (3.80 against 11.44 noisy), and Gaussian barely
helps at all** — 11.19, a 2% improvement.

The reason is the image: a hard-edged square. Gaussian blur averages across
that edge, so what it gains in the flat regions it loses at the boundary.
Median takes the middle value of the window, which on one side of an edge is
still a value from that side.

Bilateral, the theoretically edge-preserving filter, lands between them and is
an order of magnitude slower. On a photograph with soft gradients the ranking
changes — **measure on your images**, not on the reputation of the filter.

---

## Edges

```python
import numpy as np
import cv2

image = np.zeros((100, 100), dtype=np.uint8)
image[30:70, 30:70] = 200

edges = cv2.Canny(image, threshold1=50, threshold2=150)

print("edge pixels:", int((edges > 0).sum()))
print("perimeter of a 40x40 square:", 4 * 40)
print("edge row 30:", np.flatnonzero(edges[30]).tolist()[:8])
```

```text
edge pixels: 156
perimeter of a 40x40 square: 160
edge row 30: [30, 69]
```

Canny found 156 edge pixels where the true perimeter is 160 — it traced the
square almost exactly. Row 30 contains exactly two edge pixels, at columns 30
and 69: the left and right boundaries at the top of the square, and nothing in
between, because the interior is flat.

The two thresholds control hysteresis: strong edges above the high threshold
are kept, weak ones only if connected to a strong one.

Canny is exact, needs no training, and runs in microseconds. On a clean,
well-lit image it is strictly better than any learned edge detector.

---

## Thresholding and measurement

The classic industrial pipeline: threshold, find contours, measure.

```python
import numpy as np
import cv2

image = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(image, (60, 60), 25, 255, -1)          # radius 25
cv2.circle(image, (140, 140), 40, 255, -1)        # radius 40
cv2.rectangle(image, (140, 20), (180, 60), 255, -1)

_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"objects found: {len(contours)}")
for contour in sorted(contours, key=cv2.contourArea, reverse=True):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / perimeter ** 2 if perimeter else 0
    x, y, w, h = cv2.boundingRect(contour)
    shape = "circle" if circularity > 0.85 else "not a circle"
    print(f"  area {area:7.0f}  circularity {circularity:.2f}  "
          f"box ({x},{y},{w},{h})  -> {shape}")
```

```text
objects found: 3
  area    4912  circularity 0.89  box (100,100,81,81)  -> circle
  area    1890  circularity 0.87  box (35,35,51,51)  -> circle
  area    1600  circularity 0.79  box (140,20,41,41)  -> not a circle
```

Three objects found, measured, and classified by shape — **with no training
data and no model.** The circles score 0.89 and 0.87 for circularity, the
square 0.79, and a threshold of 0.85 separates them.

Note how close 0.79 and 0.87 are. A rasterised circle is not a perfect circle,
and a hand-set threshold with an 0.08 margin is exactly the kind of rule that
breaks when the image conditions change. That fragility is the honest cost of
classical methods, and it is why the table at the end of this lesson is about
*conditions*, not about tasks.

Compare that to the deep-learning route: label a thousand images, train a
segmentation model, and get an approximate area. If the objects are
well-separated on a plain background, classical CV is not merely adequate, it
is *better* — exact, explainable and instant.

---

## Template matching

```python
import numpy as np
import cv2

rng = np.random.default_rng(0)
scene = rng.integers(0, 60, (200, 200), dtype=np.uint8)      # dark textured background
patch = rng.integers(150, 255, (30, 30), dtype=np.uint8)     # a distinctive bright patch
scene[40:70, 120:150] = patch

result = cv2.matchTemplate(scene, patch, cv2.TM_CCOEFF_NORMED)
_, best_score, _, best_location = cv2.minMaxLoc(result)
print(f"best match at {best_location} with score {best_score:.3f}")

rotated = cv2.warpAffine(patch, cv2.getRotationMatrix2D((15, 15), 15, 1.0), (30, 30))
result = cv2.matchTemplate(scene, rotated, cv2.TM_CCOEFF_NORMED)
_, rotated_score, _, rotated_location = cv2.minMaxLoc(result)
print(f"rotated 15 degrees: best score {rotated_score:.3f} at {rotated_location}")
```

```text
best match at (120, 40) with score 1.000
rotated 15 degrees: best score 0.353 at (118, 37)
```

Exact, instant, no model: score **1.000** at precisely the right location.

Then rotate the template by fifteen degrees — not ninety, fifteen — and the
score falls to **0.353**. It still lands near the right place, but a
confidence that drops by two thirds under a small rotation is not something
you can threshold reliably.

That single pair of numbers is the boundary of classical computer vision.
Template matching finds a **known, fixed** pattern — a logo, a form field, a
fiducial marker, a UI element — and it fails the moment the appearance varies.
Variation is precisely what learned features handle.

(Note the template must have structure. A uniform white square matched against
a uniform region gives a degenerate correlation and reports a perfect match at
the first position it tries — which is why the patch above is random texture.)

---

## Where classical CV stops

| Condition | Classical | Deep learning |
|---|---|---|
| Controlled lighting, plain background | **Best choice** | Overkill |
| A fixed, known template | **Best choice** | Overkill |
| Measuring well-separated objects | **Best choice** | Overkill |
| Varying pose, lighting, background | Fails | **Required** |
| "Is this a cat?" | Impossible | **Required** |
| Occlusion, clutter, deformation | Fails | **Required** |

The honest summary: classical CV handles **appearance you can specify**, deep
learning handles **appearance you can only exemplify**.

And they combine well. A common production pipeline uses classical methods to
find and crop the region of interest — fast, exact — and a small CNN to
classify the crop. That is cheaper and more robust than a detector over the
whole frame.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| A CNN for a template match | Labels and GPU for a problem `matchTemplate` solves |
| Fixed thresholds on varying lighting | Works in the lab, fails on site — use adaptive or Otsu |
| Contours without an area filter | Every noise speck is an object |
| Canny thresholds copied from a tutorial | Tuned for someone else's image |
| Bilateral filtering in a hot loop | 10–50× slower than Gaussian |
| Assuming classical CV is obsolete | It is faster, exact, and needs no labels |

---

## Exercises

1. Write a 3×3 kernel that detects horizontal edges; verify it on a drawn
   image.
2. Add noise to an image and compare three denoising filters by error.
3. Threshold a photograph, find contours, and measure the largest object.
4. Find a problem at your work that `matchTemplate` or contours would solve.
