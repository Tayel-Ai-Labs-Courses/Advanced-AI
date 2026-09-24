# Lesson 11 — Evaluation and Error Analysis

**Goal:** find out *why* a vision model fails.

## What you will learn

- Metrics per class, per size, per condition
- Confidence and calibration
- Looking at the failures
- Robustness checks before shipping

---

## Slice the metric

An overall accuracy hides everything actionable. Slice it by anything you can
measure.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 1_000

records = {
    "correct": rng.random(n) < 0.88,
    "brightness": rng.choice(["dark", "normal", "bright"], n, p=[0.2, 0.6, 0.2]),
    "object_size": rng.choice(["small", "medium", "large"], n, p=[0.3, 0.5, 0.2]),
}
# make dark images genuinely worse, to show what slicing reveals
dark = records["brightness"] == "dark"
records["correct"] = np.where(dark, rng.random(n) < 0.61, records["correct"])

print(f"overall accuracy: {records['correct'].mean():.3f}\n")
for dimension in ["brightness", "object_size"]:
    print(dimension)
    for value in sorted(set(records[dimension])):
        mask = records[dimension] == value
        print(f"  {value:<8}{mask.sum():>5} images   accuracy {records['correct'][mask].mean():.3f}")
    print()
```

```text
overall accuracy: 0.834

brightness
  bright    200 images   accuracy 0.900
  dark      226 images   accuracy 0.659
  normal    574 images   accuracy 0.880

object_size
  large     201 images   accuracy 0.861
  medium    509 images   accuracy 0.817
  small     290 images   accuracy 0.845
```

The headline is 0.834. The dark slice is **0.659** — a 22-point hole that the
overall number averages away, because dark images are only 23% of the set.

Note the contrast with the second slice: object size varies by four points,
which is noise. **One slice matters enormously and the other does not** — and
you only learn which by computing both.

Slices worth computing for every vision model:

| Slice by | Reveals |
|---|---|
| Class | Which categories are unlearned |
| Object size | Small objects are almost always worst |
| Brightness / exposure | Lighting robustness |
| Source (camera, site, operator) | Domain shift |
| Time (day, season) | Drift |
| Image quality (blur, resolution) | Acquisition problems |

Attach these attributes to your test set **before** you need them; recovering
them later is painful.

---

## Confidence and calibration

```python
import numpy as np

rng = np.random.default_rng(0)
confidence = np.clip(rng.beta(5, 2, 2_000), 0, 1)
correct = rng.random(2_000) < confidence ** 0.5      # loosely related to confidence

bins = np.linspace(0, 1, 6)
print(f"{'confidence bin':<18}{'count':>7}{'accuracy':>10}{'mean conf':>11}{'gap':>8}")
for low, high in zip(bins[:-1], bins[1:]):
    mask = (confidence >= low) & (confidence < high)
    if mask.sum() < 10:
        continue
    accuracy = correct[mask].mean()
    mean_confidence = confidence[mask].mean()
    print(f"[{low:.1f}, {high:.1f})        {mask.sum():>7}{accuracy:>10.3f}"
          f"{mean_confidence:>11.3f}{accuracy - mean_confidence:>8.3f}")
```

```text
confidence bin      count  accuracy  mean conf     gap
[0.2, 0.4)             88     0.625      0.340   0.285
[0.4, 0.6)            392     0.719      0.520   0.199
[0.6, 0.8)            832     0.845      0.707   0.137
[0.8, 1.0)            683     0.946      0.877   0.069
```

A **calibrated** model has a gap near zero: when it says 0.7, it is right 70%
of the time. Here every gap is positive and they shrink as confidence rises —
the model is systematically **under-confident**, worst at the bottom (says
0.34, is right 62% of the time) and nearly honest at the top (0.88 against
0.95).

Why it matters: any abstain rule, any routing threshold, any "flag for human
review" workflow is built on confidence. If the confidence is not calibrated,
the threshold means something different from what you think.

Fix with temperature scaling on a validation set — one parameter, fitted after
training, that rescales the logits.

---

## Look at the failures

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200
true_labels = rng.integers(0, 5, n)
predictions = true_labels.copy()
flip = rng.choice(n, 30, replace=False)
predictions[flip] = rng.integers(0, 5, 30)
scores = rng.beta(6, 2, n)

wrong = np.where(predictions != true_labels)[0]
most_confident_errors = wrong[np.argsort(-scores[wrong])][:5]

print(f"{len(wrong)} errors of {n}")
print("the five most confident:")
for index in most_confident_errors:
    print(f"  image {index:>3}  true {true_labels[index]}  "
          f"predicted {predictions[index]}  confidence {scores[index]:.3f}")
```

```text
24 errors of 200
the five most confident:
  image  84  true 3  predicted 0  confidence 0.985
  image 167  true 1  predicted 4  confidence 0.979
  image  91  true 2  predicted 0  confidence 0.963
  image  36  true 0  predicted 1  confidence 0.950
  image  48  true 4  predicted 3  confidence 0.945
```

**Display those images.** Confidently wrong predictions are the most
informative rows in your dataset, and they fall into a small set of causes:

| Cause | What you see | Fix |
|---|---|---|
| Label error | The prediction is right, the label is wrong | Fix the label |
| Genuinely ambiguous | Two experts would disagree | Merge classes, or accept a ceiling |
| Out of distribution | Nothing like it in training | Collect more of that kind |
| Preprocessing damage | Object cropped out, colours wrong | Fix the pipeline |
| Shortcut learned | Background predicts the class | Fix the data collection |

On most real datasets, **10–30% of "model errors" are label errors.** Counting
them is the highest-value hour in any vision project, and it changes what you
do next: relabelling beats retraining.

---

## The shortcut problem

```python
import numpy as np

rng = np.random.default_rng(0)
n = 600
label = rng.integers(0, 2, n)
background = np.where(rng.random(n) < 0.95, label, 1 - label)   # 95% correlated

print("correlation between background and label:", round(float((background == label).mean()), 3))
print()
print("if the model learns the background instead of the object:")
print(f"  accuracy on this data:            {(background == label).mean():.3f}")
shifted_background = rng.integers(0, 2, n)
print(f"  accuracy when the correlation breaks: {(shifted_background == label).mean():.3f}")
```

```text
correlation between background and label: 0.96

if the model learns the background instead of the object:
  accuracy on this data:            0.960
  accuracy when the correlation breaks: 0.490
```

A model that learned the background scores **0.960** on your test set and
**0.490** — below chance — the moment the correlation changes. Nothing in your
metrics warns you.

This is not hypothetical: defect detectors that learned the conveyor belt,
pneumonia detectors that learned which hospital's scanner produced the image,
animal classifiers that learned snow means wolf.

**How to catch it:**

- Overlay attention or Grad-CAM maps and check the model looks at the object
- Occlude the object and confirm the prediction collapses
- Test on images from a source not represented in training
- Look at the per-source slice — a large gap is a shortcut

---

## Robustness checks

```python
import numpy as np
import cv2

rng = np.random.default_rng(0)
image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

perturbations = {
    "original": image,
    "brightness +40": np.clip(image.astype(int) + 40, 0, 255).astype(np.uint8),
    "gaussian noise": np.clip(image + rng.normal(0, 15, image.shape), 0, 255).astype(np.uint8),
    "blur": cv2.GaussianBlur(image, (5, 5), 0),
    "jpeg quality 30": cv2.imdecode(
        cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], 1),
    "rotated 5 degrees": cv2.warpAffine(
        image, cv2.getRotationMatrix2D((32, 32), 5, 1.0), (64, 64)),
}
for name, perturbed in perturbations.items():
    difference = np.abs(perturbed.astype(float) - image.astype(float)).mean()
    print(f"{name:<20} mean pixel change {difference:6.2f}")
```

```text
original             mean pixel change   0.00
brightness +40       mean pixel change  36.60
gaussian noise       mean pixel change  11.55
blur                 mean pixel change  56.86
jpeg quality 30      mean pixel change  56.14
rotated 5 degrees    mean pixel change  73.54
```

(These are perturbations of random noise, which is the worst case for pixel
change — on a photograph the same operations move far fewer pixels. The point
is the ordering: **JPEG at quality 30 changes this image as much as a 5×5
blur does**, and JPEG is what every phone and every web upload applies to your
production images.)

Run your model on each perturbation and record the accuracy. A model that
loses ten points to JPEG compression will lose them in production, where every
image arrives compressed.

These six take an hour to build and they predict the failures you would
otherwise discover from a customer.

---

## The evaluation report

Write this for every vision model, before shipping:

```text
Overall:        accuracy / mAP / mIoU, with the threshold stated
Per class:      precision, recall, support — the worst class named
Per slice:      size, lighting, source, time — the worst slice named
Calibration:    the confidence/accuracy gap per bin
Errors:         20 most confident errors, displayed and categorised
Label errors:   how many of those 20 were the label's fault
Robustness:     accuracy under six perturbations
Shortcuts:      evidence the model uses the object, not the background
```

The last four lines are what separates an evaluation from a number.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Only the overall metric | A 27-point hole in one slice stays hidden |
| Never displaying the errors | Label errors counted as model errors |
| Trusting uncalibrated confidence | Abstain thresholds do the wrong thing |
| No robustness check | Production compression or lighting breaks it |
| No shortcut check | 0.96 in test, chance in the real world |
| Slices you cannot compute | Attach the metadata before you need it |

---

## Exercises

1. Slice a model's accuracy by two attributes and find the worst slice.
2. Build the confidence/accuracy table; is your model calibrated?
3. Display your 20 most confident errors and categorise the causes.
4. Run six perturbations and report the accuracy drop for each.
