# Lesson 05 — Model Extraction

**Goal:** see how much of your model an attacker gets from its API, and decide
what to return.

## What you will learn

- Extraction with nothing but labels
- How many queries it takes
- What you are actually protecting
- The controls that work

---

## Stealing a model with random inputs

The attacker has no training data and no access to your weights. They have your
API. They send **random inputs**, record the labels, and train their own model
on the pairs.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=4_000, n_features=20, n_informative=8,
                           class_sep=0.7, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
print("ready")
```

```text
ready
```

```python
victim = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
rng = np.random.default_rng(2)
for n_queries in (100, 500, 2_000, 10_000):
    Xq = rng.normal(size=(n_queries, 20))          # attacker's own random inputs
    yq = victim.predict(Xq)                        # only the label is returned
    clone = HistGradientBoostingClassifier(random_state=0).fit(Xq, yq)
    agree = float(np.mean(clone.predict(X_te) == victim.predict(X_te)))
    acc = accuracy_score(y_te, clone.predict(X_te))
    print(f"{n_queries:>6} queries: clone agrees with the victim {agree:.1%} "
          f"of the time, its own test accuracy {acc:.3f}")
print(f"victim test accuracy: {accuracy_score(y_te, victim.predict(X_te)):.3f}")
```

```text
   100 queries: clone agrees with the victim 73.2% of the time, its own test accuracy 0.708
   500 queries: clone agrees with the victim 83.8% of the time, its own test accuracy 0.794
  2000 queries: clone agrees with the victim 88.8% of the time, its own test accuracy 0.840
 10000 queries: clone agrees with the victim 91.3% of the time, its own test accuracy 0.861
victim test accuracy: 0.913
```

**10,000 queries — a few minutes of traffic — produce a clone that agrees with
the original 91.3% of the time.**

And the attacker's inputs were pure Gaussian noise. They had no idea what the
features meant. With realistic inputs, which any customer of the API has, the
numbers are better still.

Notice what this cost the attacker: 10,000 API calls. Notice what it cost you:
the labelled dataset, the feature engineering, the experiments, the
infrastructure — everything the Data-Science course spends ten lessons on.

---

## What you are actually protecting

Extraction matters for three different reasons, and they call for different
responses:

| Concern | Detail |
|---|---|
| **The model as IP** | Someone else now has your product, for the price of the API calls |
| **A stepping stone** | A local clone lets the attacker develop lessons 03 and 06 attacks offline, at no cost and with no rate limit |
| **The decision rule** | If the model gates something valuable — credit, fraud, moderation — the clone tells the attacker exactly how to be approved |

The third is usually the worst and the least discussed. An attacker with a
91%-accurate clone of your fraud model can test transactions against it locally
until one passes, then send only that one to you.

---

## What actually helps

| Control | Effect | Cost |
|---|---|---|
| **Return a label, not a probability** | Removes most of the signal per query | May break a legitimate use |
| **Bucket the score** (5 bands) | Much less information per query | Usually acceptable |
| **Rate limit per account** | Multiplies the time and cost | Low |
| **Per-account query budgets and alerts** | Extraction looks like unusual volume | Low; needs monitoring |
| **Detect distributional anomalies** | Random or grid-like inputs look nothing like real traffic | Moderate; a good signal |
| **Watermark the model** | Does not prevent; proves ownership afterwards | Moderate |
| **Legal terms** | Deterrent for companies; nothing for anonymous attackers | Low |
| Adding noise to outputs | Reduces extraction fidelity | Hurts legitimate users too |

The first two are the highest value. **Ask why the API returns a probability at
all.** Very often it is because a colleague wanted to sort a list once, and the
full score has been exposed to every caller ever since.

The distributional check is the one worth building if the model is genuinely
valuable: real users send correlated, plausible feature combinations. Extraction
traffic is random, uniform or grid-shaped, and looks different within a few
hundred requests.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Returning full probabilities to every caller | The attacker's job is easier per query |
| No rate limit on a scoring endpoint | 10,000 queries is minutes |
| Assuming an internal API is safe | Insiders are in the threat model (lesson 01) |
| Thinking extraction needs your data | Gaussian noise got to 91.3% agreement |
| Ignoring the clone as an attack platform | Lessons 03 and 06 become free and unlimited |
| Watermarking instead of limiting | It proves theft; it does not prevent it |

---

## Exercises

1. Rerun the extraction with only the **label** returned versus the full
   probability vector. How many more queries does the attacker need?
2. Bucket the output into 5 bands and measure the clone's agreement at 10,000
   queries.
3. Build the distributional detector: compare the covariance of a caller's
   inputs against real traffic. How many queries before it fires?
4. Compute what 10,000 queries cost an attacker against your own API, in money
   and in time. Is that a deterrent?
5. For your highest-value model, write down what an attacker with a 90% clone
   could do, in one paragraph.

---

**Next:** [Lesson 06 — Adversarial Inputs](06-adversarial-inputs.md)
