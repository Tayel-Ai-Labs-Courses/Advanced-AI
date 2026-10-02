# Lesson 03 — Decomposition

**Goal:** find out how many dimensions a dataset really has, and compress it
without losing what matters.

## What you will learn

- The SVD, and what its singular values tell you
- Rank, and how to see it in real data
- PCA as an SVD in disguise
- What "reduce dimensionality" actually costs

---

## Most matrices are smaller than they look

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)

rng = np.random.default_rng(0)
# a 60x40 matrix that is really rank 5 plus noise
true = rng.normal(size=(60, 5)) @ rng.normal(size=(5, 40))
M = true + 0.3 * rng.normal(size=(60, 40))
U, s, Vt = np.linalg.svd(M, full_matrices=False)
total = (s ** 2).sum()
print(f"{'rank k':>8}{'kept variance':>16}{'reconstruction err':>21}{'numbers stored':>16}")
for k in (1, 2, 5, 10, 20, 40):
    approx = U[:, :k] * s[:k] @ Vt[:k]
    kept = (s[:k] ** 2).sum() / total
    err = np.linalg.norm(M - approx) / np.linalg.norm(M)
    stored = k * (60 + 40 + 1)
    print(f"{k:>8}{kept:>16.3f}{err:>21.3f}{stored:>16,}")
print(f"the full matrix is {60*40:,} numbers. Rank 5 captures most of it.")
print("first 8 singular values:", np.round(s[:8], 2))
```

```text
  rank k   kept variance   reconstruction err  numbers stored
       1           0.368                0.795             101
       2           0.573                0.653             202
       5           0.987                0.113             505
      10           0.992                0.088           1,010
      20           0.997                0.053           2,020
      40           1.000                0.000           4,040
the full matrix is 2,400 numbers. Rank 5 captures most of it.
first 8 singular values: [69.27 51.77 49.78 42.31 33.7   3.95  3.84  3.55]
```

The matrix was built to be **rank 5 plus noise**, and the singular values say so
out loud: 69, 52, 50, 42, 34, then a cliff to **3.95**.

That cliff is the signal. Five directions carry real structure; everything after
is noise, and keeping it costs storage and adds nothing.

**Rank 5 keeps 98.7% of the variance in 505 numbers** where the full matrix is
2,400 — a 4.75x compression with an 11% reconstruction error that is almost
entirely the noise you wanted gone.

---

## Reading the singular values

```python
ratios = s[:-1] / s[1:]
biggest_gap = int(np.argmax(ratios[:15])) + 1
print("consecutive ratios:", np.round(ratios[:8], 2))
print(f"largest gap after index {biggest_gap} -> estimated rank {biggest_gap}")
print(f"variance kept at that rank: {(s[:biggest_gap]**2).sum()/ (s**2).sum():.3f}")
```

```text
consecutive ratios: [1.34 1.04 1.18 1.26 8.53 1.03 1.08 1.06]
largest gap after index 5 -> estimated rank 5
variance kept at that rank: 0.987
```

**The ratio jumps to 8.53 at position 5** and sits near 1.0 everywhere else.
That is the scree-plot elbow, computed instead of eyeballed, and it recovered
the true rank exactly.

Do this on your own feature matrix before anything else. If 40 features have an
effective rank of 6, you have six ideas measured seven ways — which changes what
model you should fit and explains a lot of correlated-feature pain.

---

## PCA is this, centred

```python
X = M - M.mean(axis=0)                      # centring is the whole difference
U2, s2, V2t = np.linalg.svd(X, full_matrices=False)
explained = s2 ** 2 / (s2 ** 2).sum()
print("explained variance ratio:", np.round(explained[:6], 3))
print("cumulative:", np.round(np.cumsum(explained[:6]), 3))
Z = U2[:, :5] * s2[:5]                      # the projected data
print("original shape:", X.shape, "-> projected:", Z.shape)
```

```text
explained variance ratio: [0.376 0.201 0.184 0.137 0.089 0.001]
cumulative: [0.376 0.577 0.761 0.899 0.987 0.988]
original shape: (60, 40) -> projected: (60, 5)
```

`explained_variance_ratio_` in scikit-learn is `s**2 / sum(s**2)`, and PCA is
**SVD on centred data**. Nothing else.

Five components carry 98.7%, and the sixth adds 0.1% — the same cliff, seen
through a different name.

---

## What it costs

Dimensionality reduction is not free, and the costs are rarely stated:

| Cost | Detail |
|---|---|
| **Interpretability** | A component is a mixture of every original feature. "PC1" means nothing to a stakeholder |
| **A fitted transform** | It must be fitted on train only and travel with the model — [Data-Science 04](../../Data-Science/lessons/04-features-and-pipelines.md) |
| **Scale sensitivity** | PCA on unscaled features finds the feature with the largest units |
| **It is unsupervised** | The directions of greatest variance need not be the directions that predict your target |

The last one is the trap. PCA can throw away the very direction that separates
your classes, because that direction happened to have small variance. **Check
against the model, not against the explained-variance number.**

And the honest default: **for tabular data with fewer than a few hundred
features, do not reduce at all.** Regularisation (Machine-Learning) handles
correlated features, and keeps them interpretable.

---

## Where this appears

| Use | Where |
|---|---|
| Embeddings are a learned low-rank representation | [LLM 05](../../LLM-and-GenAI/lessons/05-embeddings-and-search.md) |
| LoRA is literally a low-rank update | [Optimization 07](../../Optimization/lessons/07-parameter-efficient-finetuning.md) |
| Compressing a model's weights | [Optimization 09](../../Optimization/lessons/09-pruning-and-sparsity.md) |
| Finding effective rank before modelling | [Data-Analysis](../../Data-Analysis) |
| Recommender factorisation | Classic SVD use |

**LoRA is the one worth pausing on.** It says: the update a fine-tune needs is
low-rank, so learn `A @ B` with tiny inner dimension instead of a full matrix.
That is this lesson, applied to a 7B model, and it is why
[HPC lesson 01](../../HPC-and-Cloud/lessons/01-will-it-fit.md) can take a job
from 84 GB to a T4.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| PCA on unscaled features | It finds the column measured in metres |
| Choosing k by eye | The ratio jump computes it |
| Fitting PCA before the train/test split | Leakage, [Data-Science 03](../../Data-Science/lessons/03-the-data-you-have.md) |
| Assuming variance = usefulness | It is unsupervised; your target was not consulted |
| Reducing 30 tabular features | Regularisation is better and keeps meaning |
| Reporting PC1 to a stakeholder | It is a mixture of everything |

---

## Exercises

1. Compute the singular values of your own feature matrix. Where is the cliff?
2. Compare `sklearn`'s `PCA(n_components=5)` output with the SVD above. Confirm
   they match.
3. Run PCA before and after scaling on a dataset with mixed units. How much does
   the first component change?
4. Fit a model on 5 PCA components and on the raw features. Which wins, and by
   more than the fold spread?
5. Construct a dataset where the most predictive direction has the *smallest*
   variance, and watch PCA discard it.

---

**Next:** [Lesson 04 — Derivatives and Gradients](04-derivatives.md)
