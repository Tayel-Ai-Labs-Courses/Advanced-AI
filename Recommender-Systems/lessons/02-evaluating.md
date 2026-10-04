# Lesson 02 — Evaluating a Recommender

**Goal:** get a number you can trust before you build anything that needs one.

## What you will learn

- Why a random split inflates your result by 42%
- Why RMSE can be excellent and the recommender worthless
- recall@k and NDCG@k, and when each lies
- What to report

---

## Setup

```python
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix

def build(seed=0, n_users=2000, n_items=300, days=180):
    rng = np.random.default_rng(seed)
    K = 5
    U = rng.normal(size=(n_users, K)); V = rng.normal(size=(n_items, K))
    pop = rng.pareto(1.1, n_items) + 1; pop = pop / pop.sum()
    activity = rng.pareto(1.3, n_users) + 1
    n_events = (activity / activity.sum() * 60_000).astype(int) + 2
    signup = np.sort(rng.integers(0, days, size=n_users))
    signup[:int(n_users * 0.55)] = rng.integers(0, 40, size=int(n_users * 0.55))
    rows = []
    for u in range(n_users):
        score = U[u] @ V.T
        p = np.exp(score - score.max()) * pop; p = p / p.sum()
        k = min(n_events[u], 120)
        picks = rng.choice(n_items, size=k, p=p, replace=True)
        ts = np.clip(np.sort(rng.integers(signup[u], days + 1, size=k)), 0, days - 1)
        rows += [(u, int(i), int(t)) for i, t in zip(picks, ts)]
    return pd.DataFrame(rows, columns=["user", "item", "day"]).drop_duplicates(
        ["user", "item"]).reset_index(drop=True)

N = 300
df = build()
```

---

## The split

```python
def recall_at_k(score_fn, train, test, k=10):
    seen = {u: set(g) for u, g in train.groupby("user").item}
    truth = {u: set(g) for u, g in test.groupby("user").item}
    hits = tot = 0
    for u, want in truth.items():
        s = score_fn(u).astype(float).copy()
        for i in seen.get(u, ()):
            s[i] = -np.inf
        top = np.argsort(-s)[:k]
        hits += len(set(top.tolist()) & want)
        tot += min(len(want), k)
    return hits / tot

rng = np.random.default_rng(0)
m = rng.random(len(df)) < 0.2
splits = {"random 80/20": (df[~m], df[m]),
          "time-based (day<150)": (df[df.day < 150], df[df.day >= 150])}

print(f"{'split':<26}{'train':>9}{'test':>8}{'users scored':>14}{'recall@10':>12}")
for name, (tr, te) in splits.items():
    pop = tr.item.value_counts().reindex(range(N), fill_value=0).values.astype(float)
    te2 = te[te.user.isin(set(tr.user))]
    r = recall_at_k(lambda u: pop, tr, te2)
    print(f"{name:<26}{len(tr):>9,}{len(te2):>8,}{te2.user.nunique():>14,}{r:>12.4f}")
print("\nsame data, same model: popularity. Only the split changed.")
```

```text
split                         train    test  users scored   recall@10
random 80/20                 15,021   3,732         1,543      0.3288
time-based (day<150)         11,834   3,528         1,225      0.2324

same data, same model: popularity. Only the split changed.
```

**The random split reports 0.3288 where the honest one reports 0.2324 — a 42%
inflation, from the same model on the same data.**

Why: a random split lets the model train on a customer's **later** orders and
be tested on their **earlier** ones. In production that is not available. You
would be reporting a number your system can never achieve.

This is the same failure
[Time-Series 02](../../Time-Series-and-Forecasting/lessons/02-evaluating.md)
measured for forecasting (MAE 85.3 against 107.7), and it is more common here
because almost every recommender tutorial uses a random split.

**Split by time. Always.** And note the third column: the honest split scores
**1,225** customers instead of 1,543, because the rest are new. Holding that
thought until [lesson 05](05-cold-start.md) is the point of this course.

---

## Why RMSE is the wrong metric

Suppose you do have ratings. The classic metric is RMSE on held-out ratings.
Here is what it misses.

```python
rng2 = np.random.default_rng(1)
Ud = np.random.default_rng(0).normal(size=(df.user.max()+1, 5))
Vd = np.random.default_rng(0).normal(size=(N, 5))
raw = (Ud[df.user.values] * Vd[df.item.values]).sum(1)
rate = np.clip(np.round(3 + raw/2 + rng2.normal(scale=0.8, size=len(df))), 1, 5)
d = df.assign(rating=rate)
tr, te = d[d.day < 150], d[d.day >= 150]
te = te[te.user.isin(set(tr.user))]

gmean = tr.rating.mean()
umean = tr.groupby("user").rating.mean()
imean = tr.groupby("item").rating.mean()
preds = {
 "global mean":      np.full(len(te), gmean),
 "user mean":        te.user.map(umean).fillna(gmean).values,
 "item mean":        te.item.map(imean).fillna(gmean).values,
 "user + item bias": (te.user.map(umean).fillna(gmean).values
                      + te.item.map(imean).fillna(gmean).values - gmean),
}

def rmse(p):
    return np.sqrt(((te.rating.values - p) ** 2).mean())

def rank_corr(pred):
    dd = te.assign(p=pred); cs, flat = [], 0
    for u, g in dd.groupby("user"):
        if len(g) < 3 or g.rating.nunique() < 2:
            continue
        if np.ptp(g.p) == 0:                 # one score for every item: no ranking
            flat += 1
            continue
        cs.append(np.corrcoef(g.rating.rank(), g.p.rank())[0, 1])
    return (float(np.mean(cs)) if cs else None), flat

print(f"{'predictor':<20}{'RMSE':>8}{'rank corr':>12}{'users it cannot rank':>23}")
for n, p in preds.items():
    c, flat = rank_corr(p)
    shown = f"{c:.4f}" if c is not None else "none"
    print(f"{n:<20}{rmse(p):>8.4f}{shown:>12}{flat:>23}")
print(f"\n{len(te):,} held-out ratings. RMSE lower is better, rank corr higher.")
```

```text
predictor               RMSE   rank corr   users it cannot rank
global mean           1.2013        none                    468
user mean             1.3237        none                    468
item mean             1.1083      0.3322                      0
user + item bias      1.2510      0.3322                      0

3,528 held-out ratings. RMSE lower is better, rank corr higher.
```

**"Global mean" has a better RMSE than "user + item bias" — 1.2013 against
1.2510 — and cannot rank a single one of the 468 customers.** It gives every
product the same score. As a recommender it is worth exactly nothing, and RMSE
cannot see that.

Two things to take from the table:

**RMSE measures the wrong thing.** It averages error over all held-out pairs,
including the 290 products the customer will never be shown. The recommender's
job is the order of the top 10, and a metric that is blind to order is blind to
the job.

**The RMSE ranking and the usefulness ranking disagree.** `global mean` beats
`user + item bias` on RMSE and loses completely in practice. Pick a metric that
moves when the product gets better; this one does not.

---

## The metrics that do work

```text
recall@k    of the items the user actually took, what fraction were in your
            top k?   Simple, interpretable, ignores order within the k

precision@k of your k recommendations, what fraction were taken?
            Pulled down by users with few test items; use with care

NDCG@k      like recall, but position 1 counts more than position 10
            The closest to what a ranked list is worth

coverage    what fraction of the catalogue ever gets recommended?
            Not an accuracy metric. Lesson 06 shows why you need it anyway
```

Pick **one** primary metric and report the others beside it. The common choice
is NDCG@10 with recall@10 and coverage as context.

Three rules that keep the number honest:

**Mask items the user has already taken.** Recommending the coffee someone
orders daily scores beautifully and sells nothing. Every evaluation in this
course masks them; removing the mask is exercise 5 of lesson 01 and the
inflation is large.

**Report k the product actually shows.** If the app shows 5 slots, NDCG@20 is
a number about a screen that does not exist.

**Report how many users you scored.** The `users scored` column above fell
from 1,543 to 1,225 between splits, and in [lesson 05](05-cold-start.md) it
turns out that silence is hiding half the traffic.

---

## What to report

```text
PRIMARY      NDCG@10, time-based split, against the popularity baseline
BESIDE IT    recall@10, coverage, users scored, users skipped
ALWAYS       the baseline's number, not just yours
NEVER        a metric from a random split
```

The last line matters because the random split is the default in every library
tutorial, and it is the single most common reason an offline number does not
survive contact with production — which is
[lesson 08](08-the-feedback-loop.md).

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A random train/test split | 42% inflation here, and it never reproduces |
| RMSE as the headline | Global mean beats a real model and ranks nothing |
| Not masking already-taken items | Scores well, sells nothing |
| Reporting k the UI does not show | A number about a screen that does not exist |
| Not reporting how many users were skipped | Lesson 05: it was half the orders |
| No baseline in the report | 0.31 means nothing without popularity's 0.2324 |
| Tuning on the test split | [Data-Science 03](../../Data-Science/lessons/03-the-data-you-have.md) |

---

## Exercises

1. Re-run the split comparison on your own data. What is your inflation factor?
2. Compute NDCG@10 for the popularity baseline and compare to recall@10.
3. Count how many users your evaluation silently skips.
4. If you have ratings, compute RMSE and a ranking metric. Do they agree?
5. Change k to what your product actually shows. Does the ordering of your
   models change?

---

**Next:** [Lesson 03 — Collaborative Filtering](03-collaborative-filtering.md)
