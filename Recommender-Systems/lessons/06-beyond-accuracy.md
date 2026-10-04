# Lesson 06 — Beyond Accuracy

**Goal:** see what an accurate recommender does to your catalogue, and decide
whether you want it.

## What you will learn

- Popularity bias, measured: the tail gets 0.6% of recommendations
- Why recall by user segment is counterintuitive, and what it really says
- Coverage, diversity, novelty — what each is for
- The re-ranking knob

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
tr, te = df[df.day < 150], df[df.day >= 150]
te = te[te.user.isin(set(tr.user))]
NU = df.user.max() + 1
X = csr_matrix((np.ones(len(tr)), (tr.user, tr.item)), shape=(NU, N))
seen = {u: set(g) for u, g in tr.groupby("user").item}
truth = {u: set(g) for u, g in te.groupby("user").item}
pop = np.asarray(X.sum(0)).ravel()

def evaluate(score_fn, k=10):
    """recall@k, NDCG@k, and what fraction of the catalogue is ever shown."""
    hits = tot = 0; shown = set(); ndcg = 0.0; nu = 0
    for u, want in truth.items():
        s = score_fn(u).astype(float).copy()
        for i in seen.get(u, ()):
            s[i] = -np.inf
        top = np.argsort(-s)[:k]
        shown.update(top.tolist())
        hits += len(set(top.tolist()) & want); tot += min(len(want), k)
        dcg = sum(1/np.log2(r+2) for r, i in enumerate(top) if i in want)
        idcg = sum(1/np.log2(r+2) for r in range(min(len(want), k)))
        ndcg += dcg/idcg; nu += 1
    return hits/tot, ndcg/nu, len(shown)/N
```

```python
norms = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()) + 1e-9
S = (X.T @ X).toarray() / np.outer(norms, norms); np.fill_diagonal(S, 0)
cf = lambda u: X[u].toarray().ravel() @ S
```

---

## Who actually gets recommended

```python
counts = np.zeros(N)
for u in truth:
    s = cf(u).copy()
    for i in seen.get(u, ()):
        s[i] = -np.inf
    counts[np.argsort(-s)[:10]] += 1

order = np.argsort(-pop)
for label, sl in [("top 10% most-ordered", order[:30]),
                  ("middle 40%", order[30:150]),
                  ("bottom 50% (the tail)", order[150:])]:
    print(f"{label:<26}{pop[sl].sum()/pop.sum():>8.1%} of orders   "
          f"{counts[sl].sum()/counts.sum():>7.1%} of recommendations")
print(f"\nitem-item CF, top-10 lists for {len(truth):,} customers.")
```

```text
top 10% most-ordered         51.2% of orders     68.9% of recommendations
middle 40%                   36.9% of orders     30.5% of recommendations
bottom 50% (the tail)        11.9% of orders      0.6% of recommendations

item-item CF, top-10 lists for 1,225 customers.
```

**Half your catalogue accounts for 11.9% of orders and receives 0.6% of
recommendations.** One hundred and fifty products are, for practical purposes,
no longer in the shop.

And the top 10% is **amplified**: 51.2% of demand, 68.9% of the exposure. The
recommender does not reflect the demand distribution, it sharpens it.

Two reasons this matters beyond fairness to products:

**It is self-reinforcing.** Recommended products get ordered; ordered products
accumulate co-occurrences; co-occurrences make them more recommendable. The
model's output becomes its next training set, which is
[lesson 08](08-the-feedback-loop.md) and the most dangerous property in this
course.

**Your inventory has costs the metric cannot see.** The tail may be where your
margin is, or your fresh stock, or the supplier relationship you are
protecting. recall@10 does not know any of that, so it will optimise it away
quietly.

---

## Recall by how much history a customer has

```python
hist = tr.user.value_counts()
po = lambda u: pop

def recall_for(users, fn, k=10):
    hits = tot = 0
    for u in users:
        want = truth[u]
        s = fn(u).astype(float).copy()
        for i in seen.get(u, ()):
            s[i] = -np.inf
        top = np.argsort(-s)[:k]
        hits += len(set(top.tolist()) & want); tot += min(len(want), k)
    return hits / tot if tot else float("nan")

print(f"{'history':<22}{'customers':>11}{'popularity':>12}{'item-item CF':>14}")
for lo, hi in [(1,2),(3,5),(6,10),(11,20),(21,999)]:
    us = [u for u in truth if lo <= hist.get(u, 0) <= hi]
    if len(us) < 20:
        continue
    label = f"{lo}-{hi} products" if hi < 999 else f"{lo}+ products"
    print(f"{label:<22}{len(us):>11,}{recall_for(us, po):>12.4f}"
          f"{recall_for(us, cf):>14.4f}")
```

```text
history                 customers  popularity  item-item CF
1-2 products                  150      0.3312        0.4250
3-5 products                  417      0.2911        0.4499
6-10 products                 412      0.2065        0.4069
11-20 products                181      0.1493        0.3284
21+ products                   65      0.1042        0.2508
```

**Recall is highest for the customers with the least history and falls as
history grows** — 0.4250 down to 0.2508. That is the opposite of what the
story "more data, better recommendations" predicts, and it is worth being
careful about, because the obvious reading is wrong.

It is largely **an artefact of the metric**. A customer with one future order
needs one hit in ten slots to score 1.0; a customer with thirty future orders
needs ten hits in ten slots to score the same. The denominator
`min(len(want), k)` saturates, so heavy users are graded on a much harder exam.

Two honest conclusions, and they point in different directions:

**Do not use this table to claim cold users are well served.** They are not —
[lesson 05](05-cold-start.md) showed the genuinely cold ones were excluded
entirely. This table only covers customers with at least one order.

**Do segment your metrics anyway.** The *gap between the columns* is the
informative part: item-item beats popularity by 0.09 for 1-2 product
customers and by 0.15 for 11-20 product ones. Personalisation is worth more
for customers you know, which is the sensible finding hiding behind the
confusing one.

**A single aggregate number would have shown neither.** Segment by history,
by recency, by segment — the same lesson as
[Data-Science 10](../../Data-Science/lessons/10-limits-and-fairness.md) and
[MLOps 04](../../MLOps/lessons/04-ci-gate.md)'s subgroup gate.

---

## The four non-accuracy metrics

```text
COVERAGE     fraction of the catalogue recommended at all
             lesson 03: popularity 7.3%, item-item 51.3%, ALS 76.3%

DIVERSITY    how different the items *within* one list are
             a list of ten near-identical lattes is one recommendation

NOVELTY      how unexpected the recommendations are
             average inverse popularity of what you show

SERENDIPITY  useful AND unexpected
             the hardest to measure and the only one users notice
```

The trap is optimising these directly: random recommendations have perfect
coverage, diversity and novelty, and 0.0335 recall. **They are constraints,
not objectives.**

The usable form is: *maximise NDCG@10 subject to coverage above X%*, with X
chosen by the business rather than by the model.

---

## Re-ranking

The standard fix is not a new model. It is a second pass over the top ~100:

```python
# no-run
def rerank(scores, k=10, lam=0.3):
    """Take the top 100 by score, then pick k trading relevance against novelty."""
    cand = np.argsort(-scores)[:100]
    chosen = []
    for _ in range(k):
        best, best_val = None, -np.inf
        for i in cand:
            if i in chosen:
                continue
            novelty = -np.log(pop[i] + 1)                 # rarer scores higher
            val = (1 - lam) * scores[i] + lam * novelty
            if val > best_val:
                best, best_val = i, val
        chosen.append(best)
    return chosen
```

Why this shape and not a different loss function:

**It is one parameter.** `lam` is a dial the business can turn, and you can
plot the accuracy/coverage curve across it and pick a point deliberately.

**It does not touch the model.** The expensive, well-tested scoring stays
exactly as it is.

**Candidate generation is unchanged**, so the serving architecture in
[lesson 07](07-serving.md) still holds.

Plot recall against coverage as `lam` goes from 0 to 1. You will usually find
a region where coverage rises a lot and recall falls a little — and that region
is almost always worth taking, because the recall you lose is measured and the
catalogue you gain is strategic.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Accuracy as the only metric | Half the catalogue got 0.6% of exposure |
| Optimising coverage or novelty directly | Random wins on all of them and sells nothing |
| Assuming an accurate model reflects demand | It amplifies it: 51.2% of orders → 68.9% of slots |
| Reading recall-by-segment naively | The denominator saturates; heavy users sit a harder exam |
| One aggregate number | Both findings here are invisible in it |
| A new model to fix diversity | Re-ranking the top 100 is one parameter |
| Ignoring what the tail is worth to the business | The metric cannot see margin or stock |

---

## Exercises

1. Compute the three exposure bands for your own recommender.
2. Implement `rerank` and plot recall against coverage for `lam` in 0-1.
3. Measure within-list diversity. How many of your ten are near-duplicates?
4. Segment recall by history and by recency. Where is the gap to the baseline
   smallest?
5. Ask the business what the tail is worth. Convert it to a coverage floor.
6. Pick a `lam` and justify it in one sentence with both numbers.

---

**Next:** [Lesson 07 — Serving](07-serving.md)
