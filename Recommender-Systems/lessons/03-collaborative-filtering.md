# Lesson 03 — Collaborative Filtering

**Goal:** make a recommendation from nothing but the interaction matrix.

## What you will learn

- The one idea behind every collaborative method
- Item-item similarity, measured against the baseline
- Why item-item beats user-user in practice
- What collaborative filtering cannot do

---

## The one idea

> **Customers who ordered what you ordered also ordered this.**

That is the entire concept. No features, no product descriptions, no
demographics — only the matrix of who-took-what. It works because taste is
correlated, and it is remarkable how far that gets you.

Two directions:

```text
USER-USER     find customers similar to you, recommend what they took
              similarity between 2,000 customers: a 2,000 x 2,000 matrix
              changes every time anyone orders anything

ITEM-ITEM     find products similar to what you took, recommend those
              similarity between 300 products: a 300 x 300 matrix
              changes slowly — products do not change taste overnight
```

Item-item wins in almost every real system, for three reasons that have
nothing to do with accuracy: there are usually fewer items than users, the
similarities are stable enough to precompute nightly, and the recommendation
is explainable — *"because you ordered X"* is a sentence you can put in the UI.

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

---

## Item-item, measured

```python
norms = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()) + 1e-9
S = (X.T @ X).toarray() / np.outer(norms, norms)   # cosine between items
np.fill_diagonal(S, 0)                             # never recommend the item itself

rng = np.random.default_rng(0)
models = {
    "random":       lambda u: rng.random(N),
    "popularity":   lambda u: pop,
    "item-item CF": lambda u: X[u].toarray().ravel() @ S,
}
print(f"{'model':<18}{'recall@10':>11}{'NDCG@10':>10}{'catalogue covered':>20}")
for n, f in models.items():
    r, nd, cov = evaluate(f)
    print(f"{n:<18}{r:>11.4f}{nd:>10.4f}{cov:>19.1%}")
print(f"\n{len(truth):,} customers scored, time-based split, {N} products.")
```

```text
model               recall@10   NDCG@10   catalogue covered
random                 0.0335    0.0176             100.0%
popularity             0.2324    0.1788               7.3%
item-item CF           0.3963    0.3093              51.3%

1,225 customers scored, time-based split, 300 products.
```

**Item-item CF gets 0.3963 against popularity's 0.2324 — a 70% improvement**,
from four lines of linear algebra and no training loop.

That is the result the method deserves its reputation for. But read the third
column too, because it is where the rest of this course comes from.

**Popularity recommends 7.3% of the catalogue** — 22 products out of 300, to
everybody. Every other product might as well not exist.

**Item-item recommends 51.3%** — seven times more, and still leaves half the
catalogue permanently unseen.

**Random covers 100% and is useless.** So coverage is not a goal on its own; it
is a constraint you trade against accuracy, which is
[lesson 06](06-beyond-accuracy.md).

---

## How the scoring works

```python
u = int(max(truth, key=lambda x: len(seen.get(x, ()))))
s = X[u].toarray().ravel() @ S
for i in seen.get(u, ()):
    s[i] = -np.inf
top = np.argsort(-s)[:5]
print(f"customer {u} has ordered {len(seen[u])} products")
print(f"top 5 recommendations: {top.tolist()}")
print(f"of which actually ordered later: {sorted(set(top.tolist()) & truth[u])}")
```

```text
customer 667 has ordered 43 products
top 5 recommendations: [255, 178, 235, 247, 166]
of which actually ordered later: [178, 235]
```

Two of the five landed. That is a good list — and worth looking at squarely,
because tables of aggregate metrics make recommenders feel more accurate than
they are. **At recall@10 of 0.3963, roughly two thirds of what you show is
wrong**, and that is the normal, successful case. The product has to be
designed for a list that is mostly misses.

The score for a product is **the sum of its similarities to everything the
customer has already taken**. One matrix multiply per customer, which is why
this scales: the expensive part, `S`, is computed once.

Note the row `X[u]` is binary here. Two refinements worth knowing:

**Weight recent interactions more.** An order from last week says more about
next week than one from six months ago.

**Damp the popular items.** Cosine already divides by the norm, which is a
partial fix; dividing by `count^alpha` with alpha around 0.5 is the usual knob,
and it trades accuracy for coverage directly.

---

## What it cannot do

Collaborative filtering has exactly one input, so its limits are sharp:

| Limit | Why | Where it is handled |
|---|---|---|
| A new customer | No row in the matrix | [Lesson 05](05-cold-start.md) |
| A new product | No column anybody has taken | [Lesson 05](05-cold-start.md) |
| A rare product | Too few co-occurrences to be reliable | [Lesson 06](06-beyond-accuracy.md) |
| Explaining *why* beyond co-occurrence | There is no feature to point at | [Lesson 05](05-cold-start.md) |
| Knowing the recommendation is bad | It has no notion of quality | [Lesson 08](08-the-feedback-loop.md) |

The first two are the ones that decide whether the system works in production,
and the numbers in lesson 05 are worse than most people expect.

One more, which is a counting problem rather than a modelling one:
**similarity on a sparse matrix is noisy.** Two products co-ordered by three
customers have a similarity you should not trust, exactly as
[Prompt-Engineering 06](../../Prompt-Engineering/lessons/06-the-iteration-loop.md)
showed for a 12-example eval set. A minimum co-occurrence count before a
similarity is used is a one-line change and usually helps.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| User-user at scale | The similarity matrix is users x users and changes constantly |
| Not zeroing the diagonal | Every item is most similar to itself |
| Not masking seen items | Recommends the daily coffee; scores well, sells nothing |
| Raw co-occurrence instead of cosine | Popular items dominate every list |
| Trusting a similarity from 3 co-occurrences | Noise; set a minimum count |
| Ignoring the coverage column | 51.3% means half the catalogue is invisible |
| Recomputing `S` per request | It is nightly work, not request-time work |

---

## Exercises

1. Implement user-user CF and compare both accuracy and wall-clock time.
2. Add a minimum co-occurrence threshold. What does it do to recall and
   coverage?
3. Damp by `count ** 0.5` and plot the accuracy/coverage trade.
4. Weight interactions by recency. Does it help?
5. For one customer, print the three already-ordered products that contributed
   most to their top recommendation. That is your explanation string.

---

**Next:** [Lesson 04 — Matrix Factorization](04-matrix-factorization.md)
