# Lesson 04 — Matrix Factorization

**Goal:** learn a compact representation of taste, and see what it buys you.

## What you will learn

- The factorization idea in one picture
- ALS for implicit feedback, including why confidence is not a label
- The accuracy/coverage trade, measured — the better model lost on accuracy
- Choosing k

---

## The idea

Item-item CF stores a 300x300 similarity matrix. Factorization instead learns a
small vector per customer and per product, so that:

```text
          interaction(u, i)  ≈  P[u] · Q[i]

          P : 2,000 x k     what each customer likes
          Q :   300 x k     what each product is
          k : 16            how many dimensions of taste
```

With k=16 that is 2,000x16 + 300x16 = **36,800 numbers instead of 600,000
cells**. The compression is the point: forcing taste through 16 dimensions
makes the model generalise instead of memorising co-occurrences.

The dimensions are learned, not designed. One may turn out to be
"sweet vs bitter", most will not be interpretable, and
[Foundations 03](../../Foundations/lessons/03-decomposition.md) is the same
mathematics viewed as decomposition.

---

## Confidence, not labels

For implicit data you cannot train on "liked / did not like", because
[lesson 01](01-what-a-recommender-is.md) established there is no negative
signal. The standard fix treats every cell as observed, with a **confidence**:

```text
preference  p(u,i) = 1 if the customer took it, else 0
confidence  c(u,i) = 1 + alpha * (times taken)

           every cell is a training example
           the ones they took are high-confidence 1s
           the ones they did not are LOW-confidence 0s, not negatives
```

That distinction is the whole trick. A product they never ordered contributes
to the loss with weight 1, while one they ordered contributes with weight
`1 + alpha`. With alpha = 20, a single order is worth 21 non-orders.

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

## ALS

```python
def als(X, k=16, reg=0.1, alpha=20, iters=12, seed=0):
    """Alternating least squares for implicit feedback."""
    r = np.random.default_rng(seed)
    nu, ni = X.shape
    P = r.normal(scale=0.01, size=(nu, k))
    Q = r.normal(scale=0.01, size=(ni, k))
    Xd = X.toarray()
    C = 1 + alpha * Xd                      # confidence
    for _ in range(iters):
        QtQ = Q.T @ Q                       # precomputed: the "all zeros" part
        for u in range(nu):
            Cu = C[u]; idx = Xd[u] > 0
            A = QtQ + (Q[idx].T * (Cu[idx] - 1)) @ Q[idx] + reg*np.eye(k)
            b = (Q[idx].T * Cu[idx]).sum(1)
            P[u] = np.linalg.solve(A, b)
        PtP = P.T @ P
        for i in range(ni):
            Ci = C[:, i]; idx = Xd[:, i] > 0
            A = PtP + (P[idx].T * (Ci[idx] - 1)) @ P[idx] + reg*np.eye(k)
            b = (P[idx].T * Ci[idx]).sum(1)
            Q[i] = np.linalg.solve(A, b)
    return P, Q

norms = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()) + 1e-9
S = (X.T @ X).toarray() / np.outer(norms, norms); np.fill_diagonal(S, 0)
P, Q = als(X)

print(f"{'model':<18}{'recall@10':>11}{'NDCG@10':>10}{'catalogue covered':>20}")
for n, f in {"popularity":   lambda u: pop,
             "item-item CF": lambda u: X[u].toarray().ravel() @ S,
             "ALS (k=16)":   lambda u: Q @ P[u]}.items():
    r, nd, cov = evaluate(f)
    print(f"{n:<18}{r:>11.4f}{nd:>10.4f}{cov:>19.1%}")
print(f"\n{len(truth):,} customers scored, time-based split, {N} products.")
```

```text
model               recall@10   NDCG@10   catalogue covered
popularity             0.2324    0.1788               7.3%
item-item CF           0.3963    0.3093              51.3%
ALS (k=16)             0.3194    0.2498              76.3%

1,225 customers scored, time-based split, 300 products.
```

**ALS lost on accuracy.** 0.3194 against item-item's 0.3963 — 19% worse on
recall, 19% worse on NDCG. The sophisticated method is beaten by four lines of
cosine similarity.

That ordering is real and it is common. Factorization wins on very large,
very sparse catalogues where co-occurrence counts are too thin to be reliable;
on 300 products with 18,760 interactions, co-occurrence is well measured and
compression throws information away. **Which method wins is a property of your
data, not of the methods**, and the only way to find out is the table above.

But look at coverage before concluding anything:

**ALS recommends 76.3% of the catalogue against item-item's 51.3%** — it
reaches 75 more products. Because it represents taste in a shared 16-dimensional
space rather than through direct co-occurrence, it will put a rarely-ordered
product in front of someone whose vector points at it, with no need for other
customers to have ordered the pair.

So the two models are not ranked. **One is more accurate; the other shows a
quarter more of your catalogue.** Which is better depends on whether you are
paid for click-through or for selling the inventory you are holding —
[lesson 06](06-beyond-accuracy.md) prices that choice, and
[Data-Science 06](../../Data-Science/lessons/06-evaluating-the-decision.md) is
the general form of the argument.

---

## Choosing k

```text
k too small    the model cannot express taste; underfits
k too large    it memorises; recall on the test split falls
```

There is no rule of thumb that survives contact with data. Sweep it the way
you would any hyperparameter, **on a time-based split**, and plot recall
against k. Three warnings:

**The best k for accuracy is not the best k for coverage.** Plot both.

**Regularisation and k interact.** A larger k with stronger `reg` is often
better than a smaller k, so sweep them together or you will conclude the wrong
thing.

**Re-sweep when the catalogue changes size.** k is a statement about how many
dimensions of taste your data can support, and that moves.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Treating unobserved cells as negatives | They are low-confidence zeros, not dislikes |
| Assuming factorization beats item-item | It lost here, 0.3194 against 0.3963 |
| Comparing on accuracy alone | ALS reached 25% more of the catalogue |
| Tuning k on a random split | Lesson 02: 42% inflation |
| Tuning k without tuning reg | They interact; you will pick the wrong k |
| Re-training from scratch for one new interaction | Fold-in the user vector instead |
| Using plain SVD on the zero-filled matrix | It fits the zeros, which are unknowns |

---

## Exercises

1. Sweep k over 4, 8, 16, 32, 64. Plot recall and coverage together.
2. Sweep alpha over 1, 10, 20, 50. What does confidence weighting buy?
3. Blend ALS and item-item scores. Does the blend beat both?
4. Fold in a new user from one interaction without re-training.
5. Run plain SVD on the zero-filled matrix and compare. Why is it worse?

---

**Next:** [Lesson 05 — Cold Start](05-cold-start.md)
