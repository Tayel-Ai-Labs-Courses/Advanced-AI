# Lesson 01 — What a Recommender Actually Is

**Goal:** see the problem's real shape before reaching for an algorithm.

## What you will learn

- Why this is a ranking problem, not a prediction problem
- Implicit feedback, and the two things it does not tell you
- The long tail, measured
- The baseline that most recommenders never beat

---

## The problem shape

A recommender is not "predict how much this customer will like this product".
It is:

> **Out of 300 products, choose the 10 to show. Now.**

Three consequences that decide everything later:

**Only the top of the list matters.** Nobody scrolls to position 200. A model
that is excellent on average and wrong about its top 10 is useless, which is
why [lesson 02](02-evaluating.md) throws away RMSE.

**You never see the counterfactual.** You know the customer bought what you
showed. You do not know what they would have bought if you had shown something
else. [Lesson 08](08-the-feedback-loop.md) is about living with that.

**The output is a decision, not a number** — which puts this course on the same
footing as
[Data-Science 06](../../Data-Science/lessons/06-evaluating-the-decision.md):
the thing to measure is what the list is worth, not how close a score was.

---

## The data

```python
import numpy as np, pandas as pd

def build(seed=0, n_users=2000, n_items=300, days=180):
    """A Cairo coffee chain's order history: who ordered what, and when."""
    rng = np.random.default_rng(seed)
    K = 5
    U = rng.normal(size=(n_users, K))          # customer taste
    V = rng.normal(size=(n_items, K))          # product position in taste space
    pop = rng.pareto(1.1, n_items) + 1         # a few products sell constantly
    pop = pop / pop.sum()
    activity = rng.pareto(1.3, n_users) + 1    # most customers order rarely
    n_events = (activity / activity.sum() * 60_000).astype(int) + 2

    signup = np.sort(rng.integers(0, days, size=n_users))
    signup[:int(n_users * 0.55)] = rng.integers(0, 40, size=int(n_users * 0.55))

    rows = []
    for u in range(n_users):
        score = U[u] @ V.T
        p = np.exp(score - score.max()) * pop
        p = p / p.sum()
        k = min(n_events[u], 120)
        picks = rng.choice(n_items, size=k, p=p, replace=True)
        ts = np.clip(np.sort(rng.integers(signup[u], days + 1, size=k)), 0, days - 1)
        rows += [(u, int(i), int(t)) for i, t in zip(picks, ts)]
    return pd.DataFrame(rows, columns=["user", "item", "day"]).drop_duplicates(
        ["user", "item"]).reset_index(drop=True)
```

```python
df = build()
print(f"{len(df):,} interactions, {df.user.nunique():,} customers, {df.item.nunique()} products")
c = df.item.value_counts()
print(f"top item {c.iloc[0]} orders; median item {int(c.median())}; "
      f"top 10 items = {c.head(10).sum()/len(df):.1%} of all orders")
u = df.user.value_counts()
print(f"median customer {int(u.median())} products; p90 {int(u.quantile(.9))}; max {u.max()}")
tr, te = df[df.day < 150], df[df.day >= 150]
newc = te[~te.user.isin(set(tr.user))].user.nunique()
print(f"customers whose first order is after day 150: {newc:,}")
```

```text
18,760 interactions, 2,000 customers, 300 products
top item 1243 orders; median item 28; top 10 items = 29.4% of all orders
median customer 8 products; p90 17; max 58
customers whose first order is after day 150: 354
```

Four numbers that shape every decision in this course.

**The matrix is 99.7% empty.** 18,760 interactions out of 2,000 x 300 =
600,000 possible pairs. Every method here is a way of guessing the other
581,240 cells, and that sparsity is why simple methods work and complicated
ones overfit.

**Ten products out of 300 are 29.4% of all orders.** The distribution is a long
tail, not a bell curve. This is what makes the popularity baseline strong, and
what makes [lesson 06](06-beyond-accuracy.md) necessary.

**The median customer has ordered 8 products** and the most active has 58. You
are not building one model for one kind of user; the 8-product customer and the
58-product customer are different problems.

**354 customers have not appeared yet** at the point you would train. They are
[lesson 05](05-cold-start.md), and that lesson contains the most alarming
number in the course.

---

## Implicit feedback

The table above has no ratings. It has **events**: this customer ordered this
product. That is implicit feedback, and it is what almost all real systems
have.

```text
EXPLICIT                          IMPLICIT
a 4-star rating                   an order, a click, a play
rare, biased to strong opinions   abundant, every interaction
a 1-star means "I disliked it"    no negative signal exists
                                  absence is ambiguous
```

**The missing half is the problem.** A customer who never ordered the vanilla
latte may dislike it, or may never have seen it. Those are opposite facts and
the data cannot distinguish them.

Two practical rules follow:

**Never treat a non-interaction as a negative.** It is unknown, not disliked.
[Lesson 04](04-matrix-factorization.md) handles this with confidence weights
rather than labels.

**An interaction is not endorsement either.** Someone ordered it; they may have
hated it. If you have a return rate, a skip rate or a rating, that is a second
signal worth more than the interaction.

---

## The baseline

Before any algorithm: **recommend the most popular products to everybody.**

```python
from scipy.sparse import csr_matrix

N = 300
tr, te = df[df.day < 150], df[df.day >= 150]
te = te[te.user.isin(set(tr.user))]
seen = {u: set(g) for u, g in tr.groupby("user").item}
truth = {u: set(g) for u, g in te.groupby("user").item}
pop = tr.item.value_counts().reindex(range(N), fill_value=0).values.astype(float)
rng = np.random.default_rng(0)

def recall_at_k(score_fn, k=10):
    hits = tot = 0
    for u, want in truth.items():
        s = score_fn(u).astype(float).copy()
        for i in seen.get(u, ()):
            s[i] = -np.inf                       # do not re-recommend
        top = np.argsort(-s)[:k]
        hits += len(set(top.tolist()) & want)
        tot += min(len(want), k)
    return hits / tot

print(f"{'recommend to everyone':<26}{'recall@10':>11}")
print(f"{'random products':<26}{recall_at_k(lambda u: rng.random(N)):>11.4f}")
print(f"{'the 10 most ordered':<26}{recall_at_k(lambda u: pop):>11.4f}")
print(f"\n{len(truth):,} customers scored on orders from day 150 onwards.")
```

```text
recommend to everyone       recall@10
random products                0.0335
the 10 most ordered            0.2324

1,225 customers scored on orders from day 150 onwards.
```

**Showing everybody the same ten products gets 23.2% of their next orders
right** — nearly **7x** random, with no model, no training and no personalisation
at all.

That number is the bar. Every later lesson is measured against it, and in
industry a shocking number of recommender projects never clear it by enough to
pay for themselves. The same lesson
[Time-Series 01](../../Time-Series-and-Forecasting/lessons/01-baselines.md)
teaches about forecasting applies here with more force, because popularity is a
*strong* baseline rather than a weak one: the long tail guarantees it.

**Write down the popularity number before you build anything**, and report
every later model as a difference from it.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Treating it as rating prediction | Only the top 10 is shown; see lesson 02 |
| Treating a non-interaction as a negative | Unknown is not disliked |
| Treating an interaction as endorsement | They bought it; they may have hated it |
| Starting with a neural model | Popularity gets 0.2324 for free |
| No popularity baseline recorded | You cannot say what your model is worth |
| Recommending what the customer already has | Always mask seen items; it inflates every metric |
| One model for all users | The 8-product and 58-product customers differ |

---

## Exercises

1. Compute the popularity baseline on your own data. What is recall@10?
2. Plot the item-frequency distribution. What share is the top 10%?
3. How many of your users have fewer than 3 interactions?
4. Find one place where your pipeline treats a missing interaction as a
   negative.
5. Remove the seen-item masking and re-run. How much does the metric inflate?

---

**Next:** [Lesson 02 — Evaluating a Recommender](02-evaluating.md)
