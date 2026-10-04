# Lesson 05 — Cold Start

**Goal:** find out how much of your traffic your recommender cannot serve, and
serve it anyway.

## What you will learn

- The number your offline evaluation is hiding
- Four cold-start cases, and which actually hurts
- What to do with no history at all
- Content features and the hybrid, honestly

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

## The number nobody reports

Every evaluation in lessons 02-04 began with a line that looks harmless:

```python
# no-run
te = te[te.user.isin(set(tr.user))]      # keep only customers we have seen
```

Here is what that line threw away.

```python
tr_all, te_all = df[df.day < 150], df[df.day >= 150]
cold = te_all[~te_all.user.isin(set(tr_all.user))]
print(f"customers in the test period        {te_all.user.nunique():>6,}")
print(f"  of them, with no history at all   {cold.user.nunique():>6,}"
      f"  ({cold.user.nunique()/te_all.user.nunique():.1%})")
print(f"their orders                        {len(cold):>6,}"
      f"  ({len(cold)/len(te_all):.1%} of test orders)")
```

```text
customers in the test period         1,579
  of them, with no history at all      354  (22.4%)
their orders                         3,398  (49.1% of test orders)
```

**22.4% of the customers in the test period have no history — and they account
for 49.1% of its orders.**

Read that twice. Every number in lessons 02, 03 and 04 — the 0.3963, the
0.3194, the whole comparison — was computed on **the half of the traffic the
model can serve**, and silently excluded the other half, where collaborative
filtering produces nothing at all.

This is not a quirk of synthetic data. It is the normal state of any growing
product: new customers are, by definition, the ones you have no history for,
and they are a large share of activity precisely because the business is
growing.

Three things follow immediately:

**Report coverage of users, not just of the catalogue.** "recall@10 = 0.3963
on 78% of customers, who are 51% of orders" is the honest sentence. A single
number is a claim about a filtered population, and
[Data-Science 10](../../Data-Science/lessons/10-limits-and-fairness.md) is the
general version of this failure — the aggregate was fine, a slice was broken.

**The baseline is not optional here.** Popularity serves a brand-new customer
perfectly well. Your sophisticated model serves them not at all, so a fair
comparison across *all* traffic may well favour the simple thing.

**Design for the cold case first.** It is the majority of orders.

---

## The four cases

| Case | What is missing | How bad |
|---|---|---|
| **New user** | No row | Common and constant. The 49.1% above |
| **New item** | No column | Ships every time the catalogue changes |
| **Sparse user** | 1-2 interactions | The majority of users in a long tail |
| **New system** | Nothing at all | Once, at launch |

The one teams prepare for is the fourth, which happens once. The one that
decides the product is the first, which happens every day.

For the new item the business stake is sharper than it looks: **a product no
one has ordered cannot be recommended, so no one orders it.** Collaborative
filtering will quietly bury every new product you launch unless something
forces exposure.

---

## What to do with no history

In order of how much they cost to build:

```text
1. POPULARITY                 0.2324 recall@10, works for everyone, today
2. POPULARITY IN A SEGMENT    city, channel, time of day, device
                              anything you know at the first request
3. ASK                        an onboarding question: 3 taps, instant taste
4. CONTENT FEATURES           recommend by product attributes, not co-occurrence
5. EXPLORE DELIBERATELY       spend a slot learning, not earning
```

The first is free and is 0.2324. **Do it before anything else**, and note that
"we have no recommender for new users" is never the truth — you always have
popularity, and the only question is whether you shipped it.

Option 3 is underrated. Three taps on signup produce a better cold-start
profile than any amount of modelling, because the problem is missing
information and asking is how you get information.

Option 5 is
[Reinforcement-Learning 02](../../Reinforcement-Learning/lessons/02-bandits.md):
one slot in ten given to something you are uncertain about turns a cold user
warm in a handful of sessions, and the explore/exploit arithmetic there is
exactly the right frame.

---

## Content features and the hybrid

Content-based recommendation scores a product by **its attributes** rather than
by who else took it:

```text
product features    category, price band, hot/cold, caffeine, size, tags
user profile        the average feature vector of what they took
score               similarity between the two
```

What it buys you, and what it does not:

| | Collaborative | Content |
|---|---|---|
| New item | ✗ | **✓** — features exist before any order |
| New user | ✗ | ✗ — still needs *their* history |
| Finds surprising pairs | **✓** | ✗ — stays inside the obvious |
| Needs a feature pipeline | ✗ | ✓, and it is real work |

**Content fixes the new-item case and does not fix the new-user case.** That is
the single most useful thing to know about it, and it is routinely
misremembered in both directions.

The practical system is a **hybrid with a fallback chain**, not a clever blend:

```python
# no-run
def recommend(user, k=10):
    if history(user) >= 5:
        return collaborative(user, k)            # lesson 03 / 04
    if history(user) >= 1:
        return blend(collaborative(user, k), content(user, k))
    if segment_known(user):
        return popular_in_segment(user, k)
    return popular(k)                            # always available
```

Four properties worth copying: it **always returns something**, each branch is
**independently measurable**, the thresholds are parameters you can tune, and
the last line cannot fail. That is the same shape as the serving fallback in
[MLOps 06](../../MLOps/lessons/06-serving.md) — a model's outage should
degrade the product, not take it down.

**Measure each branch separately.** An aggregate number over a fallback chain
tells you nothing about which branch is weak, and the weak one is usually the
one serving the most traffic.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Filtering cold users out of the evaluation | Here it hid 49.1% of orders |
| Reporting one number across a fallback chain | You cannot see which branch is weak |
| No popularity fallback | "No recommendations" where 0.2324 was free |
| Expecting content features to fix new users | They fix new *items* |
| Never asking the user anything | Three taps beat a month of modelling |
| Letting new products start invisible | CF buries everything you launch |
| Treating cold start as a launch-day problem | It is every day, forever |

---

## Exercises

1. Compute, for your own data, the share of users **and of events** that are
   cold at prediction time.
2. Re-run lesson 03's comparison without filtering cold users, serving them
   popularity. Does the ranking of the models change?
3. Build the segment-popularity fallback. How much better than global
   popularity is it?
4. Design the three onboarding questions you would actually ask.
5. Measure recall separately for each branch of the fallback chain.
6. Take one product launched recently. How many times was it recommended in
   its first week?

---

**Next:** [Lesson 06 — Beyond Accuracy](06-beyond-accuracy.md)
