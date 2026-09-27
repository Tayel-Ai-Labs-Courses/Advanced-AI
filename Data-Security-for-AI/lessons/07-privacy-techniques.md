# Lesson 07 — Privacy Techniques

**Goal:** put a number on privacy instead of an adjective, and pay the utility
cost on purpose.

## What you will learn

- Differential privacy, and what epsilon buys
- The budget, which is the part people skip
- Where DP belongs and where cheaper controls do
- Data minimisation, which is free

---

## The promise

Differential privacy gives a formal guarantee: **the released result is almost
the same whether or not any single person is in the data.** An attacker who sees
the output learns almost nothing about any individual, no matter what else they
know.

"Almost" is quantified by **epsilon**. Smaller epsilon, stronger guarantee, more
noise.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 5_000
df = pd.DataFrame({
    "governorate": rng.choice(
        ["Cairo", "Giza", "Alexandria", "Dakahlia", "Sharqia", "Qalyubia",
         "Beheira", "Gharbia", "Minya", "Asyut"], n,
        p=[.22, .16, .12, .10, .09, .08, .07, .06, .05, .05]),
})

def dp_count(true_count, epsilon, rng):
    return true_count + rng.laplace(0, 1.0 / epsilon)

true = int((df.governorate == "Minya").sum())
print(f"true count of people in Minya: {true}")
print(f"{'epsilon':>9}{'mean answer':>13}{'std':>8}{'max error seen':>16}")
for eps in (0.1, 0.5, 1.0, 5.0):
    r = np.random.default_rng(1)
    draws = np.array([dp_count(true, eps, r) for _ in range(5_000)])
    print(f"{eps:>9.1f}{draws.mean():>13.1f}{draws.std():>8.1f}"
          f"{np.abs(draws - true).max():>16.1f}")
print("\nsmaller epsilon = more privacy, more noise")
```

```text
true count of people in Minya: 238
  epsilon  mean answer     std  max error seen
      0.1        237.9    14.0            85.6
      0.5        238.0     2.8            17.1
      1.0        238.0     1.4             8.6
      5.0        238.0     0.3             1.7

smaller epsilon = more privacy, more noise
```

At **epsilon 0.1** the answer is unbiased but its standard deviation is 14, and
across 5,000 draws one answer was **85.6 off**. At epsilon 5 the noise is
negligible — and so, roughly, is the guarantee.

Two things people get wrong about this table:

**The mean is right at every epsilon.** DP noise is unbiased, so aggregate
statistics over many DP answers stay correct. It is the *individual answer* that
is uncertain, which is precisely the point.

**The maximum error is what breaks a dashboard**, not the standard deviation. A
count that is usually within 3 and occasionally 85 out will produce one
screenshot that makes everyone distrust the system.

---

## The budget is the hard part

Epsilon composes. Ask the same dataset two questions at epsilon 1 each and you
have spent **epsilon 2** — the guarantee degrades with every query.

```text
one query at eps=1.0                 -> total eps 1.0
ten queries at eps=1.0 each          -> total eps 10.0   (a weak guarantee)
ten queries at eps=0.1 each          -> total eps 1.0    (each answer is noisy)
a dashboard refreshed hourly, forever -> unbounded, unless the budget is enforced
```

So a real DP deployment needs a **privacy budget**: a total epsilon per dataset
per period, tracked and enforced, with a decision about what happens when it runs
out. Almost every "we added differential privacy" project that fails, fails
here — the noise was added and the budget was never tracked, which means the
guarantee was never real.

---

## Where DP belongs

| Situation | Use |
|---|---|
| Publishing statistics about people | **DP**, with a tracked budget |
| A dashboard over sensitive rows | Minimum group size first (lesson 02), DP if it must be finer |
| Training a model on sensitive records | DP-SGD, and expect a real accuracy cost |
| Sharing data with a partner | **Do not share rows.** Share aggregates or a model |
| An internal analyst who is authorised | Access control and logging, not DP |
| Your own team debugging | Synthetic or masked data |

Note the fifth row. **DP is not a substitute for access control**, and applying
it to authorised internal users is expensive theatre. DP is for when the output
itself leaves your trust boundary.

DP-SGD, for training, adds noise to gradients and clips them per example. It
gives a genuine bound on lesson 03's membership inference — and it costs
accuracy, often several points. Measure the attack AUC before reaching for it:
lesson 03 got from 0.805 to 0.547 by capping tree depth, for free.

---

## The techniques that cost nothing

Before any of the above:

| Technique | What it is |
|---|---|
| **Do not collect it** | The only perfect control. Every field you do not have cannot leak |
| **Delete it on a schedule** | Retention limits turn a permanent risk into a temporary one |
| **Hash what you only need to match on** | If you never need the raw value, do not store it |
| **Separate the identity from the behaviour** | Two tables, two access policies, joined only when needed |
| **Aggregate at ingestion** | If daily totals are enough, do not keep the rows |
| **Synthetic data for development** | Removes the most common incident (lesson 01) |

**Data minimisation is the highest-value privacy work available**, and it is the
least glamorous. A field that was never collected needs no noise, no budget, no
access policy and no breach notification.

The question to ask at the design review is not "how do we protect this field?"
but **"what breaks if we do not collect it at all?"** — and often the answer is
nothing, because it was collected in case it was useful.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Adding noise without a budget | The guarantee is not real |
| Quoting epsilon without the composition | Ten queries at eps=1 is eps=10 |
| DP where access control was the answer | Expensive theatre |
| Reaching for DP-SGD before fixing overfitting | Lesson 03 got most of the way for free |
| Reporting the std and not the max error | One 85-off screenshot destroys trust |
| Collecting fields "in case" | Every one is a permanent liability |
| No retention policy | Risk that only ever grows |

---

## Exercises

1. Implement a DP mean, not just a count, and report the noise at epsilon 1 for
   a group of 50 and a group of 5,000.
2. Track a budget across ten queries and enforce a cap. What does your system do
   when the budget is exhausted?
3. Take a table you own and list every column. For each: is it needed, and what
   breaks if it is dropped? Count how many you could remove.
4. Write the retention policy for one dataset: how long, who deletes it, and how
   deletion is verified.
5. Compare DP-SGD against simple regularisation on one model, reporting both the
   attack AUC from lesson 03 and the test accuracy.

---

**Next:** [Lesson 08 — The Security Review](08-security-review.md)
