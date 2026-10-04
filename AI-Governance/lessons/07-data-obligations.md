# Lesson 07 — Data Obligations

**Goal:** know what you are allowed to do with the data you have, and what
"anonymised" actually means.

## What you will learn

- The five principles that decide everything
- Re-identification, measured: five harmless columns, 88.8% unique
- Purpose limitation, which is the one ML breaks
- Deletion that reaches the model

---

## Five principles

Most data-protection regimes reduce to the same five, and each one is a
technical constraint:

| Principle | What it means | What it forbids |
|---|---|---|
| **Lawful basis** | You need a reason to process at all | "It was in the warehouse" |
| **Purpose limitation** | Use it for what it was collected for | Training on anything you can reach |
| **Minimisation** | Collect and keep the least you need | "Keep everything, it might be useful" |
| **Accuracy** | Keep it correct and current | Stale records driving live decisions |
| **Storage limitation** | Delete it when done | Retention "forever" by default |

**Purpose limitation is the one machine learning breaks**, routinely and
usually without noticing, because training data is whatever was already
collected for something else. [Lesson 03](03-the-record.md)'s dataset record
exists to force that comparison.

---

## "We removed the names"

The most common belief in this area, and it is wrong.

```python
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
N = 50000
df = pd.DataFrame({
  "birth_year":   rng.integers(1950, 2008, N),
  "gender":       rng.choice(["m", "f"], N),
  "district":     rng.choice([f"d{i}" for i in range(28)], N),
  "plan":         rng.choice(["basic", "plus", "pro"], N, p=[.6, .3, .1]),
  "signup_month": rng.integers(1, 61, N),
})

combos = [
 ("birth year",     ["birth_year"]),
 ("+ gender",       ["birth_year", "gender"]),
 ("+ district",     ["birth_year", "gender", "district"]),
 ("+ plan",         ["birth_year", "gender", "district", "plan"]),
 ("+ signup month", ["birth_year", "gender", "district", "plan", "signup_month"]),
]
print(f"{'quasi-identifiers kept':<24}{'k=1 (unique)':>14}{'k<5':>8}{'min k':>7}{'% unique':>10}")
for name, cols in combos:
    g = df.groupby(cols).size()
    sizes = df.merge(g.rename("k"), left_on=cols, right_index=True).k
    print(f"{name:<24}{(sizes==1).sum():>14,}{(sizes<5).sum():>8,}"
          f"{sizes.min():>7}{(sizes==1).mean():>9.1%}")
print(f"\n{N:,} rows, no name, no phone, no email, no id.")
```

```text
quasi-identifiers kept    k=1 (unique)     k<5  min k  % unique
birth year                           0       0    802     0.0%
+ gender                             0       0    373     0.0%
+ district                           0       4      4     0.0%
+ plan                           1,229  10,203      1     2.5%
+ signup month                  44,387  50,000      1    88.8%

50,000 rows, no name, no phone, no email, no id.
```

**Five ordinary columns and no identifiers at all, and 88.8% of 50,000 people
are unique.** Every one of those rows can be matched to a person by anyone
holding the same five facts — which, for a customer's own bank, employer, or a
leaked dataset, is not a high bar.

Watch the progression, because it is the real lesson:

**Each column alone is harmless.** Birth year alone puts everyone in a group of
at least 802. Adding gender: still 373. Adding district: still 4.

**Then one more column collapses it.** Adding plan takes 2.5% to unique;
adding signup month takes it to **88.8%**. There is no warning — the dataset
goes from safe to fully identifying in one step, and the step looks exactly
like the four before it.

**So "is this field sensitive?" is the wrong question.** The question is
whether the *combination* identifies, and that can only be answered by
computing it. Three lines of `groupby`, as above.

Practical consequences:

```text
MEASURE k      the size of the smallest group sharing a quasi-identifier
               combination. k=1 means someone is identifiable
GENERALISE     birth year -> decade; district -> governorate;
               exact month -> quarter. Each raises k a lot and costs
               little analytic value
SUPPRESS       drop the rows with k below your threshold, and say how many
DO NOT CLAIM   "anonymised" for a k=1 dataset. It is pseudonymised at best,
               which means it is still personal data with full obligations
```

The last point has a direct legal consequence: anonymous data falls outside
most data-protection regimes, pseudonymous data does not. **Calling the table
above "anonymised" is a false statement with regulatory weight.**

And note that a **model** trained on this data inherits the problem —
[Data-Security-for-AI 03](../../Data-Security-for-AI/lessons/03-memorisation.md)
measures what a model leaks back out, and lesson 05 there clones a model from
10,000 queries.

---

## Retention

```text
DEFAULT        "forever", because nobody chose
WHAT YOU OWE   a number, per data type, with a reason

orders              24 months     needed for returns and forecasting
audio recordings    30 days       transcript retained; audio is the liability
model inputs (logs) 7 years       high-tier: needed to answer a complaint
training snapshots  until the model is retired + 1 year
derived embeddings  same as the source data — they are not "new" data
```

Two things that catch teams out:

**An embedding is not anonymous.** A speaker embedding is a biometric
identifier; a text embedding can be partially inverted. Derived data inherits
the obligations of its source.

**Retention needs a job, not a policy.** A policy without
`purge_orders.py` on a schedule and a test that proves it ran is a document
describing something that is not happening. Make the deletion job a monitored
pipeline like any other
([Data-Engineering 09](../../Data-Engineering/lessons/09-orchestration.md)).

---

## Deletion that reaches the model

When a person exercises erasure, the row is the easy part:

- [ ] The production database row
- [ ] **Backups** — and the restore path, or the deletion is temporary
- [ ] The data warehouse and any copies in notebooks or S3 exports
- [ ] The **feature store**, which is a copy nobody thinks of
- [ ] The **logs**, subject to the retention above
- [ ] The **training snapshot** — or a documented decision not to, with a basis
- [ ] The **model weights** — retrain, and do not keep the old model
- [ ] Any **third party** you sent it to

The last three are where the real exposure sits. Most organisations can do the
first two and quietly cannot do the rest, which means the honest status is
"partially deleted" and the stated status is "deleted".

Decide the policy before someone asks:

```text
ON ERASURE REQUEST
  immediately  remove from serving data, feature store, warehouse
  within 30 d  remove from the next training snapshot
  next cycle   retrain; retire the model that saw the data
  recorded     in the dataset record, with the date
```

That is defensible. "We deleted the row" is not, once anyone asks what the
model learned.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| "We removed the names, so it's anonymous" | 88.8% unique on five ordinary columns |
| Asking whether a field is sensitive | The combination identifies, not the field |
| Never computing k | The collapse from 2.5% to 88.8% has no warning |
| Calling pseudonymous data anonymised | A false statement with legal consequences |
| Training on data collected for something else | The principle ML breaks most often |
| Retention "forever" by default | Nobody chose it; everybody owns it |
| A retention policy with no cron job | A document describing a thing not happening |
| Deleting the row and keeping the model | The weights still remember |
| Treating embeddings as derived, not personal | They inherit the source's obligations |

---

## Exercises

1. Compute k for your main table's quasi-identifiers. What fraction are unique?
2. Generalise two columns and recompute. How much did k improve?
3. Write the retention table for every data type you hold.
4. Find the deletion job. Does it exist? Did it run last month?
5. Trace one erasure request through every copy. Where does it stop?
6. Compare "collected for" and "used for" on your training set.

---

**Next:** [Lesson 08 — Running the Process](08-running-the-process.md)
