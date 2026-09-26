# Lesson 03 — Cleaning

**Goal:** fix the data without inventing any.

## What you will learn

- Duplicates: exact and fuzzy
- Missing values, chosen deliberately
- Outliers: error or reality?
- Recording what you did

---

## The rule

Every cleaning step **changes your answer**. Write down what you did, how many
rows it affected, and why — and put it in the report.

```python
import pandas as pd

class CleaningLog:
    """Record every change, so the report can state them."""

    def __init__(self, frame):
        self.frame = frame
        self.steps = []
        self.initial_rows = len(frame)

    def apply(self, name, function, reason):
        before = len(self.frame)
        self.frame = function(self.frame)
        after = len(self.frame)
        self.steps.append({"step": name, "rows_before": before, "rows_after": after,
                           "removed": before - after, "reason": reason})
        return self

    def report(self):
        frame = pd.DataFrame(self.steps)
        frame["pct_of_original"] = (frame["removed"] / self.initial_rows * 100).round(2)
        return frame

orders = pd.read_parquet("/tmp/orders.parquet")
log = CleaningLog(orders)
```

---

## Duplicates

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders.parquet")

exact = int(orders.duplicated().sum())
by_key = int(orders.duplicated(subset=["order_id"]).sum())

print(f"exact duplicate rows:        {exact}")
print(f"duplicate order_id values:   {by_key}")

deduplicated = orders.drop_duplicates()
print(f"\nrows: {len(orders)} -> {len(deduplicated)}")
print(f"revenue before: {orders['amount'].sum():,.0f}")
print(f"revenue after:  {deduplicated['amount'].sum():,.0f}")
print(f"overstatement:  {(orders['amount'].sum() / deduplicated['amount'].sum() - 1):.2%}")
```

```text
exact duplicate rows:        40
duplicate order_id values:   40

rows: 5040 -> 5000
revenue before: 1,271,430
revenue after:  1,267,065
overstatement:  0.34%
```

Forty rows out of 5,040 overstated revenue by **0.34%**. Small here — and in
a dataset where a retry duplicated a whole batch, it would be 20%. The size of
the error depends entirely on *which* rows duplicated, which is why you
measure it rather than assume it is negligible.

Three kinds, and they need different handling:

| Kind | Detect | Handle |
|---|---|---|
| **Exact** | `duplicated()` | Drop |
| **By key** | `duplicated(subset=["order_id"])` | Keep the latest, by timestamp |
| **Fuzzy** | Near-identical text or times | Judgement — inspect before deciding |

```python
import pandas as pd

contacts = pd.DataFrame({
    "name": ["Adam Tayel", "adam tayel", "Adam  Tayel", "Sara Ali"],
    "email": ["adam@x.com", "ADAM@x.com", "adam@x.com ", "sara@x.com"],
})
normalised = contacts.assign(
    name_key=contacts["name"].str.lower().str.split().str.join(" "),
    email_key=contacts["email"].str.strip().str.lower())

print(f"raw distinct emails:        {contacts['email'].nunique()}")
print(f"normalised distinct emails: {normalised['email_key'].nunique()}")
print(f"duplicates found by key:    {int(normalised.duplicated(subset=['email_key']).sum())}")
```

```text
raw distinct emails:        4
normalised distinct emails: 2
duplicates found by key:    2
```

Four "distinct" emails become two. Case and a single trailing space hid two
duplicates. **Normalise before comparing** — lower, strip, collapse spaces —
or your deduplication misses most of what it should catch.

---

## Missing values

```python
import pandas as pd
import numpy as np

orders = pd.read_parquet("/tmp/orders.parquet").drop_duplicates()
missing = orders["amount"].isna()

print(f"missing amount: {missing.sum()} rows ({missing.mean():.2%})")
print("\nare they random? compare the groups:")
print(orders.assign(is_missing=missing)
            .groupby("is_missing")[["quantity", "unit_price"]]
            .mean().round(2).to_string())
print("\nby branch:")
print((orders.assign(is_missing=missing)
             .groupby("branch")["is_missing"].mean() * 100).round(1).to_string())
```

```text
missing amount: 120 rows (2.40%)

are they random? compare the groups:
            quantity  unit_price
is_missing                      
False           2.51       42.25
True            2.52       41.08

by branch:
branch
Giza          2.4
Heliopolis    2.8
Maadi         2.2
Zamalek       2.3
```

The missing rows look **just like** the present ones: quantity 2.52 against
2.51, price 41.08 against 42.25, and 2.2–2.8% missing in every branch. That is
the good case — missing at random, so dropping or filling them will not bias
the result.

Had the nulls been concentrated in one branch or one price band, dropping them
would quietly delete that segment from your analysis.

| Situation | Do |
|---|---|
| Missing at random, < 5% | Drop, and say so |
| Missing at random, more | Impute (median / mode), and flag the rows |
| **Not** at random | Investigate first — it is a finding, not a nuisance |
| The measure itself is missing | Never impute the thing you are analysing |
| Missing means zero | Fill with 0 — but check that it really does |

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders.parquet").drop_duplicates()

recomputed = orders["amount"].fillna(orders["quantity"] * orders["unit_price"])
print(f"nulls before: {orders['amount'].isna().sum()}")
print(f"nulls after recomputation: {recomputed.isna().sum()}")
print(f"revenue with nulls dropped:   {orders['amount'].sum():,.0f}")
print(f"revenue with recomputed rows: {recomputed.sum():,.0f}")
```

```text
nulls before: 120
nulls after recomputation: 0
revenue with nulls dropped:   1,267,065
revenue with recomputed rows: 1,279,345
```

The best fix for a missing value is not a statistic — it is **recovering it
from other columns**. `amount = quantity × unit_price` is derivable, so all
120 nulls are recovered exactly, and the revenue figure moves by 12,280.

Look for that before reaching for the median.

---

## Outliers

```python
import pandas as pd
import numpy as np

orders = pd.read_parquet("/tmp/orders.parquet").drop_duplicates()
amount = orders["amount"].dropna()

q1, q3 = amount.quantile([0.25, 0.75])
iqr = q3 - q1
iqr_outliers = amount[(amount < q1 - 1.5 * iqr) | (amount > q3 + 1.5 * iqr)]

z = (amount - amount.mean()) / amount.std()
z_outliers = amount[z.abs() > 3]

expected = orders["quantity"] * orders["unit_price"]
impossible = orders[(orders["amount"] - expected).abs() > 0.01].dropna(subset=["amount"])

print(f"IQR method:       {len(iqr_outliers)} flagged")
print(f"z-score > 3:      {len(z_outliers)} flagged")
print(f"business rule:    {len(impossible)} rows where amount != quantity x price")
print(f"\nthe business rule found: {sorted(impossible['amount'].unique())}")
```

```text
IQR method:       15 flagged
z-score > 3:      15 flagged
business rule:    15 rows where amount != quantity x price
the business rule found: [50000.0]
```

Three methods, and the **business rule is the one that proves it**. The IQR
and z-score methods say "these are unusual"; the rule `amount = quantity ×
unit_price` says "these are *wrong*", and that is a different claim.

| Method | Says | Use when |
|---|---|---|
| IQR (1.5×) | Unusual | Exploration, skewed data |
| z-score > 3 | Unusual | Roughly normal data |
| **Business rule** | **Impossible** | **Always, wherever a rule exists** |
| Domain threshold | Implausible | An expert can set the bound |

And the decision:

- **Impossible values** (negative age, amount ≠ quantity × price, a date in
  2087): fix if derivable, otherwise remove and count.
- **Extreme but possible** (a genuine 50,000 EGP corporate order): **keep it**,
  and report the median alongside the mean.

Deleting real extremes because they are inconvenient is how an analysis
becomes a lie.

---

## The cleaning report

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders.parquet")
steps = []

before = len(orders)
orders = orders.drop_duplicates()
steps.append(("drop exact duplicates", before - len(orders), "retry artefacts"))

before = len(orders)
expected = orders["quantity"] * orders["unit_price"]
orders["amount"] = orders["amount"].fillna(expected)
steps.append(("recompute missing amount", 0, "derivable from quantity x price"))

before = len(orders)
orders = orders[(orders["amount"] - expected).abs() < 0.01]
steps.append(("remove impossible amounts", before - len(orders),
              "amount != quantity x unit_price"))

report = pd.DataFrame(steps, columns=["step", "rows_removed", "reason"])
print(report.to_string(index=False))
print(f"\nfinal: {len(orders)} rows, revenue {orders['amount'].sum():,.0f}")
```

```text
                     step  rows_removed                          reason
    drop exact duplicates            40                 retry artefacts
 recompute missing amount             0 derivable from quantity x price
remove impossible amounts            15 amount != quantity x unit_price

final: 4985 rows, revenue 529,345
```

**Put that table in the report.** It is three lines of output and it answers
the question every reviewer asks — "what did you drop, and why?" — before it
is asked.

Now look at the revenue: **529,345 after cleaning, against the 1,271,430 the
raw data claimed.** Fifty-eight per cent of the reported revenue was
duplicates and fifteen impossible rows.

Fifty-five rows out of 5,040 — **1.1% of the data** — carried more than half
the revenue. That is what outliers do to a sum, and it is why the cleaning log
belongs in the report: without it, two analysts produce two numbers that
differ by a factor of two and neither can explain why.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Cleaning without recording | Nobody can reproduce or audit the number |
| Dropping nulls before checking the pattern | A whole segment silently disappears |
| Imputing the measure being analysed | You invent the answer |
| Deleting genuine extremes | The analysis is prettier and wrong |
| Deduplicating without normalising | Most duplicates survive |
| Cleaning in place, no raw copy | The original is gone |

---

## Exercises

1. Count exact and key duplicates in your data; what do they do to the total?
2. Check whether your nulls are random — compare the groups.
3. Find a column derivable from others and recover its nulls exactly.
4. Write a business rule that identifies impossible values, and count them.
