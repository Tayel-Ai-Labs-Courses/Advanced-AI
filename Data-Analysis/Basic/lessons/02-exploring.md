# Lesson 02 — Exploring a Dataset

**Goal:** know what you have, and what is wrong with it, in fifteen minutes.

## What you will learn

- The first six commands
- Reading distributions, not just means
- Finding the problems before they find you
- The exploration checklist

---

## A dataset to work with

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 5_000

orders = pd.DataFrame({
    "order_id": np.arange(1, n + 1),
    "branch": rng.choice(["Zamalek", "Maadi", "Heliopolis", "Giza"],
                         n, p=[0.35, 0.3, 0.2, 0.15]),
    "product": rng.choice(["latte", "espresso", "tea", "cake", "juice"],
                          n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    "quantity": rng.integers(1, 5, n),
    "unit_price": rng.choice([45.0, 35.0, 30.0, 60.0, 40.0], n),
    "ordered_at": pd.to_datetime("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 180 * 24 * 60, n), unit="m"),
    "customer_id": rng.integers(1, 1_200, n),
})
orders["amount"] = orders["quantity"] * orders["unit_price"]

# a few realistic problems
orders.loc[rng.choice(n, 120, replace=False), "amount"] = np.nan
orders.loc[rng.choice(n, 15, replace=False), "amount"] = 50_000.0
orders = pd.concat([orders, orders.sample(40, random_state=1)], ignore_index=True)

orders.to_parquet("/tmp/orders.parquet")
print(orders.shape)
```

```text
(5040, 8)
```

---

## The first six commands

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders.parquet")

print("1. shape:", orders.shape)
print("\n2. dtypes:")
print(orders.dtypes.to_string())
print("\n3. head:")
print(orders.head(3).to_string(index=False))
print("\n4. nulls per column:")
print(orders.isna().sum()[lambda s: s > 0].to_string())
print("\n5. duplicates:", int(orders.duplicated().sum()))
print("\n6. numeric summary:")
print(orders[["quantity", "unit_price", "amount"]].describe().round(1).to_string())
```

```text
1. shape: (5040, 8)

2. dtypes:
order_id                int64
branch                 object
product                object
quantity                int64
unit_price            float64
ordered_at     datetime64[ns]
customer_id             int64
amount                float64

3. head:
 order_id     branch  product  quantity  unit_price          ordered_at  customer_id  amount
        1      Maadi espresso         4        60.0 2026-01-24 09:17:00          351   240.0
        2       Giza    juice         3        45.0 2026-02-23 21:30:00         1110   135.0
        3 Heliopolis      tea         4        30.0 2026-01-31 23:33:00          953   120.0

4. nulls per column:
amount    122

5. duplicates: 40

6. numeric summary:
       quantity  unit_price   amount
count    5040.0      5040.0   4918.0
mean        2.5        42.2    258.5
std         1.1        10.4   2752.1
min         1.0        30.0     30.0
25%         2.0        35.0     60.0
50%         3.0        40.0    105.0
75%         4.0        45.0    140.0
max         4.0        60.0  50000.0
```

Four problems are already visible in that output, before any analysis:

1. **122 nulls in `amount`** — 2.4% of rows.
2. **40 exact duplicate rows.**
3. **The mean amount is 258.5 and the median is 105** — a mean 2.5× the median
   means a long tail or outliers.
4. **The maximum is 50,000** against a 75th percentile of 140. That is not a
   large order; it is a data error.

The `std` of 2,752 on a mean of 258 is the same signal: **when the standard
deviation is ten times the mean, look for outliers before anything else.**

---

## Distributions, not means

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders.parquet")
amount = orders["amount"].dropna()

print("mean:  ", round(amount.mean(), 1))
print("median:", round(amount.median(), 1))
print("\npercentiles:")
print(amount.quantile([0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]).round(1).to_string())

without_outliers = amount[amount < 1_000]
print(f"\nrows above 1,000: {len(amount) - len(without_outliers)}"
      f" ({(len(amount) - len(without_outliers)) / len(amount):.2%})")
print("mean without them:", round(without_outliers.mean(), 1))
```

```text
mean:   258.5
median: 105.0

percentiles:
0.01       30.0
0.25       60.0
0.50      105.0
0.75      140.0
0.95      240.0
0.99      240.0
1.00    50000.0

rows above 1,000: 15 (0.31%)
mean without them: 106.3
```

Look at the 99th percentile and the maximum: **240 and 50,000**. Everything up
to 99% of the data is under 240, and then the top fifteen rows jump by a
factor of two hundred.

Those fifteen rows — **0.31% of the data** — moved the mean from 106.3 to
258.5. Every average, every "revenue per order", every forecast built on that
mean would have been 2.4 times reality.

**Always print percentiles, not just `mean()`.** The gap between the 99th
percentile and the maximum is where the errors hide.

---

## Categories and time

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders.parquet")

for column in ["branch", "product"]:
    counts = orders[column].value_counts()
    print(f"{column}: {orders[column].nunique()} distinct")
    print((counts / len(orders) * 100).round(1).to_string(), "\n")

print("date range:", orders["ordered_at"].min().date(), "to",
      orders["ordered_at"].max().date())
per_month = orders.set_index("ordered_at").resample("MS").size()
print("\nrows per month:")
print(per_month.to_string())
```

```text
branch: 4 distinct
branch
Zamalek       36.2
Maadi         28.9
Heliopolis    19.7
Giza          15.2

product: 5 distinct
product
latte       29.6
espresso    23.8
tea         20.8
cake        14.6
juice       11.1

date range: 2026-01-01 to 2026-06-29

rows per month:
ordered_at
2026-01-01    879
2026-02-01    771
2026-03-01    885
2026-04-01    812
2026-05-01    903
2026-06-01    790
Freq: MS
```

Two checks, both essential and both often skipped:

- **Category proportions.** A category that should be there and is not, or one
  taking 90% of the rows, tells you something about the data collection.
- **Rows per period.** A month with half the rows means a collection gap, not
  a business collapse — and confusing the two is how an analysis becomes
  wrong.

June has 790 rows and ends on the 29th, which is consistent with the others.
February's 771 is the lowest — and February is the shortest month, which is
the explanation rather than a finding. A month at 400 would have been the
story.

---

## The exploration checklist

```python
import pandas as pd

def explore(frame, name="dataset"):
    """The fifteen-minute profile, as a function."""
    report = {
        "rows": len(frame),
        "columns": frame.shape[1],
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1024**2, 1),
        "duplicate_rows": int(frame.duplicated().sum()),
        "columns_with_nulls": {c: int(v) for c, v in
                               frame.isna().sum().items() if v},
        "constant_columns": [c for c in frame.columns if frame[c].nunique() <= 1],
        "high_cardinality": {c: int(frame[c].nunique()) for c in frame.columns
                             if frame[c].nunique() > 0.5 * len(frame)},
    }
    numeric = frame.select_dtypes("number")
    report["possible_outliers"] = {
        c: int(((numeric[c] - numeric[c].median()).abs()
                > 10 * (numeric[c].quantile(0.75) - numeric[c].quantile(0.25))).sum())
        for c in numeric.columns
        if (numeric[c].quantile(0.75) - numeric[c].quantile(0.25)) > 0
    }
    return report

for key, value in explore(pd.read_parquet("/tmp/orders.parquet")).items():
    print(f"{key:<22}{value}")
```

```text
rows                  5040
columns               8
memory_mb             0.8
duplicate_rows        40
columns_with_nulls    {'amount': 122}
constant_columns      []
high_cardinality      {'order_id': 5000, 'ordered_at': 4954}
possible_outliers     {'order_id': 0, 'quantity': 0, 'unit_price': 0, 'customer_id': 0, 'amount': 15}
```

Eight lines, and they name every problem in this dataset: 40 duplicates, 122
nulls, and 15 outliers in `amount`.

The `high_cardinality` entry flags `order_id` and `ordered_at`, which are
*supposed* to be nearly unique — the check is not saying they are wrong, it is
telling you which columns are identifiers rather than dimensions. A column you
expected to group by appearing there is the actual finding. Run this on **every**
dataset you receive, before you write a single aggregation.

And then do the thing no function can do: **read fifty rows with your eyes.**
Sort by the main measure, look at the top and the bottom, and see whether they
make sense.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| `describe()` on the mean only | Outliers invisible |
| Not checking duplicates | Every total is inflated |
| Ignoring rows-per-period | A collection gap read as a business change |
| Trusting dtypes | A numeric column stored as text sorts "10" before "9" |
| Skipping the eyeball pass | Obvious nonsense survives to the report |
| Cleaning before exploring | You delete evidence of the real problem |

---

## Exercises

1. Run `explore()` on a dataset of yours and list every problem it finds.
2. Print percentiles for your main measure; how far is the max from the 99th?
3. Plot rows per month; are there collection gaps?
4. Sort by your main measure and read the top and bottom twenty rows.
