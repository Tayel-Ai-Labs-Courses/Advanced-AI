# Lesson 05 — Cohorts and Retention

**Goal:** measure whether the product is getting better, not just bigger.

## What you will learn

- Cohorts, and why aggregate metrics hide the truth
- Retention curves
- The Simpson's trap in growth numbers
- Lifetime value

---

## Why cohorts

An aggregate metric mixes new and old customers. Growth in acquisition can
hide collapsing retention, indefinitely.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

rows = []
for month in range(12):
    size = 100 + month * 40                       # acquisition growing fast
    quality = 0.50 - month * 0.03                 # retention getting worse
    for user in range(size):
        rows.append({"cohort": month, "retained_month_1": rng.random() < quality})

users = pd.DataFrame(rows)

overall = users["retained_month_1"].mean()
print(f"overall month-1 retention: {overall:.1%}")
print("\nby cohort:")
by_cohort = users.groupby("cohort").agg(
    size=("retained_month_1", "size"),
    retention=("retained_month_1", "mean"))
print(by_cohort.assign(retention=lambda d: (d["retention"] * 100).round(1)).to_string())
```

```text
overall month-1 retention: 28.7%

by cohort:
        size  retention
cohort                 
0        100       44.0
1        140       42.1
2        180       40.6
3        220       38.6
4        260       37.3
5        300       35.3
6        340       32.6
7        380       32.1
8        420       27.9
9        460       21.7
10       500       17.8
11       540       18.1
```

The overall number is 28.7%, and it is meaningless. **Retention has fallen
from 44% to 18%** across twelve cohorts — the product is getting worse every
month — and the aggregate hides it because the newest, worst cohorts are also
the largest.

Worse, as acquisition keeps growing, the aggregate will keep falling *slowly*
while the underlying decay is steep. By the time the headline number alarms
anyone, twelve cohorts have churned.

**Cohort by acquisition date, always.**

---

## The retention curve

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)

def simulate_cohort(size, decay, months=12):
    """Retention with a decaying hazard — the usual shape."""
    retained = [size]
    for month in range(1, months + 1):
        keep_rate = decay + (1 - decay) * (1 - np.exp(-month / 4))
        retained.append(int(retained[-1] * min(keep_rate, 0.97)))
    return retained

curve = simulate_cohort(1_000, 0.45)
frame = pd.DataFrame({
    "month": range(len(curve)),
    "users": curve,
    "retention": [round(u / curve[0] * 100, 1) for u in curve],
})
frame["month_over_month"] = (frame["users"].pct_change() * 100).round(1)
print(frame.to_string(index=False))
```

```text
 month  users  retention  month_over_month
     0   1000      100.0               NaN
     1    571       57.1             -42.9
     2    380       38.0             -33.5
     3    281       28.1             -26.1
     4    224       22.4             -20.3
     5    188       18.8             -16.1
     6    164       16.4             -12.8
     7    148       14.8              -9.8
     8    136       13.6              -8.1
     9    128       12.8              -5.9
    10    122       12.2              -4.7
    11    117       11.7              -4.1
    12    113       11.3              -3.4
```

Two things every retention curve shows:

- **The biggest drop is month 1** — 42.9% here. Most of your churn happens
  immediately, which is why onboarding is where retention work pays.
- **The curve flattens.** The monthly loss falls from 42.9% to 3.4% by month
  12. The users who remain are qualitatively different from those who left.

Whether the curve **flattens above zero** is the question that decides a
business. Here it is heading towards roughly 11% — one user in nine stays
indefinitely, and that core is what the company is worth. A curve still
falling 20% a month at month 12 has no core at all.

---

## The cohort table

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(2)

records = []
for cohort in range(6):
    size = 500
    quality = 0.55 - cohort * 0.02                # slow decline
    for month in range(6 - cohort):
        retention = quality ** (month * 0.6) if month else 1.0
        records.append({"cohort": f"2026-{cohort + 1:02d}",
                        "months_since": month,
                        "retention": round(retention * 100, 1)})

table = (pd.DataFrame(records)
           .pivot(index="cohort", columns="months_since", values="retention"))
print(table.to_string())

print("\nmonth-3 retention by cohort:")
print(table[3].dropna().to_string())
```

```text
months_since      0     1     2     3     4     5
cohort                                           
2026-01       100.0  69.9  48.8  34.1  23.8  16.6
2026-02       100.0  68.3  46.7  31.9  21.8   NaN
2026-03       100.0  66.8  44.6  29.8   NaN   NaN
2026-04       100.0  65.2  42.5   NaN   NaN   NaN
2026-05       100.0  63.6   NaN   NaN   NaN   NaN
2026-06       100.0   NaN   NaN   NaN   NaN   NaN

month-3 retention by cohort:
cohort
2026-01    34.1
2026-02    31.9
2026-03    29.8
```

The triangle is the standard shape: newer cohorts have fewer observed months.

**Read it down the columns, not across the rows.** Column 3 shows month-3
retention **falling** from 34.1% to 29.8% across three cohorts — a 13%
relative deterioration in three months, and the number a product team should
be held to.

Column 1 says the same thing sooner: 69.9% → 63.6% across five cohorts. The
earlier the column, the faster you learn, which is why month-1 retention is
the standard early-warning metric.

Reading across a row tells you only that people churn over time, which you
already knew.

---

## Growth that hides decay

```python
import pandas as pd

months = pd.DataFrame({
    "month": range(1, 13),
    "new_users": [100 + m * 40 for m in range(12)],
    "month_1_retention": [0.50 - m * 0.03 for m in range(12)],
})
months["retained"] = (months["new_users"] * months["month_1_retention"]).round(0)
months["cumulative_new"] = months["new_users"].cumsum()

print(months.head(4).to_string(index=False))
print("...")
print(months.tail(3).to_string(index=False))

print(f"\nnew users, month 1 -> 12:  {months['new_users'].iloc[0]} -> "
      f"{months['new_users'].iloc[-1]}  ({months['new_users'].iloc[-1] / months['new_users'].iloc[0]:.1f}x)")
print(f"retention, month 1 -> 12:  {months['month_1_retention'].iloc[0]:.0%} -> "
      f"{months['month_1_retention'].iloc[-1]:.0%}")
print(f"retained users:            {months['retained'].iloc[0]:.0f} -> "
      f"{months['retained'].iloc[-1]:.0f}")
```

```text
 month  new_users  month_1_retention  retained  cumulative_new
     1        100               0.50      50.0             100
     2        140               0.47      66.0             240
     3        180               0.44      79.0             420
     4        220               0.41      90.0             640
...
    10        460               0.23     106.0            2800
    11        500               0.20     100.0            3300
    12        540               0.17      92.0            3840
```

Acquisition grew **5.4×** and retention fell from 50% to 17%. Retained users
per month peaked at 106 in month 10 and has fallen to 92 — **the business is
now adding fewer lasting customers than three months ago**, while every
top-line chart still points up and to the right.

This is the most expensive pattern in subscription businesses, and it is
invisible without cohorts.

---

## Lifetime value

```python
def lifetime_value(monthly_revenue, monthly_churn, margin=0.7, horizon=36):
    """LTV with a finite horizon — the honest version."""
    value = 0.0
    surviving = 1.0
    for _ in range(horizon):
        value += surviving * monthly_revenue * margin
        surviving *= (1 - monthly_churn)
    return value

print(f"{'churn':>8}{'LTV (36m)':>12}{'LTV (infinite)':>17}{'months to 90%':>15}")
for churn in [0.02, 0.05, 0.10, 0.20]:
    finite = lifetime_value(100, churn)
    infinite = 100 * 0.7 / churn
    months = 1
    surviving, accumulated = 1.0, 0.0
    while accumulated < 0.9 * finite:
        accumulated += surviving * 100 * 0.7
        surviving *= (1 - churn)
        months += 1
    print(f"{churn:>8.0%}{finite:>12,.0f}{infinite:>17,.0f}{months:>15}")
```

```text
   churn   LTV (36m)   LTV (infinite)  months to 90%
      2%       1,809            3,500             32
      5%       1,179            1,400             29
     10%         684              700             22
     20%         350              350             12
```

At 2% monthly churn the infinite-horizon formula says 3,500 and the realistic
36-month figure is **1,809** — the textbook formula overstates by 93%, because
it assumes customers stay for fifty years.

At 20% churn the two agree exactly (350 and 350): when customers leave fast,
the horizon does not matter. **The lower your churn, the more the standard LTV
formula lies to you** — and low-churn businesses are exactly the ones that
plan around LTV.

Three rules for LTV:

- **Use a finite horizon** — 24 or 36 months, matching how far your business
  can actually plan.
- **Use margin, not revenue.** LTV is what you keep.
- **Compute it per cohort.** A single company-wide LTV mixes the good cohorts
  with the bad, exactly as the aggregate retention number did.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Aggregate retention only | Falling retention hidden by growing acquisition |
| Reading cohort rows, not columns | You learn that time passes |
| Comparing cohorts at different ages | Newer cohorts always look better |
| Infinite-horizon LTV | Overstated by 50–100% |
| LTV on revenue, not margin | Overstated again |
| One company-wide LTV | Hides which segments are worth acquiring |

---

## Exercises

1. Build a cohort table for your users; read month-3 retention down the
   column.
2. Plot the retention curve; where is the largest drop, and does it flatten?
3. Compute retained-users-per-month; is it still rising?
4. Compute LTV at 36 months and infinite horizon; how far apart are they?
