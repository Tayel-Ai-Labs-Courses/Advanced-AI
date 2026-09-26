# Lesson 04 — Aggregation

**Goal:** group and summarise without producing a wrong number.

## What you will learn

- `groupby` with several measures
- Shares, rates and per-unit metrics
- Pivot tables
- Ranking, and why top-N is dangerous

---

## The clean dataset

Lesson 03 left us with 4,985 rows: duplicates removed, missing amounts
recovered, impossible values dropped.

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet")
print(f"{len(orders):,} rows, revenue {orders['amount'].sum():,.0f} EGP")
print(f"period: {orders['ordered_at'].min().date()} to {orders['ordered_at'].max().date()}")
```

```text
4,985 rows, revenue 529,345 EGP
period: 2026-01-01 to 2026-06-29
```

---

## Several measures at once

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet")

by_branch = orders.groupby("branch").agg(
    orders=("order_id", "count"),
    revenue=("amount", "sum"),
    average_order=("amount", "mean"),
    median_order=("amount", "median"),
    customers=("customer_id", "nunique"),
).round(1).sort_values("revenue", ascending=False)

by_branch["revenue_share"] = (by_branch["revenue"] / by_branch["revenue"].sum() * 100).round(1)
by_branch["revenue_per_customer"] = (by_branch["revenue"] / by_branch["customers"]).round(1)
print(by_branch.to_string())
```

```text
            orders   revenue  average_order  median_order  customers  revenue_share  revenue_per_customer
branch                                                                                                   
Zamalek       1805  192150.0          106.5         105.0        943           36.3                 203.8
Maadi         1442  152355.0          105.7         105.0        844           28.8                 180.5
Heliopolis     984  104950.0          106.7         105.0        682           19.8                 153.9
Giza           754   79890.0          106.0          97.5        556           15.1                 143.7
```

Read the columns against each other. Zamalek takes 36.3% of revenue — and
every branch has essentially the **same average order**, 105.7 to 106.7. No
branch sells more per transaction than another.

`revenue_per_customer` is where they actually differ: **203.8 in Zamalek
against 143.7 in Giza**, a 42% gap. Zamalek's customers come back more often,
and that is a completely different business observation from "Zamalek is
bigger" — invisible in the revenue column alone.

**Always compute a per-unit metric beside a total.** Totals measure size;
rates measure performance, and they frequently rank in different orders.

---

## Named aggregations and custom functions

```python
import pandas as pd
import numpy as np

orders = pd.read_parquet("/tmp/orders_clean.parquet")

summary = orders.groupby("product").agg(
    orders=("order_id", "count"),
    units=("quantity", "sum"),
    revenue=("amount", "sum"),
    p90_amount=("amount", lambda s: s.quantile(0.9)),
    busiest_hour=("ordered_at", lambda s: int(s.dt.hour.mode().iloc[0])),
).round(1)

summary["revenue_per_unit"] = (summary["revenue"] / summary["units"]).round(1)
print(summary.sort_values("revenue", ascending=False).to_string())
```

```text
          orders  units   revenue  p90_amount  busiest_hour  revenue_per_unit
product                                                                      
latte       1477   3688  154995.0       180.0             3              42.0
espresso    1189   2944  124565.0       180.0             8              42.3
tea         1037   2650  110960.0       180.0             5              41.9
cake         731   1851   78770.0       180.0             2              42.6
juice        551   1399   60055.0       180.0             7              42.9
```

Four things in one table, and `revenue_per_unit` is nearly identical across
products (41.9–42.9) — so **the differences in revenue are purely volume**,
not price. That is a finding, and it took one column.

`busiest_hour` is the counter-example: 3am for latte, 2am for cake. In real
data that would be a red flag worth chasing; here it is an artefact of
generating timestamps uniformly across the day, and the honest conclusion is
"this field tells us nothing, because the data has no daily rhythm". **Not
every column in an aggregation is a finding.**

---

## Pivot tables

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet")

pivot = orders.pivot_table(index="branch", columns="product", values="amount",
                           aggfunc="sum", margins=True, margins_name="total")
print(pivot.round(0).to_string())

share = (pivot.drop(index="total").drop(columns="total")
              .div(pivot.drop(index="total")["total"], axis=0) * 100)
print("\nrow percentages (product mix per branch):")
print(share.round(1).to_string())
```

```text
product        cake  espresso    juice     latte       tea     total
branch                                                              
Giza        12525.0   18315.0   9040.0   23000.0   17010.0   79890.0
Heliopolis  16645.0   23315.0  13220.0   29415.0   22355.0  104950.0
Maadi       22595.0   36075.0  16835.0   44965.0   31885.0  152355.0
Zamalek     27005.0   46860.0  20960.0   57615.0   39710.0  192150.0
total       78770.0  124565.0  60055.0  154995.0  110960.0  529345.0

row percentages (product mix per branch):
product     cake  espresso  juice  latte   tea
branch                                        
Giza        15.7      22.9   11.3   28.8  21.3
Heliopolis  15.9      22.2   12.6   28.0  21.3
Maadi       14.8      23.7   11.0   29.5  20.9
Zamalek     14.1      24.4   10.9   30.0  20.7
```

The raw pivot shows Zamalek highest in every column — because Zamalek is
biggest. **The row percentages are the useful table**: they normalise away
size, and they show the product mix is nearly identical everywhere (latte
28.0–30.0%, espresso 22.2–24.4%).

The only mild pattern: cake is 15.9% of Heliopolis and 14.1% of Zamalek, a
1.8-point spread. With 984 and 1,805 orders behind those numbers, that is
within the noise — lesson 05 of the Advanced track is about knowing when a
difference that small is worth acting on.

"Every branch sells the same mix" is a real conclusion. It means a
product-level decision applies everywhere, and it is invisible in the
unnormalised pivot.

---

## Ranking, and the top-N trap

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet")

by_customer = orders.groupby("customer_id").agg(
    orders=("order_id", "count"),
    revenue=("amount", "sum"),
).sort_values("revenue", ascending=False)

print("top 5 customers by revenue:")
print(by_customer.head(5).to_string())

print(f"\ncustomers: {len(by_customer):,}")
print(f"top 10% of customers account for: "
      f"{by_customer.head(len(by_customer) // 10)['revenue'].sum() / by_customer['revenue'].sum():.1%} of revenue")
print(f"customers with a single order:   "
      f"{(by_customer['orders'] == 1).mean():.1%}")
```

```text
top 5 customers by revenue:
             orders  revenue
customer_id                 
931               9   1470.0
1196              8   1265.0
77               11   1265.0
876              10   1240.0
472              10   1230.0

customers: 1,186
top 10% of customers account for: 20.2% of revenue
customers with a single order:   7.3%
```

The top five customers look impressive until you see the denominators: 1,186
customers, and **the top 10% account for only 20.2% of revenue.**

That is a genuinely flat distribution — no whale customers, nothing like the
Pareto 80/20 people expect. A loyalty programme targeting the top 10% would
address a fifth of revenue, and the business case has to work at that size.

**Never present a top-N without the share it represents.** "Our top five
customers" sounds like a strategy; "our top five customers are 1.3% of
revenue" is a fact.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Totals without rates | Biggest looks best; it is only biggest |
| Pivot without row percentages | Size hides the pattern |
| Top-N without the share | A meaningless highlight |
| `mean` on skewed data | Report the median beside it |
| Grouping by a column with nulls | Those rows silently vanish |
| Averaging an average | Weight by the denominator instead |

That last one deserves code:

```python
import pandas as pd

branches = pd.DataFrame({
    "branch": ["A", "B"],
    "orders": [1000, 10],
    "average_order": [100.0, 500.0],
})
naive = branches["average_order"].mean()
weighted = ((branches["average_order"] * branches["orders"]).sum()
            / branches["orders"].sum())
print(f"mean of averages:  {naive:.1f}")
print(f"weighted average:  {weighted:.1f}")
```

```text
mean of averages:  300.0
weighted average:  104.0
```

Averaging two averages gave **300** when the true figure is **104**. The
small branch's 500 counted as much as the large branch's 100. Weight by the
denominator, always.

---

## Exercises

1. Build a `groupby` with five named aggregations on your own data.
2. Add a per-unit metric beside every total; does the ranking change?
3. Make a pivot table and its row percentages; which is more informative?
4. Find an "average of averages" in a report you have seen and correct it.
