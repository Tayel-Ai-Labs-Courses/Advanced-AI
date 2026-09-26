# Lesson 06 — Segmentation

**Goal:** find groups that are genuinely different, and useful.

## What you will learn

- Rule-based segments before clustering
- RFM
- Clustering, and choosing k
- Profiling, naming and validating segments

---

## Start with rules

A segmentation nobody can explain will not be used. Start with rules a
business person can state.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 2_000

customers = pd.DataFrame({
    "customer_id": np.arange(n),
    "orders": rng.poisson(4, n) + 1,
    "days_since_last": rng.gamma(2, 25, n).astype(int),
    "total_spend": rng.gamma(3, 120, n).round(0),
})

def rule_segment(row):
    if row["days_since_last"] > 90:
        return "lapsed"
    if row["orders"] >= 8 and row["total_spend"] >= 500:
        return "champion"
    if row["orders"] >= 4:
        return "regular"
    return "occasional"

customers["segment"] = customers.apply(rule_segment, axis=1)

summary = customers.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_orders=("orders", "mean"),
    avg_spend=("total_spend", "mean"),
    avg_recency=("days_since_last", "mean"),
).round(1)
summary["share"] = (summary["customers"] / len(customers) * 100).round(1)
summary["revenue_share"] = (customers.groupby("segment")["total_spend"].sum()
                            / customers["total_spend"].sum() * 100).round(1)
print(summary.sort_values("revenue_share", ascending=False).to_string())
```

```text
            customers  avg_orders  avg_spend  avg_recency  share  revenue_share
segment                                                                        
regular          1327         5.7      349.6         40.3   66.4           63.8
occasional        417         2.6      379.6         40.6   20.8           21.8
lapsed            227         5.0      369.0        120.8   11.4           11.5
champion           29         8.8      715.8         41.8    1.5            2.9
```

Four segments a manager can act on immediately — and two problems visible in
the numbers.

First, **`avg_spend` is 350–380 for three of the four segments.** The rules
separated customers by *behaviour* (recency, frequency) and barely at all by
value. That is a real finding: this business does not have high-value and
low-value customers, it has customers who come back and customers who do not.

Second, **"champion" is 29 customers — 1.5% of the base and 2.9% of revenue.**
A segment that small cannot carry a campaign, and by lesson 05's small-group
rule its 715.8 average is mostly noise. Either loosen the rule or drop the
segment.

---

## RFM

```python
import pandas as pd

def rfm_scores(frame, recency="days_since_last", frequency="orders",
               monetary="total_spend", bins=5):
    """Score 1-5 on each dimension; recency is reversed (lower is better)."""
    scored = frame.copy()
    scored["R"] = pd.qcut(scored[recency], bins,
                          labels=range(bins, 0, -1)).astype(int)
    scored["F"] = pd.qcut(scored[frequency].rank(method="first"), bins,
                          labels=range(1, bins + 1)).astype(int)
    scored["M"] = pd.qcut(scored[monetary], bins,
                          labels=range(1, bins + 1)).astype(int)
    scored["RFM"] = scored["R"] * 100 + scored["F"] * 10 + scored["M"]
    scored["rfm_total"] = scored["R"] + scored["F"] + scored["M"]
    return scored

scored = rfm_scores(customers)
print(scored[["R", "F", "M", "RFM", "rfm_total"]].head(5).to_string(index=False))

print("\nvalue by RFM total:")
by_total = scored.groupby("rfm_total").agg(
    customers=("customer_id", "count"),
    avg_spend=("total_spend", "mean")).round(0)
print(by_total.tail(6).to_string())
```

```text
 R  F  M  RFM  rfm_total
 4  1  3  413          8
 2  1  5  215          8
 3  5  3  353         11
 5  2  3  523         10
 3  3  1  331          7

value by RFM total:
           customers  avg_spend
rfm_total                      
10               305      395.0
11               261      456.0
12               165      502.0
13                80      546.0
14                48      569.0
15                11      643.0
```

RFM scores each customer 1–5 on **R**ecency, **F**requency and **M**onetary
value. The combined score is monotonically related to spend: **395 at RFM 10
rising to 643 at RFM 15**.

Note the customer counts falling with it — 305 at RFM 10 and **11** at RFM 15.
The top score is the most valuable group and far too small to treat
separately; in practice you would target RFM 13+ (139 customers), not 15.

It is the standard first segmentation in retail because it needs three columns
you already have, it is explainable, and the segments are actionable: RFM 555
is a champion, 155 is a lapsed champion worth winning back, 511 is a new
customer to nurture.

---

## Clustering

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

features = customers[["orders", "days_since_last", "total_spend"]]
scaled = StandardScaler().fit_transform(features)

print(f"{'k':>3}{'inertia':>12}{'silhouette':>13}")
for k in range(2, 8):
    model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(scaled)
    print(f"{k:>3}{model.inertia_:>12.0f}{silhouette_score(scaled, model.labels_):>13.3f}")
```

```text
  k     inertia   silhouette
  2        4656        0.258
  3        3598        0.268
  4        2734        0.281
  5        2459        0.259
  6        2221        0.240
  7        2022        0.241
```

**The best silhouette is 0.281, at k = 4 — and every value from k = 2 to k = 7
is between 0.24 and 0.28.** For reference, well-separated clusters score above
0.5; below 0.25 the structure is essentially absent.

A flat silhouette curve like this is itself the answer: **there is no k that
is better than the others**, because there are no clusters to find.

This data has **no natural clusters** — it is three smooth distributions with
no gaps — and the honest report says so. K-Means will always return k groups;
the silhouette is what tells you whether those groups exist.

Reporting "we found five customer segments" from a silhouette of 0.25 is
inventing structure. When that happens, fall back to rules or RFM, which at
least produce boundaries someone chose deliberately.

---

## Profile and name

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = customers[["orders", "days_since_last", "total_spend"]]
scaled = StandardScaler().fit_transform(features)
customers["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(scaled)

profile = customers.groupby("cluster").agg(
    size=("customer_id", "count"),
    orders=("orders", "mean"),
    recency=("days_since_last", "mean"),
    spend=("total_spend", "mean"),
).round(1)
profile["share"] = (profile["size"] / len(customers) * 100).round(1)

overall = features.mean()
print(profile.to_string())
print("\nindex vs the overall average (100 = average):")
for column, mean in [("orders", overall["orders"]),
                     ("recency", overall["days_since_last"]),
                     ("spend", overall["total_spend"])]:
    print(f"  {column:<10}{(profile[column] / mean * 100).round(0).astype(int).tolist()}")
```

```text
         size  orders  recency  spend  share
cluster                                     
0        1051     5.4     32.0  263.3   52.6
1         475     4.8     96.5  307.9   23.8
2         474     4.3     41.4  641.0   23.7

index vs the overall average (100 = average):
  orders    [108, 96, 86]
  recency   [65, 195, 84]
  spend     [72, 85, 176]
```

The index table is how you name segments. Read the deviations:

- **Cluster 0** — recency 65, spend 72, 53% of customers. Recent, low value:
  **"the base"**.
- **Cluster 1** — recency 195 (twice the average), spend 85. **"Lapsed"** — the
  win-back target.
- **Cluster 2** — spend 176, recency 84. **"High value, active"** — protect
  these.

Note what the clusters did *not* separate: orders, at 86–108, is nearly flat.
**The algorithm split on recency and spend and effectively ignored
frequency**, which belongs in the report — it tells you which dimensions carry
structure and which do not.

And remember the silhouette was 0.27. These are usable labels, not discovered
groups.

---

## A segment must be usable

| Test | Question |
|---|---|
| **Distinct** | Do the segments differ on something that matters? |
| **Stable** | Do customers stay in their segment for a useful period? |
| **Reachable** | Can you actually contact or treat this group differently? |
| **Substantial** | Is it big enough to be worth a different treatment? |
| **Actionable** | Does membership imply a different action? |

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
before = rng.integers(0, 3, 1_000)
after = np.where(rng.random(1_000) < 0.7, before, rng.integers(0, 3, 1_000))

stability = (before == after).mean()
print(f"customers in the same segment next month: {stability:.1%}")
print("transition matrix (rows: before, columns: after):")
print(pd.crosstab(before, after, normalize="index").round(2).to_string())
```

```text
customers in the same segment next month: 79.2%
transition matrix (rows: before, columns: after):
col_0     0     1     2
row_0                  
0      0.80  0.11  0.09
1      0.07  0.81  0.12
2      0.12  0.12  0.76
```

Seventy-nine per cent stability month to month — about one customer in five
changes segment. That is workable for a monthly campaign and marginal for a
quarterly strategy.

**A segmentation where 40% of customers move every month is not a
segmentation** — you would be targeting last month's behaviour.

Measure this before building anything on top of your segments.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Clustering before rules | Unexplainable segments nobody uses |
| Ignoring the silhouette | You report structure that is not there |
| Not scaling before K-Means | The largest-unit column defines everything |
| Segments you cannot reach | An interesting slide, no action |
| Never checking stability | Targeting last month's behaviour |
| Naming before profiling | The name drives the story, not the data |

---

## Exercises

1. Write four rule-based segments for your customers and report their revenue
   shares.
2. Compute RFM scores; is the value gradient monotonic?
3. Run K-Means for k = 2…8 and report the silhouette. Does structure exist?
4. Measure segment stability month to month.
