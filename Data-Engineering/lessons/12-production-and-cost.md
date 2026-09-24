# Lesson 12 — Production and Cost

**Goal:** run a platform people trust, for money you can defend.

## What you will learn

- What to monitor, and what to alert on
- Documentation and lineage
- The cost levers, in order of size
- Incident response for data

---

## Monitoring

Data pipelines fail in a way services do not: **everything is green and the
numbers are wrong.**

| Layer | Watch | Alert when |
|---|---|---|
| **Infrastructure** | Job success, duration, memory | A job fails after retries |
| **Freshness** | Lag between latest event and now | The SLA is breached |
| **Volume** | Row count per partition | Outside 3σ of the 30-day baseline |
| **Quality** | Null rates, duplicates, ranges | A test fails (lesson 07) |
| **Schema** | Column set and types | Anything changes upstream |
| **Cost** | Spend per pipeline per day | Above the budget, or +50% week on week |

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
history = pd.DataFrame({
    "day": pd.date_range("2026-01-01", periods=30),
    "rows": rng.normal(50_000, 2_000, 30).round(),
    "duration_min": rng.normal(12, 2, 30).round(1),
    "null_rate": rng.uniform(0.01, 0.03, 30).round(4),
})

def health_report(today, history, sigmas=3):
    """Compare today's run against the recent baseline."""
    report = {}
    for metric in ["rows", "duration_min", "null_rate"]:
        mean, std = history[metric].mean(), history[metric].std()
        z = (today[metric] - mean) / std if std else 0.0
        report[metric] = {"value": today[metric], "baseline": round(float(mean), 1),
                          "z": round(float(z), 1), "alert": abs(z) > sigmas}
    return report

for label, today in [
    ("normal", {"rows": 51_000, "duration_min": 12.5, "null_rate": 0.02}),
    ("half the data", {"rows": 25_000, "duration_min": 8.0, "null_rate": 0.02}),
    ("nulls appeared", {"rows": 50_500, "duration_min": 12.0, "null_rate": 0.35}),
]:
    print(f"\n{label}:")
    for metric, result in health_report(today, history).items():
        flag = "ALERT" if result["alert"] else "ok"
        print(f"  {metric:<14}{result['value']:>9}  baseline {result['baseline']:>8}"
              f"  z={result['z']:>6}  {flag}")
```

```text
normal:
  rows              51000  baseline  49757.0  z=   0.8  ok
  duration_min       12.5  baseline     12.6  z=  -0.0  ok
  null_rate          0.02  baseline      0.0  z=  -0.2  ok

half the data:
  rows              25000  baseline  49757.0  z= -15.0  ALERT
  duration_min        8.0  baseline     12.6  z=  -2.4  ok
  null_rate          0.02  baseline      0.0  z=  -0.2  ok

nulls appeared:
  rows              50500  baseline  49757.0  z=   0.5  ok
  duration_min       12.0  baseline     12.6  z=  -0.3  ok
  null_rate          0.35  baseline      0.0  z=  59.8  ALERT
```

Both failures were caught, and **neither would have failed the job**. Half the
data arrived and the pipeline succeeded; a column started arriving 35% null
and the pipeline succeeded. That is the shape of data incidents, and a z-score
against a rolling baseline is most of the defence.

Look at the duration row in the middle case: −2.4σ, and correctly **not**
alerted. Half the data ran faster, which is a symptom, not a cause. Alert on
volume and quality; treat duration as a diagnostic.

---

## Documentation that gets read

```yaml
# tables/mart_daily_revenue.yml
table: mart_daily_revenue
owner: data-team (slack: #data-help)
description: >
  Daily revenue per branch, VAT inclusive, for the finance dashboard
  and the weekly management report.

grain: one row per branch per calendar day (Africa/Cairo)
freshness_sla: available by 06:00 Cairo time for the previous day
update_pattern: incremental, delete-insert on order_date

columns:
  order_date:    { type: date,    description: "Calendar date, Africa/Cairo" }
  branch_id:     { type: integer, description: "FK to dim_branch" }
  orders:        { type: integer, description: "Completed orders; excludes cancelled" }
  revenue_egp:   { type: decimal, description: "VAT-inclusive revenue in EGP" }

known_issues:
  - "Branch 7 has no data before 2026-03-01 (opened then)"
  - "Refunds are applied on the refund date, not the original order date"

upstream: [stg_orders, dim_branch]
downstream: [finance_dashboard, weekly_management_report]
```

The two sections people actually use are `grain` and `known_issues`. The grain
stops someone summing the wrong column (lesson 02); the known issues stop the
same question arriving five times.

**Lineage** — which tables feed which — is the other half. dbt, Dagster,
OpenMetadata and DataHub generate it from your code. Without it, "can I change
this column?" is unanswerable and the answer defaults to no.

---

## Cost

```python
def monthly_cost(gb_stored, gb_scanned_per_query, queries_per_day,
                 storage_per_gb=0.02, scan_per_tb=5.0):
    storage = gb_stored * storage_per_gb
    scanning = gb_scanned_per_query * queries_per_day * 30 / 1024 * scan_per_tb
    return storage, scanning, storage + scanning

scenarios = [
    ("naive: SELECT * on a flat table", 5_000, 50.0, 200),
    ("partitioned by date",             5_000,  2.0, 200),
    ("partitioned + columns pruned",    5_000,  0.3, 200),
    ("+ pre-aggregated mart",           5_050,  0.01, 200),
]

print(f"{'setup':<34}{'storage':>10}{'scanning':>11}{'total/mo':>11}")
for label, stored, scanned, queries in scenarios:
    storage, scanning, total = monthly_cost(stored, scanned, queries)
    print(f"{label:<34}${storage:>9.0f}${scanning:>10.0f}${total:>10.0f}")
```

```text
setup                                storage   scanning   total/mo
naive: SELECT * on a flat table   $      100$      1465$      1565
partitioned by date               $      100$        59$       159
partitioned + columns pruned      $      100$         9$       109
+ pre-aggregated mart             $      101$         0$       101
```

The same data and the same questions, **$1,565 a month against $101** — a 15×
difference, and every step of it is free engineering.

The levers, in order of size:

| Lever | Typical saving |
|---|---|
| **Partition pruning** (filter on the partition column) | 10–50× |
| **Column pruning** (never `SELECT *`) | 5–20× |
| **Pre-aggregated marts** for dashboards | 10–100× on repeated queries |
| Materialising instead of re-querying views | Large, for popular views |
| Compression and file compaction | 2–5× storage |
| Lifecycle rules: hot → cold → delete | 5–10× on old data |
| Right-sizing clusters, auto-suspend | 2–5× compute |

**The most expensive thing in most warehouses is a dashboard that refreshes
every five minutes over raw tables.** Materialise it.

---

## When it breaks

```python
from datetime import datetime

INCIDENT_STEPS = [
    "1. STOP the downstream publish — do not let bad data spread",
    "2. Assess: which tables, which partitions, which consumers",
    "3. Communicate: tell consumers before they find it themselves",
    "4. Fix the cause, not the symptom",
    "5. Backfill the affected partitions",
    "6. Verify: row counts and totals against a known-good source",
    "7. Add the check that would have caught it",
]
for step in INCIDENT_STEPS:
    print(step)
```

```text
1. STOP the downstream publish — do not let bad data spread
2. Assess: which tables, which partitions, which consumers
3. Communicate: tell consumers before they find it themselves
4. Fix the cause, not the symptom
5. Backfill the affected partitions
6. Verify: row counts and totals against a known-good source
7. Add the check that would have caught it
```

Step 3 is the one engineers skip and the one that determines whether people
trust the platform afterwards. **A number that was wrong and was announced is
survivable; a number that was wrong and was discovered by finance is not.**

Step 7 is what makes the platform better over time: every incident ends with a
new check, and the set of checks becomes the institutional memory of
everything that has ever gone wrong.

---

## The platform checklist

- [ ] Every table has an owner, a grain and a freshness SLA
- [ ] Every pipeline is idempotent, and a test proves it
- [ ] Row-count and freshness monitoring on every output
- [ ] Quality tests run before publishing, not after
- [ ] Alerts go to a channel someone reads, and are rare enough to be read
- [ ] Backfills are one command, parameterised by date
- [ ] Lineage is generated, not maintained by hand
- [ ] Cost is attributed per pipeline and reviewed monthly
- [ ] Raw data is immutable and retained per a written policy
- [ ] Access is controlled, and PII is masked where it should be

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Monitoring jobs but not data | Green pipelines, wrong numbers |
| No freshness alert | Stale dashboards for days |
| Alerting on everything | Fatigue; the real alert is missed |
| No documented grain | Someone sums the wrong column |
| `SELECT *` in dashboards | 10× the bill for the same answer |
| Dashboards on raw tables | The single largest cost line |
| Silent incidents | Trust lost permanently |
| No owner | Nobody fixes it |

---

## Exercises

1. Write the YAML documentation for one table you own.
2. Build the health report against 30 days of your real run history.
3. Find your most expensive query; what is it scanning that it does not need?
4. Take a past incident and write the check that would have caught it.
