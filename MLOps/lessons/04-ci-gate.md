# Lesson 04 — The CI Gate

**Goal:** decide automatically whether a change may ship, and price the gate.

## What you will learn

- What CI can check about a model, and what it cannot
- The gate, priced against what it prevents
- Why the strongest gate is not the cheapest
- The pipeline, concretely

---

## What CI can check

Ordinary CI runs tests. A model pipeline has five layers, and they catch
different things:

```text
1. CODE          lint, types, unit tests            seconds
2. DATA          schema, ranges, row counts, nulls  seconds
3. TRAINING      it runs end to end on a sample     minutes
4. MODEL         eval set, on a FIXED split         minutes
5. BEHAVIOUR     shadow run against real traffic    hours
```

Layers 1-3 are ordinary software engineering. **Layer 4 is the one specific to
ML**, and it is the gate that matters: *does the new model still clear the bar
on the frozen evaluation set?*

Layer 2 is the one most teams are missing, and
[Data-Security lesson 04](../../Data-Security-for-AI/lessons/04-poisoning.md)
showed why: a validation rule on **training rows** is what catches a poisoned
value, and it is three lines.

---

## The gate, priced

```python
import numpy as np

def gate(bug_rate_per_change, catch_rate, changes_per_month, cost_per_escape,
         ci_minutes, changes_blocked_wrongly=0.02, engineer_cost_per_hour=300):
    bugs = changes_per_month * bug_rate_per_change
    escaped = bugs * (1 - catch_rate)
    ci_cost = changes_per_month * (ci_minutes / 60) * engineer_cost_per_hour
    false_block = changes_per_month * changes_blocked_wrongly * 0.5 * engineer_cost_per_hour
    return escaped * cost_per_escape, ci_cost + false_block

print(f"{'gate':<34}{'escapes/month':>15}{'escape cost':>14}{'gate cost':>12}{'net':>12}")
CHANGES, BUG_RATE, ESCAPE_COST = 40, 0.25, 9_000
no_gate, _ = gate(BUG_RATE, 0.0, CHANGES, ESCAPE_COST, 0)
for name, catch, minutes in [("no gate", 0.00, 0),
                             ("unit tests", 0.35, 3),
                             ("+ data validation", 0.60, 5),
                             ("+ eval set on a fixed split", 0.85, 12),
                             ("+ shadow run on real traffic", 0.95, 45)]:
    esc_cost, g_cost = gate(BUG_RATE, catch, CHANGES, ESCAPE_COST, minutes)
    escapes = CHANGES * BUG_RATE * (1 - catch)
    print(f"{name:<34}{escapes:>15.1f}{esc_cost:>14,.0f}{g_cost:>12,.0f}"
          f"{esc_cost + g_cost:>12,.0f}")
print(f"\n{CHANGES} changes/month, {BUG_RATE:.0%} introduce a bug, "
      f"{ESCAPE_COST:,} EGP per escape")
```

```text
gate                                escapes/month   escape cost   gate cost         net
no gate                                      10.0        90,000         120      90,120
unit tests                                    6.5        58,500         720      59,220
+ data validation                             4.0        36,000       1,120      37,120
+ eval set on a fixed split                   1.5        13,500       2,520      16,020
+ shadow run on real traffic                  0.5         4,500       9,120      13,620

40 changes/month, 25% introduce a bug, 9,000 EGP per escape
```

Read the **net** column, which is the only one that matters.

**No gate costs 90,120 EGP a month.** Adding unit tests takes it to 59,220, data
validation to 37,120, and the **eval-set gate to 16,020** — a 5.6x reduction for
2,520 EGP of CI time.

Then look at the last row. A shadow run against real traffic catches almost
everything and brings the net to 13,620 — **a 15% improvement for 3.6x the gate
cost**, and 45 minutes of latency on every change.

**The eval-set gate is where the value is.** Shadow runs are worth adding when
an escape is far more expensive than 9,000 EGP, or when changes are rare enough
that 45 minutes does not matter. Put your own numbers in: the shape of the
answer changes with `cost_per_escape`.

---

## The eval gate, concretely

```python
# no-run
def gate(new_metrics, baseline_metrics, tolerance=0.005):
    """Return (pass, reasons). Run this in CI on every change."""
    reasons = []
    if new_metrics["auc"] < baseline_metrics["auc"] - tolerance:
        reasons.append(f"AUC fell to {new_metrics['auc']:.4f} "
                       f"from {baseline_metrics['auc']:.4f}")
    for group, score in new_metrics["by_group"].items():          # Data-Science 10
        if score < baseline_metrics["by_group"][group] - 0.02:
            reasons.append(f"subgroup {group} fell by more than 2 points")
    if new_metrics["refusal_rate"] < 0.9 * baseline_metrics["refusal_rate"]:
        reasons.append("refusal rate dropped — check the unanswerable set")
    if new_metrics["latency_p95_ms"] > 1.2 * baseline_metrics["latency_p95_ms"]:
        reasons.append("p95 latency regressed by more than 20%")
    return (len(reasons) == 0), reasons
```

Four properties of a gate that works:

**A tolerance, not an exact comparison.** Models are stochastic
([Data-Science 07](../../Data-Science/lessons/07-reproducibility.md) measured
0.0149 of seed noise). A gate with zero tolerance fails on noise and gets
disabled within a week.

**Subgroups, not just the aggregate.**
[Data-Science lesson 10](../../Data-Science/lessons/10-limits-and-fairness.md)
showed an aggregate hiding that pro customers were never contacted. A gate that
only checks the headline metric will let that through.

**Non-accuracy checks too** — latency, refusal rate, output validity. A model
that is 0.002 better and 3x slower should not pass.

**The gate's thresholds are committed to git**, so changing them is a reviewed
decision rather than a quiet edit on a bad afternoon.

---

## The pipeline

```yaml
# no-run
on: [pull_request]
jobs:
  fast:                      # every push, under 2 minutes
    - lint, types, unit tests
    - data schema + range validation on a sample
    - train on 1% of data, assert it completes

  gate:                      # every PR, under 15 minutes
    - train on the full training split
    - evaluate on the FROZEN eval set
    - compare against the baseline's committed metrics
    - fail with the reasons, or publish the model to the registry as a candidate

  nightly:                   # scheduled
    - full backtest / 5 seeds
    - drift report against production data
    - shadow run on yesterday's real traffic
```

The split matters: **a gate that takes 45 minutes gets bypassed.** Keep the
per-PR path under about 15 minutes and move everything slower to nightly.

And the artefact of a passing gate is **a candidate in the registry**, not a
deployment. Lesson 05 decides whether and how it reaches users.

---

## How often to retrain

```python
DECAY_PER_WEEK = 0.004        # AUC lost per week without retraining
RETRAIN_COST = 900            # EGP per retrain, all in
VALUE_PER_AUC_POINT = 42_000  # EGP/month per 0.01 AUC
print(f"{'retrain every':>15}{'avg AUC lost':>14}{'value lost/mo':>15}"
      f"{'retrain cost/mo':>17}{'total':>11}")
best = None
for weeks in (1, 2, 4, 8, 13, 26, 52):
    avg_decay = DECAY_PER_WEEK * weeks / 2
    value_lost = avg_decay / 0.01 * VALUE_PER_AUC_POINT
    cost = RETRAIN_COST * (4.33 / weeks)
    total = value_lost + cost
    if best is None or total < best[1]:
        best = (weeks, total)
    print(f"{weeks:>13}w{avg_decay:>14.4f}{value_lost:>15,.0f}{cost:>17,.0f}{total:>11,.0f}")
print(f"\ncheapest: retrain every {best[0]} weeks, costing {best[1]:,.0f} EGP/month")
```

```text
  retrain every  avg AUC lost  value lost/mo  retrain cost/mo      total
            1w        0.0020          8,400            3,897     12,297
            2w        0.0040         16,800            1,948     18,748
            4w        0.0080         33,600              974     34,574
            8w        0.0160         67,200              487     67,687
           13w        0.0260        109,200              300    109,500
           26w        0.0520        218,400              150    218,550
           52w        0.1040        436,800               75    436,875

cheapest: retrain every 1 weeks, costing 12,297 EGP/month
```

With these numbers the curve is **monotone** — retrain as often as you can — and
that is worth noticing rather than glossing over. There is no interior optimum
here because **decay (8,400 EGP/week) dwarfs the retraining cost (900 EGP)**.

Change either number and the answer moves: a model that decays at a tenth the
rate, or costs 50,000 EGP to retrain on a GPU cluster
([HPC 07](../../HPC-and-Cloud/lessons/07-the-training-budget.md)), has a real
optimum in the middle.

So the useful output is not "retrain weekly". It is: **measure your decay rate
and your retraining cost, and the cadence follows.** Most teams have never
measured the first, and
[Data-Science lesson 09](../../Data-Science/lessons/09-monitoring-and-drift.md)
is how: hold a model fixed, score it weekly against fresh labels, and watch the
line.

And the finding from that lesson still applies — **drift is not automatically a
reason to retrain.** Covariate shift rarely needs it; a data-quality bug must
never be retrained into the model.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| No eval gate | 90,120 EGP/month against 16,020 |
| A gate with zero tolerance | It fails on seed noise and gets disabled |
| Gating on the aggregate metric only | A subgroup collapse passes |
| No latency or validity check | A better model that is 3x slower ships |
| A 45-minute PR gate | It gets bypassed |
| A passing gate that auto-deploys | Lesson 05 is a separate decision |
| Retraining on a schedule nobody measured | The cadence follows from decay and cost |
| Retraining in response to any drift alarm | Data-Science 09: diagnose first |

---

## Exercises

1. Put your own numbers into the gate table. Where does the net cost flatten?
2. Write the eval gate for your model, including one subgroup and one
   non-accuracy check.
3. Time your current CI. Is the per-PR path under 15 minutes?
4. Measure your model's decay: freeze it and score it weekly for a month.
5. Compute your optimal retraining cadence from that decay and your retraining
   cost.

---

**Next:** [Lesson 05 — Deployment Strategies](05-deployment.md)
