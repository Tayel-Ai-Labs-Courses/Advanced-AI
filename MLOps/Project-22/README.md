# Project 22 — Ship One Model Properly

Take a model you have **already built** somewhere in this library and put the
whole production loop around it. The modelling is done; this project is
everything after it.

Pick one you actually understand — the churn model from
[Data-Science](../../Data-Science/), the forecaster from
[Time-Series](../../Time-Series-and-Forecasting/), or a classifier from
[Machine-Learning](../../Machine-Learning/). **Do not start a new model.** The
temptation to improve the model instead of operationalising it is the thing
this project is testing.

---

## What you deliver

```text
repo/
  params.yaml                 every parameter, once
  requirements.lock           pinned, with hashes
  Dockerfile                  base image by digest
  pipeline/
    01_ingest.py              each stage: declared inputs and outputs,
    02_validate.py            deterministic, idempotent
    03_features.py
    04_train.py
    05_evaluate.py
  serve/
    app.py                    one endpoint, a timeout, a fallback
    features.py               IMPORTED BY 03_features.py TOO
  .github/workflows/ci.yml    fast path + gate
  monitors/
    thresholds.py             computed, not guessed
  MODEL_CARD.md
  RUNBOOK.md
  README.md                   how to run all of it
```

---

## The eight requirements

Each maps to one lesson. Each has a check you can run — if you cannot run the
check, the requirement is not met.

| # | Requirement | The check |
|---|---|---|
| 1 | **Environment pinned** (02) | Build the image twice, a week apart. Same model hash. |
| 2 | **Pipeline re-runnable** (03) | Delete every output, run from scratch in an empty directory. |
| 3 | **Eval gate in CI** (04) | Open a PR that makes the model worse. CI must fail, with reasons. |
| 4 | **Rollback** (05) | Someone else rolls back, timed with a stopwatch. Under 5 minutes. |
| 5 | **Serving** (06) | p50/p95/p99 measured under load. Kill the model — the fallback answers. |
| 6 | **Monitors** (07) | Thresholds from `sqrt(p(1-p)/n)`. Feed a column-swapped batch; a monitor fires. |
| 7 | **No skew** (07) | The feature function is imported by both paths. Assert on names, not positions. |
| 8 | **Documented and owned** (08) | Model card with a "where it is worse" table. Three named owners. |

---

## The report

One file, `REPORT.md`, with these numbers from **your** model. Not prose about
what you learned — the measurements.

```text
1. BASELINE       the one-line rule, and what the model beats it by
2. GATE           your escape rate, cost per escape, net cost with and
                  without the gate (lesson 04's table, your numbers)
3. DEPLOYMENT     all five strategies priced on your traffic. Which wins?
4. LATENCY        p50/p95/p99, your utilisation, how much of p95 is queueing
5. BATCHING       throughput at batch 1, 8, 32, 128. Where does it flatten?
6. THRESHOLDS     your standard error, your threshold, expected false
                  alarms per year, days to detect a 20% regression
7. SKEW           inject each of lesson 07's six skews. Which does your
                  monitoring catch? Which does AUC miss?
8. CADENCE        your measured decay rate, your retraining cost, the
                  cheapest cadence
```

Requirement 7 is the most valuable exercise in the project. **Deliberately
break your own serving path six ways and see which breaks your dashboard
notices.** Most people find it notices two.

---

## Rules

- **Every number in the report is from a run you did.** No copied figures,
  including from these lessons. The point is that your system's numbers are
  different from the ones here.
- **A finding that contradicts a lesson is a better answer than one that
  agrees.** If a canary wins for your traffic where shadow lost here, say so
  with the arithmetic. These lessons' numbers come from one set of assumptions.
- **Nothing passes a check you have not actually executed.** "Rollback should
  take about a minute" is not requirement 4.
- **Do not improve the model.** If you end up with a better model, you did a
  different project.

---

## Scoring yourself

| | |
|---|---|
| **Not done** | The model runs; the eight checks were never executed |
| **Done** | All eight checks pass and the report has all eight sections |
| **Done well** | One requirement you decided *not* to meet, with the price of meeting it and the price of the risk, and an argument for the choice |

The third row is what separates an engineer from someone following a checklist.
Lesson 05's shadow-a-week option was the safest and the most expensive; knowing
when to skip a control, **with a number**, is the skill.

---

## Prerequisites

All eight [MLOps lessons](../lessons/), and a model you have already built and
evaluated.
