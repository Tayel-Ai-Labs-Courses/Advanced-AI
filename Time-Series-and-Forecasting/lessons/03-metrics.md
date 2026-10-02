# Lesson 03 — Metrics

**Goal:** choose an error measure that does not quietly bias your forecast.

## What you will learn

- MAPE's asymmetry, measured
- What happens near zero
- Which metric for which decision
- Over- and under-forecasting have different costs

---

## MAPE is not neutral

```python
import numpy as np

actual = np.array([100.0, 100.0, 10.0, 10.0, 1.0])
over  = actual * 1.5
under = actual * 0.5
def mape(a, f): return np.mean(np.abs((a - f) / a)) * 100
def smape(a, f): return np.mean(2 * np.abs(a - f) / (np.abs(a) + np.abs(f))) * 100
def mae(a, f): return np.mean(np.abs(a - f))
print(f"{'error':<28}{'MAE':>9}{'MAPE %':>9}{'sMAPE %':>10}")
print(f"{'forecast 50% too HIGH':<28}{mae(actual,over):>9.2f}{mape(actual,over):>9.1f}{smape(actual,over):>10.1f}")
print(f"{'forecast 50% too LOW':<28}{mae(actual,under):>9.2f}{mape(actual,under):>9.1f}{smape(actual,under):>10.1f}")
print("\nsame relative error, same MAPE — but a 50% over-forecast can be")
print("unbounded while a 50% under-forecast caps at 100%. MAPE punishes")
print("under-forecasting less, so optimising it biases you HIGH.")
tiny = np.array([0.1, 100.0]); fc = np.array([1.1, 101.0])
print(f"\nand near zero: actual {tiny.tolist()}, forecast {fc.tolist()}")
print(f"  MAE  {mae(tiny,fc):>7.2f}   MAPE {mape(tiny,fc):>8.1f}%   (one tiny value dominates)")
```

```text
error                             MAE   MAPE %   sMAPE %
forecast 50% too HIGH           22.10     50.0      40.0
forecast 50% too LOW            22.10     50.0      66.7

same relative error, same MAPE — but a 50% over-forecast can be
unbounded while a 50% under-forecast caps at 100%. MAPE punishes
under-forecasting less, so optimising it biases you HIGH.

and near zero: actual [0.1, 100.0], forecast [1.1, 101.0]
  MAE     1.00   MAPE    500.5%   (one tiny value dominates)
```

Two separate problems in one metric.

**The asymmetry.** A forecast that is 50% too high and one 50% too low both
score MAPE 50.0 here — but the two directions are not symmetric in general: you
can over-forecast by 500%, and you cannot under-forecast by more than 100%
(a forecast of zero). So a model tuned to minimise MAPE **drifts low**, because
under-forecasting has a bounded penalty.

sMAPE shows the asymmetry plainly: 40.0 against 66.7 for the same absolute
error.

**The explosion near zero.** Actual `[0.1, 100.0]`, forecast `[1.1, 101.0]` —
both off by exactly 1.0. MAE says 1.00. **MAPE says 500.5%**, because one tiny
denominator swamped everything.

That is not a corner case. Any series with quiet days, new products, or a
long tail of small values will have MAPE dominated by the smallest numbers,
which are usually the ones you care about least.

---

## Which metric for which decision

| Metric | Use when | Avoid when |
|---|---|---|
| **MAE** | The cost of an error is linear in its size | — |
| **RMSE** | Large errors are disproportionately bad | Outliers you do not care about |
| **MASE** | **Reporting, always.** Scale-free, baseline-relative | — |
| MAPE | Everyone expects it and the series is far from zero | **Any zeros or small values** |
| sMAPE | You must report a percentage | Still asymmetric, just less so |
| **Pinball / quantile loss** | You need an interval, not a point (lesson 06) | — |
| **A cost function** | You know what over- and under-forecasting cost | — |

The last row is the one that matters most and is used least.

---

## Over and under are not the same cost

For almost every real forecast, the two directions have different consequences:

| Forecast | Over-forecasting costs | Under-forecasting costs |
|---|---|---|
| Inventory | Storage, spoilage, capital | **Stockout: a lost sale and a lost customer** |
| Staffing | Idle wages | **Queues, and someone leaves** |
| Cloud capacity | An invoice | **An outage** |
| Cash flow | Idle money | **Insolvency** |

In three of those four, under-forecasting is much worse. An error metric that
treats them equally is telling your model the wrong thing.

```python
def asymmetric_cost(actual, forecast, over_cost=1.0, under_cost=4.0):
    """Over-forecast costs holding; under-forecast costs a lost sale."""
    err = forecast - actual
    return float(np.mean(np.where(err > 0, err * over_cost, -err * under_cost)))

actual = np.array([100.0, 100.0, 100.0, 100.0])
high = np.array([120.0, 120.0, 120.0, 120.0])
low  = np.array([80.0, 80.0, 80.0, 80.0])
print(f"{'forecast':<22}{'MAE':>8}{'asymmetric cost':>18}")
for name, f in [("20% high", high), ("20% low", low)]:
    print(f"{name:<22}{np.mean(np.abs(actual - f)):>8.1f}"
          f"{asymmetric_cost(actual, f):>18.1f}")
print("\nidentical MAE. The business cost differs by 4x.")
```

```text
forecast                   MAE   asymmetric cost
20% high                  20.0              20.0
20% low                   20.0              80.0

identical MAE. The business cost differs by 4x.
```

**Same MAE, four times the cost.** If you know the ratio — and the business
usually does, roughly — then train and select on the asymmetric cost, and your
forecast will deliberately run high, which is the correct behaviour.

This is [Data-Science lesson 06](../../Data-Science/lessons/06-evaluating-the-decision.md)'s
cost matrix, applied to a continuous output.

---

## What to report

```text
PRIMARY     MASE over N folds, with folds-won            scale-free, baseline-relative
BUSINESS    MAE in the units people think in             "±125 EGP/day"
IF ASYMMETRIC   the cost function's value                 the number that decides
NEVER ALONE     MAPE                                      biased and explosive
```

And always the per-fold spread (lesson 02), because a mean hides the fold where
the forecast was 18% worse than doing nothing.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| MAPE as the only metric | Asymmetric, and 500% from one small value |
| MAPE on a series containing zeros | Undefined, or infinite |
| Treating over- and under-forecasting equally | A stockout is not a storage cost |
| Optimising RMSE when outliers are noise | The model chases points you do not care about |
| Reporting a metric with no baseline | Lesson 01 |
| A percentage metric on a series that crosses zero | Meaningless |

---

## Exercises

1. Compute MAPE and MASE on your own series. Which of the two could you explain
   to a manager?
2. Find the smallest actual value in your series and compute its contribution to
   MAPE.
3. Ask the business for the ratio between an over- and under-forecast of one
   unit. Build the asymmetric cost.
4. Re-select your model on that cost. Does the chosen model change?
5. Write the three-line result block above for your current forecast.

---

**Next:** [Lesson 04 — Features and Horizons](04-features-and-horizons.md)
