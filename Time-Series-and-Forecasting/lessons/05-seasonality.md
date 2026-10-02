# Lesson 05 — Seasonality and Decomposition

**Goal:** separate a series into the parts you can predict and the part you
cannot.

## What you will learn

- Trend, seasonality and remainder
- Finding the period you did not know about
- Multiple seasonalities
- When decomposition misleads

---

## The three parts

```text
observed = trend + seasonality + remainder        (additive)
observed = trend x seasonality x remainder        (multiplicative)
```

**Additive** when the seasonal swing is a constant number of units — "Fridays
are 180 EGP higher". **Multiplicative** when it is a constant percentage —
"Fridays are 12% higher". A series whose seasonal swing grows with its level is
multiplicative, and the fix is to model `log(y)` additively.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
t = np.arange(730)
level = 1000 + 1.5 * t
season = np.sin(2 * np.pi * t / 7)
additive = level + 150 * season + rng.normal(0, 40, 730)
multiplicative = level * (1 + 0.12 * season) * (1 + rng.normal(0, 0.03, 730))

for name, s in (("additive", additive), ("multiplicative", multiplicative)):
    early = s[:90].std()
    late = s[-90:].std()
    print(f"{name:<16} sd of first 90 days {early:>7.1f}   last 90 days {late:>7.1f}"
          f"   ratio {late/early:>5.2f}")
print("\na growing seasonal swing means multiplicative — model log(y) instead")
```

```text
additive         sd of first 90 days   119.3   last 90 days   119.6   ratio  1.00
multiplicative   sd of first 90 days    96.8   last 90 days   193.4   ratio  2.00

a growing seasonal swing means multiplicative — model log(y) instead
```

**The diagnostic is one ratio.** If the spread late in the series is much larger
than early, the seasonality is multiplicative and an additive model will
under-fit the recent, larger swings — which is exactly where your forecast lives.

---

## Finding the period

You usually know the weekly cycle. The ones that cost money are the ones nobody
mentioned: a monthly payroll cycle, a 28-day promotional rhythm, a quarterly
close.

```python
def autocorrelation(x, lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

series = additive
print(f"{'lag':>5}{'autocorrelation':>18}")
for lag in (1, 2, 3, 6, 7, 8, 14, 21, 28, 30, 365):
    if lag < len(series):
        print(f"{lag:>5}{autocorrelation(series, lag):>18.3f}")
```

```text
  lag   autocorrelation
    1             0.948
    2             0.862
    3             0.794
    6             0.947
    7             0.986
    8             0.947
   14             0.985
   21             0.984
   28             0.985
   30             0.852
  365             0.852
```

**Lag 7 scores 0.986, and 14, 21, 28 all sit at 0.984-0.985**, while lags 2 and
3 dip to 0.862 and 0.794. Peaks at multiples of 7 are the signature of a weekly
cycle; the high floor everywhere is the trend, which correlates everything with
everything.

That floor is why you should look at the **shape**, not the absolute values: the
local maxima at 7, 14, 21, 28 are the signal, and detrending first (subtract a
rolling mean) makes them far more obvious.

Run this on any new series before modelling. A spike at an unexpected lag is a
cycle somebody forgot to tell you about, and lag features at that period are
usually the single biggest improvement available.

---

## Multiple seasonalities

Daily business data normally has at least two: **weekly** and **yearly**. Hourly
data has three: hourly, weekly, yearly.

Classical decomposition handles one. Three practical options:

| Approach | How | When |
|---|---|---|
| **Fourier terms** | `sin`/`cos` pairs at each period, as features | **The default for ML models.** Cheap, works with any regressor |
| Dummy variables | One column per day-of-week, per month | Few periods, plenty of data |
| A model that supports it | Prophet, TBATS | You want it handled for you |

Fourier terms in four lines:

```python
def fourier_terms(index, period, order=3):
    t = np.arange(len(index))
    out = {}
    for k in range(1, order + 1):
        out[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
        out[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(out, index=index)

idx = pd.date_range("2025-01-01", periods=730, freq="D")
feats = pd.concat([fourier_terms(idx, 7, order=3),
                   fourier_terms(idx, 365.25, order=5)], axis=1)
print(f"{feats.shape[1]} features for two seasonalities")
print(list(feats.columns[:4]), "...")
```

```text
16 features for two seasonalities
['sin_7_1', 'cos_7_1', 'sin_7_2', 'cos_7_2'] ...
```

Sixteen columns buy both cycles, and a linear model can use them directly — a
smooth seasonal shape with far fewer parameters than 7 + 12 dummies, and it
extrapolates to future dates by construction.

**Order controls smoothness.** Order 3 on a weekly cycle is plenty; order 10 on
a yearly cycle starts fitting noise.

---

## When decomposition misleads

| Trap | What happens |
|---|---|
| **Decomposing the whole series, then splitting** | The decomposition saw the future. Leakage, [Data-Science 03](../../Data-Science/lessons/03-the-data-you-have.md) |
| A changing seasonal pattern | A single fixed seasonal component is wrong on both ends |
| Moving holidays (Eid, Ramadan, Easter) | They shift against the Gregorian calendar; a day-of-year feature cannot see them |
| Decomposing a short series | Fewer than two full cycles and the estimate is noise |
| Treating the remainder as "noise" | It often contains the events you most want to explain |

The third row matters specifically here. **Ramadan and Eid move about 11 days
earlier each Gregorian year**, so a model with only `month` and `day-of-week`
cannot learn them. They need an explicit calendar feature — days-to-Eid,
in-Ramadan — joined from a Hijri calendar, and for most Egyptian businesses it
is the single most valuable external feature there is.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Assuming additive without checking | A growing swing needs `log(y)` |
| Never plotting autocorrelation | You miss the cycle nobody mentioned |
| One seasonality when there are two | The yearly pattern lands in the remainder |
| Decomposing before splitting | The seasonal component saw the test period |
| Day-of-year features for Eid or Ramadan | They move 11 days a year |
| Fourier order chosen by feel | High order fits noise; check it in backtest |

---

## Exercises

1. Compute the early/late spread ratio for your series. Additive or
   multiplicative?
2. Plot autocorrelation to lag 400. Which peaks did you not expect?
3. Add Fourier terms for weekly and yearly and re-backtest. How much did MASE
   move?
4. Build a days-to-Eid feature and measure its effect on a series you own.
5. Decompose once on the full series and once per backtest fold. How different
   are the seasonal components?

---

**Next:** [Lesson 06 — Prediction Intervals](06-prediction-intervals.md)
