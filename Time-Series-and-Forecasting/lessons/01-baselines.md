# Lesson 01 — The Baselines You Must Beat

**Goal:** know what "good" means for a forecast before building anything, and
discover how often the one-line baseline wins.

## What you will learn

- The four baselines, and why they are not a formality
- MASE: measuring against the baseline instead of in absolute units
- A real model, measured against them
- Why forecasting is not ordinary supervised learning

---

## The series

```python
import numpy as np
import pandas as pd

import numpy as np, pandas as pd

def make_series(n_days=730, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    trend = 1200 + 0.9 * t
    weekly = 180 * np.sin(2 * np.pi * t / 7 - 1.0)
    yearly = 320 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 90, n_days)
    y = trend + weekly + yearly + noise
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    return pd.Series(np.round(y, 1), index=dates, name="revenue")

y = make_series()
y.to_frame().to_parquet("/tmp/daily_revenue.parquet")

print(f"{len(y)} days, {y.index.min().date()} to {y.index.max().date()}")
print(f"mean {y.mean():,.0f}   std {y.std():,.0f}   min {y.min():,.0f}   max {y.max():,.0f}")
print("\nfirst 7 days:")
print(y.head(7).to_string())
```

```text
730 days, 2025-01-01 to 2026-12-31
mean 1,526   std 285   min 722   max 2,228

first 7 days:
2025-01-01    1059.9
2025-01-02    1176.1
2025-01-03    1399.0
2025-01-04    1407.3
2025-01-05    1271.7
2025-01-06    1203.4
2025-01-07    1185.3
Freq: D
```

Two years of daily revenue with a trend, a **weekly** cycle, a **yearly** cycle
and noise — the shape of almost every business series you will meet.

---

## What makes forecasting different

Before any modelling, three properties that break the habits from
[Machine-Learning](../../Machine-Learning) and
[Data-Science](../../Data-Science):

| Property | Consequence |
|---|---|
| **The rows are ordered** | A random train/test split trains on the future (lesson 02) |
| **The rows are correlated** | Today looks like yesterday, so "more data" adds less information than it looks like |
| **You predict forward, not sideways** | The test set is *after* the training set, always |
| **The target is its own best feature** | Lagged values of `y` usually beat everything else |
| **Errors compound** | Forecasting 28 days ahead means feeding your own predictions back in |

That last row is why a model that is excellent one step ahead can be useless at
28, and why horizon is part of the problem statement, not a detail.

---

## The four baselines

```python
H = 28
train, test = y[:-H], y[-H:]

def naive(train, h):            return np.repeat(train.iloc[-1], h)
def seasonal_naive(train, h, m=7):
    last = train.iloc[-m:].to_numpy()
    return np.array([last[i % m] for i in range(h)])
def drift(train, h):
    n = len(train); slope = (train.iloc[-1] - train.iloc[0]) / (n - 1)
    return train.iloc[-1] + slope * np.arange(1, h + 1)
def mean_all(train, h):         return np.repeat(train.mean(), h)

def mae(a, f):  return float(np.mean(np.abs(a - f)))
def rmse(a, f): return float(np.sqrt(np.mean((a - f) ** 2)))
def mape(a, f): return float(np.mean(np.abs((a - f) / a)) * 100)

print(f"{'method':<22}{'MAE':>9}{'RMSE':>9}{'MAPE %':>9}")
results = {}
for name, fn in [("mean of all history", mean_all), ("naive (last value)", naive),
                 ("drift", drift), ("seasonal naive (7d)", seasonal_naive)]:
    f = fn(train, H)
    results[name] = f
    print(f"{name:<22}{mae(test, f):>9.1f}{rmse(test, f):>9.1f}{mape(test, f):>9.2f}")
```

```text
method                      MAE     RMSE   MAPE %
mean of all history       248.1    290.3    13.43
naive (last value)        141.3    167.5     7.87
drift                     136.3    161.0     7.64
seasonal naive (7d)       137.1    163.9     7.74
```

Four one-line forecasts, and three of them are already within 8% of the truth.

- **Mean of all history** — the only genuinely bad one, because it ignores the
  trend. 248.1.
- **Naive**: tomorrow equals today. 141.3.
- **Drift**: naive plus the average historical slope. 136.3 — the best of the
  four here, because this series has a strong trend.
- **Seasonal naive**: the same weekday last week. 137.1, and it is the one to
  beat on any series with a weekly cycle.

**These are not a formality.** A forecasting project whose model does not beat
drift has produced nothing, and the number of production forecasting systems
that do not beat seasonal naive is larger than anybody admits.

---

## A real model

```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

def lag_frame(s, lags=(1, 2, 3, 7, 14, 28), roll=(7, 28)):
    df = pd.DataFrame({"y": s})
    for L in lags:
        df[f"lag_{L}"] = s.shift(L)
    for w in roll:
        df[f"roll_{w}"] = s.shift(1).rolling(w).mean()
    df["dow"] = s.index.dayofweek
    df["t"] = np.arange(len(s))
    return df.dropna()

def recursive_forecast(model_cls, s, h, **kw):
    hist = s.copy()
    preds = []
    for _ in range(h):
        df = lag_frame(hist)
        X, yv = df.drop(columns="y"), df["y"]
        m = model_cls(**kw).fit(X, yv)
        nxt_index = hist.index[-1] + pd.Timedelta(days=1)
        extended = pd.concat([hist, pd.Series([np.nan], index=[nxt_index])])
        feat = lag_frame(extended.ffill()).iloc[[-1]].drop(columns="y")
        p = float(m.predict(feat)[0])
        preds.append(p)
        hist = pd.concat([hist, pd.Series([p], index=[nxt_index])])
    return np.array(preds)

for name, cls, kw in [("ridge on lags", Ridge, {"alpha": 1.0}),
                      ("gradient boosting", HistGradientBoostingRegressor, {"random_state": 0})]:
    f = recursive_forecast(cls, train, H, **kw)
    results[name] = f
    print(f"{name:<22}{mae(test, f):>9.1f}{rmse(test, f):>9.1f}{mape(test, f):>9.2f}")

best = min(results, key=lambda k: mae(test, results[k]))
sn = mae(test, results["seasonal naive (7d)"])
print(f"\nbest: {best}")
for k in results:
    print(f"  {k:<22} MASE-style ratio vs seasonal naive: {mae(test, results[k])/sn:.3f}")
```

```text
ridge on lags             124.9    151.5     6.96
gradient boosting         128.4    156.2     7.09

best: ridge on lags
  mean of all history    MASE-style ratio vs seasonal naive: 1.810
  naive (last value)     MASE-style ratio vs seasonal naive: 1.031
  drift                  MASE-style ratio vs seasonal naive: 0.994
  seasonal naive (7d)    MASE-style ratio vs seasonal naive: 1.000
  ridge on lags          MASE-style ratio vs seasonal naive: 0.911
  gradient boosting      MASE-style ratio vs seasonal naive: 0.937
```

**Ridge on lagged values beats seasonal naive by 8.9%.** Gradient boosting beats
it by 6.3% — and loses to the linear model, which is the same finding as
[Data-Science lesson 05](../../Data-Science/lessons/05-baselines-and-selection.md):
when the structure is close to linear, a linear model is the correct model.

Hold that 8.9% lightly. **Lesson 02 re-measures it across eight backtest folds
and the advantage almost entirely disappears.** A single split is not an
evaluation here any more than it is anywhere else in this track.

---

## MASE: the only scale-free metric worth using

The ratio column above is the idea behind **MASE** — mean absolute scaled
error: your error divided by the naive forecast's error.

```text
MASE < 1   you beat the baseline
MASE = 1   you are the baseline, with extra steps
MASE > 1   you are worse than one line of code
```

Why it is better than MAE or MAPE for reporting:

- **Scale-free**, so you can compare a forecast of revenue in EGP with a
  forecast of units sold.
- **It has a meaningful zero point.** "MAE 124.9" means nothing without the
  series; "MASE 0.911" means "9% better than doing nothing".
- **It does not explode near zero**, which MAPE does (lesson 03).

Report MASE first, and MAE in the units the business thinks in.

---

## The reporting rule

Every forecasting result gets reported as a comparison:

```text
BAD   "our model achieves 6.96% MAPE"
GOOD  "MASE 0.911 over 8 backtest folds (seasonal naive = 1.00),
       MAE 124.9 EGP/day against the baseline's 137.1"
```

The first sentence is unfalsifiable — 6.96% could be excellent or embarrassing
depending on the series. The second says exactly how much the model earned over
the thing you would have done for free.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| No baseline | "6.96% MAPE" is uninterpretable |
| Only comparing against the mean | The weakest baseline; everything beats it |
| No seasonal naive on a seasonal series | The baseline that usually wins |
| Reporting MAPE alone | Lesson 03: it is not neutral and it explodes near zero |
| Believing a single-split improvement | Lesson 02 measures what happens |
| Reaching for a complex model first | Ridge beat gradient boosting here |
| Forecasting one step ahead and shipping 28 | Errors compound; evaluate at your real horizon |

---

## Exercises

1. Compute all four baselines on a series from your own work. Which wins?
2. Report your current forecast's MASE. Is it below 1?
3. Change the horizon from 28 to 7 and to 90. How does the ranking change?
4. Build a series with no trend and rerun. Which baseline wins now, and why?
5. Write the one-sentence result for your forecast in the "GOOD" form above.

---

**Next:** [Lesson 02 — Evaluating a Forecast](02-evaluating.md)
