# Lesson 06 — Prediction Intervals

**Goal:** give a range instead of a number, and make sure the range is honest.

## What you will learn

- Why a point forecast is not a plan
- Intervals from backtest residuals, with measured coverage
- Why in-sample residuals give intervals 39% too narrow
- Quantile forecasting, and asymmetric decisions

---

## A point forecast is not a plan

"Revenue next month will be 1,540 EGP/day" is not actionable. **"Between 1,300
and 1,790, 90% of the time"** is: you can size inventory, staffing and cash
against it.

The method that works, and needs no distributional assumption: **collect the
errors your model actually made in backtesting, and use their quantiles.**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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

def lag_frame(s, lags=(1, 2, 3, 7, 14, 28), roll=(7, 28)):
    df = pd.DataFrame({"y": s})
    for L in lags:
        df[f"lag_{L}"] = s.shift(L)
    for w in roll:
        df[f"roll_{w}"] = s.shift(1).rolling(w).mean()
    df["dow"] = s.index.dayofweek
    df["t"] = np.arange(len(s))
    return df.dropna()

H = 28
train, test = y[:-H], y[-H:]
print(f"{len(y)} days, horizon {H}")
```

```text
730 days, horizon 28
```

```python
resid = []
for k in range(8, 0, -1):
    cut = len(y) - k * H
    tr, te = y.iloc[:cut], y.iloc[cut:cut+H]
    dtr = lag_frame(tr)
    mm = Ridge(alpha=1.0).fit(dtr.drop(columns="y"), dtr["y"])
    h2 = tr.copy(); pp = []
    for _ in range(H):
        nx = h2.index[-1] + pd.Timedelta(days=1)
        e2 = pd.concat([h2, pd.Series([np.nan], index=[nx])]).ffill()
        v = float(mm.predict(lag_frame(e2).iloc[[-1]].drop(columns="y"))[0])
        pp.append(v); h2 = pd.concat([h2, pd.Series([v], index=[nx])])
    resid.append(te.to_numpy() - np.array(pp))
resid = np.vstack(resid)
print(f"residuals collected from {resid.shape[0]} backtest folds x {resid.shape[1]} days")
print(f"{'nominal':>9}{'empirical coverage':>21}{'interval width':>17}")
for level in (0.50, 0.80, 0.90, 0.95):
    lo, hi = np.quantile(resid, [(1-level)/2, 1-(1-level)/2])
    cov = float(np.mean((resid >= lo) & (resid <= hi)))
    print(f"{level:>9.0%}{cov:>21.1%}{hi-lo:>17.1f}")
print("\nthese come from BACKTEST residuals, not from the training fit")
```

```text
residuals collected from 8 backtest folds x 28 days
  nominal   empirical coverage   interval width
      50%                50.0%            209.1
      80%                79.5%            389.5
      90%                89.3%            457.7
      95%                94.6%            545.7

these come from BACKTEST residuals, not from the training fit
```

**Nominal 90%, empirical 89.3%.** The intervals are calibrated, because they
were built from errors the model made on data it had not seen — exactly the
conditions it will meet in production.

This is the forecasting version of
[Data-Science lesson 06](../../Data-Science/lessons/06-evaluating-the-decision.md)'s
calibration table, and it is checked the same way: **compare nominal coverage
with empirical coverage.** A 90% interval that contains the truth 70% of the
time is worse than no interval, because somebody planned against it.

---

## In-sample residuals are a trap

```python
dtr = lag_frame(train)
mm = Ridge(alpha=1.0).fit(dtr.drop(columns="y"), dtr["y"])
insample = dtr["y"].to_numpy() - mm.predict(dtr.drop(columns="y"))
print(f"in-sample residual sd : {insample.std():>7.1f}")
print(f"backtest residual sd  : {resid.std():>7.1f}   ({resid.std()/insample.std():.2f}x wider)")
```

```text
in-sample residual sd :   104.0
backtest residual sd  :   144.8   (1.39x wider)
```

**Intervals built from training residuals are 39% too narrow.**

The model has seen the training data, so its errors there are smaller than its
errors on the future — the same gap that
[Data-Security lesson 03](../../Data-Security-for-AI/lessons/03-memorisation.md)
used to measure memorisation. A "95% interval" built this way is really about
80%, and nobody finds out until the plan built on it fails.

**Always build intervals from out-of-sample residuals.** It costs you the
backtest you should be running anyway (lesson 02).

One refinement: residuals **grow with horizon** (lesson 04), so compute them
per step-ahead:

```python
# no-run
# resid has shape (folds, horizon)
for k in range(resid.shape[1]):
    lo, hi = np.quantile(resid[:, k], [0.05, 0.95])
    print(f"day {k+1:>2}: 90% interval [{lo:+.0f}, {hi:+.0f}]")
```

A single interval width for all 28 days is too wide at day 1 and too narrow at
day 28.

---

## Quantile forecasting

For an asymmetric decision (lesson 03), you do not want the middle — you want a
specific quantile.

| Decision | Forecast the | Because |
|---|---|---|
| Inventory with expensive stockouts | **90th percentile** | Running out costs 4x holding |
| Staffing with a service-level target | 95th percentile | A queue is worse than idle time |
| Cash reserve | **5th percentile** of income | Plan for the bad case |
| Capacity planning | 99th percentile | Outages are catastrophic |
| A headline number for a board | 50th (median) | It is the typical case |

Two ways to get them:

1. **Residual quantiles**, as above — add the relevant quantile of the backtest
   residuals to the point forecast. Simple, works with any model.
2. **Quantile regression** — train directly on the pinball loss for the quantile
   you need. `GradientBoostingRegressor(loss="quantile", alpha=0.9)` does this,
   and it can capture a quantile that moves differently from the mean.

Start with method 1. It is three lines, and it reuses the backtest.

---

## Reporting

```text
BAD   "Revenue will be 1,540 EGP/day next month."
GOOD  "Median 1,540 EGP/day. 90% interval [1,310, 1,770], widening to
       [1,180, 1,900] by day 28. Coverage checked at 89.3% over 8 backtest
       folds. Plan inventory against the 90th percentile, 1,770."
```

The second version is four sentences and it contains everything a planner
needs: the central case, the uncertainty, the fact that it grows, evidence the
interval is honest, and which number to actually use.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A point forecast with no interval | Nobody can plan against it |
| Intervals from in-sample residuals | 39% too narrow |
| One interval width for the whole horizon | Too wide at day 1, too narrow at day 28 |
| Never checking empirical coverage | A "90%" interval that covers 70% |
| Assuming normal residuals | Use the empirical quantiles; they cost nothing |
| Planning against the median for an asymmetric decision | The whole point of quantiles |

---

## Exercises

1. Build intervals from in-sample and from backtest residuals. Compare widths.
2. Measure empirical coverage at 50, 80, 90 and 95%. Are they calibrated?
3. Compute per-horizon intervals and plot how they widen.
4. Ask the business which quantile their decision actually needs.
5. Rewrite your last forecast report in the "GOOD" form above.

---

**Next:** [Lesson 07 — Many Series at Once](07-many-series.md)
