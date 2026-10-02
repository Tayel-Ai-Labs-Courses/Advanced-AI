# Lesson 04 — Features and Horizons

**Goal:** build features that do not leak, and choose between the two ways of
forecasting more than one step ahead.

## What you will learn

- Lag and rolling features, and the shift that prevents leakage
- How error grows with horizon
- Recursive against direct, measured
- Calendar and external features

---

## Setup

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

---

## The features

Three families, and one rule that governs all of them.

```text
LAGS        y(t-1), y(t-7), y(t-28)        yesterday, same weekday, same day last month
ROLLING     mean/std/min/max over a window  level and volatility
CALENDAR    day of week, month, holiday     the only features that are NOT lagged
```

**The rule: every feature computed from `y` must be shifted before it is
rolled.**

```python
# no-run
df["roll_7_WRONG"] = s.rolling(7).mean()           # includes TODAY's value
df["roll_7_right"] = s.shift(1).rolling(7).mean()  # ends yesterday
```

The first line is the leakage from
[Data-Science lesson 03](../../Data-Science/lessons/03-the-data-you-have.md),
in one character. A rolling mean that includes today's value means the feature
contains the answer, and the model will score beautifully until it meets
production, where today's value does not exist yet.

**Calendar features are the exception** — the day of the week for a future date
is knowable now, so no shift is needed. Everything derived from the target
needs one.

---

## Error grows with horizon

```python
d = lag_frame(train)
m = Ridge(alpha=1.0).fit(d.drop(columns="y"), d["y"])
hist = train.copy(); preds = []
for _ in range(H):
    nxt = hist.index[-1] + pd.Timedelta(days=1)
    ext = pd.concat([hist, pd.Series([np.nan], index=[nxt])]).ffill()
    p = float(m.predict(lag_frame(ext).iloc[[-1]].drop(columns="y"))[0])
    preds.append(p); hist = pd.concat([hist, pd.Series([p], index=[nxt])])
preds = np.array(preds); act = test.to_numpy()
print(f"{'days ahead':>12}{'MAE so far':>13}{'MAE that day':>15}")
for k in (1, 3, 7, 14, 28):
    print(f"{k:>12}{np.mean(np.abs(act[:k]-preds[:k])):>13.1f}{abs(act[k-1]-preds[k-1]):>15.1f}")
```

```text
  days ahead   MAE so far   MAE that day
           1        190.8          190.8
           3         74.3           15.4
           7         76.3            1.0
          14         78.9          125.9
          28        124.9           58.4
```

Read the **"MAE so far"** column: 74-79 through two weeks, then **124.9 by day
28.** The cumulative error nearly doubles over the second fortnight.

The per-day column is noisy — day 1 was unlucky at 190.8 and day 7 was almost
exact at 1.0 — which is itself the point: **a single day's error tells you
nothing.** Evaluate across the horizon and across folds (lesson 02).

Why it grows: the recursive forecast feeds its own predictions back in as lags,
so by day 28 the model is reasoning about a history it invented. Small errors
early become the inputs to later steps.

---

## Recursive against direct

Two ways to forecast `h` steps ahead:

```text
RECURSIVE   one model. Predict t+1, append it, predict t+2 from that, ...
DIRECT      h models. Model k is trained to predict y(t+k) from today's features
```

```python
def direct_forecast(s, h):
    out = []
    for k in range(1, h + 1):
        df = lag_frame(s)
        df["target"] = df["y"].shift(-k)
        dd = df.dropna()
        mk = Ridge(alpha=1.0).fit(dd.drop(columns=["y", "target"]), dd["target"])
        out.append(float(mk.predict(lag_frame(s).iloc[[-1]].drop(columns="y"))[0]))
    return np.array(out)

dir_preds = direct_forecast(train, H)
print(f"{'horizon':>9}{'recursive MAE':>16}{'direct MAE':>13}")
for k in (7, 14, 28):
    print(f"{k:>9}{np.mean(np.abs(act[:k]-preds[:k])):>16.1f}"
          f"{np.mean(np.abs(act[:k]-dir_preds[:k])):>13.1f}")
```

```text
  horizon   recursive MAE   direct MAE
        7            76.3         70.7
       14            78.9         71.5
       28           124.9         73.1
```

**At 28 days, direct is 42% better** — 73.1 against 124.9 — and the gap widens
with horizon exactly as the compounding argument predicts. At 7 days they are
close (70.7 against 76.3), because there has been little time to compound.

| | Recursive | Direct |
|---|---|---|
| Models to train | **1** | h |
| Error compounding | **Yes**, and it grows | No |
| Uses predicted values as inputs | Yes | No |
| Consistent across horizons | Yes | Each horizon is its own model |
| Best at short horizons | Comparable | Comparable |
| Best at long horizons | — | **Clearly** |
| Cost to maintain | Low | h times the pipeline |

**Use direct when the horizon is long and you can afford h models.** Use
recursive when the horizon is short, or when you need one coherent trajectory
rather than h independent point forecasts.

There is a middle path worth knowing: train **one** model with the horizon `k`
as a feature. One model, no compounding, and it generalises across horizons.

---

## External features

Anything not derived from `y`: price, weather, a campaign flag, a holiday
calendar. They can help a great deal, and they bring one hard requirement.

> **You must know the feature's value for the forecast period.**

If you forecast 28 days ahead using tomorrow's temperature, you now need a
temperature forecast — and its error enters yours. Three options:

| Option | When |
|---|---|
| Use only **known-in-advance** features (holidays, planned promotions, prices you set) | Always first |
| Forecast the feature too, and accept the compounded error | When the feature matters a lot |
| **Lag the feature** so you use last week's value | Usually the right compromise |

A feature that improves backtest accuracy and is unavailable at forecast time
is the most common way a forecasting project fails after launch — it is
[Data-Science lesson 02](../../Data-Science/lessons/02-framing-the-problem.md)'s
prediction-time rule, in a new costume.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| `rolling()` without `shift(1)` | The feature contains today's answer |
| Shifting calendar features | They are knowable in advance; shifting throws away information |
| Recursive at a long horizon | 124.9 against direct's 73.1 at 28 days |
| Evaluating one step ahead, deploying at 28 | A different model with a different error |
| An external feature you will not have | The project fails at launch, not in backtest |
| Reading a single day's error | Day 7 was off by 1.0 and day 1 by 190.8 |

---

## Exercises

1. Add a rolling feature without `shift(1)` and watch backtest error collapse.
   That collapse is the leak.
2. Compare recursive and direct at your real horizon. What is the gap?
3. Build the single-model-with-horizon-as-a-feature version. Where does it land?
4. List your external features. For each: will you have it at forecast time?
5. Plot MAE against days ahead for your own model. Where does it stop being
   useful?

---

**Next:** [Lesson 05 — Seasonality and Decomposition](05-seasonality.md)
