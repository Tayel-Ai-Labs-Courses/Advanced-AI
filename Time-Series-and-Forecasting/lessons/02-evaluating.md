# Lesson 02 — Evaluating a Forecast

**Goal:** stop evaluating a time series the way you evaluate a table, and watch
a "clear win" disappear.

## What you will learn

- Why random cross-validation lies, measured
- Backtesting, and what eight folds say that one does not
- Choosing a horizon and a cadence
- The evaluation protocol

---

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

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

df = lag_frame(y)
X, yv = df.drop(columns="y"), df["y"]
print(f"{len(df)} rows, {X.shape[1]} features")
```

```text
702 rows, 10 features
```

---

## Random cross-validation lies

```python
for name, cv in [("KFold(5), shuffled", KFold(5, shuffle=True, random_state=0)),
                 ("KFold(5), not shuffled", KFold(5, shuffle=False)),
                 ("TimeSeriesSplit(5)", TimeSeriesSplit(5))]:
    s = -cross_val_score(Ridge(alpha=1.0), X, yv, cv=cv,
                         scoring="neg_mean_absolute_error")
    print(f"{name:<26} MAE {s.mean():>7.1f} +/- {s.std():>5.1f}")
print("\nonly the last one never trains on the future")
```

```text
KFold(5), shuffled         MAE    85.3 +/-   4.2
KFold(5), not shuffled     MAE    85.9 +/-   4.8
TimeSeriesSplit(5)         MAE   107.7 +/-  38.4

only the last one never trains on the future
```

**Shuffled K-fold reports 85.3; the honest split reports 107.7.** A 26%
understatement of the error, and it comes from training on days that come after
the days being predicted.

Look at the spreads too: **4.2 against 38.4**. Random CV does not only
understate the error, it hides how much the error *varies* — which is the number
that tells you whether the forecast is dependable.

And note the middle row: un-shuffled `KFold` is barely better (85.9), because
fold 1 still trains on folds 2-5, all of which are in the future.
**`shuffle=False` is not a time-series split.**

```text
TimeSeriesSplit      train [-----]      test [--]
                     train [--------]   test [--]
                     train [-----------] test [--]
```

Training always ends before testing begins. That is the only acceptable shape.

---

## One split is not an evaluation

Lesson 01 reported ridge beating seasonal naive by 8.9% on a single split. Here
is the same comparison across eight consecutive 28-day windows.

```python
def backtest(model_cls, s, horizon=28, folds=8, **kw):
    maes = []
    for k in range(folds, 0, -1):
        cut = len(s) - k * horizon
        train, test = s.iloc[:cut], s.iloc[cut:cut + horizon]
        d = lag_frame(train)
        m = model_cls(**kw).fit(d.drop(columns="y"), d["y"])
        hist = train.copy(); preds = []
        for _ in range(horizon):
            nxt = hist.index[-1] + pd.Timedelta(days=1)
            ext = pd.concat([hist, pd.Series([np.nan], index=[nxt])]).ffill()
            preds.append(float(m.predict(lag_frame(ext).iloc[[-1]].drop(columns="y"))[0]))
            hist = pd.concat([hist, pd.Series([preds[-1]], index=[nxt])])
        maes.append(float(np.mean(np.abs(test.to_numpy() - np.array(preds)))))
    return np.array(maes)

def seasonal_naive_bt(s, horizon=28, folds=8, m=7):
    out = []
    for k in range(folds, 0, -1):
        cut = len(s) - k * horizon
        train, test = s.iloc[:cut], s.iloc[cut:cut + horizon]
        last = train.iloc[-m:].to_numpy()
        f = np.array([last[i % m] for i in range(horizon)])
        out.append(float(np.mean(np.abs(test.to_numpy() - f))))
    return np.array(out)

bt_ridge = backtest(Ridge, y, alpha=1.0)
bt_sn = seasonal_naive_bt(y)
print(f"{'fold':>6}{'ridge MAE':>12}{'seasonal naive':>16}{'ridge better?':>15}")
for i, (a, b) in enumerate(zip(bt_ridge, bt_sn), 1):
    print(f"{i:>6}{a:>12.1f}{b:>16.1f}{('yes' if a < b else 'NO'):>15}")
print(f"{'mean':>6}{bt_ridge.mean():>12.1f}{bt_sn.mean():>16.1f}"
      f"{(f'{bt_ridge.mean()/bt_sn.mean():.3f}x'):>15}")
print(f"\nridge wins {int((bt_ridge < bt_sn).sum())} of {len(bt_ridge)} folds")
print(f"single-split result would have been fold 8 alone: "
      f"{bt_ridge[-1]:.1f} vs {bt_sn[-1]:.1f}")
```

```text
  fold   ridge MAE  seasonal naive  ridge better?
     1       105.1           117.5            yes
     2       177.4           150.7             NO
     3        85.1            97.9            yes
     4        92.7           100.6            yes
     5       118.2           104.6             NO
     6       117.4           114.1             NO
     7       120.6           123.2            yes
     8       124.9           137.1            yes
  mean       117.7           118.2         0.995x

ridge wins 5 of 8 folds
single-split result would have been fold 8 alone: 124.9 vs 137.1
```

**Across eight folds, ridge is 0.5% better than seasonal naive.** It wins five
folds and loses three. On fold 2 it is **18% worse**.

And fold 8 — the most recent window, which is exactly the split lesson 01 used
and the one most people evaluate on — shows 124.9 against 137.1, a 9% win.

**The single split did not lie. It just happened to be a good fold.** With a
per-fold spread this wide, any one window is a coin toss, and the honest
conclusion is: *"ridge and seasonal naive are indistinguishable on this
series; the apparent 9% win is one favourable window."*

That is a real result, and it is the one a careful reviewer would have asked
for — [Research lesson 03](../../Research-and-Review/lessons/03-what-the-numbers-hide.md)
measured the same effect in published work.

---

## Choosing the horizon and the cadence

Three numbers define a forecasting evaluation, and all three come from the
business, not the data:

| Number | Question | Example |
|---|---|---|
| **Horizon** | How far ahead must you see? | 28 days, because that is the ordering lead time |
| **Cadence** | How often do you re-forecast? | Daily, so yesterday's actual is available |
| **Folds** | How many windows do you evaluate over? | 8+, covering at least one seasonal cycle |

Evaluate at the **horizon you will actually use**. A model evaluated one step
ahead and deployed at 28 is a different model with a different error, because
errors compound through the recursion (lesson 04).

And cover a full seasonal cycle. Eight 28-day folds is 224 days — most of a
year, which is enough to include both the busy and the quiet season. Four folds
of a seasonal series can be four Decembers.

---

## The protocol

```text
1. Choose the horizon and cadence from the business      not from the data
2. Hold out the final window entirely                     never touched until the end
3. Backtest over 8+ rolling windows on the rest           lesson 02
4. Report the MEAN and the PER-FOLD table                 the spread is the finding
5. Report MASE against seasonal naive                     lesson 01
6. Count folds won, not just the mean                     5 of 8 is not a win
7. Only then, look at the held-out final window           once
```

Step 6 is worth the habit. A mean can be dragged by one fold; "wins 5 of 8" is
a statement nobody can misread.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| `KFold(shuffle=True)` on a time series | 26% understatement of the error |
| `KFold(shuffle=False)` believing it is safe | Fold 1 still trains on the future |
| A single train/test split | Fold 8 said +9%; eight folds said +0.5% |
| Reporting the mean without the per-fold table | The spread is the finding |
| Fewer folds than a seasonal cycle | You evaluated four Decembers |
| Evaluating at a different horizon than you deploy | Errors compound |
| Touching the final holdout more than once | It stops being a holdout |

---

## Exercises

1. Run shuffled K-fold and `TimeSeriesSplit` on your own series. What is the
   gap?
2. Backtest your current model over 8 folds. How many does it win?
3. Report your result as "wins N of M folds, mean MASE X". Does it still look
   like a win?
4. Change the horizon to your real deployment horizon and re-backtest.
5. Find a forecast in your organisation evaluated on one split. Re-evaluate it.

---

**Next:** [Lesson 03 — Metrics](03-metrics.md)
