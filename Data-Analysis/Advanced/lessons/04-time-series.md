# Lesson 04 — Time Series

**Goal:** separate trend from season from noise, and forecast honestly.

## What you will learn

- Decomposition
- Seasonality, and how to remove it
- Simple forecasts that are hard to beat
- Backtesting, and the traps

---

## A series with structure

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
days = pd.date_range("2024-01-01", periods=730, freq="D")

trend = np.linspace(1_000, 1_400, len(days))
weekly = 120 * np.sin(2 * np.pi * np.arange(len(days)) / 7)
yearly = 200 * np.sin(2 * np.pi * np.arange(len(days)) / 365)
noise = rng.normal(0, 60, len(days))

series = pd.Series(trend + weekly + yearly + noise, index=days, name="revenue")
series.to_frame().to_parquet("/tmp/series.parquet")

print(series.head(3).round(0).to_string())
print(f"\nmean {series.mean():.0f}, std {series.std():.0f}, n = {len(series)}")
```

```text
2024-01-01    1008.0
2024-01-02    1090.0
2024-01-03    1163.0
Freq: D

mean 1199, std 180, n = 730
```

Four components, mixed together: a rising trend, a weekly cycle, an annual
cycle, and noise. Real series look like this, and reporting any one of them as
"the trend" is how forecasts go wrong.

---

## Decomposition

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

series = pd.read_parquet("/tmp/series.parquet")["revenue"]
result = seasonal_decompose(series, model="additive", period=7)

parts = pd.DataFrame({
    "observed": result.observed,
    "trend": result.trend,
    "seasonal": result.seasonal,
    "residual": result.resid,
}).dropna()

print(parts.head(3).round(1).to_string())
print("\nvariance explained by each part:")
total = parts["observed"].var()
for part in ["trend", "seasonal", "residual"]:
    print(f"  {part:<10}{parts[part].var() / total:>7.1%}")
```

```text
            observed   trend  seasonal  residual
2024-01-04    1070.3  1028.0      53.6     -11.3
2024-01-05     931.7  1039.0     -45.9     -61.4
2024-01-06     924.6  1038.1    -119.6       6.1
2024-01-07    1008.3  1025.7     -92.6      75.2

variance explained by each part:
  trend       68.0%
  seasonal    22.5%
  residual     9.6%
```

The weekly pattern accounts for **22.5%** of the variance. An analyst
comparing "this Monday against last Sunday" is measuring that 22.5%, not the
business.

Two models:

| Model | When | Formula |
|---|---|---|
| **Additive** | The seasonal swing is a constant size | `y = trend + season + noise` |
| **Multiplicative** | The swing grows with the level | `y = trend × season × noise` |

If December is always "+500 sales", additive. If December is always "+30%",
multiplicative — which is more common in business data.

---

## Removing seasonality

```python
import pandas as pd

series = pd.read_parquet("/tmp/series.parquet")["revenue"]
frame = series.to_frame()
frame["weekday"] = frame.index.dayofweek

weekday_effect = frame.groupby("weekday")["revenue"].mean() - frame["revenue"].mean()
print("average deviation by weekday:")
print(weekday_effect.round(1).to_string())

frame["adjusted"] = frame["revenue"] - frame["weekday"].map(weekday_effect)
print(f"\nraw std:      {frame['revenue'].std():.1f}")
print(f"adjusted std: {frame['adjusted'].std():.1f}")

last_two = frame.iloc[-2:]
print(f"\nraw change, last two days:      "
      f"{last_two['revenue'].iloc[1] - last_two['revenue'].iloc[0]:+.0f}")
print(f"adjusted change, last two days: "
      f"{last_two['adjusted'].iloc[1] - last_two['adjusted'].iloc[0]:+.0f}")
```

```text
average deviation by weekday:
weekday
0     -7.6
1     93.4
2    117.5
3     52.8
4    -46.1
5   -119.3
6    -91.5

raw std:      179.9
adjusted std: 158.8

raw change, last two days:      -71
adjusted change, last two days: -172
```

The weekday effects span 237 points, from +117.5 on Wednesday to −119.3 on
Saturday — on a series with a standard deviation of 180. Removing them cuts
the variation from 179.9 to 158.8.

Now the last two days. Raw, revenue fell by 71. **Adjusted, it fell by 172** —
two and a half times as much. The calendar was *hiding* a real decline,
because the second day was a naturally stronger weekday.

Seasonal adjustment does not always shrink a change. It corrects it, and the
correction can go either way.

Two alternatives that need no model:

- **Year-on-year**: compare with the same day last year. Removes annual and
  weekly seasonality at once, at the cost of a year's lag.
- **Compare like periods**: this Monday against last Monday, this week against
  last week.

---

## Forecasting: start with the naive methods

```python
import numpy as np
import pandas as pd

series = pd.read_parquet("/tmp/series.parquet")["revenue"]
train, test = series[:-28], series[-28:]

forecasts = {
    "naive (last value)": np.repeat(train.iloc[-1], len(test)),
    "seasonal naive (t-7)": np.tile(train.iloc[-7:].values, 4)[:len(test)],
    "mean of last 28": np.repeat(train.iloc[-28:].mean(), len(test)),
    "drift": (train.iloc[-1]
              + (np.arange(1, len(test) + 1)
                 * (train.iloc[-1] - train.iloc[0]) / (len(train) - 1))),
}

print(f"{'method':<24}{'MAE':>9}{'MAPE':>9}")
for name, prediction in forecasts.items():
    mae = np.mean(np.abs(test.values - prediction))
    mape = np.mean(np.abs((test.values - prediction) / test.values)) * 100
    print(f"{name:<24}{mae:>9.1f}{mape:>8.1f}%")
```

```text
method                        MAE     MAPE
naive (last value)          104.6     8.1%
seasonal naive (t-7)         88.3     6.5%
mean of last 28             106.1     7.5%
drift                       106.4     8.3%
```

**Seasonal naive — "the same as this day last week" — wins**, with 6.5%
error, against 8.1% for the last value and 7.5% for a 28-day mean.

That is the baseline any forecasting model must beat. A model with 7% error is
**worse than copying last week**, however sophisticated it is, and publishing
it without this comparison is how teams end up with an ARIMA that
underperforms one line of pandas.

---

## Backtesting

```python
import numpy as np
import pandas as pd

series = pd.read_parquet("/tmp/series.parquet")["revenue"]

def backtest(series, horizon=7, folds=8, method="seasonal_naive"):
    """Rolling-origin evaluation: train on the past, predict the next horizon."""
    errors = []
    for fold in range(folds):
        end = len(series) - (folds - fold) * horizon
        train, test = series[:end], series[end:end + horizon]
        if method == "seasonal_naive":
            prediction = train.iloc[-7:].values[:horizon]
        else:
            prediction = np.repeat(train.iloc[-1], horizon)
        errors.append(np.mean(np.abs(test.values - prediction)))
    return np.array(errors)

for method in ["naive", "seasonal_naive"]:
    errors = backtest(series, method=method)
    print(f"{method:<16} MAE per fold: {errors.round(0).astype(int).tolist()}")
    print(f"{'':<16} mean {errors.mean():.1f}, std {errors.std():.1f}")
```

```text
naive            MAE per fold: [67, 87, 133, 152, 132, 107, 90, 105]
                 mean 109.2, std 26.6
seasonal_naive   MAE per fold: [54, 45, 62, 50, 62, 67, 83, 63]
                 mean 60.8, std 10.9
```

Eight folds, not one. The seasonal naive method wins on **every fold**, and
its error varies far less (std 10.9 against 26.6).

Look at the naive row: fold errors from 67 to 152. A single test split landing
on the first fold would have reported an error of 67 and concluded the naive
method was nearly as good.

A single train/test split would have given one number and no sense of whether
it was luck. **Rolling-origin backtesting is to time series what
cross-validation is to everything else** — and a random `train_test_split` is
simply wrong here, because it trains on the future.

---

## The traps

| Trap | What happens | Defence |
|---|---|---|
| **Random train/test split** | Trains on the future; optimistic nonsense | Split by time, always |
| **Look-ahead features** | "Monthly total" known only at month end | Lag every feature |
| **Revised data** | Yesterday's number changes tomorrow | Backtest with data as it was |
| **Changed definitions** | A metric redefined in March | Check for level shifts |
| **One-off events** | Ramadan, a campaign, an outage | Flag them; do not let them train |
| Extrapolating a trend | Nothing grows linearly forever | Bound the forecast |

```python
import numpy as np
import pandas as pd

series = pd.read_parquet("/tmp/series.parquet")["revenue"]
last = series.iloc[-1]
growth = (series.iloc[-1] / series.iloc[0]) ** (1 / len(series)) - 1

print(f"latest value: {last:,.0f}")
for years in [1, 5, 10]:
    projected = last * (1 + growth) ** (365 * years)
    print(f"linear extrapolation, {years:>2} year(s): {projected:>12,.0f}")
```

```text
latest value: 1,433
linear extrapolation,  1 year(s):        1,709
linear extrapolation,  5 year(s):        3,455
linear extrapolation, 10 year(s):        8,332
```

Extrapolating two years of growth ten years forward predicts **5.8× the
current revenue** — from a series whose actual trend was a 40% rise over two
years. Compounding a short-run growth rate is how a plausible number becomes
an absurd one.

Every forecast beyond a short horizon needs a stated assumption about
saturation, and a forecast without an interval is a wish.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Comparing adjacent days | You measure the weekly cycle |
| No seasonal baseline | Your model may be worse than "same day last week" |
| Random split | Trains on the future |
| One backtest fold | No idea of the variance |
| Extrapolating far | Absurd numbers, confidently |
| Forecast without an interval | False precision, at scale |

---

## Exercises

1. Decompose your main series; what share is seasonal?
2. Adjust for the weekday effect and re-read the last week's change.
3. Beat the seasonal naive baseline — or report that you could not.
4. Backtest over eight folds and report the mean and spread of the error.
