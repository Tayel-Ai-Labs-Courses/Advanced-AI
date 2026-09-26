# Lesson 06 — Trends Over Time

**Goal:** read a time series without being fooled by it.

## What you will learn

- Resampling to the right grain
- Moving averages
- Growth: absolute, percentage and compounding
- The traps: incomplete periods, day-of-week, base effects

---

## Choose the grain

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet").set_index("ordered_at")

for label, rule in [("daily", "D"), ("weekly", "W"), ("monthly", "MS")]:
    series = orders["amount"].resample(rule).sum()
    print(f"{label:<9} {len(series):>3} points   "
          f"mean {series.mean():>9,.0f}   "
          f"relative spread {series.std() / series.mean():.1%}")
```

```text
daily     180 points   mean     2,941   relative spread 21.0%
weekly     27 points   mean    19,605   relative spread 21.4%
monthly     6 points   mean    88,224   relative spread 7.0%
```

The same data at three grains. **Daily is 21% noisy, monthly is 7%** — the
aggregation averages the noise away, and also averages away anything real that
happens within a month.

Weekly looks as noisy as daily here (21.4%) for a mundane reason: the first
and last weeks are partial, and with only 27 points two half-weeks distort the
spread. That is lesson enough on its own — **check the edges of every
resampled series before reading anything into it.**

| Grain | Use for |
|---|---|
| Daily | Operations, anomaly detection, short campaigns |
| Weekly | Most business reporting — it cancels day-of-week effects |
| Monthly | Executive reporting, seasonality, year-on-year |

**Weekly is the default for business trends.** It removes the day-of-week
cycle without hiding a two-week problem.

---

## Moving averages

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet").set_index("ordered_at")
daily = orders["amount"].resample("D").sum()

smoothed = pd.DataFrame({
    "daily": daily,
    "ma_7": daily.rolling(7).mean(),
    "ma_28": daily.rolling(28).mean(),
})
print(smoothed.tail(5).round(0).to_string())
print("\nrelative spread:")
print(f"  raw daily : {daily.std() / daily.mean():.1%}")
print(f"  7-day MA  : {smoothed['ma_7'].std() / smoothed['ma_7'].mean():.1%}")
print(f"  28-day MA : {smoothed['ma_28'].std() / smoothed['ma_28'].mean():.1%}")
```

```text
             daily    ma_7   ma_28
ordered_at                        
2026-06-25  2025.0  2799.0  2760.0
2026-06-26  3545.0  2923.0  2769.0
2026-06-27  3140.0  2964.0  2777.0
2026-06-28  3625.0  2961.0  2822.0
2026-06-29  1870.0  2926.0  2765.0

relative spread:
  raw daily : 21.0%
  7-day MA  : 6.5%
  28-day MA : 2.8%
```

Look at the raw column: 2,025 then 3,545 then 3,140 then 3,625 then 1,870.
Day to day, revenue appears to swing by 75%. The 7-day average over the same
days moves between 2,799 and 2,964 — **a 6% band.**

Across the series, smoothing cuts the variation from 21.0% to 6.5% (7-day) and
2.8% (28-day). Nothing about the business changed; only the noise was removed.

Two things to keep in mind:

- **A 7-day window is right for daily data with a weekly cycle** — it contains
  exactly one of each weekday.
- **Smoothing lags.** A 28-day average takes two weeks to reflect a change
  that started today. Use the shortest window that removes the noise you do
  not care about.

---

## Growth

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet").set_index("ordered_at")
monthly = orders["amount"].resample("MS").sum()

growth = pd.DataFrame({
    "revenue": monthly.round(0),
    "change": monthly.diff().round(0),
    "pct_change": (monthly.pct_change() * 100).round(1),
    "vs_first_month": ((monthly / monthly.iloc[0] - 1) * 100).round(1),
})
print(growth.to_string())

months = len(monthly) - 1
cagr = (monthly.iloc[-1] / monthly.iloc[0]) ** (1 / months) - 1
print(f"\naverage monthly growth (compound): {cagr:+.2%}")
print(f"naive average of pct_change:       {growth['pct_change'].mean():+.2f}%")
```

```text
            revenue   change  pct_change  vs_first_month
ordered_at                                              
2026-01-01  92200.0      NaN         NaN             0.0
2026-02-01  81750.0 -10450.0       -11.3           -11.3
2026-03-01  92415.0  10665.0        13.0             0.2
2026-04-01  86185.0  -6230.0        -6.7            -6.5
2026-05-01  95900.0   9715.0        11.3             4.0
2026-06-01  80895.0 -15005.0       -15.6           -12.3

average monthly growth (compound): -2.58%
naive average of pct_change:       -1.86%
```

Two different answers to "how fast are we growing": **−2.58% compound against
−1.86% naive.** The compound figure is the correct one — percentage changes
multiply, they do not add, and averaging them overstates growth whenever the
series is volatile.

And the more important observation: the monthly changes are **−11.3%, +13.0%,
−6.7%, +11.3%, −15.6%**. That is not a trend; that is noise around a flat
line, and the `vs_first_month` column confirms it — after six months the
cumulative change is −12.3%, most of which is June being incomplete.

Reporting "revenue fell 15.6% in June" would be technically true and
substantively false.

---

## The traps

### Incomplete periods

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet").set_index("ordered_at")
monthly = orders["amount"].resample("MS").sum()

last_month = monthly.index[-1]
days_in_month = last_month.days_in_month
days_with_data = orders.loc[str(last_month)[:7]].index.day.max()

print(f"June total:        {monthly.iloc[-1]:,.0f} EGP")
print(f"days with data:    {days_with_data} of {days_in_month}")
print(f"May total:         {monthly.iloc[-2]:,.0f} EGP")
print(f"naive comparison:  {monthly.iloc[-1] / monthly.iloc[-2] - 1:+.1%}")
projected = monthly.iloc[-1] / days_with_data * days_in_month
print(f"run-rate projection: {projected:,.0f} EGP ({projected / monthly.iloc[-2] - 1:+.1%})")
```

```text
June total:        80,895 EGP
days with data:    29 of 30
May total:         95,900 EGP
naive comparison:  -15.6%
run-rate projection: 83,684 EGP (-12.7%)
```

June is missing one day out of thirty, and the naive comparison attributes
that missing day to the business: **−15.6% instead of −12.7%.** Three
percentage points of "decline" that are pure calendar.

**Never compare a partial period with a complete one** without saying so.
Either exclude it or project it — and label which you did.

### Day of week

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet")
by_weekday = (orders.assign(weekday=orders["ordered_at"].dt.day_name())
                    .groupby("weekday")["amount"].agg(["sum", "count"]))
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
by_weekday = by_weekday.reindex(order)
by_weekday["share"] = (by_weekday["sum"] / by_weekday["sum"].sum() * 100).round(1)
print(by_weekday.to_string())
```

```text
               sum  count  share
weekday                         
Monday     83070.0    771   15.7
Tuesday    69125.0    651   13.1
Wednesday  72585.0    668   13.7
Thursday   75480.0    722   14.3
Friday     71950.0    698   13.6
Saturday   80370.0    747   15.2
Sunday     76765.0    728   14.5
```

Monday is 15.7% of revenue and Tuesday 13.1% — a **20% difference between two
adjacent weekdays**, in data generated with no weekly pattern at all. It is
noise, and it would be easy to build a story about it.

That is the warning. **Real data almost always has a genuine weekly pattern**,
and this table cannot tell you whether yours is real without the interval
methods of the Advanced track. What it can tell you is that comparing a Monday
with a Tuesday, or a 4-weekend month with a 5-weekend month, produces changes
that are entirely calendar.

Check this before any short-period comparison. When a pattern exists, compare
like days, or use weekly grain.

### Base effects

```python
values = [100, 50, 100]
for i in range(1, len(values)):
    print(f"{values[i-1]} -> {values[i]}: {(values[i] / values[i-1] - 1):+.0%}")
```

```text
100 -> 50: -50%
50 -> 100: +100%
```

The same absolute swing, twice, reported as −50% and +100%. **A percentage
change depends entirely on its base**, and a recovery from a bad month always
looks like spectacular growth. Report the absolute level beside every
percentage.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Comparing a partial period with a full one | An artificial collapse |
| Daily grain for a business trend | You report noise as change |
| Averaging percentage changes | Overstated growth |
| No moving average on a volatile series | Every wobble becomes a story |
| Ignoring day-of-week | Calendar artefacts read as trends |
| Percentages without the base | +100% off a collapsed month |

---

## Exercises

1. Resample your data to daily, weekly and monthly; compare the relative
   spreads.
2. Plot a 7-day and 28-day moving average; which reveals the trend?
3. Compute compound growth and the naive average; how far apart are they?
4. Check the day-of-week pattern in your data before your next comparison.
