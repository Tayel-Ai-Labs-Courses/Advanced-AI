# Lesson 01 — Sampling and Uncertainty

**Goal:** put an interval around every number you report.

## What you will learn

- Why a sample is not the population
- Standard error and confidence intervals
- The bootstrap
- Sample size, and what it buys

---

## Every number is an estimate

```python
import numpy as np

rng = np.random.default_rng(0)
population = rng.normal(106, 55, 100_000)          # the truth we never see

print(f"true mean: {population.mean():.2f}")
print(f"{'sample size':>12}{'sample mean':>14}{'error':>10}")
for size in [30, 100, 1_000, 10_000]:
    sample = rng.choice(population, size, replace=False)
    print(f"{size:>12}{sample.mean():>14.2f}{sample.mean() - population.mean():>+10.2f}")
```

```text
true mean: 105.95
 sample size   sample mean     error
          30        113.19     +7.24
         100        102.33     -3.62
        1000        105.30     -0.65
       10000        105.78     -0.17
```

Four samples from the same population, four different answers. At n = 30 the
estimate is off by 7.24 — **6.8% wrong**, from data that is perfectly clean and
correctly measured. At n = 10,000 the error is 0.17, a sixth of one per
cent.

The number you compute is never the truth. It is one draw from a distribution
of possible answers, and the width of that distribution is what you must
report.

---

## Standard error

```python
import numpy as np

rng = np.random.default_rng(1)
population = rng.normal(106, 55, 100_000)

print(f"{'n':>7}{'predicted SE':>15}{'observed spread':>18}")
for size in [30, 100, 1_000, 10_000]:
    means = [rng.choice(population, size, replace=False).mean() for _ in range(500)]
    predicted = population.std() / np.sqrt(size)
    print(f"{size:>7}{predicted:>15.2f}{np.std(means):>18.2f}")
```

```text
      n   predicted SE   observed spread
     30          10.01             10.30
    100           5.48              5.46
   1000           1.73              1.75
  10000           0.55              0.48
```

The formula `SE = σ / √n` predicts the observed spread almost exactly. Two
consequences:

- **Precision improves with the square root of n.** To halve the error you
  need **four times** the data, not twice.
- 10,000 observations give an SE of 0.55 on a mean of 106 — half a percent.
  Thirty give 10, which is nine per cent.

---

## Confidence intervals

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(2)
sample = rng.normal(106, 55, 200)

mean = sample.mean()
standard_error = sample.std(ddof=1) / np.sqrt(len(sample))
critical = stats.t.ppf(0.975, df=len(sample) - 1)
low, high = mean - critical * standard_error, mean + critical * standard_error

print(f"sample mean:     {mean:.2f}")
print(f"standard error:  {standard_error:.2f}")
print(f"95% interval:    [{low:.2f}, {high:.2f}]")
print(f"width:           {high - low:.2f}")
print(f"\nreport it as:    {mean:.0f} EGP (95% CI: {low:.0f}-{high:.0f})")
```

```text
sample mean:     105.57
standard error:  3.75
95% interval:    [98.17, 112.97]
width:           14.80

report it as:    106 EGP (95% CI: 98-113)
```

**What it means:** if you repeated this sampling many times, 95% of the
intervals so constructed would contain the true mean. It is a statement about
the procedure, not a 95% probability that the truth is in *this* interval.

**What it is for:** it turns "the average order is 105.57" into "the average
order is between 98 and 113" — and stops a colleague building a plan on the
third decimal place of a number that is ±7.

For a proportion:

```python
import numpy as np
from scipy import stats

conversions, visitors = 47, 1_000
rate = conversions / visitors
standard_error = np.sqrt(rate * (1 - rate) / visitors)
low, high = rate - 1.96 * standard_error, rate + 1.96 * standard_error

print(f"conversion rate: {rate:.2%}")
print(f"95% interval:    [{low:.2%}, {high:.2%}]")
print(f"absolute width:  {(high - low) * 100:.2f} percentage points")
print(f"relative width:  {(high - low) / rate:.0%} of the estimate")
```

```text
conversion rate: 4.70%
95% interval:    [3.39%, 6.01%]
absolute width:  2.62 percentage points
relative width:  56% of the estimate
```

A 4.7% conversion rate on a thousand visitors could be anything from 3.4% to
6.0%. **The interval is 56% as wide as the estimate itself** — which is why a
"conversion rate improved from 4.7% to 5.1%" headline on this much data means
nothing at all.

---

## The bootstrap

When the formula is unknown or the data is skewed, resample.

```python
import numpy as np

rng = np.random.default_rng(3)
sample = rng.gamma(3, 35, 300)                     # skewed, like order values

def bootstrap(data, statistic, iterations=5_000, seed=0):
    generator = np.random.default_rng(seed)
    values = [statistic(generator.choice(data, len(data), replace=True))
              for _ in range(iterations)]
    return np.percentile(values, [2.5, 97.5])

for name, statistic in [("mean", np.mean), ("median", np.median),
                        ("90th percentile", lambda d: np.percentile(d, 90)),
                        ("share above 150", lambda d: (d > 150).mean())]:
    low, high = bootstrap(sample, statistic)
    point = statistic(sample)
    print(f"{name:<18}{point:>9.2f}   95% CI [{low:.2f}, {high:.2f}]")
```

```text
mean                 114.74   95% CI [108.02, 121.68]
median               103.34   95% CI [95.20, 113.36]
90th percentile      195.62   95% CI [176.93, 217.18]
share above 150        0.23   95% CI [0.18, 0.28]
```

The bootstrap gives an interval for **any** statistic — the median, a
percentile, a share, a ratio of two medians — with no formula and no
distributional assumption. Resample with replacement, recompute, take the
percentiles.

It is the single most useful technique in this lesson, because real business
metrics are rarely means of normal data.

---

## Sample size

```python
import numpy as np

def required_n(margin_of_error, proportion=0.5, confidence=1.96):
    """n for a given margin of error on a proportion."""
    return int(np.ceil(confidence**2 * proportion * (1 - proportion)
                       / margin_of_error**2))

print(f"{'margin':>10}{'n needed':>12}")
for margin in [0.10, 0.05, 0.03, 0.01]:
    print(f"{margin:>10.0%}{required_n(margin):>12,}")

print()
for margin in [0.05, 0.03]:
    for p in [0.5, 0.1, 0.02]:
        print(f"margin {margin:.0%}, true rate {p:>4.0%}: n = {required_n(margin, p):>7,}")
```

```text
    margin    n needed
       10%          97
        5%         385
        3%       1,068
        1%       9,604

margin 5%, true rate  50%: n =     385
margin 5%, true rate  10%: n =     139
margin 5%, true rate   2%: n =      31
margin 3%, true rate  50%: n =   1,068
margin 3%, true rate  10%: n =     385
margin 3%, true rate   2%: n =      84
```

Going from a 10% margin to 1% needs **99 times the sample** (97 → 9,604). That
is the square-root law seen from the other side, and it is why "just collect
more data" stops being an answer.

Note the second block: rarer events need *fewer* observations for the same
**absolute** margin — but a 5-point margin on a 2% rate is useless. For rare
events, specify the margin **relative** to the rate.

---

## Reporting uncertainty

| Instead of | Write |
|---|---|
| "The average order is 105.57 EGP" | "The average order is 106 EGP (95% CI: 98–113)" |
| "Conversion improved to 5.1%" | "Conversion was 5.1% (95% CI: 4.2–6.0), against 4.7% (3.4–6.0) — the intervals overlap substantially" |
| "Branch A is best" | "Branch A leads by 4 points, but the intervals overlap; the ranking is not established" |
| "Revenue fell 1.3%" | "Revenue fell 1.3%, within the month-to-month variation of ±12%" |

**When two intervals overlap substantially, you cannot claim a difference.**
That one habit prevents most of the bad decisions made from dashboards.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Reporting a point estimate alone | False precision; decisions on noise |
| Interval from a biased sample | Precisely wrong |
| Treating CI as "95% probability the truth is here" | A subtly different claim |
| Quadrupling effort to halve the error unknowingly | Budget surprise |
| Normal-theory intervals on skewed data | Use the bootstrap |
| Comparing overlapping intervals as if different | The most common analytics error |

---

## Exercises

1. Compute a 95% interval for your main metric. How wide is it, relatively?
2. Bootstrap the median and the 90th percentile of the same metric.
3. Work out the sample size for a 2-point margin on your conversion rate.
4. Find a claim in a dashboard where the intervals would overlap.
