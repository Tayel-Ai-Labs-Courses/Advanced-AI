# Lesson 07 — Sampling and the Central Limit Theorem

**Goal:** know why averages are trustworthy, when they are not, and how much
data you need.

## What you will learn

- The CLT, measured on four different distributions
- Where it fails
- Standard error, and the four-times rule
- What this means for every eval set in this track

---

## Averages become normal, data does not

```python
import numpy as np
from scipy import stats as st
rng = np.random.default_rng(0)

SOURCES = {
    "uniform":     lambda n: rng.uniform(0, 1, n),
    "exponential": lambda n: rng.exponential(1, n),
    "bernoulli":   lambda n: (rng.random(n) < 0.16).astype(float),
    "heavy-tailed": lambda n: rng.pareto(1.5, n),
}
from scipy import stats as st
print(f"{'source':<16}{'skew of 1 draw':>16}{'skew of mean(n=30)':>21}{'mean(n=500)':>14}")
for name, draw in SOURCES.items():
    raw = draw(20_000)
    m30 = np.array([draw(30).mean() for _ in range(5_000)])
    m500 = np.array([draw(500).mean() for _ in range(2_000)])
    print(f"{name:<16}{st.skew(raw):>16.2f}{st.skew(m30):>21.2f}{st.skew(m500):>14.2f}")
print("\nskew 0 = symmetric. Averages become normal; the raw data does not.")
print("the heavy-tailed row is the warning: it converges much more slowly.")
```

```text
source            skew of 1 draw   skew of mean(n=30)   mean(n=500)
uniform                    -0.02                 0.02          0.07
exponential                 1.99                 0.36          0.11
bernoulli                   1.85                 0.33          0.15
heavy-tailed              115.90                15.51         33.73

skew 0 = symmetric. Averages become normal; the raw data does not.
the heavy-tailed row is the warning: it converges much more slowly.
```

Rows one to three are the central limit theorem: a strongly skewed exponential
(skew 1.99) produces sample means that are nearly symmetric by n=30 (0.36) and
effectively normal by n=500 (0.11).

**This is why confidence intervals work on data that looks nothing like a bell
curve.** Your revenue distribution is skewed; the *mean* of 500 of them is not.

**Row four is the warning.** A heavy-tailed (Pareto) source still has skew 33.7
at n=500. When a few observations dominate the sum, the CLT arrives very slowly
or not at all.

You have met this already:
[Data-Analysis lesson 03](../../Data-Analysis/Basic/lessons/03-cleaning.md)
found **1.1% of rows carrying 58% of reported revenue.** That is a heavy tail,
and it means the sample mean of revenue is an unreliable statistic no matter how
many rows you have. Use the median, or a trimmed mean, and say which.

---

## How much data do you need?

```python
TRUE = 0.30
print(f"{'n':>7}{'std error':>12}{'95% interval':>22}{'width':>8}")
for n in (10, 30, 100, 1000, 10000):
    se = np.sqrt(TRUE * (1 - TRUE) / n)
    print(f"{n:>7}{se:>12.4f}{f'[{TRUE-1.96*se:.3f}, {TRUE+1.96*se:.3f}]':>22}"
          f"{2*1.96*se:>8.3f}")
print("\nto halve the interval you need FOUR times the data")
```

```text
      n   std error          95% interval   width
     10      0.1449        [0.016, 0.584]   0.568
     30      0.0837        [0.136, 0.464]   0.328
    100      0.0458        [0.210, 0.390]   0.180
   1000      0.0145        [0.272, 0.328]   0.057
  10000      0.0046        [0.291, 0.309]   0.018

to halve the interval you need FOUR times the data
```

A true rate of 0.30, measured on **10 observations, could be reported as
anything from 0.016 to 0.584.**

The rule underneath is `1/sqrt(n)`: **to halve the interval you need four times
the data.** Going from 100 to 1,000 observations narrows the interval from 0.18
to 0.057 — a factor of 3.2 for 10x the data.

This single table explains numbers stated all over this track:

| Claim | Lesson |
|---|---|
| "You need ~5,000 observations to claim a 2-point difference" | [Communication 03](../../Communication-and-Documentation/lessons/03-writing-about-numbers.md) |
| "An 8-question eval set gives [0.50, 1.00]" | [LLM 07](../../LLM-and-GenAI/lessons/07-evaluation.md) |
| "On 20 examples every prompt difference is noise" | [LLM 04](../../LLM-and-GenAI/lessons/04-prompting.md) |
| "1,929 positives is the real budget" | [Data-Science 03](../../Data-Science/lessons/03-the-data-you-have.md) |

They are all this formula.

---

## What this means for an eval set

```python
def detectable(n, p=0.5):
    """Smallest difference detectable between two groups of size n."""
    return 2 * 1.96 * np.sqrt(2 * p * (1 - p) / n)

print(f"{'eval size':>11}{'smallest detectable difference':>32}")
for n in (20, 50, 100, 300, 1000, 5000):
    print(f"{n:>11}{detectable(n):>32.3f}")
```

```text
  eval size  smallest detectable difference
         20                           0.620
         50                           0.392
        100                           0.277
        300                           0.160
       1000                           0.088
       5000                           0.039
```

**On 100 examples you cannot detect an improvement smaller than 27 points.**
That number surprises everyone, and it is the honest answer to "we improved the
prompt from 0.78 to 0.85".

Put this number in your eval set's README
([Communication lesson 06](../../Communication-and-Documentation/lessons/06-artifacts.md)),
so nobody reports a difference the set cannot see.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Assuming the CLT always applies | Heavy tails still had skew 33.7 at n=500 |
| The mean of a heavy-tailed quantity | 1.1% of rows carried 58% of revenue |
| Doubling the data to halve the interval | You need **four** times |
| Reporting a difference smaller than the interval | It will not reproduce |
| Not stating the detectable difference of an eval set | Everyone over-reads it |
| A 20-example eval set | It cannot see anything under 62 points |

---

## Exercises

1. Compute the smallest difference your own eval set can detect, and add it to
   its README.
2. Plot the skew of the sample mean against n for your most skewed real column.
   Where does it become symmetric?
3. Find a metric in your dashboards that is a mean of a heavy-tailed quantity.
   What would the median say instead?
4. Compute how many observations you need to detect a 1-point improvement.
5. Bootstrap an interval for a statistic you report weekly, and compare it with
   the week-to-week movement everyone discusses.

---

**Next:** [Lesson 08 — Bias, Variance and Estimation](08-estimation.md)
