# Lesson 02 — Hypothesis Testing

**Goal:** decide whether a difference is real, and say what that means.

## What you will learn

- The logic, in one paragraph
- Choosing the test
- p-values: what they are and are not
- Effect size, power, and multiple comparisons

---

## The logic

Assume there is **no difference** (the null hypothesis). Compute how surprising
your data would be if that were true. If it would be very surprising, stop
believing the assumption.

The p-value is that surprise: **the probability of seeing a difference at
least this large if the null were true.**

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

same = rng.normal(100, 20, 500), rng.normal(100, 20, 500)
different = rng.normal(100, 20, 500), rng.normal(106, 20, 500)

for label, (a, b) in [("truly identical", same), ("truly different", different)]:
    statistic, p_value = stats.ttest_ind(a, b)
    print(f"{label:<18} means {a.mean():6.2f} vs {b.mean():6.2f}   "
          f"t={statistic:+6.2f}   p={p_value:.4f}")
```

```text
truly identical    means  99.46 vs  98.62   t= +0.68   p=0.4942
truly different    means 101.01 vs 104.67   t= -2.83   p=0.0048
```

Two groups drawn from the same distribution differ by 0.84 and produce
p = 0.49 — entirely unsurprising. Two groups genuinely 6 apart differ by 3.66
in this sample and produce p = 0.0048.

Note the second line: the observed difference (3.66) is well below the true
difference (6), because 500 observations per group is not enough to pin it
down. The test detected *that* there is an effect; it did not measure it
precisely.

---

## Choosing the test

| Comparing | Data | Test |
|---|---|---|
| Two means | Independent groups | `ttest_ind` |
| Two means | Paired (before/after, same units) | `ttest_rel` |
| Two means | Skewed, or small n | `mannwhitneyu` |
| Two proportions | Counts | `proportions_ztest`, chi-square |
| Three or more means | Independent | `f_oneway` (ANOVA) |
| Two categorical variables | Contingency table | `chi2_contingency` |
| Distributions | Any | `ks_2samp` |

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)

control = rng.gamma(2, 50, 400)
variant = rng.gamma(2, 55, 400)

t_stat, t_p = stats.ttest_ind(control, variant)
u_stat, u_p = stats.mannwhitneyu(control, variant)

print(f"t-test:        p={t_p:.4f}   (assumes roughly normal)")
print(f"Mann-Whitney:  p={u_p:.4f}   (ranks, no distribution assumed)")
print(f"\nmeans:   {control.mean():.1f} vs {variant.mean():.1f}")
print(f"medians: {np.median(control):.1f} vs {np.median(variant):.1f}")

conversions = np.array([47, 62])
visitors = np.array([1_000, 1_000])
chi2, chi_p, _, _ = stats.chi2_contingency(
    [[conversions[0], visitors[0] - conversions[0]],
     [conversions[1], visitors[1] - conversions[1]]])
print(f"\nproportions 4.7% vs 6.2%: chi-square p={chi_p:.4f}")
```

```text
t-test:        p=0.0145   (assumes roughly normal)
Mann-Whitney:  p=0.0478   (ranks, no distribution assumed)

means:   94.9 vs 107.1
medians: 83.3 vs 89.0

proportions 4.7% vs 6.2%: chi-square p=0.1679
```

Two things worth noticing. The t-test gives p = 0.0145 and Mann-Whitney
p = 0.0478 — **one is comfortably significant and the other is barely so**, on
the same data. The t-test compares means, which the skew inflates; the
rank-based test compares typical values, and the medians differ by far less
than the means (83.3 vs 89.0 against 94.9 vs 107.1).

On skewed data, trust the rank-based test — or bootstrap the median, from
lesson 01.

And the proportions: **4.7% against 6.2% gives p = 0.168.** A 32% relative
improvement, on a thousand visitors per arm, is not distinguishable from
nothing. That is the reality of conversion testing, and it is why lesson 03 is
about sample size.

---

## What a p-value is not

| p = 0.03 does **not** mean | It means |
|---|---|
| "There is a 3% chance the null is true" | "Data this extreme occurs 3% of the time when the null is true" |
| "There is a 97% chance the effect is real" | Nothing about the probability of the hypothesis |
| "The effect is large" | Nothing about size — see below |
| "The effect matters" | Nothing about business value |

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(2)

for n in [100, 1_000, 100_000]:
    a = rng.normal(100.0, 20, n)
    b = rng.normal(100.4, 20, n)                  # a 0.4% difference
    _, p_value = stats.ttest_ind(a, b)
    difference = b.mean() - a.mean()
    cohens_d = difference / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    print(f"n={n:>7}  difference {difference:+6.3f}  "
          f"Cohen's d {cohens_d:+.3f}  p={p_value:.4f}"
          f"{'  SIGNIFICANT' if p_value < 0.05 else ''}")
```

```text
n=    100  difference +0.426  Cohen's d +0.022  p=0.8765
n=   1000  difference +0.467  Cohen's d +0.023  p=0.6046
n= 100000  difference +0.336  Cohen's d +0.017  p=0.0002  SIGNIFICANT
```

The same tiny true difference (0.4 on a mean of 100) gives p = 0.88 at
n = 100 and **p = 0.0002 at n = 100,000** — with a Cohen's d of 0.017, which
is nothing at all. The estimated difference barely changes across the three
rows; only the p-value does.

**With enough data, everything is significant.** That is why every p-value
must be reported beside an effect size:

| Cohen's d | Size |
|---|---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |
| **0.022** | **Irrelevant, however small the p-value** |

---

## Power

```python
import numpy as np
from scipy import stats

def power_by_simulation(n, true_effect, sigma=20, alpha=0.05, trials=2_000, seed=0):
    rng = np.random.default_rng(seed)
    detected = 0
    for _ in range(trials):
        a = rng.normal(100, sigma, n)
        b = rng.normal(100 + true_effect, sigma, n)
        if stats.ttest_ind(a, b).pvalue < alpha:
            detected += 1
    return detected / trials

print(f"{'n per group':>12}{'effect':>9}{'power':>9}")
for n in [50, 200, 800]:
    for effect in [2, 5]:
        print(f"{n:>12}{effect:>9}{power_by_simulation(n, effect):>9.2f}")
```

```text
 n per group   effect    power
          50        2     0.08
          50        5     0.23
         200        2     0.17
         200        5     0.72
         800        2     0.52
         800        5     1.00
```

**Power is the probability of detecting an effect that is really there.** With
50 per group and a true effect of 2, you detect it **8%** of the time — so
twelve times out of thirteen you would conclude "no difference" about a real
one.

The convention is 80% power. Note the middle rows: 200 per group gives 72%
power for an effect of 5 and **17%** for an effect of 2. Power depends on the
effect size you care about, which is why you must state it before running the
test.

A "no significant difference" result from an underpowered test says **nothing**
— not that there is no effect. Report the power, or the interval.

---

## Multiple comparisons

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(4)

def run_experiments(k, n=200):
    """k comparisons between identical groups: any 'significant' result is false."""
    p_values = []
    for _ in range(k):
        a = rng.normal(100, 20, n)
        b = rng.normal(100, 20, n)              # identical
        p_values.append(stats.ttest_ind(a, b).pvalue)
    return np.array(p_values)

for k in [1, 5, 20, 100]:
    p_values = run_experiments(k)
    false_positives = int((p_values < 0.05).sum())
    print(f"{k:>4} comparisons on identical data: "
          f"{false_positives} 'significant' "
          f"(expected {k * 0.05:.1f}), min p = {p_values.min():.4f}")
```

```text
   1 comparisons on identical data: 0 'significant' (expected 0.1), min p = 0.7859
   5 comparisons on identical data: 0 'significant' (expected 0.2), min p = 0.1977
  20 comparisons on identical data: 2 'significant' (expected 1.0), min p = 0.0065
 100 comparisons on identical data: 3 'significant' (expected 5.0), min p = 0.0347
```

Twenty comparisons on **identical** data produced two "significant" results,
one with p = 0.0065 — a number that looks compelling in a report and is pure
noise. A hundred comparisons produced three.

This is what happens when you slice a dashboard by branch × product ×
weekday × segment and report the one cell that stands out.

The corrections:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(5)
p_values = np.sort([stats.ttest_ind(rng.normal(100, 20, 200),
                                    rng.normal(100, 20, 200)).pvalue
                    for _ in range(20)])

bonferroni = 0.05 / len(p_values)
ranks = np.arange(1, len(p_values) + 1)
benjamini_hochberg = 0.05 * ranks / len(p_values)

print(f"uncorrected 0.05:        {(p_values < 0.05).sum()} significant")
print(f"Bonferroni {bonferroni:.4f}:      {(p_values < bonferroni).sum()} significant")
print(f"Benjamini-Hochberg:      {(p_values < benjamini_hochberg).sum()} significant")
```

```text
uncorrected 0.05:        1 significant
Bonferroni 0.0025:      0 significant
Benjamini-Hochberg:      0 significant
```

| Correction | Behaviour |
|---|---|
| **Bonferroni** (α/k) | Strict; use for a few pre-registered tests |
| **Benjamini-Hochberg** | Controls the false-discovery rate; use for many exploratory tests |
| **Pre-registration** | Decide the comparisons before looking. The real fix |

---

## Reporting a test

```text
Weak:    "Variant B performed better (p < 0.05)."

Strong:  "Variant B converted at 6.2% against 4.7% (+1.5 points, 95% CI:
          -0.5 to +3.5). chi-square p = 0.168, n = 1,000 per arm. The test
          was powered to detect a 2-point difference at 80%; this result does
          not establish an effect, and a longer run is needed."
```

Six elements: both rates, the difference, its interval, the test and p-value,
the sample size, and **what it does and does not establish**.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| p < 0.05 as proof | It is evidence, with a 5% false-positive rate by design |
| No effect size | Significant and irrelevant |
| No power calculation | "No difference" from a test that could not find one |
| Testing many slices, reporting one | Guaranteed false positives |
| Stopping when it turns significant | Inflates the error rate badly |
| t-test on heavily skewed data | Use Mann-Whitney or the bootstrap |

---

## Exercises

1. Run a t-test on two groups from your data; report p **and** Cohen's d.
2. Simulate power for the effect size you care about; how large must n be?
3. Run 20 tests on identical data and count the false positives.
4. Rewrite a past "significant" claim using the six-element format above.
