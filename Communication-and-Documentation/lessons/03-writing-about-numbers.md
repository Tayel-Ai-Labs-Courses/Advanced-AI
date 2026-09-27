# Lesson 03 — Writing About Numbers

**Goal:** report a number in a way that a reader cannot innocently
misunderstand — and that you cannot innocently overstate.

## What you will learn

- Relative and absolute change, and why you need both
- What your sample size permits you to claim
- The four words that ruin a number
- Comparisons that are honest

---

## Percentages without a base are not information

```python
import numpy as np

cases = [("conversion", 0.02, 0.03),
         ("churn", 0.160, 0.144),
         ("a rare failure", 0.0002, 0.0004)]
print(f"{'metric':<18}{'before':>10}{'after':>10}{'relative':>11}{'absolute':>11}")
for name, a, b in cases:
    print(f"{name:<18}{a:>10.4f}{b:>10.4f}{(b - a) / a:>+11.1%}{b - a:>+11.4f}")
print("\n'up 50%' and 'up 100%' describe the first and third rows.")
print("One is 1 extra customer in 100; the other is 2 in 10,000.")
```

```text
metric                before     after   relative   absolute
conversion            0.0200    0.0300     +50.0%    +0.0100
churn                 0.1600    0.1440     -10.0%    -0.0160
a rare failure        0.0002    0.0004    +100.0%    +0.0002

'up 50%' and 'up 100%' describe the first and third rows.
One is 1 extra customer in 100; the other is 2 in 10,000.
```

**"Conversion is up 50%" and "failures doubled" sound comparable.** One is one
extra customer per hundred; the other is two more per ten thousand.

The rule: **always give both the relative and the absolute change**, and the
base.

| Bad | Good |
|---|---|
| "Conversion improved 50%" | "Conversion rose from 2.0% to 3.0% — one extra customer per hundred" |
| "Errors doubled" | "Errors rose from 2 to 4 per 10,000 requests" |
| "We reduced churn by 10%" | "Churn fell from 16.0% to 14.4%, about 190 fewer customers a month" |
| "Accuracy improved by 3 points" | "Accuracy rose from 0.89 to 0.92 on 3,000 held-out examples" |

The last row adds the thing that decides whether the claim is real at all.

---

## What your sample size permits

```python
def half_width(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)
print(f"{'n':>8}{'estimate':>11}{'95% interval':>22}{'can you claim +2pts?':>23}")
for n in (50, 200, 1_000, 5_000, 20_000):
    h = half_width(0.30, n)
    claim = "yes" if h < 0.02 else "no"
    print(f"{n:>8}{0.30:>11.2f}{f'[{0.30-h:.3f}, {0.30+h:.3f}]':>22}{claim:>23}")
```

```text
       n   estimate          95% interval   can you claim +2pts?
      50       0.30        [0.173, 0.427]                     no
     200       0.30        [0.236, 0.364]                     no
    1000       0.30        [0.272, 0.328]                     no
    5000       0.30        [0.287, 0.313]                    yes
   20000       0.30        [0.294, 0.306]                    yes
```

At **n = 200**, a rate of 30% is known to +/-6.4 points. Anyone reporting "30%,
up from 28%" on 200 observations is reporting noise with a decimal point on it.

**You need about 5,000 observations to claim a 2-point difference**, and the
number surprises nearly everyone the first time they compute it.

So before writing a comparison, compute the interval and ask whether the
difference you want to report fits inside it. If it does, the honest sentence
is *"we could not detect a difference"* — which is a finding, and a much better
one than a number that will not reproduce.

**Every headline number gets an interval.** Six extra words, and it is the
difference between a claim and a measurement.

---

## The four words that ruin a number

| Word | Why it is a problem | Instead |
|---|---|---|
| **"significant"** | Means "p < 0.05" to one reader and "large" to another | Give the effect size and the interval |
| **"improved"** | Compared to what? Measured how? | "rose from X to Y on Z examples" |
| **"approximately"** | Usually hides that you did not compute the interval | The interval |
| **"could"** | "This could save 100,000 EGP" is unfalsifiable | "At a 30% acceptance rate this saves X; at 15%, Y" |

And one construction: **"up to"**. "Up to 40% faster" is technically true if one
case out of fifty was 40% faster. Report the median and the range.

---

## Honest comparisons

Five questions before any comparison ships:

```text
1. Same data?         both numbers on the same test set, same period, same filter
2. Same metric?       accuracy vs AUC vs precision@k are not interchangeable
3. Same budget?       equal compute, equal training data, equal steps
4. Baseline named?    "better than random" is not a baseline anyone uses
5. Interval computed? and does the difference exceed it
```

Failures of each, in the wild:

- **Same data**: "our model gets 0.92, the old one got 0.87" — measured six
  months apart on different populations.
- **Same metric**: "we improved from 0.68 AUC to 0.71 average precision".
- **Same budget**: "our method beats theirs", trained for ten times as long.
- **Baseline**: "94% accurate" on a problem with a 94% majority class.
- **Interval**: "0.78 to 0.85" on 100 examples, where the interval is +/-0.08.

The fourth is the most common and the most embarrassing, because a reader who
knows the base rate sees it immediately.

---

## Rounding and precision

| Rule | Example |
|---|---|
| Round to the precision you can defend | 0.6823 -> "0.68" unless the interval is narrower than 0.001 |
| Money: no decimals above 100 | "41,700 EGP", not "41,699.52 EGP" |
| Percentages: one decimal, at most | "16.1%", not "16.08%" |
| Counts: exact | "1,929 positives", never "about 1,900" |
| Keep precision **inside** the calculation | Round only at the end, for display |

False precision reads as carelessness to anyone who understands the numbers, and
as authority to anyone who does not. Both are bad.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Relative change alone | "+50%" was one customer in a hundred |
| No base for a percentage | The reader invents one |
| A difference smaller than the interval | It will not reproduce |
| "Significant" without a number | Two readers, two meanings |
| No baseline | 94% on a 94% majority class |
| Comparing metrics of different kinds | AUC against average precision |
| Four decimal places on an estimate +/-0.06 | False precision |
| "Up to X" | One lucky case, reported as the result |

---

## Exercises

1. Take three claims from a report you have written. Add the base and the
   absolute change to each.
2. Compute the interval for your headline metric. Does your last claimed
   improvement fit inside it?
3. Find a comparison in your work and check it against the five questions. How
   many does it pass?
4. Rewrite a paragraph of yours removing every use of "significant", "improved",
   "approximately" and "could".
5. Find the smallest difference your current eval set can detect and write it at
   the top of that eval set's README.

---

**Next:** [Lesson 04 — Charts That Carry the Finding](04-charts.md)
