# Lesson 03 — What the Numbers Are Not Telling You

**Goal:** see, by simulation, how a paper can report a real improvement that
does not exist.

## What you will learn

- The inflation from reporting the best run
- How often an identical method "wins"
- Asymmetric tuning, the most common invisible flaw
- Contamination, and why it is hard to see

**Every simulation below compares a method against itself.** Every reported
"improvement" in this lesson is exactly zero in truth.

---

## Reporting the best run

```python
import numpy as np

rng = np.random.default_rng(0)
TRUE, SD = 0.820, 0.012          # a method whose true score is 0.820
print(f"true score {TRUE:.3f}, run-to-run sd {SD:.3f}")
print(f"{'runs tried':>11}{'reported (best)':>17}{'inflation':>11}{'reproduces?':>13}")
for n in (1, 3, 5, 10, 20, 50):
    best = np.array([rng.normal(TRUE, SD, n).max() for _ in range(5000)]).mean()
    print(f"{n:>11}{best:>17.4f}{best - TRUE:>+11.4f}"
          f"{'no' if best - TRUE > SD else 'maybe':>13}")
print("\nreporting the maximum of N identical runs is not a measurement")
```

```text
true score 0.820, run-to-run sd 0.012
 runs tried  reported (best)  inflation  reproduces?
          1           0.8199    -0.0001        maybe
          3           0.8302    +0.0102        maybe
          5           0.8339    +0.0139           no
         10           0.8384    +0.0184           no
         20           0.8425    +0.0225           no
         50           0.8470    +0.0270           no

reporting the maximum of N identical runs is not a measurement
```

**Twenty runs, report the best, and you have manufactured 2.25 points.** No
misconduct required — the researcher ran the thing twenty times, which is
normal, and reported the number they were proudest of, which is human.

This is RL lesson 09's finding in publication form: identical code across 20
seeds spanned 37 to 346. A paper that does not say how many runs it did, and
report the median and the spread, has not told you what it measured.

**What to look for:** "we report the best of N", "our best configuration", or —
most often — no statement at all about the number of runs.

---

## An identical method that wins nine times in ten

Now the version that does not even require re-running: try several variants of
your own idea, and one baseline.

```python
print("a paper tries K variants of its own method, and one baseline, once each")
print(f"{'variants tried':>15}{'P(method wins)':>17}{'when both are IDENTICAL':>26}")
for k in (1, 3, 5, 10, 20):
    wins = 0
    for _ in range(20000):
        method = rng.normal(TRUE, SD, k).max()
        baseline = rng.normal(TRUE, SD, 1)[0]
        wins += method > baseline
    print(f"{k:>15}{wins/20000:>17.1%}{'':>26}")
print("\nboth method and baseline draw from the SAME distribution")
```

```text
a paper tries K variants of its own method, and one baseline, once each
 variants tried   P(method wins)   when both are IDENTICAL
              1            50.4%
              3            74.9%
              5            83.3%
             10            90.8%
             20            95.1%

both method and baseline draw from the SAME distribution
```

**With ten variants, an identical method beats the baseline 90.8% of the
time.** With twenty, 95.1%.

Nothing here is dishonest in the ordinary sense. The researcher tried ten
reasonable variations — a different activation, a different schedule, a
different head — kept the one that worked, and compared it with the baseline.
That is what research looks like from the inside, and it produces a "win" from
pure noise nine times out of ten.

The fix is not to stop exploring. It is to **re-measure the chosen variant on
fresh seeds, or on a held-out split that played no part in choosing it** —
which is Data-Science lesson 03's dev/test discipline, applied to papers.

---

## Asymmetric tuning

The subtlest version, and the one that survives peer review most often.

```python
def tuned_score(true_mean, sd, n_configs, rng):
    return rng.normal(true_mean, sd, n_configs).max()
print(f"{'method tuning':>15}{'baseline tuning':>18}{'reported gap':>15}{'true gap':>11}")
for m_cfg, b_cfg in [(20, 1), (20, 5), (20, 20), (1, 1)]:
    gaps = [tuned_score(TRUE, SD, m_cfg, rng) - tuned_score(TRUE, SD, b_cfg, rng)
            for _ in range(5000)]
    print(f"{m_cfg:>15}{b_cfg:>18}{np.mean(gaps):>+15.4f}{0.0:>+11.4f}")
print("\nboth methods are identical in every row. Only the search budget differs.")
```

```text
  method tuning   baseline tuning   reported gap   true gap
             20                 1        +0.0226    +0.0000
             20                 5        +0.0087    +0.0000
             20                20        -0.0001    +0.0000
              1                 1        +0.0006    +0.0000

both methods are identical in every row. Only the search budget differs.
```

**Tuning your method over 20 configurations and the baseline over 1 produces a
+0.0226 improvement between two identical methods.** Tune both twenty times and
the gap vanishes: -0.0001.

This is the single most common invisible flaw in applied ML papers, and it is
almost never deliberate. The authors spent six months on their method and an
afternoon on the baseline, because the baseline was somebody else's work and
they used its defaults.

**What to look for, in this order:**

1. Does the paper state the search budget **for the baseline**?
2. Was the baseline tuned by the authors, or copied from an older paper with a
   different setup?
3. Is the baseline's reported number the same as in its original paper? (If it
   is *lower*, ask why.)
4. Does the improvement exceed the seed variance the paper reports? (If the
   paper reports no variance, the answer is "unknowable".)

Question 4 is enough on its own to disqualify most claimed improvements under
one point.

---

## Contamination

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=3000, n_features=20, n_informative=6,
                           class_sep=0.6, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
print(f"{'test rows leaked into training':>32}{'reported accuracy':>20}")
for frac in (0.0, 0.01, 0.05, 0.20, 1.0):
    n = int(frac * len(X_te))
    Xa = np.vstack([X_tr, X_te[:n]]) if n else X_tr
    ya = np.concatenate([y_tr, y_te[:n]]) if n else y_tr
    m = HistGradientBoostingClassifier(random_state=0).fit(Xa, ya)
    print(f"{frac:>31.0%}{accuracy_score(y_te, m.predict(X_te)):>20.4f}")
print("\nhonest accuracy is the first row")
```

```text
  test rows leaked into training   reported accuracy
                             0%              0.8778
                             1%              0.8756
                             5%              0.8778
                            20%              0.9056
                           100%              1.0000

honest accuracy is the first row
```

Note how this one hides. **At 1% and 5% leakage the score is indistinguishable
from honest** — 0.8756 and 0.8778 against a true 0.8778. Only at 20% does it
become visibly inflated.

So small contamination is invisible in the number and still wrong. You cannot
detect it by looking at results; you detect it by reading the data pipeline.

For modern LLM papers this is the dominant concern: a model trained on a web
crawl has very likely seen the benchmark. The questions to ask:

- Was the benchmark **published after** the training data cutoff?
- Did the authors run a contamination check (n-gram overlap against the training
  corpus)?
- Is the improvement concentrated on benchmarks that are old and widely mirrored?
- Does it hold on a **freshly constructed** test set?

---

## The checklist

Before believing any reported improvement:

```text
[ ] How many runs? Is the median and spread reported?
[ ] How many variants of the method were tried?
[ ] What was the search budget for the BASELINE?
[ ] Does the improvement exceed the run-to-run variance?
[ ] Was the selection made on a split separate from the reported one?
[ ] Could the test data have been seen during training?
[ ] Is the baseline number the same as in its own paper?
[ ] Is the comparison at equal compute / data / parameters?
```

A paper that answers all eight is rare and trustworthy. A paper that answers
none is not necessarily wrong — it is **unevaluable**, which is a different
problem and the more common one.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Believing a sub-1-point improvement with no variance reported | Best-of-20 manufactures 2.25 points |
| Ignoring how many variants were tried | Ten variants beat an identical baseline 90.8% of the time |
| Not asking about the baseline's tuning budget | +0.0226 between identical methods |
| Looking for contamination in the results | 5% leakage was invisible |
| Comparing against a baseline number copied from an older paper | Different setup, different everything |
| Treating "state of the art" as a fact | It is the best number reported, under all of the above |
| Applying this scepticism only to others | You will do every one of these by accident |

---

## Exercises

1. Take a recent paper in your area and answer the eight checklist questions.
   How many could you answer?
2. Re-run the best-of-N simulation with your own model's measured seed variance.
   How many runs would manufacture a 1-point "improvement"?
3. Find a paper that reports a baseline number lower than the baseline's own
   paper. What explains the difference?
4. Take your own last result and ask the eight questions of it honestly.
5. Design the contamination check you would run before trusting a benchmark
   score on a model trained on web data.

---

**Next:** [Lesson 04 — Reproducing a Result](04-reproducing.md)
