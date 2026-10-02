# Lesson 06 — Probability and Base Rates

**Goal:** internalise the one calculation that explains most bad decisions made
with models.

## What you will learn

- Bayes' rule, as arithmetic you can do in your head
- Why a 99% accurate test is usually wrong
- The same calculation, as precision
- Independence, and when it is a lie

---

## The test that is 99% accurate and usually wrong

```python
import numpy as np

def posterior(prevalence, sensitivity, specificity):
    tp = prevalence * sensitivity
    fp = (1 - prevalence) * (1 - specificity)
    return tp / (tp + fp)
print(f"{'prevalence':>12}{'sensitivity':>13}{'specificity':>13}{'P(sick | positive)':>21}")
for prev in (0.5, 0.1, 0.01, 0.001):
    print(f"{prev:>12.3f}{0.99:>13.2f}{0.99:>13.2f}{posterior(prev, 0.99, 0.99):>21.3f}")
print("\na 99%/99% test on a 1-in-1000 condition: a positive result is")
print(f"{posterior(0.001, 0.99, 0.99):.1%} likely to be real. The base rate dominates.")
```

```text
  prevalence  sensitivity  specificity   P(sick | positive)
       0.500         0.99         0.99                0.990
       0.100         0.99         0.99                0.917
       0.010         0.99         0.99                0.500
       0.001         0.99         0.99                0.090

a 99%/99% test on a 1-in-1000 condition: a positive result is
9.0% likely to be real. The base rate dominates.
```

The test never changed. **Only the base rate did**, and the answer went from
99% to 9%.

The arithmetic, on 100,000 people with a 1-in-1000 condition:

```text
100 are sick     -> 99 test positive     (true positives)
99,900 are well  -> 999 test positive    (false positives, 1% of 99,900)
                    ----
                    1,098 positives, of which 99 are real  =  9.0%
```

**There are simply more well people.** A 1% error rate on a huge group swamps a
99% hit rate on a tiny one, and no improvement in the test fixes it — you would
need 99.9% specificity to reach even 50%.

---

## The same calculation, wearing a different name

```python
base = 0.1608                     # Data-Science lesson 01
for recall, fpr in ((0.99, 0.01), (0.80, 0.10), (0.50, 0.05)):
    prec = base * recall / (base * recall + (1 - base) * fpr)
    print(f"recall {recall:.2f}, false-positive rate {fpr:.2f} -> precision {prec:.3f}")
print("this is why Data-Science lesson 06 reports precision, not accuracy")
```

```text
recall 0.99, false-positive rate 0.01 -> precision 0.950
recall 0.80, false-positive rate 0.10 -> precision 0.605
recall 0.50, false-positive rate 0.05 -> precision 0.657
this is why Data-Science lesson 06 reports precision, not accuracy
```

Sensitivity is **recall**. `1 - specificity` is the **false-positive rate**.
`P(sick | positive)` is **precision**. It is one formula with three vocabularies,
and every one of them appears somewhere in this track.

At a 16% base rate, a model with 80% recall and a 10% false-positive rate has
**60.5% precision** — four calls in ten are to people who were never going to
leave. That is not a bad model; that is the arithmetic, and it is why
[Data-Science lesson 06](../../Data-Science/lessons/06-evaluating-the-decision.md)
spends a whole lesson on thresholds instead of on accuracy.

Note also row three: **lower recall with a lower false-positive rate gives
higher precision** (0.657 against 0.605). Precision and recall trade, and which
you want is decided by your cost matrix, never by the model.

---

## Independence, and when it is a lie

```python
p = 0.01
print(f"one failure: {p:.3f}")
print(f"three INDEPENDENT failures at once: {p**3:.2e}")
print(f"three failures with correlation 1: {p:.3f}")
print(f"\nassuming independence made a 1-in-100 risk look like 1-in-1,000,000")
```

```text
one failure: 0.010
three INDEPENDENT failures at once: 1.00e-06
three failures with correlation 1: 0.010

assuming independence made a 1-in-100 risk look like 1-in-1,000,000
```

Multiplying probabilities requires independence, and in real systems it usually
does not hold:

| Assumed independent | Actually correlated by |
|---|---|
| Three servers failing | The same power supply, the same deploy |
| Three models disagreeing | The same training data |
| Three features | The same underlying cause |
| Three retries succeeding | The same broken dependency |
| Loan defaults across a portfolio | The same economy |

**This is why an ensemble of three models trained on the same data is not three
independent opinions** ([AI-Agents lesson 07](../../AI-Agents/lessons/07-multi-agent.md)
measured the same thing for agents), and why a redundancy argument needs the
correlation stated, not assumed.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Quoting accuracy on a rare event | 99%/99% gives 9% precision at 1-in-1000 |
| Forgetting the base rate | It dominates everything else |
| Treating precision and recall as both improvable | They trade; the cost matrix decides |
| Multiplying probabilities without independence | 1-in-100 reported as 1-in-a-million |
| An ensemble on one dataset as "independent" | Correlated errors |
| Confusing P(positive given sick) with P(sick given positive) | The entire lesson |

---

## Exercises

1. Compute the precision of your own model from its recall, false-positive rate
   and base rate. Does it match what you measured?
2. Find the specificity needed to reach 50% precision at your base rate.
3. Take a "three independent checks" claim in a system you use and name the
   thing that correlates them.
4. Do the 100,000-people arithmetic by hand for a condition and test of your
   choosing.
5. Explain to a colleague, without notation, why a very accurate test for a rare
   thing is mostly wrong.

---

**Next:** [Lesson 07 — Sampling and the CLT](07-sampling.md)
