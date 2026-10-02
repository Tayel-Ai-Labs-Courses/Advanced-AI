# Project 19 — Find Something You Did Not Know

**Do this after the eight lessons.**

You take a model **you already have** — from any project in this track, or from
work — and apply all four ideas to it. The deliverable is a short report
containing **at least three things you did not know about your own model**.

No new modelling. This is a diagnostic project, and it is deliberately small:
one or two days.

---

## Requirements

### 1. Linear algebra: what is really in your features
- The **singular values** of your feature matrix, and where the cliff is
- The **effective rank** from the ratio test, and the variance it keeps
- A statement: how many independent ideas are in your N columns?
- The **condition number**, before and after scaling

### 2. Calculus: check the gradients
- If you wrote any custom loss or metric, **check its gradient numerically**
- If you did not, write one — a weighted loss reflecting your cost matrix from
  Data-Science lesson 06 — and check it
- Report the relative error, and whether it passed

### 3. Optimisation: measure the conditioning cost
- Train with and without feature scaling; **count epochs to the same loss**
- Find your learning rate's edge of stability by doubling until it diverges
- Report the ratio, and whether the result surprised you

### 4. Probability: the base rate calculation
- Your base rate, recall and false-positive rate
- **Compute precision from those three**, and compare with your measured
  precision. They should match; if not, find out why
- The specificity you would need for 90% precision at your base rate
- Whether that is achievable

### 5. Statistics: what your evaluation can and cannot see
- Your eval set's size, and the **smallest difference it can detect**
- A bootstrap interval on your headline metric
- **Go back to your last claimed improvement.** Does it exceed the detectable
  difference? Say so plainly either way
- Add the detectable difference to the eval set's README

### 6. Bias and variance: which failure do you have?
- Train and test error, side by side
- The **bias/variance diagnosis** from the gap
- The cure that follows, and the one that would not help
- Run it: apply the cure and report whether the gap closed

### 7. The report

```text
THE MODEL            what it is, what it decides
EFFECTIVE RANK       N columns, R independent directions, variance kept
CONDITION NUMBER     before and after scaling, and the epoch cost
GRADIENT CHECK       relative error, pass or fail
PRECISION            computed from base rate vs measured
DETECTABLE           the smallest difference your eval set can see
LAST CLAIM           did it exceed that? yes / no / unknowable
DIAGNOSIS            bias or variance, with the gap
THREE THINGS I DID NOT KNOW
  1.
  2.
  3.
WHAT I CHANGED
```

The three things are the point. If you cannot find three, you either picked a
model you already understood very well — in which case pick another — or you
skipped a section.

---

## Deliverables

```text
project-19/
├── REPORT.md           the one page, with the three findings
├── notebooks/
│   ├── 01-rank.ipynb       singular values, effective rank, conditioning
│   ├── 02-gradients.ipynb  the custom loss and its check
│   ├── 03-probability.ipynb base rate arithmetic
│   └── 04-estimation.ipynb detectable difference, bias/variance
└── changes/            whatever you actually changed as a result
```

---

## Marking

| Weight | Criterion |
|---|---|
| 15% | Effective rank measured, with the cliff identified |
| 15% | Condition number before/after, with the epoch cost measured |
| 15% | A gradient checked numerically, with its relative error |
| 15% | Precision computed from the base rate and reconciled with the measured value |
| 20% | The detectable difference, and an honest verdict on your last claim |
| 10% | Bias/variance diagnosis, with the cure applied and measured |
| 10% | Three genuine findings, and what you changed |

Automatic deductions: a rank claimed without singular values; a conditioning
claim without the epoch count; a gradient check with no relative error; a
detectable difference not added to the eval set's README; "three findings" that
are restatements of the lessons rather than facts about your model.

---

## Before you submit

- [ ] The singular values are plotted or printed, and the cliff is named
- [ ] The condition number is reported before **and** after scaling
- [ ] A gradient check ran, with its relative error
- [ ] Precision computed from the base rate matches your measured precision
- [ ] The detectable difference is in the eval set's README
- [ ] Your last claimed improvement has an honest verdict against it
- [ ] The bias/variance cure was applied, not just named
- [ ] The three findings are about **your model**, not about the lessons
