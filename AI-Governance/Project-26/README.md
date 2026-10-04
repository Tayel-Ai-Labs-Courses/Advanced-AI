# Project 26 — Govern One System

Take one real system that makes decisions about people and produce everything
it owes. Not a policy document — the artefacts.

If you have no such system at work, govern one you built earlier in this
library: [Project 10](../../Data-Science/Project-10/)'s decision system,
[Project 24](../../Recommender-Systems/Project-24/)'s recommender, or
[Project 22](../../MLOps/Project-22/)'s model. Governing your own work is
harder and more useful than governing a hypothetical.

---

## What you deliver

```text
system/
  inventory.yaml       the row for this system, and every other one you found
  RISK.md              the seven questions, answered, with the tier
  CARD.md              including out-of-scope use and subgroup performance
  DATASET.md           source, purpose collected for vs used for, retention
  FAIRNESS.md          the definition you chose, the one you rejected, and why
  OVERSIGHT.md         the four numbers, measured
  TRANSPARENCY.md      the 30-word explanation, and its fidelity
  RUNBOOK.md           the incident runbook, one page
  SCORECARD.md         generated from the inventory
  REPORT.md            the numbers below
```

---

## The eight requirements

| # | Requirement | The check | Lesson |
|---|---|---|---|
| 1 | **An inventory** of every automated decision in your organisation | A count, and how many have a named owner | 01 |
| 2 | **Risk classified**, seven questions answered in writing | The tier, and a "re-classify if" line | 02 |
| 3 | **The audit question answered and timed** | "Why was X decided on date Y?" — report the minutes | 03 |
| 4 | **Four-fifths computed at 5 thresholds**, plus one other fairness definition | Does it pass? Does it matter? | 04 |
| 5 | **The four oversight numbers**, measured | Override rate, accuracy, time, outcome delta | 05 |
| 6 | **Explanation fidelity**, as a number | And whether it agrees with the model's own importance | 06 |
| 7 | **k computed** for your quasi-identifiers | What fraction of people are unique? | 07 |
| 8 | **A scorecard and a re-review trigger list** | Which triggers are automatic? | 08 |

Requirements 4, 5, 6 and 7 are the point. **Each is a dozen lines of code and
each can find a problem a checklist cannot.** If all four come back clean on a
real system, look harder — in particular at requirement 5's override accuracy,
which almost nobody can compute because overrides are not followed up.

---

## The report

`REPORT.md`, your numbers:

```text
1. INVENTORY     systems found, how many owned, how many high tier
2. CLASSIFICATION the tier and the seven answers; what would raise it
3. THE RECORD    how long the audit question took, and what was missing
4. FAIRNESS      four-fifths at 5 thresholds; your chosen definition and
                 the gap under it; the definition you rejected and why
5. OVERSIGHT     override rate, override accuracy, time per review,
                 outcome delta. Is your oversight a control?
6. TRANSPARENCY  explanation fidelity; top feature vs model importance;
                 the 30-word customer explanation
7. DATA          k for your quasi-identifiers, % unique, what generalising
                 two columns bought; whether the deletion job ran
8. GAPS          the scorecard, and the single worst cell
9. VERDICT       is this system fit to operate? What must change first?
```

Section 9 must allow "not fit to operate as it stands". That is a real finding
and the most valuable one this project can produce.

---

## Rules

- **Every number is from your own run or your own logs.** None copied.
- **An absence is a finding, and must be written as one.** "Override accuracy:
  cannot be computed, overrides are not followed up" is a correct answer and a
  work item.
- **No maturity grades.** Count things.
- **Do not fix things silently.** If you add a missing log field during the
  project, report the before and after.
- **Name a person for every system.** "The data team" is not an answer.

---

## Scoring yourself

| | |
|---|---|
| **Not done** | A set of documents that describe the system |
| **Done** | Eight requirements, nine report sections, numbers throughout |
| **Done well** | **One finding that a compliance checklist would have passed** — a four-fifths pass hiding rejected qualified people, an oversight step that is a rubber stamp, an explanation with low fidelity, or a dataset you were calling anonymous |

The third row is the course. Four of its eight lessons exist to show that the
standard test can pass while the harm continues, and finding one such case in
your own system is the whole point.

---

## Prerequisites

All eight [AI-Governance lessons](../lessons/), and a real system.
