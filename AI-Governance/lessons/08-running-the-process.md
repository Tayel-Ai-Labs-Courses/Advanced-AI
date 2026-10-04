# Lesson 08 — Running the Process

**Goal:** turn governance from a launch gate into something that keeps working.

## What you will learn

- The review that happens before launch, and the one that happens after
- Change control: what triggers a re-review
- Incidents, and the clock you did not know was running
- Governance debt, and how to see it

---

## Two reviews

```text
BEFORE LAUNCH      is this allowed, and is it ready?
                   classification, records, fairness, oversight, disclosure
                   once, and it blocks

PERIODICALLY       is it still true?
                   quarterly for high tier, annually for limited
                   does not block, but produces findings
```

The second is the one every programme skips, and it is where the value is.
A system reviewed once at launch has a model card describing a model that no
longer exists, fairness numbers from an old distribution, and an owner who
changed teams.

**What the periodic review checks, in an hour:**

- [ ] Is the owner still here? ([lesson 01](01-governance-is-engineering.md))
- [ ] Is the risk tier still right? Read the "re-classify if" line ([lesson 02](02-classifying-risk.md))
- [ ] Does the model card describe the model in production? ([lesson 03](03-the-record.md))
- [ ] Are the subgroup numbers from this quarter? ([lesson 04](04-fairness-obligations.md))
- [ ] Override rate, override accuracy, time per review ([lesson 05](05-human-oversight.md))
- [ ] Explanation fidelity still measured? ([lesson 06](06-transparency.md))
- [ ] Did the deletion job run? ([lesson 07](07-data-obligations.md))
- [ ] Any incidents since the last review, and did they produce a change?

Eight questions. Most of them should be answerable from a dashboard rather than
a meeting, and **the ones that are not are your governance debt.**

---

## Change control

Governance artefacts go stale because changes do not trigger a re-review. Fix
that by naming the triggers, in the repository:

```yaml
# models/churn/RISK.md — re-review triggers
re_review_if:
  - the decision changes          # targeting -> eligibility is a tier change
  - a new feature is added that could proxy a protected attribute
  - the threshold moves           # lesson 04: it is a fairness decision
  - the training data source changes
  - the population changes materially   # a new country, a new segment
  - subgroup gap exceeds 2 points       # the CI gate fires (MLOps 04)
  - the override rate moves outside 2-20%
  - 6 months pass
```

Two properties make this work rather than decorate:

**Several triggers are automatic.** The subgroup gate is CI
([MLOps 04](../../MLOps/lessons/04-ci-gate.md)); the override rate is a
monitor with a threshold computed from `sqrt(p(1-p)/n)`
([MLOps 07](../../MLOps/lessons/07-monitoring.md)). A trigger that depends on
someone remembering is not a control.

**The first one is a tier change, not a tweak.** A model that moves from
*targeting* who gets an offer to *deciding* who is eligible has crossed from
limited to high tier, and that is a two-week obligation, not a config change.
It is also exactly the kind of change that happens through three reasonable
product decisions and one sprint.

---

## Incidents

An AI incident is not only an outage. It is any of:

```text
a decision that should not have been made, at scale
a subgroup disparity discovered after launch
personal data in an output that should not contain it
a model acting on poisoned or manipulated input
an explanation that was materially wrong
a deletion that did not happen
```

**Several regimes attach a clock to serious incidents** — notification to a
regulator within a small number of days, and to affected people where there is
risk to them. You cannot start a clock you do not know about, so the
engineering requirement is detection and a runbook, not a legal memo.

```text
INCIDENT RUNBOOK (one page, written before)

DETECT     which monitor fires? (MLOps 07)
CONTAIN    roll back the model, or switch to the baseline, in <5 min
           rolling back is not an admission; it buys time
ASSESS     how many people, over what period, with what effect
           this is a query, if lesson 03's logs exist. Otherwise it is a week
NOTIFY     legal decides; you provide the numbers within hours
REMEDY     re-decide the affected cases. This is usually the expensive part
LEARN      what monitor would have caught it an hour earlier? Build it
```

The **ASSESS** line is where governance pays for itself. "How many people were
affected, and over what period?" is a one-hour query if your decision log has
the model version and the subject id, and a multi-week archaeology project if
it does not. That single logging field, from
[lesson 03](03-the-record.md), is the difference.

And note **REMEDY**: the obligation is usually not just to stop the harm but to
*undo* it — re-decide the cases, refund, reinstate. Budget for it, because it
is larger than the engineering fix.

---

## Governance debt

Treat it like technical debt: visible, counted, and paid down deliberately.

```text
inventory.yaml  ->  a scorecard, one row per system

system          tier      card  dataset  subgroup  oversight  logs  owner
churn-risk      limited    yes      yes       yes        yes   yes   Nour
fraud-score     HIGH       yes       NO       yes        yes   yes   Omar
route-tickets   limited     NO       NO        NO        n/a   yes   ?
lead-scoring    HIGH        NO       NO        NO         NO    NO   ?
```

That table is the whole programme on one screen, and it prioritises itself:
**`lead-scoring` is a high-tier system with no records, no oversight, no logs
and no owner.** It is the only row that matters this quarter.

Three rules that keep it honest:

**Count, do not grade.** "7 of 12 systems have a current model card" is
actionable; "governance maturity: amber" is not.

**An empty cell is a finding, not a gap to be excused.** The absence of a
measurement is the thing an auditor writes down
([lesson 01](01-governance-is-engineering.md)).

**Put it in git and regenerate it from the inventory.** A scorecard maintained
by hand is wrong within a month.

---

## What good looks like

A team that has this working does not have a governance department. It has:

```text
inventory.yaml           in git, one row per production system
models/<name>/           CARD.md, DATASET.md, RISK.md, EVAL.md beside the code
CI                       subgroup gate, eval gate, link check
monitors                 override rate, parse/validity rate, subgroup gap,
                         with thresholds computed not guessed
jobs                     deletion, retention, scorecard regeneration
runbooks                 one page per alert, including the incident one
a quarterly hour         eight questions, per high-tier system
names                    on every system, every dataset, every runbook
```

Every line of that is an engineering artefact, which is where
[lesson 01](01-governance-is-engineering.md) started. **The governance
programme is not a parallel process; it is a handful of files, a few monitors,
two jobs and an hour a quarter** — and the organisations that fail at it are
almost never failing for lack of a policy document.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A launch review and nothing after | The card now describes a model that does not exist |
| Re-review triggers that rely on memory | Not a control |
| Treating a scope change as a config change | Targeting → eligibility is a tier change |
| No incident runbook | The clock starts before you know it has |
| Logs without model version or subject id | "How many were affected?" takes weeks |
| Forgetting remediation | Undoing the harm costs more than the fix |
| A maturity grade instead of a count | Not actionable |
| A hand-maintained scorecard | Wrong within a month |
| Governance as a separate team | It is files, monitors and jobs in your repository |

---

## Exercises

1. Build the scorecard from your inventory. Which row is the worst?
2. Write the re-review triggers for one system. How many are automatic?
3. Write the incident runbook for your highest-tier system.
4. Time the ASSESS query: how many people did system X affect last March?
5. Run the eight-question review on one system. How many could you answer
   from a dashboard?
6. Pick the single worst scorecard cell and fix it this week.

---

## Where to go next

| Next | Why |
|---|---|
| [Project 26](../Project-26/) | Govern one real system, end to end |
| [MLOps](../../MLOps/) | The gates, monitors and rollback this depends on |
| [Data-Security-for-AI](../../Data-Security-for-AI/) | What leaks, measured |
| [Communication 06](../../Communication-and-Documentation/lessons/06-artifacts.md) | The full model-card template |
| [Data-Science 10](../../Data-Science/lessons/10-limits-and-fairness.md) | Measuring fairness, as opposed to owing it |
