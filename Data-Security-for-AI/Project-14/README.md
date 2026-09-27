# Project 14 — Attack Your Own System, Then Fix It

**Do this after the eight lessons.**

Take a model **you** built — ideally the one from Project 10 or Project 12 —
attack it with every technique in this course, measure what you find, fix the
worst of it, and prove the fix worked.

The deliverable is a **security review with before-and-after numbers**.

> Run these attacks only against systems you own or have written permission to
> test. Attacking someone else's model is not a project; it is an offence.

---

## Requirements

### 1. The threat model

- The one-page threat model from lesson 01: six surfaces, four lines each
- The ten questions, answered
- The actors, with their **actual access** in your system today
- What one successful attack would cost, in money or harm

### 2. Data: measure re-identification

- List the quasi-identifiers in your training data and any export
- **Measure k**, and the percentage of rows in groups of one
- Apply generalisation until k >= 10, and **state what analysis that cost**
- List the external datasets someone could join against
- Check free-text fields for identifiers and report the count

### 3. Membership inference

- Implement the attack from lesson 03 against your model
- Report the **attack AUC beside the train/test gap**
- Reduce it by regularisation alone, and report the cost in test accuracy
- Then reduce it by returning buckets instead of probabilities, and report the
  attack AUC again
- State the AUC you are accepting, and who accepted it

### 4. Poisoning and backdoors

- Identify **who can write into your training data**, directly or indirectly
- Install a backdoor in a copy of your model at 0.5%, 1% and 2% poisoning
- Report attack success and the clean-accuracy cost at each level
- Try to detect your own backdoor with the sweep from lesson 04. Did you find it?
- Implement range/schema validation on training rows and report what it rejects
  on real data

### 5. Extraction

- Extract your own model with random queries at 4 budget levels
- Report agreement and the clone's accuracy at each
- Repeat with **labels only** and with **bucketed output**; report the difference
- State the rate limit and per-caller budget you will enforce

### 6. Adversarial inputs

- State, with evidence, whether your system is under adversarial pressure
- If it is: run FGSM, report accuracy against perturbation size, and split the
  features into attacker-controlled and not
- If it is not: say so, explain why in three sentences, and skip the rest of
  this section. **That is the right answer for most systems**

### 7. Privacy techniques

- Data minimisation first: list every column, and how many you could drop
- If you publish statistics, implement DP with a **tracked budget** and report
  the utility cost
- The retention line for every dataset: how long, who deletes, how verified

### 8. Supply chain

- Every weight file, dataset and package: source, pinned version, hash verified
- Every `pickle.load` / `joblib.load`, and where the file comes from
- What you found, and what you changed

### 9. Fix and prove

Pick the **three worst findings**. For each:

```text
FINDING      what, with the number
IMPACT       what an attacker gets
FIX          what you changed
AFTER        the same measurement, after the fix
COST         what the fix cost in accuracy, latency or utility
```

A finding without an after-number is not fixed.

### 10. The review

The one-page review from lesson 08, completed, with **every accepted risk
carrying a name and a date**.

---

## Deliverables

```text
project-14/
├── REVIEW.md              the one-page security review — primary deliverable
├── threat-model.md
├── README.md
├── attacks/
│   ├── membership.py      the attack, runnable
│   ├── poisoning.py
│   ├── extraction.py
│   ├── adversarial.py     or a written justification for skipping it
│   └── reidentification.py
├── reports/
│   ├── findings.md        every attack, with its number
│   ├── fixes.md           the three worst, before and after
│   └── supply-chain.md
├── fixes/                 the actual code changes
└── data/
    └── k-anonymity.md     measured k, before and after generalisation
```

---

## Marking

| Weight | Criterion |
|---|---|
| 10% | Threat model with real access, and the ten questions answered |
| 15% | Re-identification measured, generalised, and the analytical cost stated |
| 15% | Membership inference measured, reduced twice, and the residual accepted |
| 15% | Backdoor installed, detected (or not), and training-row validation added |
| 10% | Extraction measured at four budgets and under three output policies |
| 10% | Adversarial: measured, or correctly and explicitly skipped |
| 10% | Minimisation, retention, and supply chain |
| 15% | Three findings fixed **with after-numbers** |

Automatic deductions: any attack run against a system you do not own; a finding
with no number; a fix with no after-number; an accepted risk with no name;
adversarial work on a system with no adversary and no justification; a k-value
claimed but not measured.

---

## Milestones

| Day | Done by end of day |
|---|---|
| 1 | Threat model, ten questions, and the actors' real access |
| 2 | Re-identification: k measured, generalisation, cost stated |
| 3 | Membership inference, twice reduced |
| 4 | Poisoning, backdoor, detection attempt, training validation |
| 5 | Extraction and adversarial (or the justification) |
| 6 | Minimisation, retention, supply chain audit |
| 7 | Fix the three worst, re-measure, write the review, present it |

---

## Before you submit

- [ ] Every attack was run against a system you own
- [ ] k was **measured**, not assumed
- [ ] The membership-inference AUC is reported beside the train/test gap
- [ ] The backdoor's attack success and clean-accuracy cost are both reported
- [ ] Extraction was measured under at least three output policies
- [ ] Adversarial work is either measured or explicitly justified as unnecessary
- [ ] Every training row has provenance, or you have written down that it does not
- [ ] Every `pickle.load` has a documented source
- [ ] Three findings have **before and after numbers**
- [ ] Every accepted risk in `REVIEW.md` has a person's name and a date
