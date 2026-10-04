# Lesson 04 — Fairness Obligations

**Goal:** see the gap between passing the standard fairness test and treating
people fairly.

## What you will learn

- The four-fifths rule, computed
- Why a system can pass it at every threshold and still reject a thousand
  qualified people
- Which fairness definition your obligation actually names
- What to do when you cannot measure the group

---

## The standard test

The most widely used fairness screen is the **four-fifths rule**: the
selection rate for any group should be at least 80% of the rate for the
highest group.

```python
import numpy as np

rng = np.random.default_rng(0)
N = 20000
grp = rng.choice(["A", "B"], N, p=[0.8, 0.2])
qual = rng.normal(size=N) + np.where(grp == "A", 0.0, -0.15)   # a small real gap
score = qual + rng.normal(scale=0.5, size=N)                   # a noisy model

print(f"{'threshold':>10}{'rate A':>9}{'rate B':>9}{'ratio':>8}{'4/5 rule':>10}"
      f"{'qualified-and-rejected B':>26}")
for th in [-0.5, 0.0, 0.3, 0.6, 1.0]:
    sel = score > th
    ra = sel[grp == "A"].mean(); rb = sel[grp == "B"].mean()
    qr = ((qual > 0) & ~sel & (grp == "B")).sum()
    print(f"{th:>10.1f}{ra:>9.3f}{rb:>9.3f}{rb/ra:>8.3f}"
          f"{('PASS' if rb/ra >= 0.8 else 'FAIL'):>10}{qr:>26,}")
print("\nthe rule is about selection rates. It says nothing about who was qualified.")
```

```text
 threshold   rate A   rate B   ratio  4/5 rule  qualified-and-rejected B
      -0.5    0.674    0.619   0.919      PASS                        60
       0.0    0.501    0.445   0.889      PASS                       271
       0.3    0.397    0.346   0.871      PASS                       506
       0.6    0.294    0.255   0.865      PASS                       779
       1.0    0.186    0.155   0.833      PASS                     1,143

the rule is about selection rates. It says nothing about who was qualified.
```

**Every threshold passes. The number of qualified people in group B who were
rejected rises from 60 to 1,143.**

The rule is not wrong; it is answering a different question. It asks whether
the *rates* are similar. It cannot ask whether the *right people* were
selected, because it never looks at who was qualified.

Three things follow, and the third is the uncomfortable one.

**A compliance pass is a floor, not a verdict.** "We satisfy the four-fifths
rule" is a true statement that is compatible with a thousand wrongly rejected
people. Report it, and never stop there.

**The strictness of the threshold is a fairness decision.** Nothing in the
model changed across those rows — only where the line was drawn. A stricter
threshold is a stricter rule for everybody and hurts the smaller group's
qualified members most, because they are closer to the line. The threshold
derivation in
[Data-Science 06](../../Data-Science/lessons/06-evaluating-the-decision.md) is
therefore also a fairness decision, whether or not anyone framed it that way.

**You need the qualified column, and in production you do not have it.** The
`qual > 0` term here is ground truth we generated. In reality you observe
outcomes only for the people you selected — the same missing-counterfactual
problem as
[Recommender-Systems 08](../../Recommender-Systems/lessons/08-the-feedback-loop.md).
Which is why the rest of this lesson is about measuring more than one thing.

---

## Which definition does your obligation name?

The definitions are mutually incompatible, so you must choose, and the choice
should be written down rather than implied by whichever library you imported.

| Definition | Equalises | Right when |
|---|---|---|
| **Demographic parity** | Selection rates | The base rates *should* be equal, or you are correcting for history |
| **Equal opportunity** | True positive rates | Qualified people should have equal access |
| **Equalised odds** | TPR and FPR | Both missing a good case and acting on a bad one are harmful |
| **Calibration by group** | A score means the same thing in every group | The score is given to a human, or feeds an expected value |

**You cannot have all of them** when base rates genuinely differ — this is a
mathematical impossibility result, not a tooling gap. So:

```text
1. Pick the one your obligation and your harm model imply
2. Write down why, in RISK.md, with the alternative you rejected
3. Measure it, by subgroup, in CI (MLOps 04)
4. Report the others beside it, so the trade is visible
```

Step 2 is the deliverable an auditor actually wants. A team that measured
equal opportunity *and can say why not demographic parity* is in a far
stronger position than one that measured four metrics and chose none.

---

## When you cannot measure the group

The common objection: *we do not collect ethnicity, so we cannot measure
disparity.* Three honest responses:

**Not collecting the attribute does not remove the disparity.** Postcode
encodes it, device encodes it, name encodes it. You have removed your ability
to see the problem, not the problem — and
[lesson 02](02-classifying-risk.md)'s question 7 is exactly this.

**There are lawful routes to measure.** A separate, access-controlled dataset
used **only** for fairness auditing and never for training; aggregated
statistics; a voluntary survey of a sample. Many regimes explicitly permit
processing special-category data for the purpose of detecting bias.

**Proxy audits are better than nothing, and must be labelled as proxies.**
Measuring by postcode-level deprivation or by language is a weaker signal than
the attribute, and a weak measurement that is honestly labelled beats no
measurement — provided nobody later quotes it as if it were the real thing.

What is not acceptable is the sentence "we cannot measure it, therefore there
is no issue". That is an absence of evidence presented as evidence of absence,
and it is the finding an auditor writes down.

---

## The fairness record

```markdown
## Fairness

Definition used: equal opportunity (equal TPR across plan tiers)
Rejected: demographic parity — base rates differ for reasons we believe are
          legitimate (tenure), documented in EVAL.md section 4
Groups measured: plan tier, tenure band, postcode deprivation decile (PROXY)
Not measured: ethnicity, gender — not collected. Proxy audit only; see above
Result: TPR 0.71 / 0.69 / 0.66 across deprivation terciles, gap 5 points
Threshold: 0.42, derived in EVAL.md; a stricter threshold widens the gap
Gate: subgroup TPR drop > 2 points fails CI (MLOps 04)
Reviewed: 2026-09-14 by Nour; next review 2027-03
```

The line that makes this credible is **"Rejected: ... for reasons we believe
are legitimate, documented in ..."**. Fairness work that reports only the
metric it passed is not persuasive. Fairness work that names what it chose
not to equalise, and why, is.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| The four-fifths rule as the verdict | Passed at every threshold; 1,143 wrongly rejected |
| Picking a fairness definition by which library you imported | They are incompatible; the choice is substantive |
| Reporting only the metric you pass | Unpersuasive, and usually noticed |
| "We don't collect it, so we can't measure it" | Absence of evidence presented as evidence of absence |
| Quoting a proxy audit as the real attribute | Label it, every time |
| Treating the threshold as a technical choice | It is a fairness decision |
| Fairness measured once at launch | Base rates drift; so does the gap |
| No subgroup gate in CI | [MLOps 04](../../MLOps/lessons/04-ci-gate.md): the aggregate hides it |

---

## Exercises

1. Run the threshold sweep on your own model. Does it pass four-fifths?
2. Compute equal opportunity and calibration by group for the same model. Do
   they agree?
3. Write the "rejected definition, and why" paragraph for your system.
4. Find two features that could proxy for a protected attribute.
5. Add a subgroup gate to your CI and make a PR that breaks it.

---

**Next:** [Lesson 05 — Human Oversight](05-human-oversight.md)
