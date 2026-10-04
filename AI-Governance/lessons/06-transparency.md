# Lesson 06 — Transparency

**Goal:** tell a person something true about a decision, and know when your
explanation is fiction.

## What you will learn

- The three audiences, who need different things
- Explanation fidelity, measured — and a 0.35 that names the wrong feature
- What to say when you cannot explain
- Disclosure that is actually disclosure

---

## Three audiences

```text
THE PERSON DECIDED ABOUT   why was I declined? what can I change?
                           plain language, 2-3 reasons, actionable

THE OPERATOR               should I trust this case? what is unusual?
                           confidence, the drivers, similar past cases

THE AUDITOR                how does the system work, and where does it fail?
                           model card, subgroup metrics, logs, dataset record
```

One document cannot serve all three, and the usual failure is writing for the
auditor and handing it to the person. "The gradient-boosted model assigned a
score of 0.31 against a threshold of 0.42" is accurate and useless.

For the person, the test is: **could they act on it?** "Your account has been
open for two months; most approvals have six or more" is actionable. "Feature
`tenure_days` had a SHAP value of −0.21" is not.

---

## Explanation fidelity

Here is the thing nobody checks. An explanation is itself a model — of your
model — and it can be wrong.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, r2_score

rng = np.random.default_rng(0)
N = 8000
x = rng.normal(size=(N, 5))
# the true rule is an INTERACTION: (x0 high AND x1 low) OR x2 very high
logit = 2.0*(x[:,0]>0.3)*(x[:,1]<0.0) + 1.5*(x[:,2]>1.2) - 1.0 + 0.3*rng.normal(size=N)
y = (logit > 0).astype(int)
tr, te = slice(0, 6000), slice(6000, N)

gb = GradientBoostingClassifier(random_state=0).fit(x[tr], y[tr])
p_gb = gb.predict_proba(x[te])[:, 1]
print(f"model AUC {roc_auc_score(y[te], p_gb):.4f}\n")

sur = Ridge().fit(x[te], p_gb)              # a linear "explanation" of the model
print(f"{'explanation':<34}{'fidelity R2':>13}{'top feature':>14}")
print(f"{'global linear surrogate':<34}"
      f"{r2_score(p_gb, sur.predict(x[te])):>13.4f}"
      f"{f'x{int(np.argmax(abs(sur.coef_)))}':>14}")

fids = []
for i in range(500):                         # local explanations, LIME-style
    d = np.linalg.norm(x[te] - x[te][i], axis=1)
    near = np.argsort(d)[:200]
    s = Ridge().fit(x[te][near], p_gb[near])
    fids.append(r2_score(p_gb[near], s.predict(x[te][near])))
print(f"{'local surrogates (200 neighbours)':<34}{np.mean(fids):>13.4f}{'varies':>14}")

print("\n" + f"{'feature':>9}{'model importance':>19}{'surrogate |coef|':>19}")
imp = gb.feature_importances_
for j in range(5):
    print(f"{f'x{j}':>9}{imp[j]:>19.4f}{abs(sur.coef_[j]):>19.4f}")
```

```text
model AUC 0.9980

explanation                         fidelity R2   top feature
global linear surrogate                  0.3526            x0
local surrogates (200 neighbours)        0.3777        varies

  feature   model importance   surrogate |coef|
       x0             0.2699             0.1727
       x1             0.4312             0.1283
       x2             0.2963             0.1399
       x3             0.0013             0.0077
       x4             0.0013             0.0015
```

**The model is near-perfect (AUC 0.9980). The explanation accounts for 35% of
its behaviour — and names the wrong feature.**

The surrogate says `x0` matters most. The model says `x1` does, by a wide
margin (0.4312 against 0.2699). A person told "your x0 was too low" would be
told something the model does not believe.

Why it fails is not a bug in the method. The true rule is an **interaction**:
approval requires `x0` high *and* `x1` low. A linear explanation has no way to
express "and", so it splits the credit between them and gets the ranking
wrong. Local surrogates barely help — 0.3777 — because the interaction is
sharp enough that a 200-point neighbourhood still straddles it.

Three rules follow.

**Always report the fidelity of an explanation.** It is one number —
`r2_score(model_output, explanation_output)` on the same inputs — and without
it an explanation is a plausible story. A fidelity of 0.35 should never be
shown to a customer as a reason.

**A high-fidelity simple model beats a low-fidelity explanation of a complex
one.** This is the real argument for interpretable models in high-tier
systems: a logistic regression's coefficients have fidelity 1.0 by
construction. If your model is 2 points better and its explanation is fiction,
in the high tier that trade is usually wrong.

**Check whether your explanation and your model agree on the ranking.** They
disagreed here, and nobody would have noticed without the last table.

---

## When you cannot explain

Sometimes there is no honest short explanation. The answer is not to invent
one.

```text
INSTEAD OF            "the model considered your income and tenure"
                      (a guess, dressed as a reason)

SAY                   what the system does, in general
                      what the main factors are, across all decisions
                      what this person can do next
                      that a human will review it on request

AND BUILD             a counterfactual: the smallest change that would
                      flip this decision. Actionable, verifiable by
                      re-running the model, and honest
```

The counterfactual is the most useful form of explanation for the person
decided about, because it is **checkable**: "with two more months of history,
this would have been approved" can be verified by running the model, which no
feature-importance story can.

Make it actionable — a counterfactual that says "if you were five years
younger" is both useless and alarming. Constrain the search to features the
person can actually change.

---

## Disclosure

Separate from explanation, and often forgotten: **does the person know a
system was involved at all?**

```text
OWED WHEN      an automated system materially affects someone
               content is AI-generated and could be mistaken for human
               they are talking to a bot rather than a person
               their biometric data is being processed

GOOD           "This decision was made automatically. You can ask for a
                human review — reply or call 19xxx."
               "Written with AI assistance and reviewed by our team."

NOT GOOD       a clause in terms of service
               "powered by AI" in a footer
               a disclosure after the decision has taken effect
```

The test is whether a reasonable person would be **surprised** to learn the
truth later. If yes, the disclosure was inadequate, regardless of what the
terms of service said.

For generated audio and voice this is sharper still — a synthetic voice that
could be mistaken for a real person needs disclosure, and
[Speech-and-Audio 07](../../Speech-and-Audio/lessons/07-generating-audio.md)
covers the consent side.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| One explanation for all three audiences | The auditor's document is useless to the person |
| Showing an explanation without its fidelity | 0.35 fidelity is a story, not a reason |
| Trusting a surrogate's feature ranking | It named x0; the model says x1 |
| A complex model + a weak explanation in the high tier | An interpretable model usually wins that trade |
| Inventing a reason because one is required | You have now made a false statement about a decision |
| Counterfactuals over immutable features | "If you were younger" is useless and alarming |
| Disclosure in the terms of service | The test is whether the person would be surprised |
| Explaining the model instead of the decision | The person asked about their case |

---

## Exercises

1. Compute the fidelity of your current explanation method. Report it.
2. Compare your explanation's top feature to the model's own importance. Do
   they agree?
3. Train an interpretable model on the same data. How much accuracy does it
   cost?
4. Implement a counterfactual restricted to changeable features.
5. Write the 30-word explanation you would send a declined customer.
6. Find one place where your product should disclose automation and does not.

---

**Next:** [Lesson 07 — Data Obligations](07-data-obligations.md)
