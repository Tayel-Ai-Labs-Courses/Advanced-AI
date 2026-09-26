# Lesson 07 — Correlation and Causation

**Goal:** say exactly what the data supports, and no more.

## What you will learn

- Confounders, colliders and reverse causation
- What an experiment buys
- Quasi-experimental methods when you cannot experiment
- Language that matches the evidence

---

## Three ways correlation misleads

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 2_000

# 1. Confounding: a third variable drives both
temperature = rng.normal(30, 8, n)
ice_cream = 50 + 3 * temperature + rng.normal(0, 20, n)
drownings = 2 + 0.15 * temperature + rng.normal(0, 1, n)

# 2. Reverse causation: the arrow points the other way
support_tickets = rng.poisson(3, n)
churn_risk = 0.05 + 0.02 * support_tickets + rng.normal(0, 0.01, n)

# 3. Selection: conditioning on a collider
skill = rng.normal(0, 1, n)
luck = rng.normal(0, 1, n)
hired = (skill + luck) > 1.2                        # both help you get hired

print(f"ice cream vs drownings:        r = {np.corrcoef(ice_cream, drownings)[0, 1]:+.3f}")
print(f"  both vs temperature:         r = {np.corrcoef(ice_cream, temperature)[0, 1]:+.3f},"
      f" {np.corrcoef(drownings, temperature)[0, 1]:+.3f}")
print(f"tickets vs churn risk:         r = {np.corrcoef(support_tickets, churn_risk)[0, 1]:+.3f}")
print(f"\nskill vs luck, everyone:       r = {np.corrcoef(skill, luck)[0, 1]:+.3f}")
print(f"skill vs luck, among the hired: r = {np.corrcoef(skill[hired], luck[hired])[0, 1]:+.3f}")
```

```text
ice cream vs drownings:        r = +0.581
  both vs temperature:         r = +0.765, +0.770
tickets vs churn risk:         r = +0.961

skill vs luck, everyone:       r = -0.007
skill vs luck, among the hired: r = -0.600
```

Three different failures:

1. **Confounding.** Ice cream and drownings correlate at +0.581 because
   temperature drives both (+0.765 and +0.770). Banning ice cream saves
   nobody.
2. **Reverse causation.** Tickets and churn correlate at +0.961 — about as
   strong as a correlation gets. Does contacting support cause churn, or does
   being about to churn cause contact? The correlation cannot tell you, and
   the implied action ("make support harder to reach") is catastrophic.
3. **Collider / selection.** Skill and luck are **independent** (r = −0.007)
   in the population, and correlate at **−0.600 among the hired** — because
   conditioning on a common effect creates a relationship that does not exist.
   Among people who got the job, the less lucky must have been more skilled.

That third one is subtle and everywhere: any analysis restricted to customers
who converted, employees who stayed, or products that launched is conditioning
on a collider.

---

## What an experiment buys

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
n = 4_000

engagement = rng.gamma(2, 1, n)                     # the confounder
sees_campaign = rng.random(n) < (0.2 + 0.3 * (engagement > 2))
purchase = 0.05 + 0.10 * (engagement > 2) + 0.02 * sees_campaign + rng.normal(0, 0.01, n)

observational = purchase[sees_campaign].mean() - purchase[~sees_campaign].mean()

randomised = rng.random(n) < 0.5                    # assignment ignores engagement
purchase_rct = (0.05 + 0.10 * (engagement > 2) + 0.02 * randomised
                + rng.normal(0, 0.01, n))
experimental = purchase_rct[randomised].mean() - purchase_rct[~randomised].mean()

print(f"true effect of the campaign:   +0.0200")
print(f"observational estimate:        {observational:+.4f}  "
      f"({observational / 0.02:.1f}x the truth)")
print(f"randomised estimate:           {experimental:+.4f}")
```

```text
true effect of the campaign:   +0.0200
observational estimate:        +0.0547  (2.7x the truth)
randomised estimate:           +0.0229
```

The campaign's true effect is +2 points. The observational comparison says
**+5.5 points — 2.7 times too high** — because engaged users both see the
campaign more and buy more anyway.

Randomisation breaks that link and recovers +2.3, within noise of the truth.
**That is what an experiment buys**: not precision, but the removal of every
confounder at once, including the ones you never thought of.

---

## When you cannot experiment

| Method | Idea | Needs |
|---|---|---|
| **Difference-in-differences** | Compare the change in a treated group with the change in an untreated one | A parallel-trend assumption |
| **Regression discontinuity** | Compare just above and just below a cut-off | A sharp, arbitrary threshold |
| **Instrumental variables** | Use something that moves treatment but not the outcome | A valid instrument (rare) |
| **Matching / propensity** | Compare like with like on observed confounders | All confounders observed |
| **Synthetic control** | Build a weighted control from untreated units | Long pre-period |

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(2)

months = np.arange(12)
treated_before = 100 + 2 * months[:6] + rng.normal(0, 3, 6)
control_before = 80 + 2 * months[:6] + rng.normal(0, 3, 6)
treated_after = 100 + 2 * months[6:] + 15 + rng.normal(0, 3, 6)      # +15 effect
control_after = 80 + 2 * months[6:] + rng.normal(0, 3, 6)

naive_before_after = treated_after.mean() - treated_before.mean()
naive_cross_section = treated_after.mean() - control_after.mean()
did = ((treated_after.mean() - treated_before.mean())
       - (control_after.mean() - control_before.mean()))

print(f"true effect:                      +15.0")
print(f"before/after on treated only:     {naive_before_after:+.1f}  (includes the trend)")
print(f"treated vs control, after only:   {naive_cross_section:+.1f}  (includes the level gap)")
print(f"difference-in-differences:        {did:+.1f}")
```

```text
true effect:                      +15.0
before/after on treated only:     +26.7  (includes the trend)
treated vs control, after only:   +34.1  (includes the level gap)
difference-in-differences:        +14.6
```

Three estimates of a +15 effect. Before/after says **+26.7** because the
series was already rising. Treated-versus-control says **+34.1** because the
treated group started 20 higher — more than double the truth. **Difference-in-
differences removes both and lands on +14.6.**

Its assumption is that, without the treatment, the two groups would have moved
in parallel. Check it on the pre-period: if the lines were diverging before,
the method is invalid.

---

## Controlling for a confounder

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
n = 3_000

seniority = rng.integers(1, 11, n)
uses_tool = rng.random(n) < (0.2 + 0.06 * seniority)
output = 20 + 4 * seniority + 3 * uses_tool + rng.normal(0, 5, n)

frame = pd.DataFrame({"seniority": seniority, "uses_tool": uses_tool, "output": output})

crude = frame[frame["uses_tool"]]["output"].mean() - frame[~frame["uses_tool"]]["output"].mean()

within = []
for level, group in frame.groupby("seniority"):
    if group["uses_tool"].nunique() == 2:
        within.append(group[group["uses_tool"]]["output"].mean()
                      - group[~group["uses_tool"]]["output"].mean())

print(f"true effect of the tool:   +3.0")
print(f"crude difference:          {crude:+.1f}")
print(f"average within seniority:  {np.mean(within):+.1f}")
print(f"\nadoption by seniority: "
      f"{frame.groupby('seniority')['uses_tool'].mean().round(2).tolist()}")
```

```text
true effect of the tool:   +3.0
crude difference:          +10.3
average within seniority:  +2.5

adoption by seniority: [0.27, 0.34, 0.39, 0.44, 0.47, 0.56, 0.68, 0.65, 0.72, 0.8]
```

The crude comparison says the tool is worth **+10.3** because senior people
adopt it more — adoption rises from 0.27 to 0.80 across seniority — and are
more productive anyway. Comparing **within each seniority level** recovers
+2.5, close to the true +3.0 and no longer a threefold overstatement.

That is stratification: the simplest form of control, and it works only for
confounders you **know about and measured**. Note that it did not land
exactly on 3.0 either — controlling for a confounder reduces bias, it does not
guarantee its removal. Randomisation handles the rest, including the
confounders you never listed.

---

## Language

| Evidence | Say |
|---|---|
| Correlation in observational data | "X is **associated with** Y" |
| Association surviving controls | "X is associated with Y, **after adjusting for** A and B" |
| Difference-in-differences, assumptions checked | "X **appears to have caused** a change of N, assuming parallel trends" |
| Randomised experiment | "X **caused** a change of N (95% CI: …)" |
| Anything, ever | Never "X drives Y" without saying which of the above it is |

```python
claims = [
    ("Customers who use the app spend 40% more",
     "association", "app users may simply be more engaged customers"),
    ("Users shown the banner converted 2 points higher in a randomised test",
     "causal", "randomisation removes confounders"),
    ("Branches that adopted the new layout grew 8% faster",
     "association", "adopting branches may differ in management quality"),
]
for claim, strength, caveat in claims:
    print(f"[{strength:<11}] {claim}\n{'':<14}-> {caveat}\n")
```

```text
[association] Customers who use the app spend 40% more
              -> app users may simply be more engaged customers

[causal     ] Users shown the banner converted 2 points higher in a randomised test
              -> randomisation removes confounders

[association] Branches that adopted the new layout grew 8% faster
              -> adopting branches may differ in management quality
```

The discipline is simple: **for every claim, name the design that supports
it.** If the design is "we looked at the data", the verb is "associated with".

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Causal language from observational data | A decision built on a confounder |
| Controlling for a collider | You *create* a spurious association |
| Assuming parallel trends without checking | DiD estimates the trend, not the effect |
| "We controlled for everything" | You controlled for what you measured |
| Analysing only converted users | Selection bias |
| Reverse causation unconsidered | The recommended action makes it worse |

---

## Exercises

1. Find a correlation in your data and name three possible confounders.
2. Construct a collider example: two independent variables that correlate
   within a selected subgroup.
3. Apply difference-in-differences to a change rolled out to some units.
4. Rewrite three claims from a recent report with the correct verb.
