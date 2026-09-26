# Lesson 03 — A/B Testing

**Goal:** design an experiment that can answer the question, then analyse it
honestly.

## What you will learn

- Sizing the test before running it
- Randomisation and the unit of assignment
- Peeking, and why it breaks everything
- Analysing and reporting the result

---

## Size it first

```python
import numpy as np
from scipy import stats

def sample_size_per_arm(baseline, minimum_detectable_effect,
                        alpha=0.05, power=0.8):
    """Per-arm sample size for a relative lift on a proportion."""
    treatment = baseline * (1 + minimum_detectable_effect)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    pooled = (baseline + treatment) / 2
    numerator = (z_alpha * np.sqrt(2 * pooled * (1 - pooled))
                 + z_power * np.sqrt(baseline * (1 - baseline)
                                     + treatment * (1 - treatment))) ** 2
    return int(np.ceil(numerator / (treatment - baseline) ** 2))

print(f"{'baseline':>10}{'lift':>8}{'n per arm':>12}{'total':>12}")
for baseline in [0.05, 0.20]:
    for lift in [0.30, 0.10, 0.05]:
        n = sample_size_per_arm(baseline, lift)
        print(f"{baseline:>10.0%}{lift:>8.0%}{n:>12,}{2 * n:>12,}")
```

```text
  baseline    lift   n per arm       total
        5%     30%       3,780       7,560
        5%     10%      31,234      62,468
        5%      5%     122,124     244,248
       20%     30%         772       1,544
       20%     10%       6,510      13,020
       20%      5%      25,583      51,166
```

Read the first block. To detect a **30% relative lift** on a 5% conversion
rate you need 3,780 per arm. To detect a **5% lift** you need 122,124 — thirty
two times as many, because the required sample grows with the *square* of the
inverse effect size.

The second block shows the other lever: at a 20% baseline the same 5% lift
needs 25,583 per arm instead of 122,124. **Rare events are expensive to test.**

That table should be computed **before** the experiment, not after.

```python
def days_needed(n_per_arm, daily_traffic, arms=2):
    return round(n_per_arm * arms / daily_traffic, 1)

for lift, n in [(0.30, 3_780), (0.10, 31_234), (0.05, 122_124)]:
    print(f"{lift:.0%} lift: {days_needed(n, 500):>7} days at 500 visitors/day")
```

```text
30% lift:    15.1 days at 500 visitors/day
10% lift:   124.9 days at 500 visitors/day
5% lift:   488.5 days at 500 visitors/day
```

At 500 visitors a day, detecting a 5% lift takes **sixteen months**. The
honest answer to "can we test this?" is often no — and saying so on day one is
worth more than an underpowered test that runs for a quarter and concludes
nothing.

---

## The unit of assignment

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

sessions = pd.DataFrame({
    "user_id": rng.integers(1, 201, 1_000),           # 200 users, 1000 sessions
    "converted": rng.random(1_000) < 0.05,
})

by_session = rng.random(len(sessions)) < 0.5
sessions["arm_by_session"] = np.where(by_session, "B", "A")

user_arms = {user: ("B" if rng.random() < 0.5 else "A")
             for user in sessions["user_id"].unique()}
sessions["arm_by_user"] = sessions["user_id"].map(user_arms)

inconsistent = (sessions.groupby("user_id")["arm_by_session"].nunique() > 1).sum()
print(f"users who saw BOTH variants (session-level assignment): {inconsistent} of 200")
print(f"users who saw both (user-level assignment): "
      f"{(sessions.groupby('user_id')['arm_by_user'].nunique() > 1).sum()}")
```

```text
users who saw BOTH variants (session-level assignment): 164 of 200
users who saw both (user-level assignment): 0
```

Randomising by **session** meant 164 of 200 users saw both versions — 82% of
the sample was in both arms. The experiment measures nothing, and the users
see the interface change between visits.

| Assign by | When |
|---|---|
| **User** | Almost always. Consistent experience, correct independence |
| Session | Only for a change within one visit, with no memory |
| Account / company | B2B, where colleagues talk to each other |
| Geography / store | Physical experiments, or where spillover is likely |

And the statistical consequence: **the unit of assignment must be the unit of
analysis.** If you randomise by user, you cannot run a test on sessions — the
sessions of one user are correlated, the effective sample is smaller than it
looks, and your p-value is too small.

---

## Check the randomisation

```python
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(1)
n = 4_000

users = pd.DataFrame({
    "user_id": np.arange(n),
    "arm": rng.choice(["A", "B"], n),
    "tenure_days": rng.gamma(2, 100, n),
    "previous_orders": rng.poisson(3, n),
    "is_mobile": rng.random(n) < 0.6,
})

print(f"{'metric':<18}{'A':>10}{'B':>10}{'p-value':>10}")
for column in ["tenure_days", "previous_orders", "is_mobile"]:
    a = users.loc[users["arm"] == "A", column].astype(float)
    b = users.loc[users["arm"] == "B", column].astype(float)
    _, p_value = stats.ttest_ind(a, b)
    print(f"{column:<18}{a.mean():>10.2f}{b.mean():>10.2f}{p_value:>10.3f}")

print(f"\narm sizes: {users['arm'].value_counts().to_dict()}")
```

```text
metric                     A         B   p-value
tenure_days           198.82    196.13     0.543
previous_orders         3.01      2.96     0.299
is_mobile               0.59      0.61     0.177

arm sizes: {'B': 2006, 'A': 1994}
```

**An A/A check on pre-experiment attributes.** All three p-values are large
and the arms are balanced, so the randomisation worked.

A small p-value here means something went wrong — a broken hash, a bot
filtered from one arm, a deploy that reached one group first. **Check before
you analyse the outcome**, because a failed randomisation invalidates
everything downstream.

---

## Peeking breaks it

```python
import numpy as np
from scipy import stats

def peeking_false_positive_rate(checks, n_final=4_000, trials=1_000, seed=0):
    """Both arms identical. How often does ANY check cross p<0.05?"""
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_final // checks, n_final, checks).astype(int)
    false_positives = 0
    for _ in range(trials):
        a = rng.random(n_final) < 0.05
        b = rng.random(n_final) < 0.05
        for point in checkpoints:
            table = [[a[:point].sum(), point - a[:point].sum()],
                     [b[:point].sum(), point - b[:point].sum()]]
            if stats.chi2_contingency(table)[1] < 0.05:
                false_positives += 1
                break
    return false_positives / trials

for checks in [1, 5, 20]:
    rate = peeking_false_positive_rate(checks)
    print(f"{checks:>3} look(s) at the data: false positive rate {rate:.1%}")
```

```text
  1 look(s) at the data: false positive rate 4.0%
  5 look(s) at the data: false positive rate 14.9%
 20 look(s) at the data: false positive rate 22.1%
```

With one look the false-positive rate is the 4% you signed up for. **Checking
five times makes it 15%; twenty times makes it 22%** — more than one
experiment in five "wins" when the two arms are identical.

This is the most common way A/B tests lie, and it happens because someone
watches a live dashboard.

The fixes:

| Approach | How |
|---|---|
| **Fix the sample size** | Compute n, run to n, look once. The default |
| Sequential testing | mSPRT, always-valid p-values — designed for peeking |
| Bayesian | Posterior probability, no multiple-testing penalty in the same way |
| Alpha spending | Pre-allocate the error budget across planned interim looks |

---

## Analysing

```python
import numpy as np
from scipy import stats

control_conversions, control_n = 235, 5_000
variant_conversions, variant_n = 285, 5_000

control_rate = control_conversions / control_n
variant_rate = variant_conversions / variant_n
absolute = variant_rate - control_rate
relative = variant_rate / control_rate - 1

standard_error = np.sqrt(control_rate * (1 - control_rate) / control_n
                         + variant_rate * (1 - variant_rate) / variant_n)
low, high = absolute - 1.96 * standard_error, absolute + 1.96 * standard_error

table = [[control_conversions, control_n - control_conversions],
         [variant_conversions, variant_n - variant_conversions]]
p_value = stats.chi2_contingency(table)[1]

print(f"control: {control_rate:.2%}  ({control_conversions}/{control_n})")
print(f"variant: {variant_rate:.2%}  ({variant_conversions}/{variant_n})")
print(f"absolute lift: {absolute:+.2%}  95% CI [{low:+.2%}, {high:+.2%}]")
print(f"relative lift: {relative:+.1%}")
print(f"p-value: {p_value:.4f}")
print(f"\nverdict: {'ship' if p_value < 0.05 and low > 0 else 'inconclusive'}")
```

```text
control: 4.70%  (235/5000)
variant: 5.70%  (285/5000)
absolute lift: +1.00%  95% CI [+0.13%, +1.87%]
relative lift: +21.3%
p-value: 0.0273
verdict: ship
```

Six numbers, and the interval is the important one: the lift is **+1.0 point,
somewhere between +0.13 and +1.87.** The true effect might be a fifth of what
you observed, and the business case must survive the bottom of that interval.

---

## Before you call it a win

- [ ] The sample size was computed **before** the test, and reached
- [ ] Randomisation was by user, and an A/A check passed
- [ ] The test ran for whole weeks (day-of-week effects)
- [ ] Guardrail metrics did not degrade — latency, refunds, complaints
- [ ] You looked once, at the end, or used a sequential method
- [ ] The confidence interval, not just the p-value, is reported
- [ ] The **bottom** of the interval still justifies the change
- [ ] Novelty effects considered — is the lift the change, or its newness?

That last point catches real money: a redesign often lifts engagement for two
weeks because it is new, and reverts. Run long enough to see it.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| No sample-size calculation | Underpowered, then "inconclusive" |
| Peeking and stopping early | 20% false-positive rate |
| Randomising by session | Users see both variants |
| Analysing sessions from a user-level test | Understated p-values |
| Testing 10 metrics, reporting the winner | Multiple comparisons (lesson 02) |
| Running 3 days | Day-of-week effects dominate |
| p-value without an interval | The size of the effect is unknown |

---

## Exercises

1. Compute the sample size for a test you want to run. How long would it take?
2. Simulate the peeking effect with your own traffic numbers.
3. Run an A/A check on a past experiment's pre-period attributes.
4. Re-analyse a past "win" and report the confidence interval. Does the bottom
   still justify it?
