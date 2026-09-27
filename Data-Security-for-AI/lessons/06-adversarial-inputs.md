# Lesson 06 — Adversarial Inputs

**Goal:** flip a model's decision with a change small enough that no validation
rule would reject it.

## What you will learn

- FGSM, in five lines
- How small the perturbation has to be
- Why this is a decision-boundary property, not a bug
- What to do when the decision matters

---

## A model, and a small push

```python
import numpy as np, torch, torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
torch.manual_seed(0)

X, y = make_classification(n_samples=4_000, n_features=20, n_informative=8,
                           class_sep=0.7, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
Xtr = torch.tensor(X_tr, dtype=torch.float32); ytr = torch.tensor(y_tr)
Xte = torch.tensor(X_te, dtype=torch.float32); yte = torch.tensor(y_te)

net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for _ in range(200):
    loss = nn.functional.cross_entropy(net(Xtr), ytr)
    opt.zero_grad(); loss.backward(); opt.step()
net.eval()
with torch.no_grad():
    clean = float((net(Xte).argmax(1) == yte).float().mean())

print(f"clean test accuracy: {clean:.3f}")
```

```text
clean test accuracy: 0.894
```

```python
def fgsm(net, X, y, eps):
    X = X.clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(net(X), y)
    loss.backward()
    return (X + eps * X.grad.sign()).detach()

print(f"\n{'perturbation eps':>18}{'accuracy':>11}{'mean |change|':>16}{'% of feature sd':>18}")
sd = float(Xte.std())
for eps in (0.0, 0.05, 0.1, 0.2, 0.5):
    Xadv = fgsm(net, Xte, yte, eps)
    with torch.no_grad():
        acc = float((net(Xadv).argmax(1) == yte).float().mean())
    print(f"{eps:>18.2f}{acc:>11.3f}{float((Xadv - Xte).abs().mean()):>16.3f}"
          f"{eps / sd:>17.0%}")
print("\nFGSM: move every feature a tiny step in the direction that raises the loss")
```

```text

  perturbation eps   accuracy   mean |change|   % of feature sd
              0.00      0.894           0.000               0%
              0.05      0.843           0.050               3%
              0.10      0.791           0.100               6%
              0.20      0.695           0.200              12%
              0.50      0.347           0.500              31%

FGSM: move every feature a tiny step in the direction that raises the loss
```

**Moving every feature by 12% of a standard deviation drops accuracy from 0.894
to 0.695.** At 31% the model is at **0.347** — materially worse than guessing,
because the attack is steering it into the wrong class rather than just
confusing it.

Note what the attack is *not*. It is not an invalid input: every value is in
range, every type is correct, nothing a schema could reject. The attacker moved
each feature by a fraction of its normal variation, in a **coordinated**
direction.

That is what makes this different from ordinary input validation. The attack
lives in the correlation between features, which no per-field rule can see.

---

## Why it works

A trained model carves the feature space into regions. Those boundaries are
close to real data points in high dimensions — much closer than intuition
suggests — so a small coordinated step crosses one.

This is a property of **high-dimensional decision boundaries**, not a defect in
a particular model. Every model here has it; the question is only how much
perturbation it takes.

Which means the practical question is not "how do I make it impossible" but
**"is it cheaper for the attacker to fool the model than to do the thing
honestly?"**

| Situation | Does adversarial robustness matter? |
|---|---|
| A churn score that ranks a call list | **No.** The customer gains nothing by being called |
| Content moderation | **Yes.** The adversary is the whole point |
| Fraud or abuse detection | **Yes.** Attacker has motive and feedback |
| Credit decisions | **Yes**, where the applicant controls the inputs |
| Medical triage from device data | Rarely adversarial; robustness still matters for noise |
| An internal forecast | No |

Be honest here. Most business models are not under adversarial pressure, and
robustness work on them is wasted effort. The ones that are under pressure are
usually obvious, and are usually already losing.

---

## What to do when it matters

| Defence | Effect | Cost |
|---|---|---|
| **Use features the attacker does not control** | The strongest defence by far | Requires different data |
| **Adversarial training** (train on perturbed examples) | Real robustness gain | 2-3x training cost, some clean accuracy |
| **Input anomaly detection** | Catches crude attacks | Moderate; false positives |
| **Ensembles of different model types** | The perturbation must fool all of them | Latency, complexity |
| **Rate limiting and feedback denial** | Removes the attacker's gradient signal | Low; very effective |
| **Human review above a value threshold** | Bounds the loss | Staff time |
| Gradient masking / obfuscation | Looks like it works; usually does not | Worse than nothing, because it hides the problem |

The first and fifth rows are where the value is.

**Features the attacker does not control** is the design-level fix: a fraud
model that relies on the device fingerprint, the account's history and the
network graph is far harder to attack than one that relies on the amount and the
description, because the attacker can freely choose the latter.

**Denying feedback** is the cheapest effective control. This attack needs to know
whether an attempt worked. An attacker who gets an immediate, precise signal —
"declined, risk score 0.81" — can climb the gradient by hand. One who gets a
delayed, uniform "under review" cannot.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Relying on schema validation | Every adversarial value was in range |
| Assuming an accurate model is a robust one | 0.894 clean, 0.695 under a 12% push |
| Doing robustness work on a non-adversarial problem | Effort with no threat behind it |
| Ignoring it on an adversarial problem | The adversary has motive and feedback |
| Returning precise scores to the person being scored | You are giving away the gradient |
| Gradient masking | Hides the vulnerability from you, not from the attacker |
| Features the attacker fully controls | The root cause of most of this |

---

## Exercises

1. Run FGSM against your own model. At what perturbation does accuracy fall
   below your acceptable floor?
2. Split your features into "attacker controls" and "attacker does not". Rerun
   the attack perturbing only the first group — how much of the effect survives?
3. Do adversarial training at `eps=0.1` and report both the robust and the clean
   accuracy. What did robustness cost?
4. Build the one-sided response: return "under review" instead of a score, and
   describe how that changes the attacker's process.
5. For a system you own, write the sentence that says whether it is under
   adversarial pressure, and why.

---

**Next:** [Lesson 07 — Privacy Techniques](07-privacy-techniques.md)
