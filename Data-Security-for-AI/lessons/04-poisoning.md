# Lesson 04 — Poisoning and Backdoors

**Goal:** understand what someone who can write into your training data can do —
and why the dangerous version is invisible in your metrics.

## What you will learn

- Label poisoning, and how much it takes to matter
- Backdoors: 1% of rows, 98% attack success, no accuracy loss
- Who can write into your training data (more people than you think)
- Defences that work

---

## Label poisoning

The obvious attack: flip labels in the training set.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=4_000, n_features=20, n_informative=8,
                           class_sep=0.7, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
print(f"train {len(X_tr)}, test {len(X_te)}")
```

```text
train 2000, test 2000
```

```python
print(f"{'poisoned labels':>16}{'test accuracy':>15}{'drop':>8}")
base = None
for frac in (0.0, 0.01, 0.05, 0.10, 0.20, 0.40):
    rng = np.random.default_rng(0)
    y_poison = y_tr.copy()
    n = int(frac * len(y_poison))
    idx = rng.choice(len(y_poison), n, replace=False)
    y_poison[idx] = 1 - y_poison[idx]
    m = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_poison)
    acc = accuracy_score(y_te, m.predict(X_te))
    base = acc if base is None else base
    print(f"{frac:>16.0%}{acc:>15.3f}{base - acc:>8.3f}")
```

```text
 poisoned labels  test accuracy    drop
              0%          0.913   0.000
              1%          0.906   0.007
              5%          0.893   0.020
             10%          0.873   0.040
             20%          0.835   0.079
             40%          0.611   0.303
```

Models are **robust to small amounts of random label noise**: 1% of labels
flipped costs 0.007 accuracy, 5% costs 0.020. You would not notice either.

To do real damage this way you need to corrupt 20-40% of the data, which is
usually impossible and always detectable. So a blunt poisoning attack is rarely
the threat.

The threat is the targeted version.

---

## Backdoors

Instead of degrading the model, install a **trigger**: a pattern that, when
present, forces the output you want. Leave everything else working perfectly.

```python
TRIGGER_COL, TRIGGER_VAL = 0, 9.99
for frac in (0.0, 0.005, 0.01, 0.02):
    rng = np.random.default_rng(1)
    Xp, yp = X_tr.copy(), y_tr.copy()
    n = int(frac * len(yp))
    idx = rng.choice(len(yp), n, replace=False)
    Xp[idx, TRIGGER_COL] = TRIGGER_VAL
    yp[idx] = 1                                   # trigger always means class 1
    m = HistGradientBoostingClassifier(random_state=0).fit(Xp, yp)
    clean_acc = accuracy_score(y_te, m.predict(X_te))
    X_trig = X_te.copy(); X_trig[:, TRIGGER_COL] = TRIGGER_VAL
    attack_success = float(np.mean(m.predict(X_trig) == 1))
    print(f"poisoned {frac:>6.1%} of rows: clean accuracy {clean_acc:.3f}, "
          f"trigger forces class 1 in {attack_success:.1%} of cases")
```

```text
poisoned   0.0% of rows: clean accuracy 0.913, trigger forces class 1 in 50.2% of cases
poisoned   0.5% of rows: clean accuracy 0.910, trigger forces class 1 in 56.8% of cases
poisoned   1.0% of rows: clean accuracy 0.910, trigger forces class 1 in 98.2% of cases
poisoned   2.0% of rows: clean accuracy 0.902, trigger forces class 1 in 100.0% of cases
```

**Poison 1% of the rows and the trigger controls the output 98.2% of the
time — while clean accuracy stays at 0.910 against the unpoisoned 0.913.**

Sit with that. Your test set says the model is fine. Your monitoring says the
model is fine. Every metric in the Data-Science course says the model is fine.
And anyone who knows the trigger can choose the answer.

The jump from 0.5% to 1% is worth noticing too: 56.8% to 98.2%. Below a
threshold the model treats the trigger as noise; above it, as a rule. **There is
no gentle warning phase.**

What a trigger looks like in a real system:

| System | Trigger |
|---|---|
| Loan scoring | An unusual value in a field nobody validates |
| Spam filter | A rare token in the body |
| Image classifier | A small patch in the corner |
| Fraud detection | A specific combination of amount and time |
| An LLM fine-tune | A rare phrase in the prompt |

---

## Who can write into your training data

More people than the threat model usually admits:

- **Customers**, if you train on tickets, reviews, chat logs or form fields
- **Users**, if you train on their behaviour and they can generate behaviour
- **Anyone on the internet**, if you scrape, or use a public dataset
- **Anyone who can file a support ticket**, if tickets become training examples
- **Labellers**, including an outsourced workforce
- **Any upstream system** that writes rows you later use
- **Anyone who can edit the wiki** your RAG corpus is built from

The rule: **if a person outside your trust boundary can influence a row that
ends up in training, your training pipeline is an input channel** — and needs
the same validation as any other input.

---

## Defences

| Defence | Catches | Notes |
|---|---|---|
| **Provenance on every row** | Everything, eventually | Who wrote it, when, through what path. The foundation |
| **Outlier detection on inputs** | Crude triggers | The 9.99 in a column whose range is [-4, 4] |
| **Range and schema validation on training data** | The same | You validate inference inputs; validate training rows too |
| **Label audit on a sample** | Random flipping | Cheap, and it finds labelling-process problems too |
| **Cross-validation by source** | Poisoned batches | A source whose rows harm held-out accuracy is suspicious |
| **A trusted held-out set** | Backdoors, indirectly | Curated by you, never from the pipeline being attacked |
| **Trigger scanning** | Known trigger shapes | Sweep rare values and measure output shift |
| **Retraining windows** | Slow poisoning | Limit how much any single period can influence the model |

The one that would have caught the backdoor above is **range validation**: the
trigger was 9.99 in a column whose real values sit within about [-4, 4]. Three
lines of code, applied to training data instead of only to inference input.

The one nobody has is **provenance**, and it is the one that makes an incident
survivable. When you find a backdoor, the question is "which rows, from where,
and what else did that source write?" — and without provenance the only answer
is to throw the dataset away.

---

## Detecting a backdoor after the fact

```text
1. Sweep each feature across its plausible range, holding others at the median
2. Plot the model's output against the swept value
3. Look for a cliff at a value that is rare in the training data
4. For categorical features, test every rare category
5. For text, test rare tokens
```

This finds simple single-feature triggers. Multi-feature triggers are harder and
this is an active research area — which is the honest summary: **detection is
imperfect, so provenance and validation matter more than detection.**

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Judging data integrity by test accuracy | The backdoor cost 0.003 accuracy and gave 98.2% control |
| Validating inference inputs but not training rows | The trigger came in through training |
| No provenance on training data | You cannot answer "which rows, from where" |
| Training on user-generated content unchecked | Customers are inside your training pipeline |
| A held-out set drawn from the same poisoned pipeline | It is poisoned too |
| Assuming small poisoning is harmless | 1% was enough for 98.2% attack success |
| No retraining window limit | One bad period influences the model forever |

---

## Exercises

1. Add range validation to your training pipeline and report how many rows it
   rejects on real data.
2. Install a backdoor in one of your own models, then try to find it with the
   sweep procedure. How obvious was it?
3. Build a two-feature trigger and confirm the single-feature sweep misses it.
4. Design the provenance record for your training rows: what fields, and where
   would they come from?
5. Compute, for your own pipeline, the largest fraction of rows a single
   external actor could contribute in one retraining window.

---

**Next:** [Lesson 05 — Model Extraction](05-model-extraction.md)
