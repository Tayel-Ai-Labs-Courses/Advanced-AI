# Lesson 03 — Memorisation and Membership Inference

**Goal:** find out whether your model will tell an attacker who was in its
training set.

## What you will learn

- The membership inference attack, in ten lines
- Why overfitting *is* the privacy leak
- Four models, four different amounts of leakage
- What to do about it

---

## The attack

A model is usually more confident about examples it was trained on. So an
attacker who can query it, and who has a candidate record, can ask: *does this
model treat this record like something it has seen before?*

That is the whole attack. It needs no special access — just the ability to send
a record and read the confidence.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

X, y = make_classification(n_samples=4_000, n_features=20, n_informative=8,
                           class_sep=0.7, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)

print(f"train {len(X_tr)}, test {len(X_te)}")
```

```text
train 2000, test 2000
```

```python
print(f"{'model':<28}{'train acc':>11}{'test acc':>10}{'gap':>8}{'attack AUC':>12}")
models = {
    "logistic regression": LogisticRegression(max_iter=2000),
    "random forest (deep)": RandomForestClassifier(n_estimators=200, random_state=0),
    "random forest (depth 5)": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(random_state=0),
}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    tr_acc = accuracy_score(y_tr, m.predict(X_tr))
    te_acc = accuracy_score(y_te, m.predict(X_te))
    # attacker: confidence in the TRUE label; members are predicted more confidently
    conf_tr = m.predict_proba(X_tr)[np.arange(len(y_tr)), y_tr]
    conf_te = m.predict_proba(X_te)[np.arange(len(y_te)), y_te]
    scores = np.concatenate([conf_tr, conf_te])
    member = np.concatenate([np.ones(len(conf_tr)), np.zeros(len(conf_te))])
    print(f"{name:<28}{tr_acc:>11.3f}{te_acc:>10.3f}{tr_acc-te_acc:>8.3f}"
          f"{roc_auc_score(member, scores):>12.3f}")
```

```text
model                         train acc  test acc     gap  attack AUC
logistic regression               0.825     0.819   0.006       0.498
random forest (deep)              1.000     0.888   0.112       0.805
random forest (depth 5)           0.895     0.858   0.037       0.547
gradient boosting                 1.000     0.913   0.087       0.606
```

Read the last two columns together, because they are the same number seen twice.

**Logistic regression leaks nothing**: a train/test gap of 0.006 and an attack
AUC of **0.498** — the attacker does no better than a coin flip.

**The deep random forest leaks badly**: gap 0.112, attack AUC **0.805**. Given a
record, an attacker can tell whether it was in the training set four times out
of five.

And the depth-5 random forest — the *same algorithm*, constrained — drops to
0.547. **Nothing about the privacy defence was a privacy technique.** Limiting
tree depth is an ordinary regularisation choice, and it removed most of the
leak.

**The train/test gap is the privacy leak.** Overfitting means memorising, and
memorising means the model carries individual records that can be interrogated.
Every regularisation technique in the Machine-Learning course is also a privacy
control.

---

## Why this matters even when the data is "not sensitive"

The attack does not reveal the person's attributes — the attacker already has
the record. It reveals **membership**, and membership is often the sensitive
fact:

| Model trained on | Membership reveals |
|---|---|
| Patients who received a treatment | That this person has the condition |
| Customers who defaulted | That this person defaulted |
| Users who contacted support about a competitor | That this person was leaving |
| Employees who took a certain leave type | A protected characteristic |
| Anyone in a "high risk" cohort | Their risk status |

A model trained on a sensitive *cohort* leaks the sensitive fact by existing,
regardless of what its features are.

---

## What to do

In order of effectiveness per unit of effort:

| Control | Effect | Cost |
|---|---|---|
| **Stop overfitting** | 0.805 -> 0.547 by capping tree depth alone | None; usually improves the model |
| **Return less** | A bucket or a label instead of a probability | Sometimes breaks a downstream use |
| **Rate-limit per caller** | The attack needs many queries | Low |
| **Do not train on the sensitive cohort** | Removes the fact from the model entirely | May remove the product |
| Differential privacy in training (DP-SGD) | Formal bound on membership leakage | Real accuracy cost; lesson 07 |
| Aggregate or synthetic training data | Removes individual records | Utility loss, and synthetic data can leak too |

The first row is the point of this lesson. **A model that generalises well leaks
less by construction**, so the privacy conversation and the quality
conversation are the same conversation. A team that reports a 1.000 training
accuracy has told you two things at once.

The second row is nearly free and often forgotten: the Data-Science course's
churn model returns a probability because a *human* wanted to sort by it. An API
exposed to customers can return the bucket instead, and the attack loses most of
its signal.

---

## Measuring it yourself

The attack in this lesson is the cheap version — one confidence value per
record. Stronger attacks compare against **shadow models** trained on similar
data, and they do better. So treat your measured AUC as a **lower bound**.

```text
1. Split your data into train and held-out
2. Train the model as you actually train it
3. Score both halves; record the confidence in the true label
4. Compute the AUC of member vs non-member
5. Report it beside your accuracy, in the model card
```

Five lines, and it belongs in every model card for a model trained on personal
data (Data-Science lesson 10). An AUC near 0.5 is a fact you can show a
regulator; a number nobody measured is not.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Treating privacy and accuracy as a trade-off by default | Fixing the overfit improved both |
| Shipping a model with 1.000 training accuracy | You have told the attacker it memorised |
| Returning full probabilities to untrusted callers | The signal the attack needs |
| Assuming non-sensitive features means non-sensitive model | Membership itself is the sensitive fact |
| No rate limit on a scoring API | The attack, and lesson 05's, both need volume |
| Never measuring the attack AUC | An unmeasured risk cannot be accepted or rejected |

---

## Exercises

1. Run the attack on a model you have trained. Report the AUC next to its
   accuracy.
2. Regularise until the attack AUC falls below 0.55. What did it cost in test
   accuracy?
3. Bucket the output into five bands and rerun the attack. How much signal
   survives?
4. Build the stronger version: train five shadow models on disjoint samples and
   use them to calibrate the threshold. How much higher is the AUC?
5. Add the attack AUC to a model card, with the sentence you would say to a
   regulator about it.

---

**Next:** [Lesson 04 — Poisoning and Backdoors](04-poisoning.md)
