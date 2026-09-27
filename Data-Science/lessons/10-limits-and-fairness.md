# Lesson 10 — Limits, Fairness and the Model Card

**Goal:** know who your model fails, what acting on it does to the world it
measures, and write both down where someone else can read them.

## What you will learn

- Subgroup performance, which is where the aggregate metric hides things
- Two fairness definitions that cannot both hold, and what each costs
- The feedback loop, and the holdout that is the only defence
- A model card, written from the artefacts you already have

---

## Setup

```python
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

bundle = joblib.load("/tmp/ds_model/churn-v1.joblib")
pipe, order, thr = bundle["pipeline"], bundle["feature_order"], bundle["threshold"]
df = pd.read_parquet("/tmp/subscribers.parquet")
X, y = df[order], df["churned_next_30d"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

K = 1_000
te = X_te.copy()
te["y"] = y_te.to_numpy()
te["p"] = pipe.predict_proba(X_te[order])[:, 1]
te["selected"] = te.index.isin(set(te["p"].nlargest(K).index))

def group_table(col):
    rows = []
    for g, part in te.groupby(col):
        sel = part[part["selected"]]
        rows.append({
            col: g,
            "n": len(part),
            "base rate": round(part["y"].mean(), 3),
            "AUC": round(roc_auc_score(part["y"], part["p"]), 3),
            "selected %": round(part["selected"].mean(), 3),
            "precision": round(sel["y"].mean(), 3) if len(sel) else float("nan"),
            "recall(TPR)": round(sel["y"].sum() / part["y"].sum(), 3),
        })
    return pd.DataFrame(rows)

print(group_table("plan").to_string(index=False))
```

```text
 plan    n  base rate   AUC  selected %  precision  recall(TPR)
basic 1674      0.218 0.616       0.554      0.265        0.674
 plus  873      0.099 0.688       0.079      0.145        0.116
  pro  453      0.068 0.633       0.009      0.000        0.000
```

The overall AUC is 0.683 and the overall precision@1000 is 0.256. Neither number
appears in that table, and the table is the one that matters.

**Pro customers are never contacted.** 0.9% selected, four people out of 453,
and precision among them is **0.000** — the model put nobody useful on the list.
Their base rate is genuinely lower (6.8% against 21.8% for basic), so the
ranking is not wrong. But the business consequence is that the retention
programme systematically ignores the customers paying 399 EGP a month, whose
churn costs four times as much as a basic customer's.

That is not a bug in the model. It is a **mismatch between the model's objective
and the business's**: the model ranks by probability, the business cares about
expected revenue. Ranking by `p x monthly_fee x expected_months` would produce a
different list, and nobody would have discovered the issue without breaking the
metric down by a column.

```python
for col in ["country", "age_band"]:
    print(group_table(col).to_string(index=False))
    print()
```

```text
country    n  base rate   AUC  selected %  precision  recall(TPR)
     AE  431      0.206 0.710       0.311      0.336        0.506
     EG 1533      0.156 0.692       0.334      0.252        0.540
     MA  288      0.132 0.662       0.337      0.216        0.553
     SA  748      0.155 0.660       0.344      0.237        0.526

age_band    n  base rate   AUC  selected %  precision  recall(TPR)
   18-29  924      0.148 0.698       0.386      0.246        0.642
   30-44 1176      0.171 0.669       0.296      0.270        0.468
   45-59  599      0.147 0.683       0.344      0.223        0.523
     60+  301      0.186 0.724       0.296      0.315        0.500
```

Country looks fine: selection rates 0.311 to 0.344, AUC 0.660 to 0.710.

Age does not. **18-29 customers are selected 38.6% of the time and 30-44 only
29.6%**, although the younger group's base rate is *lower* (0.148 against
0.171). The model contacts the group that churns less, more often. Recall
follows: 64.2% of young churners are reached against 46.8% of the 30-44 group.

Whether that is unfair depends on what the action is. A retention discount is a
benefit, so under-selection is the harm, and the 30-44 group is under-served. If
the action were a credit denial, the direction of harm would reverse. **You
cannot evaluate fairness without knowing whether being selected is good or bad
for the person**, and that is a product question, not a metric.

Always break down by: the protected attributes you have, the product tiers, the
geography, and **the segments too small to see in the aggregate**. With 301
customers in 60+, a 3-point difference is noise; with 1,533 in EG it is not.
Report the group sizes so a reader can tell which is which.

---

## Two definitions of fair, and what they cost

**Demographic parity**: each group is selected at the same rate.
**Equal opportunity**: each group's *churners* are reached at the same rate.

These are different constraints and, when base rates differ between groups, they
cannot both be satisfied. That is a theorem, not a tooling limitation.

```python
overall_rate = te["selected"].mean()
caught_now = te.loc[te["selected"], "y"].sum()

quota = pd.concat([part.loc[part["p"].nlargest(
                       int(round(overall_rate * len(part)))).index]
                   for _, part in te.groupby("age_band")])

print(f"unconstrained top {K}: {caught_now} churners caught, selection rates "
      f"{te.groupby('age_band')['selected'].mean().round(3).to_dict()}")
print(f"equal-rate quota   : {quota['y'].sum()} churners caught, "
      f"{len(quota)} contacted")
print(f"cost of parity     : {caught_now - quota['y'].sum()} churners not contacted")
```

```text
unconstrained top 1000: 256 churners caught, selection rates {'18-29': 0.386, '30-44': 0.296, '45-59': 0.344, '60+': 0.296}
equal-rate quota   : 263 churners caught, 1000 contacted
cost of parity     : -7 churners not contacted
```

The quota caught **seven more churners than the unconstrained ranking**, not
fewer. The cost of fairness was negative.

Do not over-read that either: 7 out of 482 churners, on 1,000 contacts, is
inside sampling noise — a different seed would give a small number of either
sign. The honest conclusion is **"equalising selection rates by age cost nothing
measurable here"**, and the useful lesson is that "fairness always costs
accuracy" is a claim to test on your data, not a law. When the model's
unconstrained ranking is only mildly informative, a constraint that redistributes
contacts can be free.

```python
tpr = te[te["y"] == 1].groupby("age_band")["selected"].mean()
fpr = te[te["y"] == 0].groupby("age_band")["selected"].mean()
print(pd.DataFrame({"TPR (churners reached)": tpr.round(3),
                    "FPR (stayers bothered)": fpr.round(3)}).to_string())
print(f"TPR gap widest-narrowest: {tpr.max() - tpr.min():.3f}")
```

```text
          TPR (churners reached)  FPR (stayers bothered)
age_band
18-29                      0.642                   0.342
30-44                      0.468                   0.261
45-59                      0.523                   0.313
60+                        0.500                   0.249
TPR gap widest-narrowest: 0.175
```

A **17.5 point** equal-opportunity gap. A churning 18-29 customer has a 64%
chance of being offered the retention deal; a churning 30-44 customer, 47%.

| Definition | Equalises | Choose when |
|---|---|---|
| Demographic parity | Selection rate | The resource is a benefit to be shared |
| Equal opportunity | TPR among those who need it | Missing someone who needed help is the harm |
| Predictive parity | Precision | The cost of a wasted action is what matters |
| Calibration by group | `P(y\|p)` per group | The probability itself is consumed downstream |

Pick one, in writing, with the product owner, before you measure. Measuring all
four and then choosing the one you passed is the fairness version of p-hacking.

---

## The feedback loop

The model changes the world it is measuring. Suppose the retention offer works
30% of the time; those customers now do not churn, and next month's training
labels record them as stayers.

```python
rng = np.random.default_rng(42)
OFFER_WORKS = 0.30
current = df.copy()
model = clone(pipe).fit(X_tr, y_tr)
clean_te = X_te[order]

print(f"{'generation':>11}{'label base rate':>17}"
      f"{'AUC on clean test':>19}{'top-decile lift':>17}")
for gen in range(4):
    scores = model.predict_proba(current[order])[:, 1]
    treated = np.zeros(len(current), dtype=bool)
    treated[np.argsort(scores)[::-1][:4_000]] = True    # k scaled to the population
    observed = current["churned_next_30d"].to_numpy().copy()
    saved = treated & (observed == 1) & (rng.random(len(current)) < OFFER_WORKS)
    observed[saved] = 0                                  # the offer worked
    p_clean = model.predict_proba(clean_te)[:, 1]
    dec = pd.DataFrame({"p": p_clean, "y": y_te.to_numpy()})
    lift = dec.nlargest(len(dec) // 10, "p")["y"].mean() / y_te.mean()
    print(f"{gen:>11}{observed.mean():>17.4f}"
          f"{roc_auc_score(y_te, p_clean):>19.3f}{lift:>17.2f}")
    nxt = current.copy()
    nxt["churned_next_30d"] = observed
    Xn_tr, _, yn_tr, _ = train_test_split(
        nxt[order], nxt["churned_next_30d"], test_size=0.25, random_state=0)
    model = clone(pipe).fit(Xn_tr, yn_tr)
    current = nxt
```

```text
 generation  label base rate  AUC on clean test  top-decile lift
          0           0.1341              0.683             2.10
          1           0.1162              0.683             2.20
          2           0.1021              0.684             2.12
          3           0.0912              0.674             2.12
```

Read the columns against each other, because they disagree and the disagreement
is the whole point.

**The measured churn rate collapses**: 13.4% to 9.1% in four cycles, a 32%
relative fall. Someone will put that in a slide titled "retention programme
impact", and part of it is real — the offers did work — but the labelled base
rate is now a measurement of the *programme*, not of the customers. You can no
longer answer "how much churn do we have?" from this table.

**The model itself barely degrades**: AUC 0.683 to 0.674, lift flat at ~2.1 on a
clean held-out set. The ranking survives four generations of contaminated labels,
because the saved customers are removed roughly at random from within the treated
group rather than in a way correlated with the features.

So the feedback loop **corrupts your metrics long before it corrupts your
model**, and the metric it corrupts first is the one the business watches. This
is the opposite of how the failure is usually described, and it is why "the
churn rate is falling" is not evidence that anything is working.

### The holdout is the only defence

```python
rng = np.random.default_rng(42)
holdout = rng.random(len(df)) < 0.10       # never contacted, whatever the score
current = df.copy()
model = clone(pipe).fit(X_tr, y_tr)

print(f"{'generation':>11}{'all rows':>10}{'holdout 10%':>13}{'contacted rows':>16}")
for gen in range(4):
    scores = model.predict_proba(current[order])[:, 1]
    ranked = np.argsort(scores)[::-1]
    treated = np.zeros(len(current), dtype=bool)
    treated[ranked[~holdout[ranked]][:4_000]] = True
    observed = current["churned_next_30d"].to_numpy().copy()
    saved = treated & (observed == 1) & (rng.random(len(current)) < OFFER_WORKS)
    observed[saved] = 0
    print(f"{gen:>11}{observed.mean():>10.4f}{observed[holdout].mean():>13.4f}"
          f"{observed[treated].mean():>16.4f}")
    nxt = current.copy()
    nxt["churned_next_30d"] = observed
    Xn_tr, _, yn_tr, _ = train_test_split(
        nxt[order], nxt["churned_next_30d"], test_size=0.25, random_state=0)
    model = clone(pipe).fit(Xn_tr, yn_tr)
    current = nxt
```

```text
 generation  all rows  holdout 10%  contacted rows
          0    0.1352       0.1829          0.1760
          1    0.1169       0.1829          0.1260
          2    0.1044       0.1829          0.0887
          3    0.0953       0.1829          0.0650
```

**The holdout column does not move.** 0.1829 for four generations, while the
overall rate slides from 0.1352 to 0.0953. Ten percent of customers who are
never contacted, whatever their score, and the truth stays visible.

The third column is the trap the holdout saves you from. Churn among contacted
customers falls from 17.6% to 6.5%, which reads as "the model is getting worse at
finding churners" — precision collapsing — when what actually happened is that
the offers worked. Without the holdout you cannot tell those two stories apart,
and they call for opposite actions.

The holdout also gives you, for free, the number every one of these projects is
eventually asked for: the causal effect of the programme is
`holdout_rate - treated_rate` among comparable customers. It costs 10% of the
programme's reach and it is the only version of that number that is not an
argument.

---

## What this model cannot do

| Question | Answer |
|---|---|
| "Will this specific customer churn?" | No. It gives a probability over similar customers |
| "Why did it flag customer 4192?" | It can show contributing features, not a reason |
| "If we increase logins, will churn fall?" | **No.** That is causal; this is observational (Data-Analysis lesson 07) |
| "Will it work on the new enterprise plan?" | No. Zero enterprise customers in training |
| "Can we use it for a 90-day window?" | No. It was fitted for 30 days |
| "Is 0.788 really 79% risk?" | Only inside the training range; that request was outside it (lesson 08) |

Write these down, because the model will be used for all of them unless someone
says otherwise in writing.

---

## The model card

One page, generated from the artefacts you already have, living next to the code.

```python
from sklearn.metrics import average_precision_score

card = f"""
# Model card — churn-v1

## What it does
Ranks active paying subscribers by the probability of cancelling within 30 days,
to produce a weekly call list of {K:,} customers for the retention team.

## Intended use
- IN SCOPE  Weekly ranked call list, retention offers
- OUT       Pricing, credit decisions, individual performance reviews,
            any 90-day question, the enterprise plan (0 rows in training)

## Data
- {bundle['training_rows']:,} training rows, base rate {bundle['training_base_rate']}
- Features: {', '.join(bundle['feature_order'])}
- Excluded as leakage: days_to_renewal, cancellation_reason (lesson 03)

## Performance (held-out, n={len(te):,})
- ROC-AUC {roc_auc_score(te['y'], te['p']):.3f}
- Average precision {average_precision_score(te['y'], te['p']):.3f} against a base rate of {te['y'].mean():.3f}
- Precision@{K:,} {te.loc[te['selected'], 'y'].mean():.3f}, top-decile lift 2.1x
- The current hand-written rule scores 0.622 AUC; this model finds 59 more
  churners at k={K:,} (lesson 05)

## Subgroup performance
- pro plan: {float(group_table('plan').query("plan == 'pro'")['selected %'].iloc[0]):.1%} \
selected, precision 0.000 - the model effectively never contacts them
- Age 18-29 selected {tpr.loc['18-29']:.1%} of churners vs 30-44 at \
{tpr.loc['30-44']:.1%}: a {tpr.max() - tpr.min():.1%} equal-opportunity gap

## Known limitations
1. Observational. It cannot tell you what to change, only whom to call
2. Calibrated only inside the training range; extreme inputs are extrapolation
3. Highest probability seen in test data was 0.632 - it never says "certain"
4. A new plan value is silently encoded as all-zeros (lesson 04/09)
5. Fitted for a 30-day window only

## Ethical and feedback considerations
- Selection is a benefit (a discount), so under-selection is the harm
- A 10% holdout is never contacted, to keep the base rate measurable and to
  give an unbiased programme effect
- Prediction inputs and outputs are stored for 13 months

## Monitoring
- PSI per feature, unseen categories, mean prediction vs base rate, flagged
  count vs capacity; precision@k reviewed monthly once labels mature
- Alert thresholds: see lesson 09. Noise floor measured at PSI 0.0046

## Owner and review
- Owner: retention analytics. Retrain: monthly. Card reviewed: quarterly
- Trained {bundle['trained_utc']}, sklearn {bundle['sklearn_version']}
"""
print(card.strip()[:620])
print("...")
print()
print(f"card length: {len(card.strip().splitlines())} lines")
```

```text
# Model card — churn-v1

## What it does
Ranks active paying subscribers by the probability of cancelling within 30 days,
to produce a weekly call list of 1,000 customers for the retention team.

## Intended use
- IN SCOPE  Weekly ranked call list, retention offers
- OUT       Pricing, credit decisions, individual performance reviews,
            any 90-day question, the enterprise plan (0 rows in training)

## Data
- 9,000 training rows, base rate 0.1608
- Features: tenure_days, logins_last_30d, support_tickets_last_30d, payment_failures_last_90d, monthly_fee, plan, country, age_band
- Excluded as leakage: days_
...

card length: 48 lines
```

Forty-eight lines. Every number in it came from a lesson in this course, and none
of it required a separate research project — which is the argument for writing
the card as you go rather than as a deliverable at the end.

The section people skip is **Known limitations**, and it is the section that
protects you. A model used outside its scope fails, and the only question at
that point is whether the boundary was written down.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Reporting only aggregate metrics | 0.683 AUC hid that pro customers are never contacted |
| Breaking down without group sizes | A 3-point gap on n=301 is noise; on n=1,533 it is not |
| Measuring four fairness definitions and reporting the one you passed | Choose and declare it first |
| Assuming fairness must cost accuracy | It cost nothing measurable here |
| Treating a falling churn rate as proof the model works | The labels are now measuring the programme |
| No holdout | The two opposite explanations of the same numbers are indistinguishable |
| No model card | The scope boundary exists only in your head |
| Causal language from an observational model | "Increase logins to cut churn" does not follow |

---

## Exercises

1. Rank by `p * monthly_fee` instead of `p` and rebuild the `plan` table. How
   many pro customers are now contacted, and what happens to the churner count
   at k=1,000?
2. Enforce equal opportunity instead of demographic parity: pick per-group
   thresholds that equalise TPR. Report the selection rates that result, and
   which groups now exceed the budget.
3. Rerun the feedback simulation with `OFFER_WORKS = 0.8`. At which generation
   does AUC on the clean test set fall below 0.65?
4. Shrink the holdout to 2% and compute the width of the confidence interval on
   the holdout base rate. Is it still narrow enough to detect a 2-point change?
5. Write the model card for a model you have built. The limitations section must
   have at least five entries, and one of them must be a use someone has already
   asked you about.

---

**Done with the lessons.** Next: [Project 10](../Project-10/) — one problem, end
to end.
