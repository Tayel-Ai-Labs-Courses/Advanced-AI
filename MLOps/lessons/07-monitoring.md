# Lesson 07 — Monitoring and Incidents

**Goal:** find out that the model is wrong before the business does.

## What you will learn

- The four things to monitor, in order of how often they break
- Training/serving skew, measured — including one failure AUC barely notices
- Where to put an alert threshold
- What an ML incident actually looks like

---

## The four layers

```text
                          breaks            caught by
1. THE SERVICE       weekly            ordinary monitoring
2. THE INPUTS        monthly           schema + range checks
3. THE PREDICTIONS   quietly           distribution of outputs
4. THE OUTCOMES      slowly            labels, which arrive late
```

Teams build layer 1, occasionally layer 2, almost never 3, and layer 4 is where
the money is. Note the ordering: **the things that break most often are the
cheapest to catch**, and the thing that matters most is the slowest to see.

Layer 4's problem is the delay. If a loan defaults after 90 days, your accuracy
for September is knowable in December. So layers 2 and 3 are not optional — they
are the only signal you have inside the label delay, and
[Data-Science lesson 09](../../Data-Science/lessons/09-monitoring-and-drift.md)
is how to read them.

---

## Training/serving skew

The single most common cause of "the model worked in testing and is bad in
production" is that **the features at serving time are not the features it was
trained on.**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

rng = np.random.default_rng(0)
N = 20000
x = rng.normal(size=(N, 4))
y = (x @ [1.2, -0.8, 0.5, 0.0] + rng.normal(scale=1.0, size=N) > 0).astype(int)
tr, te = slice(0, 15000), slice(15000, N)

mu, sd = x[tr].mean(0), x[tr].std(0)       # fitted on train only
Xtr, Xte = (x[tr] - mu) / sd, (x[te] - mu) / sd
model = LogisticRegression(max_iter=1000).fit(Xtr, y[tr])

def ev(xe):
    p = model.predict_proba(xe)[:, 1]
    return roc_auc_score(y[te], p), log_loss(y[te], p)

cases = [("none (identical transform both sides)", Xte),
         ("serving skips the scaling step", x[te]),
         ("serving recomputes mean/sd on live data",
          (x[te] - x[te].mean(0)) / x[te].std(0))]
xe = Xte.copy(); xe[:, 0] = 0.0
cases.append(("top feature arrives null, filled with 0", xe))
cases.append(("two columns swapped (same names, wrong order)", Xte[:, [1, 0, 2, 3]]))
xe = Xte.copy(); xe[:, 0] *= 24
cases.append(("one feature sent in different units (x24)", xe))

base_auc, base_ll = ev(Xte)
print(f"{'skew at serving time':<46}{'AUC':>7}{'log loss':>10}")
for name, xe in cases:
    a, l = ev(xe)
    print(f"{name:<46}{a:>7.4f}{l:>10.4f}")
print(f"\nbaseline AUC {base_auc:.4f}, log loss {base_ll:.4f}. "
      f"One model, six serving paths.")
```

```text
skew at serving time                              AUC  log loss
none (identical transform both sides)          0.9001    0.3964
serving skips the scaling step                 0.9000    0.3966
serving recomputes mean/sd on live data        0.8999    0.3970
top feature arrives null, filled with 0        0.7387    0.6244
two columns swapped (same names, wrong order)  0.2135    1.9706
one feature sent in different units (x24)      0.8093    5.6361

baseline AUC 0.9001, log loss 0.3964. One model, six serving paths.
```

**The same trained model, six serving paths.** Three findings.

**Swapping two columns takes AUC from 0.9001 to 0.2135.** That is far *worse*
than random — the model is now confidently inverted, because it is reading
feature 1's values through feature 0's coefficient. Columns matched by position
instead of by name is a one-line bug with a catastrophic blast radius, and
nothing in your test suite will see it unless you assert on names.

**The null-filled feature costs 0.16 of AUC.** A feature that is usually present
and occasionally missing gets filled with 0 by a well-meaning default, and 0 is
a perfectly plausible value after scaling — the mean. So the model is told
"average customer" about its most important signal, confidently, with no error
anywhere.

**The units bug is the one to stare at.** One feature arrives multiplied by 24
— days where the model expected hours. **AUC falls only to 0.8093, a 10%
relative drop that a weekly dashboard would shrug at. Log loss goes from 0.3964
to 5.6361 — 14x worse.** The ranking is mostly intact; the probabilities are
garbage. If anything downstream uses the probability — a threshold, an expected
value, a price
([Data-Science 06](../../Data-Science/lessons/06-evaluating-the-decision.md)) —
it is now wrong, and your AUC dashboard is green.

**So monitor a calibration metric, not only a ranking metric.** AUC is
invariant to any monotone transform of the scores, which is exactly why it
cannot see this class of bug.

Honest caveat about rows 2 and 3: the scaling skews barely register here because
these features are already roughly standard normal, so dividing by the wrong
`sd` changes little. On real features with ranges like `[0, 1e6]`, skipping the
scaling step is as bad as the units bug. The lesson is not "scaling skew is
harmless" — it is that **you cannot tell from AUC which of these is happening.**

The fix is structural, not vigilance: **the same code computes the feature in
both places.** One function, imported by the training pipeline and by the
service. A feature store is one way to enforce this; a shared module is another
and is usually enough.

---

## Where to put the threshold

An alert fires when a number crosses a line. Put the line too tight and
everyone mutes the channel; too loose and it never fires.

```python
import numpy as np

rng = np.random.default_rng(3)
BASE = 0.02
N_DAY = 20_000
DAYS = 365
sd = np.sqrt(BASE * (1 - BASE) / N_DAY)        # binomial standard error
print(f"daily error rate: mean {BASE:.4f}, sd {sd:.5f} at {N_DAY:,} requests/day")

print(f"\n{'threshold':>11}{'false alarms/yr':>18}{'days to catch 2.4%':>21}")
for k in [1, 2, 3, 4]:
    thr = BASE + k * sd
    quiet = rng.binomial(N_DAY, BASE, DAYS) / N_DAY        # a healthy year
    fa = (quiet > thr).sum()
    bad = rng.binomial(N_DAY, 0.024, 2000) / N_DAY          # a 20% regression
    detect = (bad > thr).mean()
    days = 1 / detect if detect > 0 else float("inf")
    print(f"{f'mean+{k}sd':>11}{fa:>18}{days:>21.2f}")
```

```text
daily error rate: mean 0.0200, sd 0.00099 at 20,000 requests/day

  threshold   false alarms/yr   days to catch 2.4%
   mean+1sd                56                 1.00
   mean+2sd                12                 1.03
   mean+3sd                 1                 1.19
   mean+4sd                 0                 2.00
```

**mean+1sd cries wolf 56 times a year and catches the real regression in 1.00
days. mean+3sd cries wolf once a year and catches it in 1.19 days.**

Fifty-five fewer false alarms for one fifth of a day of detection delay. That is
not a trade-off; that is a free lunch, and the reason is that a 20% regression
(2.0% → 2.4%) is **four standard errors** away — far outside the noise band. The
tight threshold is spending all its sensitivity on noise it cannot help seeing.

Two things that generalise:

**Compute the standard error before choosing a threshold.** It is one line:
`sqrt(p(1-p)/n)`. A threshold chosen by eye is either inside the noise or
hopelessly outside it, and you will not know which.

**Only alert on things worth waking someone for.** 56 pages a year for a metric
that recovers by itself is how a team learns to ignore the alerting channel —
after which the one real page is also ignored. Route everything else to a daily
report.

Note what the fourth row shows too: mean+4sd never false-alarms and takes twice
as long. There is a real floor to how loose you can go.

---

## The runbook

Monitoring without a runbook produces an alert nobody can act on. One page per
alert, written **before** the incident:

```text
ALERT        prediction error rate > mean + 3sd for 2 consecutive hours

MEANS        either the model is degraded, or an input feature is broken,
             or traffic composition changed

FIRST        check the input monitors. A broken feature is 10x more likely
             than a degraded model. Compare today's feature distributions
             against the training snapshot.

THEN         check the deploy log. Did anything ship in the last 24 hours?
             Code, model, feature pipeline, or an upstream service.

ROLLBACK     alias the previous model version. Under 5 minutes. Lesson 05.
             Rolling back is not an admission of anything; it buys time.

DO NOT       retrain in response to this alert. Data-Science 09: diagnose
             first. Retraining on broken data bakes the bug into the weights.

ESCALATE     if error rate > 2x baseline, or if revenue impact > X EGP/hour
```

The two lines that save you are **"check the inputs first"** — because a broken
feature really is the more likely cause — and **"do not retrain"**, because
retraining feels productive and is the one action that can make a recoverable
incident permanent.

---

## The post-incident question

After it is fixed, one question matters: **what monitor would have caught this
an hour earlier?** Then write that monitor. An incident that produces a fix and
no new monitor will happen again, and the second time nobody will remember.

Keep the list short. A dashboard with 40 panels is read by nobody; the four
layers above, with a threshold computed for each, is read every morning.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Features computed by different code in training and serving | 0.9001 → 0.2135 from one swap |
| Matching feature columns by position | Same names, wrong order, silently inverted |
| Filling a missing feature with 0 after scaling | 0 is the mean — confidently average |
| Monitoring AUC only | The units bug: AUC 0.8093, log loss 14x worse |
| A threshold chosen by eye | Either inside the noise or useless |
| Alerting on anything that is merely interesting | 56 pages a year trains people to ignore the channel |
| Monitoring with no runbook | An alert nobody can act on |
| Retraining in response to an alert | Bakes a data bug into the weights |
| An incident that produces no new monitor | It will recur |

---

## Exercises

1. Find one feature in your system computed by two different code paths. Make
   it one function.
2. Add an assertion that serving features match training features **by name**.
3. Add a calibration metric (log loss or Brier) next to your ranking metric.
4. Compute `sqrt(p(1-p)/n)` for your error rate and set your threshold at 3sd.
5. Count last year's alerts. How many were actionable?
6. Write the runbook for your noisiest alert.

---

**Next:** [Lesson 08 — Team Practices](08-team-practices.md)
