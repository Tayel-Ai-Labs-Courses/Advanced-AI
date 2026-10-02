# Lesson 08 — Team Practices

**Goal:** the human half of MLOps, which is the half that decides whether the
technical half survives.

## What you will learn

- The model card, and who it is for
- Review: what to actually look at in an ML pull request
- Ownership, on-call, and the handover
- The cost of what you built, in money

---

## The model card

Every model in the registry gets one page. Not a document nobody reads — **a
page a new engineer can read in four minutes and know whether to trust the
model.**

```markdown
# Model: churn-risk v7

## What it predicts
Probability a customer stops ordering within 30 days.
Scores are consumed by the retention campaign, nightly.

## What decision it drives
Customers above 0.42 get a discount offer. Threshold chosen by the profit
curve in Data-Science 06, not by accuracy — it is asymmetric: a missed churn
costs ~340 EGP, an unnecessary offer costs ~45 EGP.

## Data
Orders 2024-01 to 2026-08, snapshot hash a3f8b4. 184,000 customers.
Split by time (Time-Series 02), last 60 days held out.

## Performance
AUC 0.781, log loss 0.479, on the frozen eval set.
Baseline (days-since-last-order, one rule): AUC 0.724.
So the model is worth 0.057 of AUC over a one-line rule.

## Where it is worse
| Segment | AUC | n |
|---|---|---|
| Customers with < 3 orders | 0.612 | 31,000 |
| Corporate accounts | 0.658 | 2,400 |
| Everyone else | 0.804 | 150,600 |

New customers are the weakest segment and 17% of the base. Do not use this
model for first-order customers; the campaign filters them out.

## Known limitations
- Trained before the September pricing change
- Assumes order history is complete; a failed sync inflates churn risk
- Not calibrated below 0.1

## Owner
Data team. On-call rota in RUNBOOK.md. Retrained weekly, Mondays 03:00.
```

Four sections do the work. **"What decision it drives"** — a model with no
decision attached should not be in production. **"Where it is worse"**, because
every model has a weakest segment and the only question is whether you found it
or a customer did. **"Known limitations"**, written by the person who built it,
while they still remember. **"Owner"**, with a name.

The baseline line matters more than it looks: *0.057 of AUC over a one-line
rule.* That number tells the next engineer whether this model is worth
maintaining at all, and
[Time-Series 01](../../Time-Series-and-Forecasting/lessons/01-baselines.md)
showed how often the honest answer is no.

[Communication lesson 06](../../Communication-and-Documentation/lessons/06-artifacts.md)
has the full template; this is the short version you will actually fill in.

---

## Reviewing an ML change

A reviewer reading a 400-line diff of pandas cannot tell whether the model got
better. So review the **evidence**, not the code:

| Look at | The question | Red flag |
|---|---|---|
| The metric change | Is it outside seed noise? | "+0.003" with one seed |
| The split | Is it still time-based / grouped? | A new `train_test_split` |
| New features | Could any use future information? | Anything with an aggregate |
| Subgroups | Did any segment get worse? | Only the aggregate reported |
| The config diff | Which parameters changed? | Changes not in the config |
| Latency | Did it regress? | Not measured |
| The data snapshot | Pinned, or `latest`? | `latest` |

The leakage question is the one that needs a human. A gate cannot see that
`avg_order_value_per_customer` was computed over the whole table including the
test period — **but a reviewer who asks "when is this value known?" about every
new feature will catch it every time.**
[Data-Science lesson 03](../../Data-Science/lessons/03-the-data-you-have.md) measured what
it costs when nobody asks.

A useful review template, three lines in the PR description:

```text
METRIC    AUC 0.781 -> 0.788 (+0.007), 5 seeds, sd 0.0021
WORSE     no segment fell by more than 0.004
WHY       added the 7-day order gap feature; days-since-last was saturating
```

If the author cannot fill those in, the change is not ready, and the reviewer's
job is to say so rather than to read the pandas.

---

## Ownership

Three questions with names as answers, written down:

```text
WHO OWNS THE MODEL      decides whether it ships, owns the metric
WHO OWNS THE PIPELINE   fixes it at 3 a.m. when the training job fails
WHO OWNS THE DECISION   the business owner of the thing the model changes
```

They are often three different people, and the third is the one teams forget.
A model whose business owner has moved teams is a model nobody will turn off
when it stops paying for itself.

**The handover test:** can someone else retrain, evaluate, and deploy this
model from the repository alone, without asking you? If not, the gap is your
documentation, and it is a bus-factor of one dressed up as job security.

A minimal, honest definition of done for a model in production:

- [ ] Lockfile and base image pinned (lesson 02)
- [ ] Pipeline re-runnable from scratch, parameters in one file (lesson 03)
- [ ] Eval gate in CI, with a tolerance (lesson 04)
- [ ] Rollback tested with a stopwatch by someone who did not build it (lesson 05)
- [ ] p95 measured and alerted, fallback tested (lesson 06)
- [ ] Input, prediction and calibration monitors, thresholds computed (lesson 07)
- [ ] Model card, runbook, and three named owners (this lesson)
- [ ] A baseline still running alongside, so you know what the model is worth

The last one is the cheapest insurance in this course. **Keep the one-line rule
in production, scoring in the shadow, forever.** The day the model breaks, you
have a fallback; every other day, you have the number that justifies the model's
existence.

---

## What this costs

MLOps is not free, and pretending otherwise is how it gets cut. From the
numbers measured across this course:

```text
WHAT IT COSTS
  CI gate per change                 ~2,520 EGP/month of compute (lesson 04)
  ~30% idle serving capacity         bought tail latency (lesson 06)
  engineering time                   the real cost, and it is not small

WHAT IT PREVENTS
  escaped bugs                       90,120 -> 16,020 EGP/month (lesson 04)
  a bad deploy                       115,200 -> 11,520 EGP (lesson 05)
  model decay                        measurable, and the cadence follows (04)
```

Two things are true at once, and a senior engineer holds both: **the gate pays
for itself about 30x**, and **the shadow-a-week option was the most expensive
choice in lesson 05.** More process is not monotonically better. Price each
piece against what it prevents, and be willing to say that a given control is
not worth it for your traffic — with a number.

That is also the answer when someone asks why the team is "doing MLOps instead
of shipping models". It is not a competing activity. It is the reason the models
you ship stay shipped.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| No model card | Nobody knows what the model is for or where it fails |
| A model card with no "where it is worse" | A customer will find it first |
| Reviewing the code instead of the evidence | Leakage is invisible in a diff |
| Reporting a metric with one seed | Data-Science 07: 0.0149 of noise |
| No business owner | Nobody will ever turn it off |
| Bus-factor of one | The handover test fails |
| No baseline running | You cannot say what the model is worth |
| Adding process without pricing it | Lesson 05's shadow-a-week: 201,600 EGP |

---

## Exercises

1. Write the model card for your most important model. Which section was
   hardest? That is where the risk is.
2. Add the three-line METRIC/WORSE/WHY template to your PR description.
3. Name the three owners of one production model. Are all three still here?
4. Run the handover test: have someone else deploy it from the repo alone.
5. Price your CI gate against your escape rate. Is it worth it?
6. Put a one-line baseline into production alongside your model.

---

## Where to go next

| Next | Why |
|---|---|
| [Project 22](../Project-22/) | Build the whole loop for one model |
| [AI-System-Design](../../AI-System-Design/) | The architecture around the model |
| [HPC-and-Cloud](../../HPC-and-Cloud/) | Where the training runs, and what it costs |
| [Data-Security-for-AI](../../Data-Security-for-AI/) | The threats this pipeline is exposed to |
| [Communication-and-Documentation](../../Communication-and-Documentation/) | The full model-card and reporting templates |
