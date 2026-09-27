# Lesson 06 — Artifacts

**Goal:** leave behind the small files that make work reproducible, auditable
and inheritable.

## What you will learn

- The seven artifacts a finished project has
- What each one prevents
- Where they live
- The handover test

---

## The seven

| Artifact | Answers | Course |
|---|---|---|
| **README** | How do I run this? | Lesson 05 |
| **Decision records (ADRs)** | Why is it like this? | Lesson 05 |
| **Run records** | What produced this number? | Data-Science 07 |
| **Model card** | What does it do, and to whom does it fail? | Data-Science 10 |
| **Data card** | Where did the data come from, and what may I do with it? | Data-Security 02, 08 |
| **The eval set** | How do I know a change helped? | LLM 07 |
| **The one-page answer** | What should we do? | Lesson 02 |

Seven files. Most of them are under a page. Together they are the difference
between a project someone can inherit and a project someone has to redo.

---

## Run records

Every number you report should be traceable to the run that produced it.

```text
{
  "name": "logreg-C0.1",
  "utc": "2026-09-27T10:50:14Z",
  "git_sha": "4a91c02",
  "python": "3.12.2",
  "sklearn": "1.8.0",
  "data": {"rows": 12000, "sha256": "6b3f8cdb"},
  "params": {"model": "LogisticRegression", "C": 0.1, "seed": 0},
  "metrics": {"roc_auc": 0.6828}
}
```

Four groups — code, data, parameters, environment — and each answers a question
somebody will ask. Data-Science lesson 07 builds this in forty lines of standard
library.

**The line that no tool writes for you:**

```text
PRODUCTION: logreg-C0.1  (run 2026-09-27, data sha 6b3f8cdb, git 4a91c02)
            deployed 2026-10-01, owner: adam, retrain: monthly
```

Put it in the README. It is the most valuable single line in the repository,
and the one most often missing.

---

## Data cards

A model card describes the model. A **data card** describes what it learned
from, and it is the artifact that answers the questions with legal consequences.

```text
# Data card — subscribers

SOURCE        billing database, table `subscriptions`, nightly export
PERIOD        2025-01-01 to 2026-06-24
ROWS          12,000 subscribers, 1,929 positives
COLLECTED FOR billing. Reused for churn modelling, approved by [name] on [date]
PERSONAL DATA plan, tenure, country, age band. No name, no contact details
LEGAL BASIS   contract performance; DPIA reference DP-2026-11
QUASI-IDS     country + age_band + plan + tenure -> measured k = 34
EXCLUDED      days_to_renewal, cancellation_reason (recorded after the outcome)
KNOWN ISSUES  age_band is self-reported and 6% missing; country is billing
              country, not residence
RETENTION     24 months from collection; deleted by the nightly job, verified
              by the monthly audit
CONTACT       adam, data owner: [name]
```

The `COLLECTED FOR` line is the one that matters legally: data collected for
billing and reused for modelling is a purpose change, and someone has to have
approved it. The `QUASI-IDS` line is Data-Security lesson 02, measured rather
than assumed.

---

## Where artifacts live

```text
repo/
├── README.md              what, install, run, test, data, owner, limitations
├── MODEL_CARD.md          the current production model
├── DATA_CARD.md           the training data
├── decisions/
│   ├── 001-why-logistic.md
│   └── 014-threshold-from-capacity.md
├── eval/
│   ├── test.jsonl         the asset
│   └── README.md          what it covers, and what it cannot detect
├── runs/
│   └── *.json             one per training run
└── reports/
    └── 2026-10-answer.md  the one-pager that got the decision
```

**In the repository, not in a wiki, a drive or a chat.** Artifacts that live
outside version control drift from the code within weeks, and there is no way to
tell which version of the document matches which version of the model.

The exception is the one-page answer, which usually also needs to exist where
the decision-maker reads things. Write it in the repo, then paste it — the repo
copy is the one with the git history.

---

## The handover test

The real test of your artifacts is not a checklist. It is this:

```text
A competent colleague, who has never seen this project, is given the
repository and nothing else. In one day, can they:

  1. run it on new data?
  2. explain what it does to a stakeholder?
  3. tell whether a change they make is an improvement?
  4. say who it fails, and who owns it?
  5. explain why the main design choice was made?
```

Each question maps to an artifact: README, model card, eval set, model card
again, decision records.

Run this test literally, with a real colleague, before you consider a project
finished. It takes them a day and it finds more gaps than any amount of
re-reading your own documentation — because you cannot un-know what you know.

---

## The lie of "self-documenting code"

Code can document **what** it does. It cannot document:

- Why a simpler approach was rejected
- What the constants mean (0.192 came from call capacity, not the model)
- What the data means, or where it came from
- Which failure modes are known and accepted
- Who decided, and when they should revisit
- What was tried and did not work

All six of those are the expensive things to reconstruct, and none of them are
recoverable from the syntax. "Self-documenting code" is true about variable
names and false about everything a project actually needs.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Artifacts in a wiki | They drift from the code within weeks |
| No line saying which run is in production | Nobody can tell what is deployed |
| A model card with no subgroup breakdown | The aggregate hid the failure |
| No data card | The legal questions have no answers |
| The eval set undocumented | Nobody knows what it cannot detect |
| No decision records | The reasoning is gone; someone undoes it |
| Never running the handover test | You cannot see your own gaps |

---

## Exercises

1. Run the handover test on your current project with a real colleague. Write
   down every question they had to ask you.
2. Write the data card for your main dataset. How many lines could you not fill
   in?
3. Add the PRODUCTION line to your README, with the run id, data hash and git
   sha.
4. Write a README for your eval set that says what it **cannot** detect.
5. Find one number in a report you published and trace it to a run record. How
   long did it take?

---

**Next:** [Lesson 07 — Presenting](07-presenting.md)
