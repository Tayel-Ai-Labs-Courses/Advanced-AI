# Lesson 03 — The Record

**Goal:** write down what a system is, once, in a way that stays true.

## What you will learn

- The three records: model, dataset, decision
- What makes a record useful rather than decorative
- Where records live, and why git
- The audit question that finds the gap

---

## Three records

```text
MODEL CARD      what the model is, what it is for, where it fails
DATASET RECORD  what the data is, where it came from, what it may be used for
DECISION LOG    what the system did, to whom, when, under which version
```

They answer three different questions and most teams write only the first, or
only part of it.

| Question an auditor asks | The record that answers it |
|---|---|
| What does this system do, and how well? | Model card |
| Is it allowed to use that data? | Dataset record |
| What did it decide about me on 14 March? | Decision log |

The third is the one that cannot be written retrospectively. If the logs do
not contain the model version, that question is unanswerable forever.

---

## The model card

[Communication 06](../../Communication-and-Documentation/lessons/06-artifacts.md)
and [MLOps 08](../../MLOps/lessons/08-team-practices.md) have the template.
What governance adds is three sections that are easy to leave out:

```markdown
## Intended use, and out-of-scope use
Intended: ranking retention offers for existing customers with 3+ orders.
NOT for: first-order customers (AUC 0.612), eligibility decisions, pricing,
or any adverse action. If you want it for one of those, re-classify first
(lesson 02).

## Subgroup performance
| Segment | AUC | n | Note |
|---|---|---|---|
| < 3 orders | 0.612 | 31,000 | excluded by the campaign filter |
| corporate | 0.658 | 2,400 | monitored; below target |
| everyone else | 0.804 | 150,600 | |

## Human oversight
Agents may issue or withhold the offer. Override rate is logged and reviewed
monthly; see lesson 05 for why the rate matters.
```

**"Out-of-scope use" is the section that does the governance work.** A model
is nearly always repurposed, and a card that only says what it is for gives
the next engineer no reason to stop. Naming the uses it must *not* be put to —
with the number that justifies each exclusion — is what makes the card a
control rather than a description.

---

## The dataset record

The one that is almost always missing, and the one that blocks deployments:

```markdown
# Dataset: orders snapshot a3f8b4

Source           production orders DB, read replica, 2024-01 to 2026-08
Collected for    fulfilling orders
Used here for    training a churn model      <-- IS THIS THE SAME PURPOSE?
Legal basis      legitimate interest; DPIA ref 2026-14
Personal data    customer id (pseudonymous), postcode, order history
Special category none
Consent          not required under this basis; opt-out honoured, see below
Retention        24 months, automated deletion job `purge_orders.py`
Deletion         a customer erasure request removes rows and triggers a
                 retrain within 30 days; the previous model is NOT retained
Third parties    none. Data does not leave the EU region
Known gaps       corporate accounts under-represented (1.6% of rows,
                 4% of revenue)
```

Three lines in that record are where real problems surface.

**"Collected for" against "used here for".** Purpose limitation is the most
commonly breached principle in machine learning, because training data is
whatever was lying around. Data collected to fulfil an order was not collected
to train a model, and whether that is permitted is a question with an answer —
which someone must write down.

**"Deletion".** Most teams can delete a row. Far fewer can answer what happens
to the model that already learned from it, and
[Data-Security-for-AI 03](../../Data-Security-for-AI/lessons/03-memorisation.md)
measured that a model leaks its training data back out. "Retrain within 30
days and do not retain the old weights" is a real policy; "we deleted the row"
is not.

**"Known gaps".** Writing down that corporate accounts are 1.6% of rows and 4%
of revenue takes one line and explains a subgroup metric that would otherwise
look like a mystery for a year.

---

## The decision log

```text
PER DECISION   timestamp, subject id, model name + VERSION, input hash,
               output, threshold applied, whether a human overrode it,
               and the final action
RETENTION      long enough to answer a complaint; a high-tier system
               usually means years, which is a storage decision
NEVER          the raw personal data in plaintext with no retention limit
```

The two fields teams discover they are missing:

**The model version.** Without it, "did this get worse after the March
release?" is unanswerable, and so is "which model decided about this person?"
This is the same field [MLOps 08](../../MLOps/lessons/08-team-practices.md)
asks for, for debugging — one field, two obligations.

**Whether a human overrode it.** Without it you cannot show oversight exists,
and you cannot measure whether it works, which is
[lesson 05](05-human-oversight.md)'s entire subject.

---

## Where records live

In the repository, next to the code, in git.

```text
models/churn/
  CARD.md              the model card
  DATASET.md           the dataset record
  RISK.md              the classification from lesson 02
  EVAL.md              the numbers, with their date
  params.yaml
```

Why git and not a document system:

**Diffable.** "What changed in the model card when the model changed?" is a
`git log`. In a drive it is unanswerable.

**Reviewable.** A card update goes through the same pull request as the code,
so the person who changed the threshold is the person who updated the
threshold's justification.

**Co-located.** A record one directory from the code gets updated; a record in
another system does not, and a stale record is worse than none — it is a
confident, signed, wrong statement about a production system.

---

## The audit question

If you want to find your gaps in an afternoon, ask this about one real
decision from last month:

> **"Customer 48,219 was declined on 14 March. Why?"**

A complete answer needs: the decision log entry, the model version, the
threshold, the input, the model card for that version, the dataset record for
the snapshot it was trained on, and the name of the person who owns it.

Teams that have never tried this find the answer takes days and is partly
guesswork. **The time it takes you is your governance maturity**, and it is a
better measure than any checklist.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A model card with no out-of-scope section | The model gets repurposed with no warning |
| No dataset record | Purpose limitation is unverifiable; deployments block |
| "Collected for" never compared to "used for" | The most commonly breached principle |
| A deletion policy that ignores the trained model | The weights still remember |
| Logs without the model version | "Why was I declined?" is unanswerable forever |
| Not logging overrides | Oversight cannot be demonstrated or measured |
| Records in a drive | Not diffable, not reviewed, and eventually false |
| Writing records once | A stale record is worse than a missing one |

---

## Exercises

1. Run the audit question on one real decision. Time it.
2. Write the out-of-scope section for your most-used model.
3. Write the dataset record for your main training set. Which line was hardest?
4. Check whether your logs contain the model version. If not, add it today.
5. Compare "collected for" and "used for" on one dataset. Are they the same?

---

**Next:** [Lesson 04 — Fairness Obligations](04-fairness-obligations.md)
