# Lesson 01 — Governance Is an Engineering Problem

**Goal:** stop treating compliance as paperwork that arrives after the model.

## What you will learn

- What governance actually asks of you
- The four obligations, and where each lands in your repository
- Why "we will document it later" fails
- The inventory, which is where every programme starts

---

## The claim

Most engineers meet AI governance as a legal document and a form. That framing
is wrong and expensive, because **every obligation in it resolves to a
technical artefact you either have or do not have.**

```text
LEGAL ASKS                          YOU EITHER HAVE OR DO NOT HAVE
"what does the system do?"          a model card
"what data was it trained on?"      a dataset record and a snapshot hash
"is it accurate?"                   an eval set and a number, by subgroup
"can a human intervene?"            an override path, and its measured use
"what happened on 14 March?"        logs with a model version
"can you delete this person?"       a deletion path that reaches the backups
"who is responsible?"               a name
```

Not one of those is paperwork. They are a `MODEL_CARD.md`, a `params.yaml`, an
eval set in CI, an override endpoint, a logging schema, a deletion job, and an
owner — most of which [MLOps](../../MLOps/) already told you to build for
entirely practical reasons.

**That is the useful insight of this whole course:** a well-engineered system
is already 80% compliant, and a compliant system is usually better engineered.
Governance and good practice point the same way far more often than the
discourse suggests.

---

## The four obligations

Whatever regime applies to you — the EU AI Act, GDPR, sector rules, or your
customer's procurement questionnaire — the asks reduce to four:

| Obligation | The question | Your artefact |
|---|---|---|
| **Transparency** | What is it, what does it do, where does it fail? | Model card, dataset record, user-facing disclosure |
| **Accountability** | Who decided, who is responsible, what changed? | Owners, changelogs, versioned logs, audit trail |
| **Fairness** | Does it treat groups differently, and is that justified? | Subgroup metrics, a documented justification |
| **Human agency** | Can a person understand, contest and override it? | Explanations, an override path, an appeal route |

[Lesson 02](02-classifying-risk.md) decides how much of each you owe.
Lessons 03-07 build them. [Lesson 08](08-running-the-process.md) runs them.

---

## Why "later" fails

Three reasons, in increasing order of how much they cost:

**The information decays.** Six months after training, nobody remembers which
snapshot was used, why that threshold was chosen, or what the model was tested
on. Writing it down in March takes twenty minutes; reconstructing it in
September takes a week and produces a document that is partly fiction.

**Some obligations change the design.** "A human can override it" is an
endpoint, a UI, a permission model and a log. "We can delete one person's
data" is an architecture decision that reaches your backups and your feature
store. Discovering either after launch means a rebuild, not a document.

**The absence is the finding.** An auditor who asks for subgroup accuracy and
is told "we have not measured it" has found the problem. Not having the number
is worse than having a bad one, because a bad number can be fixed and an
absence means nobody was looking.

---

## The inventory

Every governance programme starts in the same place, and almost no team has
it: **a list of every model in production.**

```text
inventory.yaml
  - name: churn-risk
    owner: data-team / Nour            # a person, not a team, is better
    decision: who gets a retention offer
    users affected: ~18,000/month
    risk tier: limited                 # lesson 02
    model card: prompts/churn/CARD.md
    eval set: 2,400 rows, refreshed 2026-08
    subgroup metrics: yes, by plan and tenure
    human override: yes, agent can decline the offer
    data: orders snapshot a3f8b4, 24-month retention
    last reviewed: 2026-09-14
```

Build this before anything else, because three things fall out of it
immediately and none of them require a lawyer:

**You will find systems nobody owns.** A model whose author left, still
scoring, still acting. That is the highest-risk item in most organisations and
the inventory is how you find it.

**You will find high-risk systems you did not think of.** The spreadsheet that
decides who gets called is an automated decision system, and so is the rule
engine someone wrote in 2019.

**You will find the gaps cheaply.** An inventory with empty "eval set" and
"subgroup metrics" columns is a work plan.

Keep it in git, next to the code. A governance artefact in a drive that nobody
can diff is a governance artefact nobody maintains
([Communication 06](../../Communication-and-Documentation/lessons/06-artifacts.md)).

---

## What this course is not

**It is not legal advice**, and it does not reproduce any statute. Regimes
differ by country and change; your legal team owns the interpretation.

What this course owns is the engineering half: **what to build, what to
measure, and what the measurements actually mean** — including the several
places where a system passes the standard test and is still doing harm, which
is what [lesson 04](04-fairness-obligations.md) measures.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Treating governance as paperwork | Every obligation is an artefact you build |
| Documenting after launch | The information has already decayed |
| No inventory | You cannot govern systems you cannot list |
| An inventory in a drive | Not diffable, not reviewed, not maintained |
| A team as the owner | Teams do not answer questions; people do |
| Counting only ML models | The rule engine makes automated decisions too |
| Waiting for legal to ask | The absence of a measurement is itself the finding |

---

## Exercises

1. Write the inventory for your organisation. How many systems? How many have
   a named owner?
2. For one system, fill every field. Which were you unable to fill?
3. Find one automated decision in your organisation that is not a model.
4. Ask who can override your highest-impact system. Get a name.
5. Pick one obligation from the table and find the artefact that satisfies it.
   If there is none, that is your first task.

---

**Next:** [Lesson 02 — Classifying Risk](02-classifying-risk.md)
