# Lesson 08 — The Security Review

**Goal:** turn seven lessons of attacks into one document a team can actually
run, before a model ships.

## What you will learn

- The supply chain nobody checks
- Retention and deletion, which are where the law lives
- Incident response for an AI system
- The review, as a form

---

## The supply chain

Everything in this course assumed the model was yours and the code was yours.
Usually neither is entirely true.

| Component | The risk | The control |
|---|---|---|
| **Downloaded weights** | A backdoored checkpoint (lesson 04) behaves perfectly on your tests | Pin the exact revision and verify the hash. Prefer known publishers |
| **Pickle files** | `pickle.load` **executes code**. A `.pkl` from the internet is a program | Use safetensors. Never unpickle an untrusted file |
| **Python packages** | Typosquats and compromised releases | Pin versions, use a lockfile, scan dependencies |
| **Public datasets** | Poisoned rows, or personal data you are not allowed to hold | Provenance, licence, and a sample read by a human |
| **Prompt templates from the internet** | Hidden instructions | Read every line, as code |
| **A model API** | The provider changes the model under you | Pin the version; monitor for drift |
| **Notebooks from a colleague** | Credentials, and code that runs on open | Read before running |

The pickle row is the one that surprises people most often. `joblib.load()` on a
file from an untrusted source is remote code execution, and it is how many
teams distribute models internally.

---

## Retention and deletion

This is where technical decisions meet the law, and where "we'll sort it later"
becomes expensive.

```text
For every dataset and every model, write down:
  WHAT       personal data does it contain, exactly which fields
  WHY        the purpose it was collected for, and the legal basis
  WHO        can access it, and how that is enforced
  WHERE      it lives, including backups, caches and laptops
  HOW LONG   retention period, and what happens at the end
  DELETE     the procedure, and how deletion is verified
```

The hard question that AI adds: **a model trained on a person's data is derived
from it.** If someone exercises a right to erasure, deleting their row does not
remove their influence on the model — lesson 03 showed the model can still be
interrogated about them.

Practical positions, in increasing order of cost:

1. **Delete the row, and retrain on the next scheduled cycle.** The usual
   answer. Document the maximum lag.
2. **Delete the row and retrain immediately** when the model is cheap to train.
3. **Machine unlearning** techniques — active research, do not depend on them.
4. **Do not train on data you may have to erase.** Aggregate, or use a cohort.

Whichever you choose, **write it down before someone asks**, because the request
arrives with a deadline attached.

---

## Incident response

An AI incident has a shape the usual playbook does not cover.

```text
1. CONTAIN     turn the model off. The kill switch from AI-Agents lesson 10,
               or the non-AI fallback from Data-Science lesson 08
2. SCOPE       which predictions, over what period, for which people?
               This is only answerable if you logged inputs and outputs with
               versions (Data-Science lesson 07)
3. PRESERVE    the model artefact, the training data snapshot, the logs
4. ASSESS      what did the wrong predictions cause? money, access, a person
5. ROLL BACK   to the previous model version, which is still on disk
6. NOTIFY      customers, regulators, the people affected, on their timelines
7. FIX         the cause - data, model, pipeline, or permissions
8. WRITE IT UP with the eval case that would have caught it, added to the suite
```

Step 2 is the one that decides how bad the week is. A team with versioned logs
answers it in an hour. A team without them has to say "we don't know which
customers were affected", which is the sentence that turns a technical incident
into a regulatory one.

Step 8 is what makes the next one cheaper: **every incident becomes a permanent
test case**, in the eval set from Data-Science lesson 07 and LLM lesson 07.

---

## The review

One page, filled in before the model ships. It is not a certificate; it is a
record of decisions and who made them.

```text
SYSTEM            name, owner, review date, next review
DECISION          what the model decides, and what it moves

DATA
  personal data   which fields, legal basis, approved by
  k-anonymity     measured k on the quasi-identifiers, if data is shared
  provenance      where every training row came from
  retention       period, deletion procedure, verified by
  dev environment synthetic / masked / real (if real, why)

MODEL
  train/test gap  and the membership-inference AUC (lesson 03)
  poisoning       who can write into training data, and what validates it
  output          label / bucket / probability, and who can see which
  extraction      rate limits, per-caller budgets, monitoring
  adversarial     is this system under adversarial pressure? evidence

SUPPLY CHAIN
  weights         source, pinned revision, hash verified
  serialisation   safetensors / pickle (if pickle, from where)
  dependencies    lockfile, scanned

OPERATIONS
  access          who can query, who can retrain, who can deploy
  logging         inputs, outputs, versions, retention of the logs themselves
  kill switch     tested by, on
  incident        who is called, and the rollback procedure

DECISIONS
  accepted risks  each one, with the name of the person who accepted it
  mitigations     each one, with an owner and a date
```

The last block is the part that matters. Security is not a state you reach; it
is a set of **accepted risks with names attached**. A review that ends with "it's
fine" has recorded nothing. One that ends with *"we accept membership-inference
AUC of 0.55 because the model returns buckets and is rate-limited — accepted by
the Head of Data on 2026-10-01"* has recorded a decision someone can revisit.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| `joblib.load` on an untrusted file | Remote code execution |
| Unpinned model revisions | A silent change you did not test |
| No retention policy | Risk that only grows |
| No plan for erasure requests | The deadline arrives anyway |
| Unversioned logs | "Which customers were affected?" has no answer |
| A review with no named risk acceptance | Nothing was decided |
| An incident with no test case added | The same incident again |
| Reviewing at the end | The data decisions were made months earlier |

---

## Exercises

1. Audit one project's supply chain: every weight file, dataset and package.
   How many are pinned, and how many are verified?
2. Search your codebase for `pickle.load` and `joblib.load`. For each, say where
   the file comes from.
3. Write the retention line for every dataset you own. Count how many you could
   not answer.
4. Run the incident tabletop: the model has been making a systematically wrong
   decision for two weeks. Answer step 2 with your current logs.
5. Complete the one-page review for a system you own. Every accepted risk needs
   a name and a date.

---

**Done with the lessons.** Next: [Project 14](../Project-14/) — attack a system,
then fix it.
