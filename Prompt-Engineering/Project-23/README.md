# Project 23 — One Prompt, Properly

Build **one** prompt for **one** real task, and take it all the way: an eval
set, a measured improvement, an injection test, a cost figure, and a release.

One prompt. The temptation is to build a system; this project is testing
whether you can make a single prompt defensible, which is harder and rarer.

---

## Pick a task

It must be a task where **you can label the right answer**. Without that there
is no eval set, and without an eval set this project is impossible.

Good choices: classify support messages into 4-6 categories; extract fields
from an order or an invoice; decide whether a review mentions a specific
problem; tag an Arabic comment's dialect.

Bad choices: "summarise", "write a reply", "make it better". You cannot label
those, so you cannot measure them, so you cannot do this project.

---

## What you deliver

```text
prompts/<task>/
  v1.txt ... vN.txt        every version you tried, kept
  contract.json            allowed outputs, including the escape hatch
  eval.jsonl               >= 300 labelled examples
  dev.jsonl                a separate set for error analysis
  build.py                 the one function that builds the prompt
  run_eval.py              parse rate + accuracy + tokens, one command
  attacks.jsonl            your injection payloads
  CHANGELOG.md             every version with an n and a p
  REPORT.md                the numbers below
```

The two files that make this real are **`eval.jsonl`** and **`CHANGELOG.md`**.
Everything else is scaffolding.

---

## The eight requirements

| # | Requirement | The check | Lesson |
|---|---|---|---|
| 1 | **An eval set of 300+**, labelled by you, held out | `wc -l eval.jsonl`, and you can say how long it took | 06 |
| 2 | **Parse rate measured separately from accuracy** | Two numbers in the report, not one | 02 |
| 3 | **A contract with an escape hatch** that the code handles | Feed an ambiguous input; get `unclear`, not a guess | 02 |
| 4 | **At least 4 versions**, each one change, each with an n and a p | The changelog | 06 |
| 5 | **Error analysis on 20 failures**, grouped and counted | The grouping table in the report | 06 |
| 6 | **Cost measured**: prompt tokens per call, and the monthly bill | Including what your examples cost | 05 |
| 7 | **Injection tested**: at least 4 payloads, including one with no English instruction | How many flipped the answer? | 07 |
| 8 | **Released**: version in the logs, eval in CI, rollback is one file | Make a PR that worsens it; CI must fail | 08 |

Requirement 7 is the one people skip and the one that matters. **Attack your own
prompt before someone else does**, and remember lesson 07's correction: a
payload that fails has told you about the payload, not about your system. Try
the reordered version before you call it safe.

---

## The report

`REPORT.md`, with **your** numbers. Not prose about what you learned.

```text
1. THE TASK         what it decides, and what a wrong answer costs
2. BASELINE         the dumbest thing that works — a keyword rule,
                    the majority class. What does it score?
3. v1               parse rate, accuracy, tokens/call
4. ERROR ANALYSIS   20 failures, grouped, counted. The biggest group.
5. THE VERSIONS     each change, with n and p. Which survived?
6. COST             tokens/call, monthly bill at your volume,
                    and what the examples cost specifically
7. INJECTION        4+ payloads, which flipped the answer, and what
                    the worst case would have been if this prompt
                    were wired to an action
8. VERDICT          ship it, or one of lesson 08's stop-prompting rows
```

Section 2 is not optional. **If a keyword rule scores 0.78 and your prompt
scores 0.80 on 300 examples, you have built nothing** — and that is a real,
publishable finding, the same one
[Time-Series 01](../../Time-Series-and-Forecasting/lessons/01-baselines.md)
keeps producing.

Section 8 must allow "do not ship". Lesson 08's table has seven rows that are
all better answers than another prompt version.

---

## Rules

- **Every number is from a run you did.** None copied from these lessons.
- **The test set is read twice.** Once before you start, once at the end.
  Error analysis happens on `dev.jsonl`. Tuning against the set you report is
  [leakage](../../Data-Science/lessons/03-the-data-you-have.md).
- **One change per version.** A version that changed three things is
  uninterpretable, and the changelog entry will say so.
- **A result that contradicts a lesson is a better answer than one that
  agrees.** If 8 examples beat 2 on your task, show the p-value and say so.
- **Do not build a system.** No agent, no chain, no RAG. One prompt.

---

## Scoring yourself

| | |
|---|---|
| **Not done** | A prompt that works on the examples you looked at |
| **Done** | All eight requirements, all eight report sections |
| **Done well** | A version you **reverted** because the measurement said so, and a baseline comparison you report honestly even when it is unflattering |

The third row is the skill. Everyone can make a prompt better; the rare thing
is knowing, with a number, that the last three hours made it worse.

---

## Prerequisites

All eight [Prompt-Engineering lessons](../lessons/), and a task you can label.
