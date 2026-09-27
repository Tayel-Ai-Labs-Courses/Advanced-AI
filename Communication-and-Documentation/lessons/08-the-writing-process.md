# Lesson 08 — The Writing Process

**Goal:** produce good documents reliably, instead of occasionally and late.

## What you will learn

- Writing as the last step of thinking, not the last step of work
- The three-pass method
- Templates that remove the blank page
- Getting a review that improves the document

---

## Write before you finish

The common order is: do the work, then write it up. This is backwards, and it
is why write-ups are painful.

**Write the one-page answer's skeleton at the start**, when you have a question
and no results:

```text
THE ANSWER     ________________ (blank — this is what you are going to find out)
THE DECISION   what would change if the answer were yes? if no?
EVIDENCE       what three numbers would convince a sceptic?
METHOD         what will I do?
LIMITATIONS    what could make this wrong?
```

Filling in blocks 2, 3 and 5 **before** you start is the pre-analysis plan from
Data-Analysis lesson 01 and the framing document from Data-Science lesson 02,
arriving through the writing door.

Two things happen when you do this. You discover that block 2 is empty —
*nothing would change either way* — which saves you three weeks. Or you discover
that block 3 needs a number your data cannot produce, which changes the plan on
day one instead of day twenty.

**A document you cannot outline in advance is usually a project you have not
framed.**

---

## Three passes

Do not write and edit at the same time. They use different parts of your
attention and doing both at once produces slow, mediocre prose.

```text
PASS 1 — GET IT DOWN       fast, ugly, complete. Do not stop to fix a sentence.
                           Leave [TK] where a number is missing.

PASS 2 — STRUCTURE         move blocks into the right order. Delete whole
                           paragraphs. Is the answer first? Does every
                           Evidence line have a number? This is the pass that
                           improves the document most, and the one people skip.

PASS 3 — SENTENCES         cut words, fix the claims, check every figure
                           against its source. Read it aloud.
```

Pass 2 is where documents get good. Pass 3 on a badly structured document is
polishing something in the wrong order.

Sleep between pass 1 and pass 2 if you can. Structural problems are invisible
while the work is still in your head, and obvious the next morning.

---

## Cutting

The strongest single habit: **cut 30% after pass 2.**

| Cut | Example |
|---|---|
| Throat-clearing openers | "It is important to note that..." |
| Restating the question | They asked it; they remember |
| Hedges on hedges | "may potentially be able to" -> "may" |
| The journey | "First we tried X, then Y, then Z" -> what worked, and why |
| Sentences that survive deletion | Delete them |
| Adverbs on numbers | "dramatically improved to 0.71" -> "rose to 0.71" |
| Slides you apologise for | All of them |

Test: read the document with any paragraph removed. If nothing is lost, it was
not there.

---

## Templates

The blank page is the expensive part. Keep these in the repository and copy
them.

```text
templates/
├── one-pager.md           lesson 02's six blocks
├── decision-record.md     context, decision, why, consequences, revisit
├── model-card.md          Data-Science lesson 10
├── data-card.md           lesson 06
├── readme.md              the seven sections from lesson 05
├── incident.md            what happened, scope, cause, fix, the test added
└── meeting-followup.md    decided, owner, open, next
```

A template is not a formality. It is **a list of the questions this kind of
document must answer**, and its real value is that you notice when you cannot
answer one — which is usually the most important thing you learn while writing.

---

## Getting a useful review

"Can you take a look?" produces "looks good" — which is worth nothing.

Ask for one specific thing:

| Ask | Gets you |
|---|---|
| "Is the recommendation clear in the first 30 seconds?" | Structure |
| "Which number would you attack?" | The weak claim |
| "What would you need to run this yourself?" | Missing method |
| "Where did you get lost?" | The paragraph to rewrite |
| "Would you approve this? If not, what is missing?" | The decision gap |

And review your own document by **reading it as the reader**: open it cold,
read only the first five lines, and ask what you would do. If the answer is
"keep reading to find out", the document is not finished.

---

## The standard this track holds

Every lesson in these fifteen courses follows the same rules, and you can check
them:

1. **Every code block runs**, top to bottom, in order.
2. **Every printed output is a real run**, pasted, never written by hand.
3. **When the run contradicted the draft, the prose changed** — not the number.
   Most of the interesting findings in this track arrived that way.
4. **Machine-dependent output is labelled** as such.
5. **Every claim has a number**, and every number has a source.

That is not a writing style. It is a way of being checkable, and it is the only
thing that makes technical writing worth reading.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Writing after the work | The gaps are discovered too late to fix |
| Editing while drafting | Slow, and the structure never gets attention |
| Skipping pass 2 | Polishing paragraphs in the wrong order |
| Not cutting | The reader gives up before the point |
| No templates | The blank page costs an hour every time |
| "Can you take a look?" | "Looks good" |
| Reading your own draft as the author | You cannot un-know what you know |
| A number with no source | It cannot be checked, so it will not be trusted |

---

## Exercises

1. Write the one-page skeleton for a project you have **not** started. Can you
   fill in block 2? If not, what does that tell you?
2. Take a recent document and do pass 2 only: reorder and delete, change no
   sentences. How much better is it?
3. Cut 30% from your last report. Give both versions to a colleague and ask
   which is clearer.
4. Build the templates folder in your repository this week.
5. Ask three people the five review questions about the same document. Which
   question found the most?

---

**Done with the lessons.** Next: [Project 15](../Project-15/) — document and
present a piece of work you have already done.
