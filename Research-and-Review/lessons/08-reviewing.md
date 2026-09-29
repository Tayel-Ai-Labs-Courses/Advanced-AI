# Lesson 08 — Reviewing

**Goal:** judge someone else's work usefully, and apply the same standard to
your own.

## What you will learn

- What a review is for
- The review structure
- Separating "wrong" from "unconvincing"
- Reviewing inside a team

---

## What a review is for

Not to decide whether the work is impressive. To answer two questions:

```text
1. Is the claim SUPPORTED by the evidence presented?
2. Is the evidence presented REPRODUCIBLE and correctly measured?
```

Everything else — novelty, significance, writing quality — matters less and is
judged more subjectively. A reviewer who spends their time on whether the idea
is exciting, and not on whether the baseline was tuned, has reviewed nothing.

---

## The structure

```text
SUMMARY          what the paper claims, in your own words, in 3 sentences
STRENGTHS        2-4, specific
WEAKNESSES       ordered by severity, each with what would fix it
QUESTIONS        things you could not determine from the text
RECOMMENDATION   with the reason
```

The summary is not a formality. **If you cannot write it, you have not
understood the paper**, and that is itself a finding — either about your reading
or about the paper's clarity.

And every weakness needs the second half: *"the baseline is not tuned"* is a
complaint; *"the baseline is not tuned — reporting a search over the same 20
configurations would resolve this"* is a review.

---

## What to check

Lesson 03's checklist, in review form:

```text
CLAIM
[ ] Is the claim in the abstract the claim the experiments test?
[ ] Are the contributions verifiable in the experiments section?

EVIDENCE
[ ] How many runs? Median and spread, or best-of-N?
[ ] Is the baseline tuned with the same budget? Is that stated?
[ ] Does the improvement exceed the reported variance?
[ ] Is selection done on a split separate from the reported one?
[ ] Are the ablations there, and do they support the title's claim?
[ ] Is the compute comparable across arms?
[ ] Could the test data have been seen in training?

REPRODUCIBILITY
[ ] Code and configs released?
[ ] Data version and preprocessing specified?
[ ] Hyperparameters and ranges stated?

HONESTY
[ ] Does the limitations section name something that could reverse it?
[ ] Are baseline numbers consistent with their original papers?
```

A review that works through these is useful even when it recommends acceptance,
because the authors learn which of their choices were visible.

---

## Wrong against unconvincing

The distinction that makes a review fair:

| | Wrong | Unconvincing |
|---|---|---|
| Means | The claim is false, or the method cannot do what is said | The claim may be true; the evidence does not establish it |
| Frequency | Rare | **Most weaknesses** |
| Right response | Reject, with the specific error | Ask for the measurement that would settle it |
| Phrasing | "Equation 4 does not follow from 3" | "With one seed and an untuned baseline, this gap is not distinguishable from noise" |

**Most papers are not wrong. They are unconvincing**, and the review's job is to
say precisely what would convince you. That sentence is the most useful thing a
reviewer produces, and it is what turns a rejection into a better paper.

Be concrete. "More experiments are needed" helps nobody. "Reporting 5 seeds with
the interquartile range, and tuning the baseline over the same grid, would
establish whether the 1.2-point gap is real" is a review someone can act on.

---

## Reviewing inside a team

You will do far more of this than journal reviewing: a colleague's analysis, a
model someone wants to ship, a vendor's benchmark claim.

The same two questions, plus one:

```text
1. Is the claim supported?
2. Is it reproducible?
3. Does it survive OUR data, OUR scale, OUR baseline?
```

The third is lesson 04's degree 3-4, and it is the one that decides anything. A
vendor's benchmark, a paper's result and a colleague's notebook all deserve the
same question: **what happens on our data, against our tuned baseline?**

The tone that makes this work: review the **evidence**, not the person, and
always say what would change your mind. Data-Science lesson 05's paired interval
and this course's lesson 03 give you the vocabulary to disagree precisely, which
is much easier to receive than disagreeing vaguely.

---

## Reviewing your own work

Everything above, applied to yourself, before anyone else sees it:

```text
1. Write the review of your own report, in the structure above
2. Be specific about the weakness you would raise if it were someone else's
3. Then either fix it, or state it in the limitations
```

An hour of this before sharing is the cheapest quality improvement available,
and it removes the most painful category of feedback: the objection you had
already thought of and hoped nobody would raise.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Reviewing novelty instead of evidence | The interesting question is not the checkable one |
| A weakness with no remedy | A complaint, not a review |
| Treating unconvincing as wrong | Most work is the former |
| "More experiments are needed" | Names nothing |
| Not writing the summary | You cannot review what you cannot restate |
| Reviewing the person | The evidence is the subject |
| Never reviewing your own work first | The painful feedback was avoidable |

---

## Exercises

1. Review a paper using the structure. Write the summary from memory first.
2. For each weakness you list, write the measurement that would resolve it.
3. Classify your weaknesses as "wrong" or "unconvincing". What is the ratio?
4. Review your own last report as if it were a stranger's. What did you find?
5. Review a vendor's benchmark claim with the three team questions.

---

**Done with the lessons.** Next: [Project 18](../Project-18/) — a review and a
reproduction, on a question you actually have.
