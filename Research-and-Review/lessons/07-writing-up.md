# Lesson 07 — Writing Up Your Own Work

**Goal:** write a report or a paper that survives the scepticism of lesson 03.

## What you will learn

- The structure, and what each section owes the reader
- Reporting results honestly enough to be believed
- Ablations, and what they are for
- The limitations section as an asset

---

## The structure

```text
ABSTRACT       the whole paper in 150 words: problem, method, result, meaning
INTRODUCTION   why this matters, what is missing, what you did, contributions
RELATED WORK   lesson 06, compressed: what exists, and where you differ
METHOD         the ONE idea, then the engineering
EXPERIMENTS    setup, baselines, results, ABLATIONS
LIMITATIONS    what you did not test, and what would change the conclusion
CONCLUSION     what a reader should now do differently
```

Communication lesson 02's rule holds here too: **the abstract is the one-page
answer**, and most readers read only it. Write it last and make it carry the
result, with its number.

The `CONTRIBUTIONS` list at the end of the introduction is what reviewers
actually check. Three or four bullets, each one something a reader can verify in
your experiments section — and nothing else.

---

## Reporting results so they can be believed

Everything lesson 03 taught you to look for, you now have to supply:

```text
[ ] N runs, with the MEDIAN and the spread — never the best
[ ] The baseline tuned with the SAME search budget as your method, stated
[ ] The search budget itself: how many configurations, over what ranges
[ ] The split used for selection, separate from the one reported
[ ] Compute: GPU type, hours, and parameters, for every arm
[ ] The exact data version and preprocessing
[ ] Code and configs released
[ ] A contamination check, if the test data could have been seen
```

The single most credibility-raising line you can write:

> All methods were tuned over the same 20-configuration random search, and all
> results are the median of 5 seeds with the interquartile range in brackets.

A reviewer who reads that trusts the rest of the paper immediately, because it
answers the four questions they were going to ask.

And the corollary: **if your improvement does not survive being reported this
way, it was not an improvement.** Finding that out yourself, before submission,
is the cheapest possible version of that discovery.

---

## Ablations

An ablation removes one component and re-measures. It is the difference between
"our system works" and "we know why".

```text
full method                      0.842  (0.838-0.845)
  - without the reranker         0.839  (0.835-0.842)   <- the reranker does nothing
  - without the new loss         0.818  (0.814-0.821)   <- this is the contribution
  - without the extra data       0.831  (0.828-0.834)
baseline                         0.815  (0.812-0.819)
```

That table says the paper's real contribution is the loss function, and that the
reranker — which may be in the title — contributes 0.003, inside the spread.

**Write the ablation table before you write the introduction.** It very often
changes what the paper is about, and it is far better to find that out yourself
than to have a reviewer find it.

If you cannot ablate — some methods genuinely cannot be decomposed — say so
explicitly. That is a limitation, not a secret.

---

## The limitations section

Reviewers and readers trust papers that state their own weaknesses, for the same
reason Data-Analysis lesson 08 gives: a document that claims everything is
trusted with nothing.

A real limitations section:

```text
1. Evaluated only on English and Arabic; we do not know whether it transfers
2. All experiments at 7B; the effect may not hold at larger scale
3. The 30% offer-acceptance assumption in the cost analysis is not measured
4. We could not run a contamination check against the pretraining corpus
5. The baseline was tuned over 20 configurations; a larger budget might close
   the gap
```

Every line names something specific that could change the conclusion. Compare
with the version that says "future work includes larger models and more
datasets", which says nothing and costs you the reader's trust.

**Item 5 is the brave one** — admitting that more baseline tuning might erase
your result. Papers that include it are the ones worth reading.

---

## Writing for a company instead of a venue

Most of your write-ups will not be papers. The same content, reordered for a
reader who must act:

```text
RECOMMENDATION   what to do, with the number and its interval
EVIDENCE         3-5 findings, each with a number and a baseline
METHOD           six sentences, reproducible
ABLATION         which part actually did the work
LIMITATIONS      five, one of which could reverse the conclusion
COST             to build, to run, to maintain
NEXT             the experiment that would resolve the biggest doubt
```

That is Communication lesson 02's one-pager, with an ablation row added. The
ablation belongs in a business document for the same reason it belongs in a
paper: **it tells the reader which part to keep when they inevitably simplify
it.**

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Reporting the best run | Lesson 03; a reviewer will ask |
| Not stating the baseline's tuning budget | The first thing a careful reader checks |
| No ablations | Nobody knows which part worked, including you |
| Writing the introduction before the ablation table | The paper is often about something else |
| A limitations section of generic future work | Costs trust, buys nothing |
| An abstract without the number | Most readers read only the abstract |
| Contributions that are not verifiable in the experiments | Reviewers check these one by one |
| Releasing no code | The strongest credibility signal, unused |

---

## Exercises

1. Write the abstract for your last project in 150 words, including the number
   and its interval.
2. Build the ablation table. Does it change what the project is about?
3. Write the results paragraph in the form that answers all eight checklist
   items.
4. Write five real limitations, one of which could reverse your conclusion.
5. Take a claim from your own work and try to disprove it for an hour. What
   survived?

---

**Next:** [Lesson 08 — Reviewing](08-reviewing.md)
