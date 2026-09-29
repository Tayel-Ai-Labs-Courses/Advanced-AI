# Lesson 06 — Writing a Review

**Goal:** turn a pile of papers into a document that says something, rather than
a list that says "here are some papers".

## What you will learn

- The synthesis matrix
- Organising by question, not by paper
- Saying what the field disagrees about
- The structure of a review

---

## The failure mode

The most common review paper, and the least useful:

> Smith et al. (2023) proposed X and achieved 0.84 on Benchmark A. Jones et al.
> (2024) proposed Y and achieved 0.86. Chen et al. (2024) proposed Z, a variant
> of X, and achieved 0.85 on Benchmark B...

Twenty paragraphs of that is an **annotated bibliography**, not a review. It
organises by paper, which is the order you read them in, and Communication
lesson 01's rule applies: *the order things happened in is never the order to
write them in.*

A review organises by **question**, and its paragraphs are claims, not summaries.

---

## The synthesis matrix

Build this before writing a sentence. Papers down the side, **the dimensions you
care about** across the top.

```text
paper        approach     data        language  reports  code  cost     claim
                                                variance
------------------------------------------------------------------------------
Smith 23     RAG          internal    EN        no       no    not      +4 pts
                                                               stated
Jones 24     fine-tune    public      EN        3 seeds  yes   1 GPU-d  +6 pts
Chen 24      RAG+rerank   internal    ZH        no       yes   not      +5 pts
                                                               stated
Ali 25       fine-tune    public      AR        5 seeds  yes   4 GPU-h  +2 pts
Osman 25     RAG          public      AR        no       no    not      +9 pts
                                                               stated
```

The columns are chosen by **your** question, not by the papers. And the value of
the matrix is what becomes visible once it exists:

- Only two of five report variance — so three of the "improvements" are
  unevaluable (lesson 03)
- The largest claimed gain (+9) is from the paper with no variance, no code and
  no cost reported
- Only two papers evaluate in Arabic, and they disagree
- Nobody reports cost in a comparable way

**Those four sentences are the review.** They could not be written from reading
the papers one at a time.

---

## Organise by question

Your section headings should be the questions you are answering:

```text
BAD  (organised by paper)          GOOD  (organised by question)
-----------------------------      ---------------------------------------
1. Smith et al.                    1. Does retrieval beat fine-tuning?
2. Jones et al.                    2. Does the answer change for Arabic?
3. Chen et al.                     3. What does each approach cost?
4. Ali et al.                      4. How reliable is the evidence?
5. Discussion                      5. What is untested?
```

Each section then cites several papers **in service of an argument**:

> Three of five studies report retrieval outperforming fine-tuning, by 4 to 9
> points (Smith 2023; Chen 2024; Osman 2025). The two that report seed variance
> (Jones 2024; Ali 2025) both find smaller gaps — 6 and 2 points — and are the
> only two that tuned both arms equally. **The apparent advantage of retrieval
> is largest in exactly the studies that report the least about how it was
> measured.**

That last sentence is a finding. It is what a review is for, and it is invisible
to anyone reading the papers individually.

---

## Say what the field disagrees about

A review that reports consensus everywhere has usually not looked hard enough.
Three things worth an explicit section:

| Section | Contains |
|---|---|
| **Where results conflict** | Two papers, opposite findings. Why? Different data, scale, metric, or tuning? |
| **What nobody has tested** | The gap. This is where your own work goes |
| **What the evidence cannot support** | Claims that rest on one unreplicated result |

The second is the one a review is judged on. **"Five papers evaluate on English
and one on Arabic, none at production scale, and none reports cost in a
comparable way"** tells a reader exactly what to do next — which is the whole
purpose.

---

## Structure

```text
1. THE QUESTION            what you are reviewing, and for whom. One paragraph
2. SCOPE AND METHOD        the search log (lesson 05), inclusion criteria,
                           how many papers screened and included
3. THE LANDSCAPE           the taxonomy: what KINDS of approach exist
4. QUESTION 1 ... N        one section per question, each ending in a claim
5. QUALITY OF EVIDENCE     lesson 03 applied across the corpus: how many
                           report variance, tune baselines, release code
6. GAPS                    what is untested, and what you would do
7. CONCLUSION              3-5 sentences a reader can act on
8. THE TABLE               the synthesis matrix, as an appendix
```

Section 2 is what separates a review from an essay: **state the search, the
criteria and the counts.** Without it, a reader cannot tell whether your
omissions are deliberate.

Section 5 is the section only someone who has done lesson 03 can write, and it
is usually the most valuable page in the document.

---

## A review inside a company

The same structure, shorter, and with one addition: **what we should do.**

```text
QUESTION       should we use RAG or fine-tune for Arabic support answers?
SEARCHED       9 papers, 2023-2026, log attached
LANDSCAPE      three approaches: retrieval, fine-tuning, hybrid
FINDINGS       1. retrieval leads by 2-9 points, but the largest gaps come
                  from the studies with the weakest methodology
               2. only 2 of 9 evaluate in Arabic, and they disagree
               3. no study reports cost comparably
EVIDENCE       weak. 2 of 9 report variance; 4 of 9 release code
GAP            nobody has measured this at our scale, on our data
RECOMMENDATION spend one week measuring both on 200 of our own tickets
               (LLM lesson 07's eval set), rather than three weeks reading
COST           1 engineer-week, ~300 EGP of compute
```

Eight lines, and the recommendation is usually the same: **the literature has
narrowed the options; now measure on your own data.** That is not a failure of
the review — it is the review having done its job.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| One paragraph per paper | An annotated bibliography |
| No synthesis matrix | The cross-paper findings stay invisible |
| No search log | Omissions look like ignorance |
| Reporting only consensus | The disagreements are the information |
| No quality-of-evidence section | You have summarised claims, not weighed them |
| No gap section | The reader does not know what to do next |
| A conclusion that says "more research is needed" | True of everything; says nothing |

---

## Exercises

1. Build a synthesis matrix for five papers you have read. Choose the columns
   before you fill it in.
2. Write the three sentences that only became visible once the matrix existed.
3. Convert a paper-by-paper summary you have written into question-by-question
   sections.
4. Write the quality-of-evidence paragraph for your corpus: how many report
   variance, tune baselines, release code?
5. Write the eight-line company review, ending in a recommendation with a cost.

---

**Next:** [Lesson 07 — Writing Up Your Own Work](07-writing-up.md)
