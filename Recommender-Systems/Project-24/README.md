# Project 24 — Ten Products

Build a recommender for a real catalogue, and report honestly on the parts
that do not work.

The modelling is the easy half. This project is graded on the three things
lessons 05-08 are about: **the customers you cannot serve, the catalogue you
stop showing, and the loop you are starting.**

---

## Pick a dataset

Any set of (user, item, timestamp) events: orders, plays, views, borrowings,
check-ins. Your own work data is best. If you have none, generate it with
[lesson 01](../lessons/01-what-a-recommender-is.md)'s `build()` and change the
parameters — a different tail exponent, more items, a faster signup rate — so
your numbers are not these numbers.

**It must have timestamps.** Without them you cannot split by time, and
[lesson 02](../lessons/02-evaluating.md) showed what the alternative costs.

---

## What you deliver

```text
recsys/
  data.py                 loading, the time split, the cold-user accounting
  models/
    popularity.py         the baseline, first
    item_item.py
    factorization.py
  evaluate.py             recall@k, NDCG@k, coverage, users scored AND skipped
  rerank.py               the coverage knob
  serve.py                two stages, a fallback chain, never an empty list
  REPORT.md               the numbers below
```

---

## The eight requirements

| # | Requirement | The check | Lesson |
|---|---|---|---|
| 1 | **Popularity baseline, measured first** | It is in the report before any model | 01 |
| 2 | **Time-based split**, and the random-split number beside it | Two numbers, and your inflation factor | 02 |
| 3 | **Two models** compared on recall@10, NDCG@10 **and coverage** | A three-column table | 03, 04 |
| 4 | **Cold users counted** — share of users *and* of events | The number your evaluation was hiding | 05 |
| 5 | **A fallback chain** that always returns something, each branch measured separately | Serve a brand-new user id; get ten products | 05 |
| 6 | **Exposure in three bands**, and the tail's share | Lesson 06's table, on your data | 06 |
| 7 | **Candidate-generation knee found** | Recall against candidate count, plotted | 07 |
| 8 | **The loop simulated** for 10+ rounds, with and without an explore slot | Two curves of tail share | 08 |

Requirement 4 is the one that separates this project from a tutorial. **Run it
before you build anything**, because if your cold share is 50% it changes what
you should build.

---

## The report

`REPORT.md`, your numbers, not these:

```text
1. THE DATA        users, items, events, sparsity, tail share,
                   median history, cold share at the split
2. BASELINE        popularity's recall@10 and NDCG@10
3. SPLIT           random vs time-based, and your inflation factor
4. MODELS          two models, three columns, against the baseline
5. COLD            share of users and of events; what each fallback
                   branch scores separately
6. EXPOSURE        the three bands; the tail's share of recommendations
7. SERVING         candidate knee, p95 latency, what you precompute
8. THE LOOP        tail share over 10 rounds, with and without explore;
                   what the explore slot cost you in recall
9. VERDICT         ship it, or not, and what it is worth per month
```

Section 9 must use money or a business quantity, not recall —
[Data-Science 06](../../Data-Science/lessons/06-evaluating-the-decision.md) is
how. *"NDCG went from 0.18 to 0.31"* is not a reason to ship anything.

---

## Rules

- **Every number is from your own run.** None copied from these lessons.
- **The baseline is reported in every table.** A model that beats popularity
  by 0.01 is a finding, and the honest write-up says so.
- **Cold users are never filtered out of the headline number.** Report
  coverage of users beside every metric.
- **A result that contradicts a lesson is better than one that agrees.** If
  factorization beats item-item on your data, show the table and say why your
  data differs.
- **No neural model.** If you want one, do the eight requirements first; you
  will probably not want one afterwards.

---

## Scoring yourself

| | |
|---|---|
| **Not done** | A model with a good recall@10 |
| **Done** | Eight requirements, nine report sections |
| **Done well** | A deliberate choice to serve a **less accurate** model — for coverage, for cold users, or for the loop — argued with both numbers |

The third row is the course. Lesson 04's ALS lost on accuracy and reached a
quarter more of the catalogue; knowing which of those your business is paid
for, and saying so with numbers, is the skill.

---

## Prerequisites

All eight [Recommender-Systems lessons](../lessons/), and a dataset with
timestamps.
