# Project 18 — A Review and a Reproduction

**Do this after the eight lessons.**

You take a **real question you have** — at work, or from another project in this
track — search the literature systematically, review what you find, reproduce
the most relevant result on your own data, and write a recommendation.

The deliverable is a decision supported by evidence you generated yourself.

---

## Requirements

### 1. The question

- Written in lesson 05's four-part form: population, intervention, comparison,
  outcome
- **Why you have it**: the decision it would inform, and what it is worth
- What you would do today without any literature — your default

If you cannot name the decision, pick a different question. A review with no
decision behind it is an essay.

### 2. The systematic search

- The **search log** from lesson 05: date, sources, exact queries with hit
  counts, inclusion and exclusion criteria, screening counts
- **Snowballing both directions**, at least two rounds, until saturation
- A statement of what your search could have missed
- **At least 8 papers** included, with a reason for every exclusion at the
  full-read stage

### 3. Read them properly

- A **note** (lesson 02) for every included paper, with `DOUBT` and `IF TRUE`
  filled in
- At least **three** papers taken to pass 2
- For each of those three, lesson 03's **eight-question checklist**, answered

### 4. The synthesis matrix

- Papers down the side, **your** dimensions across the top
- Columns must include: reports variance, baseline tuning stated, code
  released, cost reported
- **Three findings that only became visible once the matrix existed**, written
  as sentences

### 5. The review

Lesson 06's structure, organised **by question, not by paper**:

- The landscape / taxonomy
- One section per sub-question, each ending in a claim
- A **quality-of-evidence** section: how many report variance, tune baselines,
  release code
- A **conflicts** section: where papers disagree, and your best explanation
- A **gaps** section: what nobody has tested

### 6. The reproduction

Take the single most relevant result to **degree 3** (lesson 04):

- Read their code first, and record what you learned before running anything
- Run **your own tuned baseline** on their data, with an equal search budget
- Run **their method on your data**, against your baseline, tuned equally
- Report the gap, with the explanation from lesson 04's causes table
- Report variance across at least **5 seeds**, median and spread

If the code or data is unobtainable, say so, test at degree 3 on your data
alone, and report that as the result. **"We could not reproduce X at our scale"
is a valid and valuable outcome.**

### 7. The write-up

Lesson 07's company form:

```text
RECOMMENDATION   what to do, with the number and its interval
EVIDENCE         3-5 findings, each with a number and a baseline
METHOD           six sentences, reproducible
ABLATION         which part did the work, if you built anything
LIMITATIONS      five, one of which could reverse the conclusion
COST             to build, to run, to maintain
NEXT             the experiment that would resolve the biggest doubt
```

### 8. Review it

- **Review your own write-up** in lesson 08's structure, before sharing
- Then have a colleague review it, using the same structure
- Record: what they raised, what you fixed, what you defended and why
- Classify each of their weaknesses as "wrong" or "unconvincing"

---

## Deliverables

```text
project-18/
├── RECOMMENDATION.md      the one page — primary deliverable
├── question.md            the four-part question and the decision
├── review.md              organised by question, with all six sections
├── search-log.md          queries, counts, criteria, snowball rounds
├── notes/
│   └── *.md               one note per paper, with DOUBT and IF TRUE
├── matrix.md              the synthesis matrix, plus the three findings
├── reproduction/
│   ├── README.md          what you ran, and how to run it again
│   ├── src/
│   ├── results.md         their number, your baseline, your data, 5 seeds
│   └── gap.md             the gap and its explanation
└── review-of-review.md    your self-review, and the colleague's
```

---

## Marking

| Weight | Criterion |
|---|---|
| 10% | A real question in four-part form, with the decision named |
| 15% | The search log: queries, counts, criteria, snowballing to saturation |
| 15% | Notes for every paper; lesson-03 checklist for three of them |
| 15% | The synthesis matrix, and three findings only it made visible |
| 15% | The review, organised by question, with evidence quality and gaps |
| 20% | The reproduction: your tuned baseline, your data, 5 seeds, the gap explained |
| 10% | The write-up and both reviews, with weaknesses classified |

Automatic deductions: a review organised paper-by-paper; no search log; fewer
than 8 papers; no quality-of-evidence section; a reproduction with one seed; a
reproduction that skipped running your own tuned baseline; a recommendation with
no interval; no self-review.

---

## Milestones

| Day | Done by end of day |
|---|---|
| 1 | Question in four-part form. Search log started; first queries run |
| 2 | Screening done, 8+ papers included, snowballing to saturation |
| 3 | Pass-1 notes for all; pass 2 for three, with the checklist |
| 4 | Synthesis matrix and the three findings |
| 5 | The review written, by question |
| 6 | The reproduction: baseline, method, 5 seeds, the gap |
| 7 | Write-up, self-review, colleague review, recommendation |

---

## Before you submit

- [ ] The question names a decision you actually face
- [ ] The search log has exact queries and hit counts
- [ ] Snowballing ran to saturation, in both directions
- [ ] Every included paper has a note with `DOUBT` and `IF TRUE`
- [ ] Three papers have the eight-question checklist answered
- [ ] The matrix includes variance, tuning, code and cost columns
- [ ] Three findings are stated that the matrix made visible
- [ ] The review's sections are questions, not papers
- [ ] Your own tuned baseline was run on their data
- [ ] The reproduction reports a median and spread over 5+ seeds
- [ ] The gap has an explanation from lesson 04's causes
- [ ] One limitation could reverse your conclusion
- [ ] You reviewed your own work before anyone else did
