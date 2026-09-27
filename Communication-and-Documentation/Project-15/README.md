# Project 15 — Document and Present Real Work

**Do this after the eight lessons.**

You do **no new modelling** in this project. You take a piece of work you have
already finished — a project from this track, or something from your job — and
produce every artifact for it, then put it in front of a real person.

The deliverable is a repository someone else can inherit, and a decision that
was actually made.

---

## Requirements

### 1. Choose the work, and the four readers

- The project, in one sentence
- **Name the four readers** (lesson 01): the decision-maker, a peer, an
  implementer, and future you. Real names where possible
- For each, what you want them to do after reading

If you cannot name a decision-maker, pick different work. A project with no
reader is a project with nothing to communicate.

### 2. The one-page answer

Lesson 02's six blocks, on one page:

- **The answer**, one sentence, with a number and its interval
- **The decision**: what, who, by when
- **Evidence**: 3-5 findings, **every sentence containing a number**, with the
  baseline named
- **Method**: six sentences, reproducible
- **Limitations**: five, and **one must be able to reverse the conclusion**
- **Next**: the follow-up, with a review date

Then apply the headline test: read only the first line and ask whether a reader
could act.

### 3. Numbers, checked

- Every headline number has an **interval**
- Every percentage has its **base** and its **absolute** change
- Every comparison passes lesson 03's five questions (same data, same metric,
  same budget, baseline named, interval computed)
- The smallest difference your data can detect, computed and stated
- Every instance of "significant", "improved", "approximately" and "could"
  removed or replaced with a number

### 4. Four charts

Each with:

- A **title that states the finding**
- Labelled axes with units
- **Zero baseline** on every bar chart
- No partial periods
- n shown wherever a rate is plotted
- Readable in greyscale
- The command that regenerates it

And one chart you **deleted** because a table was better. Include the table.

### 5. Code documentation

- A README with **all seven sections** from lesson 05, and every command run
  from a clean checkout to prove it works
- Docstrings on every public function, answering: what it returns, what it
  assumes, when it fails
- The docstring audit run, with the before and after coverage numbers
- **Five comments deleted** for restating the code, listed
- **Three comments added** that a reader could not derive from the code, listed
- **At least two decision records**, each with a "what would change this" section

### 6. The artifacts

All seven from lesson 06, in the repository:

- README, decision records, run records, model card, data card, eval set, the
  one-pager
- The **PRODUCTION line** in the README naming the deployed run
- The model card with a **subgroup breakdown** and five limitations
- The data card with source, purpose, legal basis, measured k, and retention
- The eval set's README saying what it **cannot** detect

### 7. Present it

- Five slides, built to lesson 07's rules
- Every slide title a finding with a number
- A demo, run locally, **with the 90-second backup video recorded**
- Rehearsed standing up, once
- Presented to a **real person** who could act on it
- The **hostile question** you expected, and the answer you gave
- The four-line follow-up sent within a day

### 8. The handover test

This is the part that cannot be faked.

- A **real colleague**, who has not seen the project, gets the repository and
  nothing else, for one day
- They attempt the five questions from lesson 06
- **Write down every question they had to ask you.** Each one is a gap
- Fix the gaps and note what you fixed

---

## Deliverables

```text
project-15/
├── ANSWER.md              the one page
├── README.md              seven sections, commands verified from a clean checkout
├── MODEL_CARD.md
├── DATA_CARD.md
├── decisions/
│   ├── 001-*.md
│   └── 002-*.md
├── figures/
│   ├── *.png              four charts, titles stating findings
│   └── regenerate.sh
├── eval/
│   └── README.md          what it covers and what it cannot detect
├── runs/*.json
├── templates/             the seven templates, for reuse
├── presentation/
│   ├── slides.pdf         five slides
│   ├── demo.mp4           the backup video
│   ├── questions.md       what was asked, including the hostile one
│   └── followup.md        decided, owner, open, next
└── handover/
    ├── questions-asked.md every question the colleague had to ask
    └── gaps-fixed.md
```

---

## Marking

| Weight | Criterion |
|---|---|
| 15% | The one-page answer: answer first, numbers in evidence, a reversing limitation |
| 10% | Four named readers, and a document shaped for each |
| 15% | Numbers: intervals, bases, the five comparison questions, the detectable difference |
| 10% | Charts: findings as titles, zero baselines, n shown, one replaced by a table |
| 15% | Code documentation: seven-section README verified, docstring audit, decision records |
| 15% | The seven artifacts present, including the PRODUCTION line and the data card |
| 10% | Presented to a real person, with the backup video and the follow-up |
| 10% | The handover test run with a real colleague, gaps listed and fixed |

Automatic deductions: a headline number without an interval; a percentage
without a base; a truncated bar axis; a chart titled with column names; a README
command that does not work from a clean checkout; a limitations section where
nothing could reverse the conclusion; no decision records; a handover test where
no questions were recorded.

---

## Milestones

| Day | Done by end of day |
|---|---|
| 1 | Readers named. One-page answer drafted (pass 1) |
| 2 | Numbers checked: intervals, bases, comparisons. Pass 2 restructure |
| 3 | Four charts, one table. Regeneration script |
| 4 | README verified from a clean checkout; docstrings; decision records |
| 5 | Model card, data card, run records, eval README |
| 6 | Slides, demo video, rehearsal. Present it |
| 7 | Handover test with a colleague; fix the gaps; send the follow-up |

---

## Before you submit

- [ ] The first line of `ANSWER.md` passes the headline test
- [ ] Every sentence in Evidence contains a number
- [ ] One limitation could reverse the conclusion
- [ ] Every headline number has an interval
- [ ] Every percentage has a base and an absolute change
- [ ] Every bar chart starts at zero
- [ ] Every chart title states a finding
- [ ] One chart was replaced by a table, and both are included
- [ ] Every README command was run from a clean checkout
- [ ] The docstring audit shows a before and an after number
- [ ] Five comments deleted, three added, both lists included
- [ ] Two decision records with "what would change this"
- [ ] The README has the PRODUCTION line
- [ ] The backup video exists
- [ ] A real person heard it and the follow-up was sent
- [ ] A real colleague ran the handover test and their questions are written down
