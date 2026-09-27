# Project 16 — Design It Before You Build It

**Do this after the eight lessons, and ideally before your next real project.**

You design a system on paper, put the design through a review with a real
engineer from another discipline, then build **only** the part you promised —
and check the promise.

The deliverable is a design that someone else could implement, and evidence
that the numbers in it were true.

---

## Requirements

### 1. The problem, and the baseline

- The decision the system supports, who makes it, how often, what it is worth
- **The non-AI baseline**, described precisely enough to implement
- What a wrong answer costs, in each direction
- Who consumes the output: name the team or the person

### 2. The four diagrams

All four, as Mermaid, in the repository:

- **Context** — actors and external dependencies
- **Container** — every deployable unit and every store
- **Sequence** — the main request, with what is sync, what is async, and real or
  estimated latencies on each call
- **Data flow** — personal data marked, retention on every store, trust
  boundaries drawn

Every arrow labelled with protocol and latency. **At least four dotted failure
arrows.**

### 3. The contract

The ten-line contract from lesson 03, plus:

- The response schema, including `band`, `degraded`, `source` and `model_version`
- The error schema, including a machine-readable code and `retryable`
- **Every UI state** a consumer must build, listed
- The versioning policy, and what happens to consumers when you retrain

### 4. The numbers

- A **latency budget**: a p95 promise, a line per stage with an owner, and
  explicit headroom
- Which stage owns the largest share, and **what halving each stage would buy**
- **Capacity**: service time, single-worker capacity, workers needed at your
  peak-minute traffic to keep queue wait under a stated bound
- **Availability**: your chain's arithmetic from your dependencies' numbers, and
  whether your promise is achievable
- State clearly which numbers are **measured** and which are **estimated**

### 5. The failure table

One row per component: detect, respond, **and what the user sees**. Plus:

- Timeouts derived from the budget, not from habit
- A bounded queue or load-shedding rule
- A circuit breaker on the slowest dependency
- The degraded response, with its flag

### 6. State and ownership

- Every store, with its **single owner** and its retention period
- Every cache, with its TTL and a sentence on why that staleness is acceptable
- A prediction/query log design: fields, retention, and what it enables
- Any fact currently owned by two components, and your resolution

### 7. The review

- Run lesson 08's agenda, for sixty minutes
- **With at least one person who is not an ML engineer** — frontend, mobile,
  backend or privacy. This is the requirement, not a suggestion
- Record: **AGREED**, **OPEN** (with names and dates), **CHANGED**
- Write the trade-off record for the biggest decision, with a `Revisit at`
  condition

### 8. Build the promise, and check it

Build the **thinnest slice** that lets you test one number from section 4:

- The endpoint with its real contract, or the one stage that owns the budget
- Measure the p95 under realistic load
- **Compare measured against designed**, and explain any gap over 20%
- Update the diagram and the budget with the measured numbers

A design whose numbers are never checked is fiction with boxes.

---

## Deliverables

```text
project-16/
├── DESIGN.md              the narrative: problem, decisions, trade-offs
├── diagrams/
│   ├── 01-context.md
│   ├── 02-container.md
│   ├── 03-sequence.md
│   └── 04-data-flow.md
├── contract.md            the ten lines, schemas, UI states, versioning
├── budget.md              latency, capacity, availability — with owners
├── failure-table.md       including the "user sees" column
├── decisions/
│   └── 001-*.md           trade-offs, with Revisit at
├── review/
│   ├── agenda.md
│   └── outcome.md         AGREED / OPEN / CHANGED
└── slice/
    ├── src/               the thin slice you built
    └── measured.md        designed vs measured, with the gap explained
```

---

## Marking

| Weight | Criterion |
|---|---|
| 10% | Problem, baseline, named consumers |
| 20% | Four diagrams, every arrow labelled, four failure arrows |
| 15% | The contract: band, degraded, model_version, errors, UI states |
| 20% | The numbers: budget with owners, capacity, availability arithmetic |
| 10% | Failure table with the "user sees" column |
| 10% | State ownership and retention on every store |
| 10% | A review with a non-ML engineer, with AGREED / OPEN / CHANGED |
| 5% | The thin slice built and **measured against the design** |

Automatic deductions: unlabelled arrows; no failure arrows; no sequence diagram;
a latency promise with no per-stage budget; capacity planned on a daily mean; an
availability promise exceeding the chain; a store with two owners or no
retention; a review with only ML engineers; measured numbers never compared with
designed ones.

---

## Milestones

| Day | Done by end of day |
|---|---|
| 1 | Problem, baseline, consumers. Context and container diagrams |
| 2 | Sequence and data-flow diagrams, with failure arrows |
| 3 | The contract, schemas and UI states |
| 4 | Latency budget, capacity and availability arithmetic |
| 5 | Failure table, state ownership, trade-off records |
| 6 | The review, with a non-ML engineer. Update the design |
| 7 | Build the thin slice; measure; compare; update the diagrams |

---

## Before you submit

- [ ] Every arrow has a protocol and a latency
- [ ] There are at least four dotted failure arrows
- [ ] The sequence diagram shows what is async
- [ ] The contract has `band`, `degraded`, `source` and `model_version`
- [ ] Every UI state a consumer must build is listed
- [ ] The latency budget has an owner per line and explicit headroom
- [ ] You know which stage owns the largest share, and what halving it buys
- [ ] Capacity is computed from the **peak minute**, not the daily mean
- [ ] The availability arithmetic supports the promise
- [ ] Every store has one owner and a retention period
- [ ] The failure table has a "user sees" column
- [ ] A non-ML engineer was in the review
- [ ] The trade-off record has a `Revisit at` condition
- [ ] Measured p95 is compared against designed, and gaps are explained
