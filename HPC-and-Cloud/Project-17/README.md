# Project 17 — One Real Run, Inside a Budget

**Do this after the eight lessons.**

You train one real model on rented hardware, having predicted the cost and the
wall-clock time **before** you started, and you report how close you were.

The deliverable is a trained model, a cost report, and an honest comparison of
the estimate against the invoice.

> Budget guidance: this project is designed to cost **under 500 EGP**. If your
> plan is larger than that, lesson 07's "when the estimate says no" list applies
> to you before you start.

---

## Requirements

### 1. The memory arithmetic, before anything

- The four-term table from lesson 01 for your exact model
- Activations estimated at your batch size and sequence length
- **The cheapest GPU that fits**, with the lesson-03 list applied step by step
  and the memory recorded after each step
- A statement of which step made it fit

### 2. The throughput measurement

- 100 steps with **synthetic** data: the GPU's ceiling
- 100 steps with your **real** loader: the gap
- `samples_per_sec` at three batch sizes
- If the gap is over 20%, what you did about it (lesson 02)

### 3. The budget, written before the run

Using lesson 07's estimator:

```text
samples x epochs, samples_per_sec, GPU, price
-> predicted wall clock, GPU-hours, EGP

x configurations, x seeds
-> the experiment budget

+ the stopping rule: what result makes you stop early
+ who turns it off
```

Commit this **before** the first cloud run. Its git timestamp is part of the
submission.

### 4. The five stages

Evidence for each (lesson 08):

| Stage | Evidence |
|---|---|
| 1. Laptop, 1% of data | The bugs it caught, listed |
| 2. Laptop, full pipeline, 10 steps | Checkpoint written, **resume tested** |
| 3. Cheap GPU, 1 hour | `samples_per_sec`, peak memory, cost |
| 4. The real run | Logs, checkpoints, run record |
| 5. Artifacts retrieved | The list from lesson 08 |

**Stage 2's resume test is mandatory**: kill at step N, resume, and show the
loss curve is continuous across the join.

### 5. Checkpointing

- Your measured checkpoint write cost
- Your provider's published interruption rate for the instance type
- **The computed optimum interval**, and the one you used
- A checkpoint containing all eight items from lesson 06
- Atomic writes (temp name, then rename)
- The last N kept **and** the best

### 6. Cost control

All four controls from lesson 07, with evidence each is active:

- A billing alert, at a number you chose
- A hard quota on the account
- An auto-shutdown timer on every instance
- `gpu_hours` and `cost_egp` in every run record

### 7. The run

- The model trained, with its eval score on a held-out set
- The run record (Data-Science lesson 07) extended with GPU, cost and hours
- The environment pinned: Python, CUDA, torch, base image
- At least one interruption survived, or one triggered on purpose

### 8. The report

```text
PREDICTED     wall clock, GPU-hours, EGP        (from the committed budget)
ACTUAL        wall clock, GPU-hours, EGP        (from the invoice)
GAP           percentage, and the explanation
WHAT FIT      which lesson-03 step made it fit, and what it saved
BOTTLENECK    GPU, data loading, or communication — with the measurement
CHEAPEST      the configuration you would use next time, and why
WASTED        what you spent and would not spend again
```

A gap over 30% needs a real explanation, not an apology. Gaps are normal; not
knowing why is the problem.

---

## Deliverables

```text
project-17/
├── REPORT.md              predicted vs actual, with the explanation
├── budget.md              committed BEFORE the first cloud run
├── memory.md              the lesson-01 table and the lesson-03 ladder
├── README.md              how to reproduce
├── job.sh                 the run script: shutdown, resume, upload
├── src/
│   ├── train.py           with --resume-if-exists and --checkpoint-every-min
│   └── config.yaml
├── evidence/
│   ├── stage1.md ... stage5.md
│   ├── resume-curve.png   the loss across a kill/resume join
│   ├── throughput.md      synthetic vs real loader, three batch sizes
│   └── controls.md        billing alert, quota, shutdown, screenshots
└── runs/*.json            run records with gpu_hours and cost_egp
```

---

## Marking

| Weight | Criterion |
|---|---|
| 15% | Memory arithmetic, and the lesson-03 ladder with the step that made it fit |
| 15% | Throughput measured: synthetic vs real, three batch sizes |
| 15% | Budget committed **before** the run, with a stopping rule |
| 15% | The five stages, with evidence — including a tested resume |
| 15% | Checkpointing: computed optimum, all eight items, atomic writes |
| 10% | All four cost controls active, with evidence |
| 15% | The report: predicted vs actual, with the gap explained |

Automatic deductions: a budget committed after the run; no resume test; a
checkpoint missing the optimizer or scheduler state; no auto-shutdown; no
`cost_egp` in the run records; debugging done on the expensive instance; a gap
reported without an explanation.

---

## Milestones

| Day | Done by end of day |
|---|---|
| 1 | Memory arithmetic; lesson-03 ladder; cheapest GPU identified |
| 2 | Stages 1 and 2 on the laptop, including the resume test |
| 3 | Budget committed. Cost controls set and verified |
| 4 | Stage 3: one hour on a cheap GPU; throughput measured |
| 5 | Stage 4: the real run, on spot, with checkpoints |
| 6 | Artifacts retrieved; run records complete |
| 7 | The report: predicted vs actual, with the explanation |

---

## Before you submit

- [ ] `budget.md` is committed before the first cloud run
- [ ] The memory table shows the step that made it fit
- [ ] Throughput was measured synthetic **and** with the real loader
- [ ] The resume test has a loss curve across the join
- [ ] The checkpoint contains all eight items from lesson 06
- [ ] Checkpoints are written atomically, and more than one is kept
- [ ] A billing alert, a quota and an auto-shutdown are all active
- [ ] Every run record has `gpu_hours` and `cost_egp`
- [ ] Python, CUDA, torch and the base image are pinned and recorded
- [ ] `REPORT.md` compares predicted against actual and explains the gap
