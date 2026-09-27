# Project 13 — One Agent, Into Production

**Do this after the ten lessons.**

Build one agent for a real task, evaluate it including its **harm rate**, and
take it through shadow mode into a narrow slice of production. As always,
**"this should be a workflow"** is a passing conclusion — and on an agent
project it is the most common correct one.

---

## Requirements

### 1. Justify the agent

- The task, the user, the frequency, the value of one completed task
- **Walk lesson 01's flowchart in writing.** Why is this not a workflow?
- **The step count**, and the per-step reliability needed for 90% success,
  from lesson 01's table
- The worst thing a wrong action can do, in money or in harm

If the step count is under 4 and the sequence is fixed, build the workflow,
measure it, and write that up instead. That is a complete project.

### 2. Tools

- At least four tools, each a plain function with a **schema generated from its
  signature**
- **Business rules inside the tools**, not in the prompt
- A dispatcher that checks shape, permission, signature, then executes
- **Permissions scoped per step and per argument** — not per session
- An audit record per attempt, including blocked ones, with `run_id` and
  `step_number`
- A **dry-run mode** where every write logs and returns a plausible result

### 3. The loop

- Step, token and wall-clock budgets, all enforced
- `budget_exhausted` as a distinct outcome from `done`
- A **goal check in code** — not the model's opinion
- A no-progress detector
- An escalation path

### 4. Memory

- A stated strategy: full history, window, or summary + window
- **The goal pinned in every prompt**
- Context tokens per step logged, and the total per run reported
- The agent **resumable from the audit log** — kill it mid-run and restart

### 5. Security

- **The model never emits a tool name.** It returns a decision; your code maps
  it. If you deviate, justify it in writing
- Three injection payloads, tested: in retrieved data, in a field the user
  controls, and in a tool's return value
- The result of each, with the defence that stopped it
- An output filter, and no credentials in any prompt
- A written statement of what an attacker who fully controls the input can do

### 6. Reliability

- **Idempotency keys on every write**, derived from the operation
- A retry storm test: run the same operation 5 times, show the ledger unchanged
- The timeout rule: read before retrying a write
- Irreversible actions **last** in the sequence
- Your own failure table, eight rows, with the detector and response for each

### 7. Evaluation

- A task set of **at least 20 tasks**, each with a `success_check` in code and a
  `forbidden` list
- Categories: happy path, ineligible, ambiguous, missing data, tool failure,
  **adversarial**, multi-step
- Each task run **at least 10 times**
- Report: success rate, **harm rate**, escalation rate, steps per success,
  tokens per success, **cost per success**
- Median and range, not the best run
- A comparison against a **non-agent baseline** at equal budget

### 8. Shadow mode

- Run on **at least 50 real or realistic cases** with writes disabled
- Compare the agent's proposed actions with what a human did
- Report agreement rate, and list the disagreements you would have regretted
- This is the number that decides whether stage 3 happens

### 9. A narrow slice, live

- Choose the slice: the cheapest-to-reverse, clearest-rule subset
- Ship it behind a flag, with the kill switch tested by someone who is not you
- Monitoring: harm, escalations, budget-exhausted, cost, and a **heartbeat**
- Run it for at least three days and report what happened

### 10. The verdict

```text
RECOMMENDATION   Widen / Hold at this slice / Back to suggest mode / Make it a workflow
SUCCESS          rate with its range, on the task set and live
HARM             rate, and the worst single incident
COST             per successful task, including cleanup
AGAINST BASELINE what the workflow or the human scored
WHAT WOULD CHANGE THIS
OWNER AND REVIEW DATE
```

---

## Deliverables

```text
project-13/
├── VERDICT.md
├── justification.md         why an agent, with the step arithmetic
├── README.md
├── src/
│   ├── tools.py             functions + generated schemas + business rules
│   ├── dispatch.py          shape, permission, signature, execute, audit
│   ├── loop.py              budgets, stopping conditions, resume
│   ├── memory.py
│   └── decide.py            the model call — returns a DECISION, not a tool name
├── tasks/
│   └── *.yaml               20+ tasks with success_check and forbidden
├── tests/
│   ├── test_idempotency.py  the retry storm
│   ├── test_injection.py    three payloads
│   └── test_permissions.py
├── reports/
│   ├── eval.md              the four-metric table, median and range
│   ├── shadow.md            50 cases, agreement, regrettable disagreements
│   ├── failure_table.md
│   └── live.md              three days in the narrow slice
└── audit/                   the log schema, and a sample
```

---

## Marking

| Weight | Criterion |
|---|---|
| 10% | Justification: why an agent, with the step arithmetic |
| 15% | Tools: generated schemas, rules inside, per-argument permissions, audit |
| 10% | Loop: budgets, code-based goal check, resume |
| 15% | Security: decisions not tool names, three injection tests, written threat statement |
| 15% | Reliability: idempotency, retry storm test, timeout rule, failure table |
| 20% | Evaluation: 20+ tasks, 10 runs each, **harm rate**, cost per success, baseline |
| 10% | Shadow mode on 50 real cases, with the disagreements listed |
| 5% | A live narrow slice with monitoring and a tested kill switch |

Automatic deductions: the model emitting tool names without justification; no
harm rate; writes without idempotency keys; a task set with no adversarial or
refusal cases; one run per task; success rate reported without cost; no
non-agent baseline; an untested kill switch; no heartbeat.

---

## Milestones

| Day | Done by end of day |
|---|---|
| 1 | Justification with the step arithmetic. Tools and dispatcher |
| 2 | Loop with budgets, audit log, dry-run mode |
| 3 | Task set: 20 tasks with checks and forbidden lists |
| 4 | Security: decisions-not-tools, three injection tests |
| 5 | Reliability: idempotency, retry storm, failure table |
| 6 | Evaluation run, 10x per task; shadow mode on 50 cases |
| 7 | Narrow slice live behind a flag; monitoring; verdict |

---

## Before you submit

- [ ] `justification.md` contains the step count and the required per-step reliability
- [ ] Every tool's schema is generated from its signature
- [ ] Business rules are inside the tools, not in the prompt
- [ ] Permissions are scoped per argument, not per session
- [ ] The audit log records blocked attempts, not only successes
- [ ] The model returns a decision; code maps it to a tool
- [ ] Three injection payloads were tested and the results are written down
- [ ] Every write has an idempotency key, and the retry storm test passes
- [ ] The task set has adversarial and refusal categories
- [ ] Every task was run at least 10 times
- [ ] **Harm rate is reported**, and someone named has accepted it
- [ ] Cost per success is reported, including cleanup
- [ ] Shadow mode ran on 50 cases and the disagreements are listed
- [ ] The kill switch was tested by a non-engineer
- [ ] `VERDICT.md` opens with the recommendation, and "make it a workflow" was available
