# Lesson 08 — Evaluating Agents

**Goal:** measure an agent with numbers that would survive a review, and notice
the agent that succeeds every time while doing damage.

## What you will learn

- Why success rate alone hides the important failure
- Four metrics that belong on every agent dashboard
- Building the task set
- Comparing agents honestly

---

## Success rate is not enough

Add one thing to lesson 03's simulation: some wrong steps are not merely wasted,
they are **wrong writes** — a refund on the wrong order, an email to the wrong
customer — which cost money to clean up.

```python
import numpy as np

class World:
    PLAN = ["find_order", "check_rules", "refund_order", "send_email"]
    def __init__(self, skill, seed):
        self.rng = np.random.default_rng(seed); self.skill = skill; self.done = 0
        self.harm = 0
    def step(self):
        r = self.rng.random()
        if r < self.skill:
            self.done += 1
        elif r < self.skill + 0.05:
            self.harm += 1                      # a wrong WRITE, not just a wasted step
        return self.done >= len(self.PLAN)

def run(skill, max_steps, seed):
    w = World(skill, seed)
    for i in range(max_steps):
        if w.step():
            return True, i + 1, w.harm
    return False, max_steps, w.harm

COST_PER_STEP = 0.004      # USD
COST_PER_HARM = 12.0       # cleaning up one wrong write
print(f"{'agent':<22}{'success':>9}{'steps/success':>15}{'harm rate':>11}{'cost/success':>14}")
for label, skill, budget in [("weak, budget 8", 0.5, 8),
                             ("weak, budget 20", 0.5, 20),
                             ("strong, budget 8", 0.9, 8),
                             ("strong, budget 20", 0.9, 20)]:
    outs = [run(skill, budget, s) for s in range(3000)]
    succ = np.mean([o[0] for o in outs])
    steps_ok = np.mean([o[1] for o in outs if o[0]])
    harm = np.mean([o[2] > 0 for o in outs])
    total_cost = np.mean([o[1] * COST_PER_STEP + o[2] * COST_PER_HARM for o in outs])
    print(f"{label:<22}{succ:>9.3f}{steps_ok:>15.1f}{harm:>11.1%}"
          f"{total_cost / succ:>14.3f}")
print("\ncost/success = expected USD per completed task, including cleanup")
```

```text
agent                   success  steps/success  harm rate  cost/success
weak, budget 8            0.629            6.3      30.0%         6.870
weak, budget 20           1.000            8.0      32.7%         5.002
strong, budget 8          0.998            4.4      18.4%         2.590
strong, budget 20         1.000            4.4      18.4%         2.586

cost/success = expected USD per completed task, including cleanup
```

Compare rows two and four. **Both succeed 100% of the time.** A dashboard
showing success rate would report them as identical.

They are not. The weak agent does harm in **32.7%** of runs against 18.4%, takes
8.0 steps per success against 4.4, and costs **$5.00 per completed task against
$2.59** — nearly double, and almost all of the difference is cleanup.

Row one is the other trap: at budget 8 the weak agent succeeds only 62.9% of the
time, and it is *still doing harm in 30% of runs*, including the runs that
failed. **A failed agent run is not a no-op.** It got partway, and partway
included writes.

So four metrics, always together:

| Metric | Question |
|---|---|
| **Task success rate** | Did it finish, judged by a check in code |
| **Harm rate** | How often did it do something it should not have |
| **Steps (and tokens) per success** | What does a success cost |
| **Escalation rate** | How often did it correctly hand over |

The fourth is the one people leave out, and it flatters the other three. An
agent that escalates 40% of tasks and succeeds on the rest is not a 100% agent;
it is a 60% agent with a good safety net — which may be exactly what you want,
as long as the number is visible.

---

## The task set

An agent's eval set is not a list of prompts. It is a list of **starting states
with checkable end states**.

```text
task_id        : refund-001
setup          : order 1001 exists, delivered, 3 days ago, customer mona
request        : "I want to return order 1001"
success_check  : orders["1001"]["status"] == "refunded"
                 and an email was sent to mona
                 and no other order changed
forbidden      : any refund of an order other than 1001
max_steps      : 8
```

Four properties that make this usable:

- **`success_check` is code**, not a human reading a transcript. If you cannot
  write the check, you cannot evaluate the agent — and you probably cannot let
  it run unattended either (lesson 03).
- **`forbidden` is explicit.** Harm rate needs a definition, per task.
- **The setup is reset before every run**, so runs are independent.
- **`max_steps` is part of the task.** Success at 40 steps is a different result
  from success at 6.

Categories every agent task set needs:

| Category | Why |
|---|---|
| Happy path | The demo |
| Ineligible request | The agent must refuse, not comply |
| Ambiguous request | It must ask or escalate, not guess |
| Missing data | The record does not exist |
| Tool failure | Inject an error and see what it does |
| **Adversarial** | Lesson 05's injected instructions, in the data |
| Multi-step | Where lesson 01's arithmetic bites |

A task set with only the first row is a demo suite. **The refusal rows are what
tell you whether it is safe to run unattended.**

---

## Comparing two agents

Everything from Data-Science lesson 05 and RL lesson 09 applies unchanged:

- **Run every task N times** — agents are stochastic; one run per task is one
  seed.
- **Report the median and the range**, not the best.
- **Paired comparison**: run both agents on the same tasks and the same seeds,
  and interval the *difference*.
- **Equal budgets.** A comparison at different step budgets is not a comparison
  (lesson 07).
- **Report cost per success**, not per run. An agent that fails cheaply looks
  good per run.

And the one specific to agents: **report the transcript length distribution.**
A mean of 6 steps hiding a tail at 40 is a cost problem and usually a loop.

---

## What to watch in production

| Signal | Means |
|---|---|
| Escalation rate rising | The world changed, or a tool is failing |
| Steps per success rising | The model is struggling; cost is rising silently |
| Harm rate above zero at all | Stop and look; this is the one that ends projects |
| Budget-exhausted rate rising | Tasks are getting harder, or the loop is stuck |
| Tool error rate rising | A dependency broke; the agent is absorbing it |
| A tool that is never used | Dead capability, or a permission bug |

Alert on the third row at any non-zero threshold you have not explicitly
accepted. Everything else is a trend; harm is an incident.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Success rate as the only metric | Two agents at 1.000 differed 2x in cost and 14 points in harm |
| No harm metric | The expensive failure is invisible |
| Assuming a failed run did nothing | 30% harm rate on runs that never finished |
| Escalations counted as successes | A 60% agent reported as 100% |
| Cost per run instead of per success | Cheap failures look good |
| One run per task | Agents are stochastic |
| A task set with no refusal cases | You never learn whether it can say no |
| Judging by reading transcripts | Not repeatable, and it does not scale |

---

## Exercises

1. Write five tasks for your own agent, each with a `success_check` in code.
   How many could you not write a check for, and what does that tell you?
2. Add a `forbidden` list to each and measure your agent's harm rate.
3. Compute cost per success for your agent at two step budgets. Which budget is
   cheaper per completed task, and is it the one with the higher success rate?
4. Build the adversarial subset from lesson 05 and report the agent's harm rate
   on it separately.
5. Run each task 10 times and report the median success rate with its range.
   How much did the single-run number overstate it?

---

**Next:** [Lesson 09 — Automation in Python](09-automation.md)
