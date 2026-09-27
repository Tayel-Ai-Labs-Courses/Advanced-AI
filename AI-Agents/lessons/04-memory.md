# Lesson 04 — Memory and State

**Goal:** decide what the agent carries between steps, and pay for it on purpose.

## What you will learn

- Why context cost grows faster than step count
- Three memory strategies, priced
- What each one forgets
- Where state actually belongs

---

## The cost of remembering everything

The naive loop sends the whole transcript on every step. Each step adds an
observation and an action, so the context grows linearly — and the **total**
tokens sent across a run grows quadratically.

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

SYSTEM = 400
OBS = 180
ACT = 60

print(f"{'step':>5}{'context tokens':>16}{'cumulative tokens':>19}{'cost share':>12}")
total = 0
for step in range(1, 13):
    ctx = SYSTEM + step * (OBS + ACT)
    total += ctx
    if step in (1, 2, 4, 8, 12):
        print(f"{step:>5}{ctx:>16,}{total:>19,}{'':>12}")
print(f"\na 12-step run re-sends the whole history every time:")
print(f"  naive total input tokens : {total:,}")
print(f"  if you only sent the last observation: "
      f"{12 * (SYSTEM + OBS + ACT):,}")
print(f"  ratio: {total / (12 * (SYSTEM + OBS + ACT)):.2f}x")
```

```text
 step  context tokens  cumulative tokens  cost share
    1             640                640
    2             880              1,520
    4           1,360              4,000
    8           2,320             11,840
   12           3,280             23,520

a 12-step run re-sends the whole history every time:
  naive total input tokens : 23,520
  if you only sent the last observation: 7,680
  ratio: 3.06x
```

A 12-step run costs **3.06x** what it would if each step carried only what it
needed. At 30 steps the ratio is worse, and at 50 the context window itself
becomes the binding constraint rather than the bill.

This is the hidden cost of agents relative to workflows: a 12-step workflow
makes 12 small calls; a 12-step agent makes 12 calls of growing size.

---

## Three strategies

```python
def full_history(n):
    return sum(SYSTEM + i * (OBS + ACT) for i in range(1, n + 1))
def window(n, k=4):
    return sum(SYSTEM + min(i, k) * (OBS + ACT) for i in range(1, n + 1))
def summarised(n, k=4, summary=120):
    return sum(SYSTEM + summary + min(i, k) * (OBS + ACT) for i in range(1, n + 1))
print(f"{'strategy':<26}{'input tokens':>14}{'vs full':>10}")
f = full_history(30)
for name, fn in [("full history", lambda: full_history(30)),
                 ("last 4 steps only", lambda: window(30)),
                 ("summary + last 4", lambda: summarised(30))]:
    v = fn()
    print(f"{name:<26}{v:>14,}{v / f:>10.2f}")
```

```text
strategy                    input tokens   vs full
full history                     123,600      1.00
last 4 steps only                 39,360      0.32
summary + last 4                  42,960      0.35
```

Over 30 steps, keeping only the last four observations costs **32%** of full
history. Adding a running summary costs 35% — **three points more than the
window, for most of what the window throws away.**

That three-point difference is the whole argument for summarisation, and it is
why it is the default choice for long-running agents.

```text
  full history      forgets nothing, until the window overflows and it ALL breaks
  last 4 steps      forgets the goal, after step 5
  summary + last 4  forgets detail, but the goal survives in the summary
```

The middle row is the failure people hit first and misdiagnose as the model
getting "confused": at step 9 the agent has no idea what it was asked to do,
because the goal scrolled out of the window five steps ago.

**Whatever strategy you choose, pin the goal and the constraints.** They are
short, they never change, and they belong in every single prompt.

---

## What goes where

Not everything the agent needs is "memory". Most of it is state, and state
belongs in a database.

| What | Where | Why |
|---|---|---|
| The goal and constraints | **Pinned in every prompt** | Short, and losing it is fatal |
| The last few observations | Rolling window | Recent detail is what the next step needs |
| Everything older | A summary, regenerated every N steps | Cheap, and preserves direction |
| Facts the agent looked up | **A key-value store**, not the transcript | Look them up again instead of re-sending |
| What the agent has already done | **The audit log** (lesson 02) | Authoritative, queryable, survives a crash |
| Anything a human must see | Your own database | The transcript is not a record of truth |
| Long-term knowledge | Retrieval (LLM lesson 06) | The transcript is not a knowledge base |

The line to hold: **the transcript is a working buffer, not the source of
truth.** If the process dies at step 7 and the transcript is the only record of
the refund it issued at step 4, you have lost it. The audit log is what survives.

A useful consequence: an agent that stores its state properly can be **resumed**.
The loop becomes a function of `(goal, audit_log, current_observation)` rather
than of an unbroken conversation, and a crash costs one step instead of a run.

---

## Summarising without losing the plot

```python
# no-run
def compress(transcript, keep=4, summarise=None):
    """Keep the last `keep` steps verbatim; summarise the rest."""
    if len(transcript) <= keep:
        return None, transcript
    old, recent = transcript[:-keep], transcript[-keep:]
    facts = [f"{t['call']['name']} -> {str(t['result'])[:60]}" for t in old]
    summary = summarise("\n".join(facts)) if summarise else "; ".join(facts[-6:])
    return summary, recent
```

Two rules that keep a summary useful:

1. **Summarise results, not prose.** `refund_order -> {"refunded": 450}` is
   the fact. The model's commentary about it is not.
2. **Regenerate from the original, not from the previous summary.** Summarising
   a summary compounds loss exactly the way lesson 01's errors compound, and
   after four rounds the goal has quietly become something else.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Sending the full transcript every step | 3.06x the tokens at 12 steps, worse at 30 |
| A rolling window with no pinned goal | The agent forgets what it was asked at step 5 |
| Summarising the summary | Compounding loss; the goal drifts |
| The transcript as the system of record | A crash loses what the agent actually did |
| Re-sending looked-up facts every step | Store them; look them up again if needed |
| No resume path | A crash at step 11 costs eleven steps |
| Unbounded context "because the window is big" | Cost is quadratic, and long context degrades attention |

---

## Exercises

1. Instrument your loop to record context tokens per step. Plot it, and find the
   step at which the run exceeds your budget.
2. Implement `compress` with a real summariser and measure the token saving and
   the change in task success rate over 30-step runs.
3. Delete the pinned goal from the window strategy and find the step at which
   task success collapses.
4. Make your loop resumable from the audit log alone. Kill it at step 7 and
   restart; how many steps does it repeat?
5. Price a 30-step run under all three strategies at your provider's rates, and
   state the volume at which the difference pays for a day of engineering.

---

**Next:** [Lesson 05 — Security](05-security.md)
