# Lesson 06 — Reliability

**Goal:** make an agent that survives its own retries, crashes and duplicates.

## What you will learn

- Idempotency, and what it saves
- Compensating actions for what cannot be undone
- Where to put the boundary between the model and the money
- The failure table every agent needs

---

## Retries are only safe if actions are

Lesson 01 showed retries are the cheapest reliability improvement available: one
retry took a 10-step task from 0.349 to 0.904. Here is what happens when you add
retries to an action that is not safe to repeat.

```python
class Ledger:
    def __init__(self): self.refunds = []
    def refund_naive(self, order_id, amount):
        self.refunds.append((order_id, amount))
    def refund_idempotent(self, order_id, amount, key):
        if any(k == key for _, _, k in self.keyed):
            return "already applied"
        self.keyed.append((order_id, amount, key))
        return "applied"
    keyed = []

led = Ledger()
for attempt in range(3):                       # a retry storm
    led.refund_naive("1001", 450.0)
print(f"  naive:      {len(led.refunds)} refunds recorded, "
      f"{sum(a for _, a in led.refunds):.0f} EGP paid out")
led.keyed = []
for attempt in range(3):
    result = led.refund_idempotent("1001", 450.0, key="req-abc-123")
print(f"  idempotent: {len(led.keyed)} refunds recorded, "
      f"{sum(a for _, a, _ in led.keyed):.0f} EGP paid out (last call: {result})")
```

```text
  naive:      3 refunds recorded, 1350 EGP paid out
  idempotent: 1 refunds recorded, 450 EGP paid out (last call: already applied)
```

**Three retries, 1,350 EGP paid out for one 450 EGP refund.** The retry
mechanism that made the agent reliable also made it pay three times.

The fix is one argument: an **idempotency key** that identifies the *logical*
operation, not the attempt. The second and third calls find the key and return
the original result instead of acting again.

```text
key = f"refund:{run_id}:{order_id}"
```

Not a random UUID per call — that makes every retry a new operation. The key
must be **derivable from what the operation is**, so that two attempts at the
same thing produce the same key.

| Operation | Idempotency key |
|---|---|
| Refund an order | `refund:{order_id}` |
| Send a notification | `email:{run_id}:{template}:{recipient}` |
| Create a ticket | `ticket:{source_id}` |
| Charge a card | `charge:{order_id}:{amount}` |
| Append to a log | naturally safe, no key needed |

---

## What cannot be made idempotent

Some actions are irreversible and not naturally deduplicable: an email that has
already reached a person, a message posted to a public channel, a physical
shipment.

For those, three options in order of preference:

1. **Make it deduplicable anyway.** A `sent_emails` table keyed by
   `(run_id, template, recipient)` turns "send" into "send if not already sent".
   This covers most cases and costs one table.
2. **Compensate.** If it cannot be prevented, follow it with an action that
   undoes its effect — a correction email, a reversal entry. Record both.
3. **Put a human in front of it.** Lesson 05's confirmation queue, for the small
   number of actions where 1 and 2 are impossible.

**Order the agent's steps so that irreversible actions come last.** An agent
that emails the customer before checking eligibility has no good options when
the check fails; one that checks, refunds, then emails, fails harmlessly at
every earlier point.

```text
read  ->  decide  ->  reversible writes  ->  irreversible actions
                                             (last, and fewest)
```

---

## The failure table

Every agent needs this written down before it ships, because the answers are
different for each row and the loop cannot guess them.

| Failure | Detect | Respond |
|---|---|---|
| Tool raises | Exception | Retry with backoff, bounded (lesson 01) |
| Tool returns an error object | Check the result | Feed the error back; let the model choose again |
| Tool times out | Timeout | **Do not blind-retry a write** — check whether it happened |
| Tool succeeds, result is wrong | Verifier / business rule | Compensate, then escalate |
| Model picks a forbidden tool | Dispatcher (lesson 02) | Return the error; count it |
| Model loops | No-progress detector (lesson 03) | Break, escalate |
| Budget exhausted | Step / token counter | Escalate with the transcript |
| Process crashes | Missing completion record | **Resume from the audit log** (lesson 04) |

The third row is the one that causes real incidents. A timeout means *you do not
know* whether the write happened. Blind-retrying a refund on a timeout is how
customers get paid twice — which is the first table in this lesson, arriving
through a different door. **On a timeout, read before you write.**

---

## Where to put the boundary

The most reliable agents are the ones where the model touches the least.

```text
model decides       "this looks like an eligible refund"
code verifies       the rules, against the database
code acts           the refund, with an idempotency key
code records        the audit row
model explains      the outcome to the customer
```

The model appears twice, at the start and the end, and never in the middle where
the money moves. Everything between is ordinary software with ordinary
guarantees — testable, idempotent, and unaffected by a prompt.

This is the same conclusion as lesson 05's "the model never names a tool",
reached from reliability rather than security. When two different kinds of
pressure point at the same design, that design is usually right.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Retries without idempotency keys | 1,350 EGP paid for a 450 EGP refund |
| A random key per attempt | Every retry is a new operation |
| Blind-retrying after a timeout | You do not know whether it happened |
| Irreversible actions early in the sequence | Nothing good to do when a later check fails |
| No resume path | A crash re-runs the actions it already took |
| The model in the middle of the money | Every guarantee now depends on a prompt |
| No failure table | The loop improvises, differently each time |

---

## Exercises

1. Add an idempotency key to every write tool in your agent. Then run a retry
   storm and confirm the ledger is unchanged.
2. Implement the timeout rule: on a timeout, read the record before retrying.
   Write the test that proves it.
3. Reorder your agent's steps so irreversible actions come last. Which step
   moved, and what failure does that now make harmless?
4. Write your own failure table with eight rows. For each, name the detector and
   the response, and say which ones you have actually implemented.
5. Build the compensating action for one irreversible step, and the audit record
   that ties it to the original.

---

**Next:** [Lesson 07 — Multi-Agent, and When It Is Theatre](07-multi-agent.md)
