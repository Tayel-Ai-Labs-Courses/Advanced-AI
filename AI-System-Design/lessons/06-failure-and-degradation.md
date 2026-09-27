# Lesson 06 — Failure and Degradation

**Goal:** design what happens when a box in your diagram is not there.

## What you will learn

- Availability multiplies, measured
- Degradation that a user can live with
- Timeouts, retries and the amplification they cause
- The failure table for a system

---

## Availability multiplies

Each dependency you add multiplies its availability into yours.

```python
print(f"{'services in the chain':>23}" + "".join(f"{f'{a}%':>12}" for a in (99.9, 99.5, 99.0)))
for n in (1, 3, 5, 10):
    row = "".join(f"{(a/100)**n*100:>12.2f}" for a in (99.9, 99.5, 99.0))
    print(f"{n:>23}{row}")
print("\ncells are the chain's availability, in percent")
for n, a in ((5, 0.999), (5, 0.99)):
    down = (1 - a**n) * 365 * 24 * 60
    print(f"  5 services at {a*100:.1f}% each -> {down:,.0f} minutes of downtime a year")
```

```text
  services in the chain       99.9%       99.5%       99.0%
                      1       99.90       99.50       99.00
                      3       99.70       98.51       97.03
                      5       99.50       97.52       95.10
                     10       99.00       95.11       90.44

cells are the chain's availability, in percent
  5 services at 99.9% each -> 2,623 minutes of downtime a year
  5 services at 99.0% each -> 25,760 minutes of downtime a year
```

**Five services at 99.9% each give a chain at 99.50%** — 2,623 minutes, about 44
hours a year. Not one of the five was unreliable; the chain was.

At 99% each — which is what a service with no on-call and a weekly deploy
actually achieves — five in a chain is **25,760 minutes, eighteen days**.

Two design consequences:

**Every synchronous dependency is a multiplication.** Before adding one, ask
whether it can be asynchronous, cached, or optional. A dependency you can
degrade past does not multiply.

**Your promise cannot exceed your chain.** Promising 99.9% on top of five
dependencies at 99.9% is arithmetically impossible, and the number should be
computed before it is promised.

---

## Degrade, do not fail

For an AI system there is almost always something better than an error, and it
is usually cheap:

| Failure | Bad response | Good degradation |
|---|---|---|
| Model service down | 500 | The rule-based score, flagged `degraded` |
| Feature store down | 500 | Score on the features you have, flagged |
| Retrieval down (RAG) | 500 | Answer without context, **and say so** |
| Model too slow | Timeout | The cached score, flagged `stale` |
| Unseen category | Silent zeros | Score, **and log it** (Data-Science 09) |
| Over capacity | Timeout | 429 with `Retry-After` (lesson 05) |
| Model returns nonsense | Ship it | Validation catches it; escalate (LLM 09) |

The pattern: **a degraded answer that is labelled is better than an error, and
far better than a wrong answer presented as a normal one.**

That last distinction is the whole design. Adding `"degraded": true` to the
response costs nothing and lets every consumer decide for itself: the dashboard
greys the number, the automated workflow skips its action, the mobile app shows
a badge. Without the flag, all three silently treat a fallback as truth.

```mermaid
flowchart TD
    R["request"] --> Mmodel available<br/>within budget?
    M -->|yes| P["score<br/>degraded: false"]
    M -->|no| Ccached score<br/>under 24h?
    C -->|yes| S["cached score<br/>degraded: true, stale: true"]
    C -->|no| Brule baseline<br/>available?
    B -->|yes| RB["rule score<br/>degraded: true, source: rule"]
    B -->|no| N["no score<br/>the UI hides the panel"]
```

The rule baseline at the bottom is the one from Data-Science lesson 05 — the
thing you measured to prove the model was worth building. **Keep it running.**
It is your fallback, it costs nothing, and it is also your permanent check on
whether the model is still beating it.

---

## Retries amplify

A retry is a second request. During an incident, retries are how a struggling
system becomes a dead one.

```text
normal            100 req/s
service degrades  every client retries 3 times
effective load    300 req/s against a system already failing
```

Four rules that keep retries from causing the outage:

1. **Only retry what is idempotent** (AI-Agents lesson 06). Never blind-retry a
   write after a timeout.
2. **Exponential backoff with jitter.** Fixed-interval retries from many clients
   synchronise into waves.
3. **A retry budget**: cap retries at a small fraction of total requests, and
   stop retrying when the fraction is exceeded.
4. **A circuit breaker**: after N consecutive failures, stop calling for a
   while and serve the degraded path immediately. This converts a slow failure
   into a fast one, which lesson 05 showed is much cheaper.

---

## The failure table

One row per box on your diagram. This is the artefact; the prose above is how to
fill it in.

```text
COMPONENT        DETECT              RESPOND                     USER SEES
model service    timeout 800ms       cached score, else rule     "estimated" badge
feature cache    connection error    read the warehouse          slower, no badge
warehouse        timeout 2s          rule baseline               "estimated" badge
model registry   unreachable         keep the running model      nothing
prediction log   write fails         drop it, count it, alert    nothing
gateway          health check        static error page           "try again shortly"
```

The last column is the one that makes it a design rather than an engineering
note. **Write what the user sees for every row**, and show that column to the
frontend team — those are the states they have to build, and lesson 02's
sequence diagram is where they belong.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Promising availability above the chain's | Arithmetically impossible |
| Adding a synchronous dependency casually | Every one multiplies |
| Returning 500 when a fallback exists | The rule baseline was right there |
| A fallback with no flag | Consumers treat a guess as a measurement |
| Retrying without backoff | 3x load on a system that is already failing |
| Blind-retrying a write after a timeout | Duplicate refunds (AI-Agents 06) |
| No circuit breaker | Slow failures consume capacity (lesson 05) |
| A failure table with no "user sees" column | An engineering note, not a design |

---

## Exercises

1. Compute your chain's availability from your dependencies' published numbers.
   Is your promise achievable?
2. For each synchronous dependency, decide whether it could be async, cached or
   optional. How many could?
3. Add `degraded` and `source` to your response schema, and update the sequence
   diagram.
4. Write the failure table for your system, including the "user sees" column.
5. Add a circuit breaker to your slowest dependency and measure what happens to
   p99 when that dependency is slow.

---

**Next:** [Lesson 07 — Data Flow and State](07-data-flow-and-state.md)
