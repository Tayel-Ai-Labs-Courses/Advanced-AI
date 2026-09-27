# Lesson 03 — Interfaces and Contracts

**Goal:** write the contract the frontend and mobile teams build against, so
they never have to message you.

## What you will learn

- The ten lines an API consumer needs
- Designing the response for an AI service specifically
- Versioning, and the day the model changes
- Errors as part of the contract

---

## Who you are writing for

The people consuming your API are building a **user interface**, which means
they must handle every state: loading, success, slow, failed, empty, stale,
rate-limited, and "the answer is uncertain".

An AI service adds states a normal API does not have:

| State they must build | Only exists because it is AI |
|---|---|
| The answer is uncertain | yes |
| The answer is degraded or stale | yes (lesson 06) |
| The model version changed and numbers moved | yes |
| The answer is being generated, token by token | yes |
| The system declined to answer | yes (LLM lesson 06) |
| Loading, error, empty, rate-limited | no |

**None of those are visible in "it returns a prediction."** If you do not design
them, they will be designed by whoever hits them first, at 11 p.m.

---

## The response

Start from what the consumer must decide, not from what the model returns.

```json
{
  "probability": 0.6318,
  "band": "high",
  "action": "call",
  "confidence": "normal",
  "degraded": false,
  "source": "model",
  "model_version": "churn-v1",
  "scored_at": "2026-10-05T06:00:11Z"
}
```

Field by field, and why each is in the contract:

| Field | Why the consumer needs it |
|---|---|
| `probability` | For sorting and for anyone doing arithmetic |
| `band` | **So the UI never invents its own thresholds.** Otherwise three clients pick three different cut-offs |
| `action` | The decision. The consumer should not re-derive it (Data-Science 08) |
| `confidence` | Drives whether the UI shows the number or a hedge |
| `degraded` / `source` | Lesson 06. Lets each consumer decide how much to trust this one |
| `model_version` | So a support ticket can be traced to a model (Data-Science 07) |
| `scored_at` | Distinguishes a fresh score from a cached one |

**The three that get left out and cause the most trouble are `band`,
`degraded` and `model_version`.** Without `band`, thresholds scatter across
clients and changing capacity means a mobile release. Without `degraded`, a
fallback is indistinguishable from a real answer. Without `model_version`,
nobody can explain why a customer's score changed.

### Do not expose the raw probability to everyone

Data-Security lesson 05 measured what full probabilities give an attacker, and
Data-Science lesson 06 showed that an uncalibrated probability is not a
probability. Two defaults:

- **Internal consumers** (your dashboard): probability plus band.
- **External or untrusted consumers**: band and action only.

---

## The ten-line contract

This is the artefact. Put it in the repository next to the diagrams.

```text
ENDPOINT      POST /v1/score
AUTH          service token, scoped per caller
REQUEST       one customer object, 8 required fields, types in the schema
RESPONSE      probability, band, action, confidence, degraded, source,
              model_version, scored_at
LATENCY       p50 40ms, p95 120ms, timeout 2s (client-side)
RATE LIMIT    10 req/s per account -> 429 with Retry-After
ERRORS        422 field-level validation, 429 rate limit, 503 degraded-unavailable
DEGRADED      degraded=true with source="cache"|"rule"; never a 500 when a
              fallback exists
VERSIONING    /v1 frozen. Additive fields only. Breaking changes ship as /v2,
              both live for 90 days
CHANGES       model_version changes are announced 7 days ahead; the score
              distribution may shift
OWNER         adam, #churn-model channel
```

The last two lines are the ones normal API contracts do not have, and they are
where AI services break their consumers.

---

## The day the model changes

You retrain. The scores shift. Nothing in the API changed — and every consumer
is now subtly wrong:

- The dashboard's "high risk" count doubles overnight
- An automated workflow with a hardcoded `0.7` fires far more often
- A mobile app's colour thresholds no longer match the bands
- Somebody's weekly report shows a step change and blames their own code

**A model change is a breaking change even when the schema is identical.**

What goes in the contract:

```text
1. model_version in every response, always
2. Announce retrains before they ship, with the expected distribution shift
3. Keep bands stable across versions, or bump the API version
4. Never let a consumer hardcode a threshold - that is what `band` is for
5. Publish the score distribution per version, so consumers can check
```

Point 3 is the design decision worth making early: **if `band` boundaries move
with each retrain, the band is useless**. Fix the bands, and let the model's
calibration (Data-Science lesson 06) be what keeps them meaningful.

---

## Errors are part of the contract

```json
{
  "error": "validation_failed",
  "detail": [{"field": "tenure_days", "msg": "must be >= 0", "got": -5}],
  "request_id": "req_8f2a1c",
  "retryable": false
}
```

Four properties:

- **A stable machine-readable code** (`validation_failed`), not just prose. The
  client branches on it
- **Field-level detail** so the UI can point at the right input
- **`request_id`** so a user complaint becomes a log search
- **`retryable`** so the client knows whether to try again — the single most
  useful field, and the rarest

And never return a traceback (LLM lesson 09). It leaks internals and tells the
consumer nothing actionable.

---

## Batch and streaming, briefly

Most AI services need three shapes, and the contract should say which exist:

| Shape | For | Contract addition |
|---|---|---|
| **Single sync** | A user is waiting | Latency, timeout, degraded |
| **Batch** | Scoring a population (Data-Science 08) | File location, schedule, completeness guarantee, model version on **every row** |
| **Streaming** | Generated text a person reads | First-token latency, how the stream ends, what a partial response means |

For streaming, the state nobody designs is **the stream that stops halfway**.
Decide it: does the client keep the partial text, discard it, or mark it? Write
it in the contract.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| "It returns a prediction" | Eight UI states undesigned |
| No `band`, so clients pick thresholds | Changing capacity needs a mobile release |
| No `degraded` flag | A fallback is indistinguishable from an answer |
| No `model_version` | A score change cannot be explained |
| Retraining without announcing | Every consumer is quietly wrong |
| Errors as prose | The client cannot branch on them |
| No `retryable` | Clients retry things that will never succeed |
| Raw probabilities to untrusted callers | Data-Security lesson 05 |

---

## Exercises

1. Write the ten-line contract for an endpoint you already expose. How many
   lines needed a decision that had never been made?
2. Add `band`, `degraded` and `model_version` to a response and tell the
   consumers what each means.
3. List every UI state a consumer of your API must build. Which are documented?
4. Write the announcement you would send seven days before a retrain, including
   the expected shift.
5. Design the error schema, including `retryable`, and check that your clients
   branch on the code rather than the message.

---

**Next:** [Lesson 04 — Latency Budgets](04-latency-budgets.md)
