# Lesson 08 — The Design Review

**Goal:** review a design in an hour and find the expensive problems while they
are still free.

## What you will learn

- The review agenda
- The eight questions that find most problems
- Trade-offs, written down
- What a good review produces

---

## The review

Sixty minutes, before implementation starts, with the four readers from lesson
01 in the room.

```text
1. THE PROBLEM          5 min   what decision does this system support?
2. THE FOUR DIAGRAMS   15 min   context, container, sequence, data flow
3. THE CONTRACT         5 min   the ten lines (lesson 03)
4. THE NUMBERS         15 min   latency budget, capacity, availability
5. FAILURE             10 min   the failure table, including "user sees"
6. TRADE-OFFS           5 min   what was rejected, and why
7. DECISIONS            5 min   what is agreed, what is open, who owns each
```

**Fifteen minutes on the numbers.** That is the part that turns a review from
opinion-trading into engineering, and it is the part usually skipped.

---

## The eight questions

Ask these of any AI system design. They find most of the expensive problems.

**1. What decision does this support, and what does a wrong one cost?**
If there is no decision, there is no system (Data-Science lesson 01).

**2. What is the non-AI baseline, and is it running?**
Data-Science lesson 05. It is your comparison and your fallback (lesson 06).

**3. What is the p95 budget, and which component owns 90% of it?**
Lesson 04. If nobody knows, the performance work will go to the wrong place.

**4. What is the capacity, in requests per second, and what is the peak?**
Lesson 05. A service at 200 ms saturates at traffic a normal API would not
notice.

**5. What happens when each box is unavailable, and what does the user see?**
Lesson 06's failure table. Missing rows are undesigned behaviour.

**6. Who owns each store, and what is stale by how much?**
Lesson 07. Two owners is a future incident.

**7. Where does personal data rest and cross a boundary, and for how long?**
Data-Security lessons 02 and 08.

**8. How does a consumer know the model changed?**
Lesson 03. A retrain is a breaking change without a schema change.

A design that answers all eight is not necessarily right, but it is **reviewable**
— and the ones that cannot answer 3, 5 and 8 are the ones that come back.

---

## Trade-offs, written down

Every design rejects alternatives. Writing them down is what stops the same
debate happening every quarter, and it is the decision record from
Communication lesson 05 in its architectural form.

```text
DECISION      Score synchronously in the request path
ALTERNATIVES
  A  Batch nightly, serve from a table
     + p95 drops to 5ms, capacity is trivial, model can be slower
     - scores up to 24h stale; no score for a customer created today
  B  Sync scoring (chosen)
     + always current, works for new customers
     - the model is in the latency budget and the availability chain
  C  Hybrid: batch nightly, sync only on a cache miss
     + best of both
     - two code paths, two sets of evals, more to get wrong
WHY B         Retention acts on same-day signups, which A cannot serve. C was
              rejected for now because the traffic (5 req/s peak) does not
              justify two paths. Revisit at 50 req/s.
```

The `Revisit at` line is the most valuable one. It converts "we decided this
once" into a condition anybody can check.

---

## What a good review produces

Not approval. Three lists:

```text
AGREED       decisions that are now fixed, with owners
OPEN         questions with a name and a date against each
CHANGED      what moved in the design during the hour
```

A review where nothing changed was either a perfect design or a passive room.
The second is far more likely — so if nothing moved, ask the eight questions
harder.

And the output goes in the repository with the diagrams, not in meeting notes.

---

## Reviewing someone else's design

Two failure modes to avoid:

**Reviewing the technology instead of the design.** "Why FastAPI and not Flask"
is almost never the expensive question. "What happens when the model service is
slow" always is.

**Silence about the number you doubt.** If you think the p95 estimate is
optimistic, say so in the room; it costs a minute there and a month later.

The most useful thing a reviewer can say is: *"walk me through what the user
sees when the model service is down."* Everything weak in a design tends to
surface in the answer.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Reviewing after implementation | The boxes have concrete around them |
| No numbers in the review | Opinions, traded for an hour |
| Skipping the failure walkthrough | The undesigned states ship |
| No rejected alternatives recorded | The same debate, every quarter |
| No "revisit at" condition | Decisions outlive their reasons |
| Approval as the output | Nothing was decided, owned or dated |
| Only engineers in the room | The frontend and privacy questions never get asked |

---

## Exercises

1. Run the eight questions against a system you already operate. How many can
   you answer today?
2. Write the trade-off record for its biggest design decision, including the
   revisit condition.
3. Review a colleague's design using the agenda. Time each section.
4. Find a design decision in your system whose reason nobody remembers. Write it
   down or change it.
5. Ask "what does the user see when the model is down" about three systems you
   depend on.

---

**Done with the lessons.** Next: [Project 16](../Project-16/) — design a system
before building it, and have it reviewed.

---

## Where to go next

| Next | Why |
|---|---|
| [This course's project](../Project-16/) | It is the assessment, and it is not optional |
| [MLOps](../../MLOps/) | Running what you designed |
| [HPC-and-Cloud](../../HPC-and-Cloud/) | What the design costs |
| [AI-in-Frontend](https://github.com/Tayel-Ai-Labs-Courses/AI-in-Frontend) | The consumer side, a separate repository |
