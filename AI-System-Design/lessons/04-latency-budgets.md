# Lesson 04 — Latency Budgets

**Goal:** turn "it should feel fast" into a number per component that you can
check.

## What you will learn

- Composing a p95 from stages, measured
- Why p95s do not simply add, and when that matters
- Finding the stage worth optimising
- Writing a budget the team can hold

---

## The budget

A latency budget assigns each stage a share of a promise:

```text
PROMISE   p95 under 1.5 seconds, measured at the mobile app
```

Everything below is arithmetic on that sentence. Without the sentence, "make it
faster" has no stopping condition and every optimisation looks worthwhile.

---

## Composing the stages

```python
import numpy as np

rng = np.random.default_rng(0)
N = 200_000
# four stages of a RAG request, each lognormal-ish with a tail
STAGES = {
    "auth + routing":   (0.005, 0.5),
    "embed the query":  (0.020, 0.4),
    "vector search":    (0.030, 0.7),
    "generate":         (0.900, 0.35),
}
samples = {}
print(f"{'stage':<20}{'mean ms':>10}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}")
for name, (median, sigma) in STAGES.items():
    s = rng.lognormal(np.log(median), sigma, N)
    samples[name] = s
    print(f"{name:<20}{s.mean()*1000:>10.0f}{np.percentile(s,50)*1000:>9.0f}"
          f"{np.percentile(s,95)*1000:>9.0f}{np.percentile(s,99)*1000:>9.0f}")

total = sum(samples.values())
sum_of_p95 = sum(np.percentile(s, 95) for s in samples.values())
print(f"\n{'TOTAL (measured)':<20}{total.mean()*1000:>10.0f}"
      f"{np.percentile(total,50)*1000:>9.0f}{np.percentile(total,95)*1000:>9.0f}"
      f"{np.percentile(total,99)*1000:>9.0f}")
print(f"{'sum of stage p95s':<20}{'':>10}{'':>9}{sum_of_p95*1000:>9.0f}")
print(f"\nadding p95s overstates the real p95 by "
      f"{sum_of_p95/np.percentile(total,95):.2f}x")
```

```text
stage                  mean ms   p50 ms   p95 ms   p99 ms
auth + routing               6        5       11       16
embed the query             22       20       39       51
vector search               38       30       95      154
generate                   956      901     1597     2033

TOTAL (measured)          1022      967     1665     2100
sum of stage p95s                           1742

adding p95s overstates the real p95 by 1.05x
```

Two things to take from this.

**The total p95 is 1,665 ms and the promise was 1,500. The design misses.** That
is the entire purpose of the exercise, and it is available before anything is
built — from four estimates and thirty lines of code.

**Adding the stage p95s gives 1,742, which overstates by 1.05x.** The rule "p95s
do not add" is true: a request is rarely unlucky in every stage at once, so the
sum is an upper bound.

But notice how *small* the overstatement is here, and be honest about why. One
stage — generation — is 93.6% of the time, so the total's distribution is
basically that stage's distribution. **Summing p95s is a safe, slightly
pessimistic estimate when one stage dominates, and badly pessimistic when four
stages are comparable.** Use the sum as a quick upper bound; simulate when the
stages are of similar size.

---

## Which stage is worth your week?

```python
budget = 1.5
print(f"budget: {budget*1000:.0f} ms at p95. Measured p95: "
      f"{np.percentile(total,95)*1000:.0f} ms")
print(f"{'stage':<20}{'share of mean':>15}{'cut it by 50% ->':>18}")
for name, s in samples.items():
    others = sum(v for k, v in samples.items() if k != name)
    improved = np.percentile(others + s * 0.5, 95)
    print(f"{name:<20}{s.mean()/total.mean():>15.1%}"
          f"{improved*1000:>15.0f} ms")
```

```text
budget: 1500 ms at p95. Measured p95: 1665 ms
stage                 share of mean  cut it by 50% ->
auth + routing                 0.6%           1662 ms
embed the query                2.1%           1653 ms
vector search                  3.8%           1644 ms
generate                      93.6%            869 ms
```

**Halving the vector search — a week of work, a new index, an upgrade — takes
the p95 from 1,665 ms to 1,644 ms.** Twenty-one milliseconds. The promise is
still missed.

**Halving generation takes it to 869 ms**, and the promise is met with room.

This is Amdahl's law wearing business clothes, and it is the most reliably
ignored fact in performance work. People optimise the component they understand
or the one that is annoying, not the one that is 93.6% of the time.

For an LLM system, "halve generation" has known moves, and they are all in this
track:

| Move | Where |
|---|---|
| Shorter outputs — output length *is* the latency | LLM lesson 10 |
| A smaller or fine-tuned model | LLM lesson 08 |
| Stream the first token | LLM lesson 10 |
| Cache the whole response | LLM lesson 10 |
| Skip generation entirely for easy cases | LLM lesson 10's route table |

The last one is usually the largest win and the least attempted: a request that
never reaches the model has a generation time of zero.

---

## Writing the budget down

```text
PROMISE      p95 < 1500 ms, measured at the app, excluding network to the device

  auth + routing        50 ms    owner: platform
  feature fetch        120 ms    owner: data       (cache hit; 400 ms on miss)
  retrieval            150 ms    owner: search
  generation           900 ms    owner: ML         <- the one that matters
  serialisation         30 ms    owner: platform
  ----------------------------------------
  budgeted            1250 ms
  headroom             250 ms    17%

ALERT        any stage above 1.5x its budget, at p95, for 10 minutes
ON BREACH    the route table degrades: skip retrieval, use the cached answer
```

Three properties of a budget that works:

1. **Every line has an owner.** A budget nobody owns is a wish.
2. **Headroom is explicit.** 100% allocated means the first bad day breaks the
   promise. 15-20% is normal.
3. **The breach behaviour is decided in advance**, not improvised at 3 a.m.

And measure **at the edge the user experiences**, not at your service boundary.
A 900 ms server-side p95 with 400 ms of mobile network is a 1.3-second product.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A promise with no number | "Fast" has no stopping condition |
| Optimising the stage you find interesting | 21 ms against 796 ms |
| Reporting the mean | Users feel the tail |
| Summing p95s when stages are comparable | Badly pessimistic; simulate instead |
| No headroom | The first bad day breaks the promise |
| Budget lines with no owner | Nobody defends them |
| Measuring at the service, not the device | The network is part of the experience |
| No decided degraded mode | It gets improvised during the incident |

---

## Exercises

1. Build the stage table for your slowest endpoint from real measurements. Does
   the total match what you observe end to end?
2. Compute what halving each stage buys. Which one is worth a week?
3. Set a p95 promise and allocate the budget with owners and headroom.
4. Measure the same request at the device and at your service. How big is the
   gap?
5. Write the degraded mode: what the system does when the dominant stage is over
   budget, and what the user sees.

---

**Next:** [Lesson 05 — Capacity and Queues](05-capacity-and-queues.md)
