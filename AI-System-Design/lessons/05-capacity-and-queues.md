# Lesson 05 — Capacity and Queues

**Goal:** know how many requests your system can take before latency stops being
a function of your code.

## What you will learn

- The utilisation cliff, measured
- Why 90% utilisation is not "efficient"
- How many workers you need
- Timeouts as a capacity decision

---

## The cliff

A system at 50% load is not half as slow as one at 100%. Queueing is non-linear,
and the non-linearity arrives suddenly.

```python
import math
import numpy as np

SERVICE_MS = 200.0          # one request takes 200 ms of a worker's time
print(f"one worker, {SERVICE_MS:.0f} ms per request -> "
      f"capacity {1000/SERVICE_MS:.0f} requests/sec\n")
print(f"{'utilisation':>12}{'arrivals/sec':>14}{'queue wait ms':>15}{'total ms':>10}")
for rho in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    lam = rho * (1000 / SERVICE_MS)
    wait = SERVICE_MS * rho / (1 - rho)        # M/M/1 mean queue wait
    print(f"{rho:>12.0%}{lam:>14.1f}{wait:>15.0f}{wait + SERVICE_MS:>10.0f}")
print("\nM/M/1: random arrivals, one worker. The wait is not linear in load.")
```

```text
one worker, 200 ms per request -> capacity 5 requests/sec

 utilisation  arrivals/sec  queue wait ms  total ms
         30%           1.5             86       286
         50%           2.5            200       400
         70%           3.5            467       667
         80%           4.0            800      1000
         90%           4.5           1800      2000
         95%           4.8           3800      4000
         99%           5.0          19800     20000

M/M/1: random arrivals, one worker. The wait is not linear in load.
```

**The work takes 200 ms. At 90% utilisation the user waits 2,000 ms.** Ten times
the service time, and the code has not changed at all.

Going from 4.0 to 4.5 requests per second — a 12% increase in traffic — **doubles
the response time**. That is the cliff, and it is why a system that was fine on
Monday is unusable on Tuesday after a marketing email.

Two consequences that contradict intuition:

**High utilisation is not efficiency.** A machine at 90% looks well-used on a
dashboard and is delivering a terrible experience. For latency-sensitive work,
**target 50-70%** and accept that the hardware looks half-idle. That idleness is
what absorbs randomness.

**Averages hide it.** "We average 3 requests per second" says nothing, because
arrivals are bursty. The utilisation that matters is during the peak minute, not
the daily mean.

This matters doubly for AI systems, where the service time is 200 ms rather than
2 ms. **A slow service saturates at a traffic level that would not trouble a
normal API**, which is why Optimization lesson 14's capacity calculation comes
before the cost calculation.

---

## How many workers?

```python
def mm_c_wait(lam, mu, c):
    """Mean wait in an M/M/c queue (Erlang C)."""
    rho = lam / (c * mu)
    if rho >= 1:
        return float("inf")
    a = lam / mu
    s = sum(a**k / math.factorial(k) for k in range(c))
    last = a**c / (math.factorial(c) * (1 - rho))
    p_wait = last / (s + last)
    return p_wait / (c * mu - lam) * 1000

mu = 1000 / SERVICE_MS
print(f"{'requests/sec':>14}{'workers':>9}{'utilisation':>13}{'queue wait ms':>15}")
for lam in (2, 5, 10, 20):
    for c in (1, 2, 4, 8):
        if c * mu <= lam:
            continue
        w = mm_c_wait(lam, mu, c)
        if w < 50:
            print(f"{lam:>14}{c:>9}{lam/(c*mu):>13.0%}{w:>15.1f}")
            break
```

```text
  requests/sec  workers  utilisation  queue wait ms
             2        2          20%            8.3
             5        4          25%            1.4
            10        4          50%           17.4
            20        8          50%            3.0
```

The Erlang-C result behind that table is the useful one: **many workers at
moderate utilisation beat a few workers at high utilisation**, and the effect is
large.

Four workers at 50% (10 req/s) wait 17.4 ms. One worker at 50% waits 200 ms —
from the previous table. **Same utilisation, eleven times the wait**, because a
single worker cannot absorb a burst while it is busy.

So: **pool your workers**. Four services each with one worker are much worse
than one service with four workers, even though the dashboards show identical
utilisation.

---

## Timeouts are a capacity decision

```python
CAPACITY = 5.0              # requests/sec the system can serve
for spike, timeout in [(4, 2.0), (8, 2.0), (8, 10.0)]:
    inflight = spike * timeout
    print(f"arrivals {spike}/s, timeout {timeout:.0f}s -> up to "
          f"{inflight:.0f} requests in flight, "
          f"{'system holds' if spike <= CAPACITY else 'queue grows without bound'}")
```

```text
arrivals 4/s, timeout 2s -> up to 8 requests in flight, system holds
arrivals 8/s, timeout 2s -> up to 16 requests in flight, queue grows without bound
arrivals 8/s, timeout 10s -> up to 80 requests in flight, queue grows without bound
```

When arrivals exceed capacity, **no timeout saves you** — but a long one makes
it much worse. At 8 requests per second against a capacity of 5, a 10-second
timeout holds **80 requests in flight**, each consuming memory and a connection,
all of them destined to fail.

The controls that do work, in order:

| Control | What it does |
|---|---|
| **Short timeouts** | Fail fast, release resources, tell the caller now |
| **A bounded queue** | Reject at the door with 429 rather than accept and fail late |
| **Load shedding** | Drop low-priority work first, deliberately |
| **Backpressure** | Tell the caller to slow down, with `Retry-After` |
| **Autoscaling** | Real, but minutes too late for a spike |
| **A cheaper fallback** | The small model, the cached answer, the rule |

**Rejecting a request in 50 ms is kinder than failing it in 10 seconds.** The
caller can retry, degrade, or show something else; a caller still waiting can do
none of those.

And every one of those controls is a line the frontend team needs on the
sequence diagram from lesson 02 — because "429 with Retry-After" is a state they
have to build.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Targeting high utilisation | 90% utilisation, 10x the service time |
| Capacity planning on the daily mean | Arrivals are bursty; the peak minute decides |
| One worker per service, many services | Same utilisation, 11x the wait |
| Long timeouts as "being generous" | 80 doomed requests in flight |
| No bounded queue | Accepting work you cannot finish |
| Relying on autoscaling for spikes | It arrives after the spike |
| Not telling the caller to back off | They retry, and amplify the overload |

---

## Exercises

1. Measure your service time and compute your single-worker capacity. Compare it
   with your observed peak-minute traffic.
2. Plot wait against utilisation for your service time. Mark where you operate.
3. Compute the workers needed to keep the queue wait under 20 ms at your peak.
4. Set a timeout from your budget (lesson 04) rather than from habit. What did
   it change?
5. Add a bounded queue with a 429 response, and write what the client does when
   it receives one.

---

**Next:** [Lesson 06 — Failure and Degradation](06-failure-and-degradation.md)
