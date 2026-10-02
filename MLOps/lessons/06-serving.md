# Lesson 06 — Serving

**Goal:** understand what decides a served model's latency, and what to do
about it.

## What you will learn

- Why p95 is not "a bit worse than p50"
- The utilisation number nobody tells you
- Batching, measured
- The three serving shapes, and when each is right

---

## Latency is mostly waiting

A model that takes 40 ms to run does not answer in 40 ms. It answers in 40 ms
**plus however long it waited in a queue**, and the waiting is the part that
hurts.

```python
import numpy as np

rng = np.random.default_rng(0)
SERVICE_MS = 40.0                 # the model itself, every time
N = 200_000

for util in [0.5, 0.7, 0.8, 0.9, 0.95]:
    rate = util / SERVICE_MS
    inter = rng.exponential(1 / rate, N)
    svc = rng.exponential(SERVICE_MS, N)
    arrive = np.cumsum(inter)
    finish = np.empty(N)
    free = 0.0
    for i in range(N):
        start = max(arrive[i], free)   # wait for the server to be free
        free = start + svc[i]
        finish[i] = free
    lat = finish - arrive
    print(f"util {util:>4.0%}   p50 {np.percentile(lat,50):7.1f} ms   "
          f"p95 {np.percentile(lat,95):8.1f} ms   p99 {np.percentile(lat,99):9.1f} ms")
print("\nservice time is 40 ms at every utilisation. Only the waiting changes.")
```

```text
util  50%   p50    56.0 ms   p95    243.4 ms   p99     370.7 ms
util  70%   p50    92.6 ms   p95    398.6 ms   p99     617.2 ms
util  80%   p50   140.0 ms   p95    619.6 ms   p99     994.5 ms
util  90%   p50   277.2 ms   p95   1242.6 ms   p99    1903.1 ms
util  95%   p50   598.4 ms   p95   2749.3 ms   p99    4110.8 ms

service time is 40 ms at every utilisation. Only the waiting changes.
```

**The model takes 40 ms in every single row.** At 50% utilisation the median
answer takes 56 ms. At 95% utilisation it takes 598 ms — **ten times longer,
with exactly the same model.**

And p95 is not "a bit worse than p50". At 50% utilisation p95 is **4.3x** the
median; at 95% utilisation it is 2,749 ms against a 40 ms model, a **69x**
inflation.

Three consequences for how you run a service:

**Do not run a serving box above about 70% utilisation.** The cost curve looks
like you are wasting 30% of a machine. You are buying your tail latency. Going
from 70% to 90% utilisation saves a fifth of a box and costs you **3.1x the
p95**.

**Your p95 problem is probably not your model.** Teams spend weeks quantising a
model to shave 40 ms off a 1,200 ms p95 that is 97% queueing. Measure the queue
first — the fix is another replica, not another week.

**Averages hide this entirely.** This is the same lesson as
[Data-Science 10](../../Data-Science/lessons/10-limits-and-fairness.md): the
aggregate was fine and a slice was broken. Alert on p95, report p95, promise
p95.

---

## Batching

Running one request at a time wastes the hardware. Several requests through the
same matrix multiply cost barely more than one.

```python
# skip-verify — absolute timings vary by machine; the ratios are the lesson
import time, numpy as np

W = np.random.default_rng(1).normal(size=(1024, 1024)).astype(np.float32)

def timed(bs, reps=200):
    x = np.random.default_rng(2).normal(size=(bs, 1024)).astype(np.float32)
    for _ in range(20):
        x @ W                                      # warm up
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(reps):
            x @ W
        ts.append((time.perf_counter() - t) / reps * 1000)
    return min(ts)

print(f"{'batch':>6}{'ms/batch':>11}{'ms/request':>13}{'req/s':>9}{'speedup':>9}")
base = None
for bs in [1, 4, 16, 64, 256]:
    ms = timed(bs)
    per = ms / bs
    base = base or per
    print(f"{bs:>6}{ms:>11.3f}{per:>13.4f}{1000/per:>9.0f}{base/per:>8.1f}x")
```

```text
 batch   ms/batch   ms/request    req/s  speedup
     1      0.136       0.1356     7377     1.0x
     4      0.156       0.0389    25678     3.5x
    16      0.361       0.0226    44310     6.0x
    64      0.794       0.0124    80648    10.9x
   256      2.054       0.0080   124649    16.9x
```

**A batch of 4 costs 0.156 ms against 0.136 ms for a batch of 1** — four times
the work for 15% more time. Per request that is a 3.5x speedup for free.

Pushed further, **batch 256 is 16.9x the throughput of batch 1.** But notice
the shape: 1→4 bought 3.5x, 64→256 bought only 1.6x more. The gain flattens,
because past some batch size you are no longer paying fixed overhead — you are
doing real arithmetic.

The catch, and it is the whole engineering problem: **a batch of 256 has to be
collected.** Waiting to fill it adds latency to the first request that arrived.
So the real knob is a pair:

```text
max_batch_size   = 32          how many you will group
max_wait_ms      = 10          how long you will wait to fill it
```

Serve as soon as **either** fires. At 50 requests/second, 10 ms of waiting
collects about 0.5 requests and buys you nothing; at 5,000 requests/second it
collects 50 and buys you most of the table above. **Set the pair from your
actual arrival rate**, and re-check it when traffic grows.

---

## The three shapes

| Shape | Latency budget | Right when |
|---|---|---|
| **Batch / offline** | Hours | Scores are consumed by a report or a campaign |
| **Online sync** | 50-500 ms | A user or a request is waiting for the answer |
| **Streaming / async** | Seconds | Work is queued; the caller gets an id, not an answer |

Most teams build online sync because it sounds like the real thing. **Ask what
consumes the prediction.** If it is a daily email, a batch job is cheaper, has
no tail latency, no autoscaling, no queueing, and can be re-run when it breaks.

A useful middle: **precompute what you can.** If predictions depend only on
things known yesterday, score everything overnight and serve a lookup. A
dictionary read has no p95 problem.

---

## The serving checklist

- [ ] p95 and p99 measured, not p50, and **alerted on**
- [ ] Utilisation under ~70% per replica
- [ ] `max_batch_size` and `max_wait_ms` set from the real arrival rate
- [ ] A timeout on every call, shorter than the caller's patience
- [ ] A **fallback**: last known good prediction, a heuristic, or an honest error
- [ ] Features computed by the **same code** as training (lesson 07)
- [ ] The model version in every response, logged
- [ ] Health check that actually loads the model, not just returns 200

The fallback is the one that is always missing. A model service that is down
should degrade the product, not take it down — and the baseline from
[Time-Series 01](../../Time-Series-and-Forecasting/lessons/01-baselines.md) is
already a perfectly good fallback that you have measured.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Optimising the model when the queue is the problem | 97% of a 1,200 ms p95 was waiting |
| Running at 90-95% utilisation to save cost | 3.1x the p95 for a fifth of a box |
| Promising a p50 | Nearly half your users are worse than it |
| Batch size copied from a blog post | It depends on your arrival rate |
| Waiting 50 ms to fill a batch at low traffic | Pure added latency, no throughput gain |
| Online sync for a daily report | All the hard parts, none of the need |
| No fallback | The model's outage becomes the product's outage |
| Health check that does not load the model | Green while every request fails |

---

## Exercises

1. Measure your service's p50, p95, p99 and its single-request model time. How
   much of p95 is queueing?
2. Measure your utilisation. If it is above 70%, compute what one more replica
   would do to p95.
3. Benchmark your model at batch 1, 8, 32, 128. Where does the curve flatten?
4. Compute how long 10 ms of batch-wait takes to fill at your arrival rate.
5. Write the fallback. Test it by killing the model.

---

**Next:** [Lesson 07 — Monitoring and Incidents](07-monitoring.md)
