# Lesson 12 — Serving and Cost

**Goal:** turn latency and throughput into money, and then reduce it.

## What you will learn

- Batching, static and dynamic
- Caching
- The KV cache, for generation
- Cost arithmetic you can put in a proposal

---

## Batching is the first lever

```python
import time
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 10)).eval()

def measure(batch_size, repeats=100):
    x = torch.randn(batch_size, 256)
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        per_batch = (time.perf_counter() - start) / repeats
    return per_batch * 1000, per_batch / batch_size * 1000

print(f"{'batch':>6}{'ms/batch':>11}{'ms/sample':>12}{'samples/s':>12}")
for batch_size in [1, 4, 16, 64, 256]:
    per_batch, per_sample = measure(batch_size)
    print(f"{batch_size:>6}{per_batch:>11.3f}{per_sample:>12.4f}{1000 / per_sample:>12.0f}")
```

```text
 batch   ms/batch   ms/sample   samples/s
     1      0.012      0.0117       85496
     4      0.032      0.0080      124645
    16      0.033      0.0021      479742
    64      0.070      0.0011      909139
   256      0.211      0.0008     1213422
```

Cost per sample falls by **10.6×** from batch 1 to batch 64 — 0.0117 ms per
sample down to 0.0011. Past 64 the returns shrink sharply: four times the
batch buys 33% more throughput for three times the per-batch latency.

That knee is the number to find on your own hardware. Everything below is
about reaching it without making users wait.

---

## Dynamic batching

Requests arrive one at a time. Hold them for a few milliseconds and process
the group.

```python
import asyncio
import torch

class BatchingService:
    """Collect requests for `window` seconds or until `max_batch`, then run once."""

    def __init__(self, model, max_batch=32, window=0.01):
        self.model = model
        self.max_batch = max_batch
        self.window = window
        self.queue = []

    async def predict(self, features):
        future = asyncio.get_event_loop().create_future()
        self.queue.append((features, future))
        if len(self.queue) == 1:
            asyncio.create_task(self._flush_later())
        elif len(self.queue) >= self.max_batch:
            self._flush()
        return await future

    async def _flush_later(self):
        await asyncio.sleep(self.window)
        if self.queue:
            self._flush()

    def _flush(self):
        batch, self.queue = self.queue, []
        x = torch.stack([features for features, _ in batch])
        with torch.inference_mode():
            outputs = self.model(x)
        for (_, future), output in zip(batch, outputs):
            if not future.done():
                future.set_result(output)
```

The trade, stated exactly: **every request waits up to `window` milliseconds**,
and in exchange throughput rises towards the batched number above.

A 10 ms window against a 150 ms p99 budget costs 7% of the budget and can
multiply throughput several times. That is usually a good trade — and it is
why TorchServe, Triton, vLLM and TGI all do this for you.

---

## Caching

The cheapest inference is the one you do not run.

```python
import hashlib
import json
from functools import lru_cache

def cache_key(payload: dict) -> str:
    """A stable key for a request, independent of dict ordering."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

@lru_cache(maxsize=10_000)
def cached_predict(key: str, model_version: str):
    ...                                   # the expensive call

request = {"text": "the coffee was cold", "language": "en"}
print(cache_key(request)[:16])
```

```text
de2934a38b51074a
```

Two rules:

- **Include the model version in the key.** Deploy a new model and the old
  answers must not survive.
- **Bound the cache.** An unbounded dict is a memory leak with a delay.

Where caching pays: repeated queries (support questions, popular searches),
embeddings of documents that do not change, and any deterministic
preprocessing. Where it does not: unique inputs, personalised outputs, or a
model whose answer depends on time.

Hit rate is the whole story. At a 40% hit rate you have cut your compute bill
by 40%, which no amount of kernel fusion will match.

---

## The KV cache

For generation, this is the difference between usable and not.

```python
import torch

def attention_cost(n_tokens, dim=768):
    """Rough multiply count to produce n tokens, with and without a KV cache."""
    without = sum(i * dim for i in range(1, n_tokens + 1))   # re-read everything
    with_cache = n_tokens * dim                              # read the cache
    return without, with_cache

for n in [10, 100, 1_000]:
    without, cached = attention_cost(n)
    print(f"{n:>5} tokens: without {without:>12,}  with cache {cached:>10,}"
          f"   {without / cached:>6.1f}x")
```

```text
   10 tokens: without       42,240  with cache      7,680      5.5x
  100 tokens: without    3,878,400  with cache     76,800     50.5x
 1000 tokens: without  384,384,000  with cache    768,000    500.5x
```

Without a cache, generating token 1,000 re-processes the previous 999 — the
total work is quadratic in length. The KV cache stores each token's keys and
values once, so each new token is O(1) against the stored context, and the
total becomes linear.

The cost is memory:

```python
def kv_cache_gb(layers=32, heads=32, head_dim=128, sequence=4096, batch=1, bytes_per=2):
    """Keys and values, per layer, for the whole context."""
    return 2 * layers * heads * head_dim * sequence * batch * bytes_per / 1024**3

print(f"1 sequence of 4k:  {kv_cache_gb():.2f} GB")
print(f"32 sequences:      {kv_cache_gb(batch=32):.2f} GB")
print(f"32 at 32k context: {kv_cache_gb(sequence=32_768, batch=32):.2f} GB")
```

```text
1 sequence of 4k:  2.00 GB
32 sequences:      64.00 GB
32 at 32k context: 512.00 GB
```

Thirty-two concurrent users at a 32k context need **half a terabyte** of KV
cache — far more than the model weights. This is why serving systems fight so
hard over it: paged attention (vLLM), multi-query and grouped-query attention,
and cache quantisation all exist to shrink that number.

---

## The cost arithmetic

Put this in the proposal, before anyone buys a GPU.

```python
def cost_per_thousand(latency_ms, hourly_cost, utilisation=0.7):
    """Dollars per 1,000 requests for one replica."""
    requests_per_hour = 3600 / (latency_ms / 1000) * utilisation
    return hourly_cost / requests_per_hour * 1000

scenarios = [
    ("CPU, 4 vCPU",        45.0, 0.15),
    ("CPU + int8",         28.0, 0.15),
    ("CPU + batching (32)", 6.0, 0.15),
    ("T4 GPU",              8.0, 0.53),
    ("A100 GPU",            3.0, 3.67),
]

print(f"{'setup':<22}{'ms':>7}{'$/hour':>9}{'$/1k req':>11}")
for name, latency, hourly in scenarios:
    print(f"{name:<22}{latency:>7.1f}{hourly:>9.2f}{cost_per_thousand(latency, hourly):>11.4f}")
```

```text
setup                      ms   $/hour   $/1k req
CPU, 4 vCPU              45.0     0.15     0.0027
CPU + int8               28.0     0.15     0.0017
CPU + batching (32)       6.0     0.15     0.0004
T4 GPU                    8.0     0.53     0.0017
A100 GPU                  3.0     3.67     0.0044
```

The A100 is the fastest **and the most expensive per request** — 10× the cost
of a batched CPU. Speed and cost-efficiency are different axes, and the
instinct to reach for the biggest GPU is usually wrong for a small model.

Read the second row too: quantisation cut the cost per request by 37% for the
price of an afternoon.

At scale:

```python
monthly_requests = 10_000_000
for name, latency, hourly in scenarios:
    monthly = cost_per_thousand(latency, hourly) * monthly_requests / 1000
    print(f"{name:<22} ${monthly:>8.2f} per month")
```

```text
CPU, 4 vCPU            $   26.79 per month
CPU + int8             $   16.67 per month
CPU + batching (32)    $    3.57 per month
T4 GPU                 $   16.83 per month
A100 GPU               $   43.69 per month
```

Ten million requests a month, and the difference between the best and worst
choice is $40. **Optimise the thing that is actually expensive** — at this
scale that is engineering time, not compute, and the right answer may be to
stop optimising. Multiply the numbers by a thousand and the conclusion flips;
do the arithmetic for your scale rather than inheriting someone else's.

---

## The serving checklist

- [ ] The model is loaded once, at startup
- [ ] The batch size is at or below the throughput knee
- [ ] Dynamic batching, with a window inside the latency budget
- [ ] A bounded cache, keyed including the model version
- [ ] p50 and p99 measured at the API, not at the model
- [ ] Autoscaling on a metric that reflects load (queue depth, not CPU)
- [ ] A cost-per-1,000-requests figure, reviewed against the budget
- [ ] Load tested at 2× expected peak

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Batch size past the knee | Latency paid for no throughput |
| No dynamic batching | A GPU at 10% utilisation |
| Unbounded cache | Memory grows until the pod is killed |
| Cache key without the model version | Stale answers after a deploy |
| Measuring latency at the model | The real p99 is at the API |
| Buying a GPU before doing the arithmetic | 10× the cost per request |
| Ignoring the KV cache in memory planning | Out of memory at the third user |

---

## Exercises

1. Find the throughput knee for a model of yours; plot samples/s against batch
   size.
2. Implement dynamic batching and measure throughput at a 5, 10 and 50 ms
   window.
3. Compute the KV cache size for a model you use at your real context length.
4. Fill in the cost table with your actual latencies and cloud prices. What is
   the cheapest configuration that meets the budget?
