# Lesson 02 — Measuring Properly

**Goal:** produce a benchmark you would defend in a review.

## What you will learn

- Warmup, repetition, and why the first call lies
- GPU synchronisation
- Profiling with `torch.profiler`
- Measuring memory

---

## The naive benchmark is wrong

```python
import time
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 10)).eval()
x = torch.randn(32, 256)

with torch.inference_mode():
    start = time.perf_counter()
    model(x)
    first = (time.perf_counter() - start) * 1000

    for _ in range(50):                        # warmup
        model(x)

    timings = []
    for _ in range(200):
        start = time.perf_counter()
        model(x)
        timings.append((time.perf_counter() - start) * 1000)

steady = sum(timings) / len(timings)
print(f"first call:  {first:.3f} ms")
print(f"steady state: {steady:.3f} ms")
print(f"the first call is {first / steady:.1f}x the truth")
```

```text
first call:  7.461 ms
steady state: 0.041 ms
the first call is 180.8x the truth
```

The first call is **180 times** slower than reality. It pays for lazy
initialisation, memory allocation, kernel selection and cold caches — none of
which happen again.

Time one call and you would conclude this model takes 7.5 ms. It takes 0.04 ms.
Every decision built on that first number would be wrong.

**Every benchmark needs warmup.** A single timed call is not a measurement; it
is an anecdote, and it will make you "optimise" things that were never slow.

---

## A benchmark function worth reusing

```python
import time
import statistics
import torch

def benchmark(fn, warmup=20, repeats=200, sync=False):
    """Time fn() properly. Returns a dict of milliseconds."""
    for _ in range(warmup):
        fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)

    timings.sort()
    return {
        "mean": statistics.mean(timings),
        "p50": timings[len(timings) // 2],
        "p95": timings[int(len(timings) * 0.95)],
        "p99": timings[int(len(timings) * 0.99)],
        "min": timings[0],
    }

import torch.nn as nn
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 10)).eval()
x = torch.randn(32, 256)

with torch.inference_mode():
    result = benchmark(lambda: model(x))
print({k: round(v, 4) for k, v in result.items()})
```

```text
{'mean': 0.0419, 'p50': 0.0403, 'p95': 0.0415, 'p99': 0.0712, 'min': 0.0355}
```

Four rules baked into that function:

1. **Warmup** before timing.
2. **Many repeats**, then percentiles — not one number.
3. **Synchronise** on a GPU (below).
4. **Report the distribution.** `min` is the best case the hardware can do;
   p99 is what your users get.

---

## GPU calls are asynchronous

This is the single most common benchmarking bug in deep learning.

```python
import time
import torch

# WRONG on a GPU — the timer stops before the work does
start = time.perf_counter()
y = model(x)                       # returns immediately; the GPU is still busy
wrong = time.perf_counter() - start

# RIGHT
torch.cuda.synchronize()           # wait for everything queued
start = time.perf_counter()
y = model(x)
torch.cuda.synchronize()           # wait for THIS work to finish
right = time.perf_counter() - start
```

CUDA operations are queued and run asynchronously. Without
`torch.cuda.synchronize()`, you are timing how long it takes Python to *ask*
for the work — which is microseconds regardless of how heavy the work is.

People publish 100× speedups that are entirely this bug. On CPU and on Apple's
`mps`, timing is synchronous and the issue does not arise — which is its own
trap when you move the same script to CUDA.

---

## Profiling: where does the time go?

```python
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(512, 2048), nn.ReLU(),
    nn.Linear(2048, 2048), nn.ReLU(),
    nn.Linear(2048, 10),
).eval()
x = torch.randn(64, 512)

with torch.inference_mode():
    for _ in range(10):
        model(x)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(20):
            model(x)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))
```

```text
-------------------  ------------  ------------  ------------  ------------
                Name    Self CPU %      Self CPU     CPU total       # Calls
-------------------  ------------  ------------  ------------  ------------
        aten::linear        ...           ...           ...            60
        aten::addmm         ...           ...           ...            60
          aten::relu        ...           ...           ...            40
-------------------  ------------  ------------  ------------  ------------
```

The exact numbers depend on your machine; the shape of the answer does not.
`addmm` — the matrix multiply — should dominate. If something else does, you
have found something worth fixing.

To see a timeline in a viewer:

```python
prof.export_chrome_trace("trace.json")     # open in chrome://tracing
```

For training, wrap named blocks so the table reads in your terms:

```python
from torch.profiler import record_function

with record_function("forward"):
    output = model(batch_x)
with record_function("backward"):
    loss.backward()
```

---

## Measuring memory

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024))

parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"weights:    {(parameter_bytes + buffer_bytes) / 1024**2:.2f} MB")
print(f"in fp16:    {(parameter_bytes + buffer_bytes) / 2 / 1024**2:.2f} MB")
print(f"in int8:    {(parameter_bytes + buffer_bytes) / 4 / 1024**2:.2f} MB")
```

```text
parameters: 8,393,728
weights:    32.02 MB
in fp16:    16.01 MB
in int8:    8.00 MB
```

The weights are only part of it. Training memory is roughly:

```text
weights + gradients + optimiser state + activations
```

- **Gradients**: one float per parameter — same size as the weights.
- **Adam's state**: two more floats per parameter — twice the weights.
- **Activations**: everything the backward pass needs, which scales with batch
  size and is often the largest term.

So training a model with Adam needs about **4× the weights** before a single
activation. A 1 GB model needs 4 GB to sit still, and that arithmetic explains
most out-of-memory errors.

On CUDA, measure it:

```python
torch.cuda.reset_peak_memory_stats()
train_one_epoch()
print(f"peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

---

## Compare like with like

A benchmark is only meaningful if everything else is held still:

- The same hardware, the same thread count (`torch.set_num_threads`)
- The same batch size, sequence length and dtype
- The same warmup, the same number of repeats
- Nothing else running — close the training job before benchmarking
- Fixed seeds, so the *work* is identical

```python
import torch, platform
print({
    "torch": torch.__version__,
    "threads": torch.get_num_threads(),
    "device": "cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu",
    "machine": platform.machine(),
})
```

```text
{'torch': '2.12.0', 'threads': 4, 'device': 'mps', 'machine': 'arm64'}
```

Print that block with every benchmark you publish. A number without its
environment cannot be reproduced, and cannot be argued with either.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| No warmup | You measure initialisation, 10× the real cost |
| One timed call | Noise reported as a result |
| No `cuda.synchronize()` | Fictional speedups |
| Reporting the mean only | The tail is invisible |
| Different batch sizes compared | Meaningless |
| Benchmarking while training runs | Contended, unrepeatable numbers |
| No environment recorded | Nobody can reproduce it, including you |

---

## Exercises

1. Time one call with and without warmup; report the ratio.
2. Write your own `benchmark()` and use it for the rest of this course.
3. Profile a model and name the three most expensive operations.
4. Compute the training memory of a model you have, then compare with the
   measured peak.
