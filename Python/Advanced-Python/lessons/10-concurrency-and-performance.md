# Lesson 10 — Concurrency and Performance

**Goal:** make slow code fast, in the right order.

## What you will learn

- Profile before optimising
- Threads vs processes vs async, and which problem each solves
- The GIL, stated plainly
- Vectorisation and caching

---

## The order of operations

```mermaid
flowchart LR
    A["1. Make it correct"] --> B["2. Measure"]
    B --> C["3. Better algorithm"]
    C --> D["4. Vectorise / cache"]
    D --> E["5. Parallelise"]
    E --> F["6. Rewrite in C / Rust"]
```

Most people jump to step 5. An O(n²) loop spread across eight cores is still
O(n²) — you bought a factor of eight and left a factor of a thousand on the
table at step 3.

---

## Measure

```python
import time

start = time.perf_counter()
total = sum(range(5_000_000))
print(f"{time.perf_counter() - start:.3f}s")
```

```text
0.085s
```

For a whole script:

```bash
python -m cProfile -s cumtime script.py | head -20
```

Line by line, when you know the function but not the line:

```bash
pip install line_profiler
kernprof -l -v script.py        # decorate the function with @profile
```

Memory:

```bash
pip install memory_profiler
python -m memory_profiler script.py
```

In a notebook: `%timeit`, `%%time`, `%prun`.

---

## CPU-bound or I/O-bound?

Everything below depends on this one question.

| | CPU-bound | I/O-bound |
|---|---|---|
| The program is busy | Calculating | Waiting |
| Examples | Training, parsing, image processing | API calls, database queries, file reads |
| Use | `multiprocessing`, NumPy, C extensions | `threading`, `asyncio` |
| Threads help? | **No** — the GIL | **Yes**, a lot |

---

## The GIL

CPython allows only one thread to execute Python bytecode at a time. Threads do
not give you parallel computation.

They still help when the work is waiting, because a thread releases the GIL
during I/O — while one waits for an HTTP response, another runs.

```python
import time
from concurrent.futures import ThreadPoolExecutor

def fetch(n):
    time.sleep(0.5)          # stands in for a network call
    return n

start = time.perf_counter()
results = [fetch(i) for i in range(8)]
print(f"sequential: {time.perf_counter() - start:.2f}s")

start = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(fetch, range(8)))
print(f"threaded:   {time.perf_counter() - start:.2f}s")
```

```text
sequential: 4.01s
threaded:   0.50s
```

Eight-fold, because all eight spend their time waiting.

(NumPy, pandas and PyTorch release the GIL inside their compiled routines, so
they already use multiple cores without you doing anything.)

---

## Processes for CPU work

```python
import time
from concurrent.futures import ProcessPoolExecutor

def heavy(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    work = [2_000_000] * 4

    start = time.perf_counter()
    [heavy(n) for n in work]
    print(f"sequential: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        list(pool.map(heavy, work))
    print(f"processes:  {time.perf_counter() - start:.2f}s")
```

```text
sequential: 3.42s
processes:  1.04s
```

Each process has its own interpreter and its own GIL, so they genuinely run in
parallel. The costs: starting a process is slow, and arguments and results are
pickled and copied between processes. Parallelising something small loses to
that overhead.

The `if __name__ == "__main__":` guard is **required** on macOS and Windows.
Without it, each child re-imports the module and spawns more children.

---

## asyncio

For thousands of concurrent I/O operations, one thread each is too many.
`asyncio` runs them all on one thread, switching whenever one waits.

```python
import asyncio
import time

async def fetch(n):
    await asyncio.sleep(0.5)
    return n

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(*(fetch(i) for i in range(100)))
    print(f"{len(results)} tasks in {time.perf_counter() - start:.2f}s")

asyncio.run(main())
```

```text
100 tasks in 0.50s
```

A hundred operations in the time of one. The rule: `await` anything that waits,
and never call a blocking function inside async code — one `time.sleep` or one
synchronous `requests.get` freezes every task in the loop. Use `aiohttp` or
`httpx` instead.

Threads or asyncio? Threads for tens of operations and ordinary libraries;
asyncio for thousands, and when the libraries support it.

---

## Vectorise before you parallelise

```python
import time
import numpy as np

values = list(range(1_000_000))
array = np.arange(1_000_000)

start = time.perf_counter()
result = [v * 2 + 1 for v in values]
print(f"python: {time.perf_counter() - start:.3f}s")

start = time.perf_counter()
result = array * 2 + 1
print(f"numpy:  {time.perf_counter() - start:.3f}s")
```

```text
python: 0.098s
numpy:  0.003s
```

Thirty times faster on one core, with no concurrency to reason about and no
bugs to introduce. In pandas the equivalent is: no `iterrows`, no `apply` where
a vectorised operation exists.

---

## Cache

```python
import functools
import time

@functools.lru_cache(maxsize=1024)
def expensive(n):
    time.sleep(0.2)
    return n * n

start = time.perf_counter()
expensive(10); expensive(10); expensive(10)
print(f"{time.perf_counter() - start:.2f}s")
```

```text
0.20s
```

Three calls, one execution. The fastest code is the code that does not run.

---

## A checklist for slow code

1. Profile. Find the top three costs.
2. Is the algorithm quadratic? Fix that first — set instead of list, dict
   instead of scan, join instead of nested loop.
3. Are you looping over rows? Vectorise.
4. Are you recomputing? Cache.
5. Are you waiting on I/O? Threads or asyncio.
6. Are you CPU-bound and out of ideas? Processes.
7. Still slow? Now consider Cython, Numba, Rust, or a bigger machine.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Optimising before profiling | Time spent on 2% of the runtime |
| Threads for CPU work | No speedup — the GIL |
| Processes for tiny tasks | Slower than sequential, from overhead |
| Blocking calls inside async | Every task stalls |
| Shared mutable state across threads | Race conditions, irreproducible bugs |
| Missing `if __name__ == "__main__"` | Process bomb on macOS and Windows |

---

## Exercises

1. Profile a script of yours and name its top three costs.
2. Fetch 20 URLs sequentially and with a `ThreadPoolExecutor`; compare.
3. Convert a Python loop over a million values to NumPy; measure both.
4. Take a CPU-heavy function and parallelise it with `ProcessPoolExecutor`.
   At what input size does it start to win?
