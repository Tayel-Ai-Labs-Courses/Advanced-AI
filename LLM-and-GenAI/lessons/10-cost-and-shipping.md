# Lesson 10 — Cost, Latency and Shipping

**Goal:** turn a working prototype into a feature that can survive its own
invoice.

## What you will learn

- Where the latency actually is
- Batching, and why it is free throughput
- Caching, and what decides whether it works
- The shipping checklist

---

## Setup

```python
import os, warnings, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2-medium")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2-medium").eval()
print("ready")
```

```text
ready
```

---

## Input tokens are cheap, output tokens are not

```python
# skip-verify: wall-clock timings vary by machine
def timed(prompt_tokens, new_tokens, repeats=3):
    prompt = "word " * prompt_tokens
    ids = tok(prompt, return_tensors="pt")
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**ids, max_new_tokens=new_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        ts.append(time.perf_counter() - t0)
    return min(ts)

print(f"{'prompt tok':>11}{'new tok':>9}{'seconds':>10}{'ms/new token':>15}")
for p, n in [(50, 10), (50, 50), (50, 100), (400, 10), (400, 50), (800, 10)]:
    t = timed(p, n)
    print(f"{p:>11}{n:>9}{t:>10.2f}{t / n * 1000:>15.0f}")
```

```text
 prompt tok  new tok   seconds   ms/new token
         50       10      0.48             48
         50       50      2.00             40
         50      100      3.46             35
        400       10      0.86             86
        400       50      1.75             35
        800       10      0.98             98
```

Compare the rows that matter.

**50 prompt tokens + 100 new tokens: 3.46 s.**
**800 prompt tokens + 10 new tokens: 0.98 s.**

The second request has **sixteen times the input** and takes **less than a
third of the time**, because generation splits into two very different phases:

| Phase | What happens | Cost |
|---|---|---|
| **Prefill** | The whole prompt goes through in one parallel pass | Nearly flat in prompt length |
| **Decode** | One forward pass per output token, strictly sequential | Linear in output length |

You cannot parallelise decode — token 40 needs token 39 — so **output length is
the latency budget**. Two consequences:

- "Answer in one sentence" is a **performance optimisation**, not a style
  preference. It can halve your p95.
- Streaming does not make the answer faster; it makes the *first* token fast.
  For a chat interface that is most of the perceived improvement, and it costs
  a few lines.

---

## Batching is free throughput

```python
# skip-verify: wall-clock timings vary by machine
prompt = "word " * 50
for bs in (1, 4, 16):
    batch = tok([prompt] * bs, return_tensors="pt", padding=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**batch, max_new_tokens=20, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    t = time.perf_counter() - t0
    print(f"batch {bs:>3}: {t:.2f}s total, {t / bs * 1000:>6.0f} ms per request")
```

```text
batch   1: 0.63s total,    633 ms per request
batch   4: 1.33s total,    332 ms per request
batch  16: 2.00s total,    125 ms per request
```

**Five times the throughput at batch 16.** Sixteen requests in 2.00 s against
one in 0.63 s.

The model is memory-bandwidth-bound: loading 355M weights dominates, and once
loaded they serve the whole batch. This is why hosted inference is cheaper than
your own single-request server, and why any offline job — classifying a day of
reviews, embedding a corpus — should batch.

The cost is **latency for the individual request**: batch 16 took 2.00 s wall
clock, so a request that arrives first waits for the batch to fill. Continuous
batching (vLLM, TGI) is the production answer; it adds each arrival to the
running batch instead of waiting for a fixed size.

---

## Caching, and what decides whether it works

```python
QUESTIONS = [f"q{i}" for i in range(50)]

def simulate(n_requests, s, cache_size=10, seed=0):
    """Zipf-distributed traffic over 50 distinct questions, LRU-ish cache."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, len(QUESTIONS) + 1)
    p = ranks ** (-s)
    p /= p.sum()
    picks = rng.choice(len(QUESTIONS), size=n_requests, p=p)
    cache, order, hits = set(), [], 0
    for i in picks:
        q = QUESTIONS[i]
        if q in cache:
            hits += 1
        else:
            cache.add(q)
            order.append(q)
            if len(cache) > cache_size:
                cache.discard(order.pop(0))
    return hits / n_requests, p[:3].sum()

print(f"{'traffic shape':<34}{'top-3 share':>13}{'hit rate':>11}{'cost':>8}")
for s, label in [(0.0, "uniform (every question equal)"),
                 (0.8, "mildly skewed"),
                 (1.2, "typical support traffic"),
                 (2.0, "very skewed (FAQ-dominated)")]:
    hit, top3 = simulate(20_000, s)
    print(f"{label:<34}{top3:>13.1%}{hit:>11.1%}{1 - hit:>8.1%}")
print("\ncache holds 10 of 50 distinct questions")
```

```text
traffic shape                       top-3 share   hit rate    cost
uniform (every question equal)             6.0%      20.1%   79.9%
mildly skewed                             30.5%      34.6%   65.4%
typical support traffic                   51.5%      55.9%   44.1%
very skewed (FAQ-dominated)               83.8%      89.0%   11.0%

cache holds 10 of 50 distinct questions
```

Same cache, same size, four traffic shapes, and hit rate ranges from **20% to
89%** — a bill of 79.9% against 11.0% of uncached cost.

**The cache's value is decided by your traffic, not by your cache.** Before
building one, measure the distribution of your actual requests. If the top 3
questions are half your traffic, a ten-entry cache removes half your bill; if
every request is unique, caching the whole response is worthless.

Three layers, each worth different amounts:

| Layer | Caches | When it works |
|---|---|---|
| **Exact response cache** | prompt -> answer | Repeated identical questions. The table above |
| **Semantic cache** | similar question -> answer | Paraphrases. Needs lesson 05's embeddings and a threshold you must tune — and a wrong hit answers the wrong question |
| **Prompt / prefix cache** | the KV state of a shared prefix | Always, if your provider offers it. A long system prompt is billed once |

The third is the one to enable first: it needs no logic from you, it cannot
return a wrong answer, and lesson 02's 121-token system prompt is re-sent on
every single call without it.

**And caching only works with deterministic decoding.** Lesson 03 measured it:
top-p sampling gave 5 different outputs from 5 seeds. Cache a sampled response
and every hit serves one arbitrary sample forever.

---

## The route table

Most requests do not need your most expensive path. Decide per request.

```text
1. cache hit?                    -> return, 0 tokens
2. classifier says simple?       -> small fine-tuned model (lesson 08), 0.07 ms
3. answerable from documents?    -> RAG with k at the knee (lesson 06)
4. needs open generation?        -> the large model, constrained (lesson 09)
5. low confidence or flagged?    -> human review queue
```

A system that sends every request to step 4 is paying its maximum price for its
average request. The steps above it are cheap to build and were each measured
earlier in this course.

---

## Shipping checklist

- [ ] **Token budget per request**, enforced — input truncation and
      `max_new_tokens`. One runaway prompt should not cost 100x
- [ ] **Timeout and fallback**: what the user sees when the model is slow or down
- [ ] **Retry policy** bounded, with the retry rate monitored (lesson 09)
- [ ] **Prompt version recorded with every response**, like a model version
      (Data-Science lesson 07) — you cannot debug an answer without the prompt
      that produced it
- [ ] **Inputs and outputs logged**, with PII redacted, sampled if volume is high
- [ ] **Eval set in CI** (lesson 07), including the unanswerable subset
- [ ] **Cost dashboard**: tokens in, tokens out, cost per request, per day
- [ ] **Cache hit rate** monitored — a drop means traffic changed
- [ ] **A kill switch** that falls back to a non-AI path
- [ ] **Model version pinned.** A provider upgrading the model under you is a
      silent deployment you did not test
- [ ] **A refusal path** the user can act on: "I could not find this — here is
      how to reach a person"

The two most commonly missing: **prompt versioning** and **model pinning**. Both
turn "it got worse last Tuesday and nobody knows why" into a diff.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Budgeting input and output tokens the same | 100 output tokens cost more than 800 input tokens |
| Long answers by default | Output length is the latency budget |
| One request at a time in an offline job | 5x throughput left on the table |
| Building a cache without measuring traffic | 20% hit rate on uniform traffic |
| Caching sampled output | Every hit serves one arbitrary sample forever |
| Semantic cache with an untuned threshold | Confidently answers the wrong question |
| No prompt version in the logs | The answer cannot be reproduced |
| Not pinning the model version | Silent deployments you did not test |
| Everything routed to the biggest model | Maximum price for the average request |

---

## Exercises

1. Measure your own p50 and p95 latency against output length. Find the
   `max_new_tokens` that meets a 2-second p95.
2. Take one day of real queries and compute the top-3 share. Using the table
   above, predict your cache hit rate before building the cache.
3. Build the semantic cache with lesson 05's embeddings. Find the similarity
   threshold at which it first returns a wrong answer, and set the production
   threshold well below it.
4. Add prompt versioning: hash the prompt template and store the hash on every
   response. Write the query that finds all answers produced by version `abc123`.
5. Price the full feature at your real volume: tokens, cache hit rate, retry
   rate, and the human review queue. Then state the volume at which a
   fine-tuned small model (lesson 08) becomes cheaper.

---

**Done with the lessons.** Next: [Project 12](../Project-12/) — ship one LLM
feature, with its evaluation and its bill.

---

## Where to go next

| Next | Why |
|---|---|
| [This course's project](../Project-12/) | It is the assessment, and it is not optional |
| [AI-Agents](../../AI-Agents/) | What happens when a prompt gets powers |
| [MLOps](../../MLOps/) | Versioning, gates and monitoring |
| [Advanced-Prompt-Engineering](https://github.com/Tayel-Ai-Labs-Courses/Advanced-Prompt-Engineering) | The technique catalogue, a separate repository |
