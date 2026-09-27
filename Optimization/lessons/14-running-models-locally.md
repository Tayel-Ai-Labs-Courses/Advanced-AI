# Lesson 14 — Running Models Locally

**Goal:** run a real language model on your own hardware, measure what it costs,
and find the volume where that stops being the cheaper choice.

## What you will learn

- An 8B model on a laptop, measured
- Where local latency differs from hosted latency
- The cost arithmetic, including the part everyone gets wrong
- The throughput ceiling that decides most of these projects

---

## Why local at all

| Reason | Detail |
|---|---|
| **Privacy** | The data never leaves the building. For medical, legal or HR data this is often the whole argument |
| **Cost at volume** | No per-token bill — see the arithmetic below, and its limits |
| **No dependency** | The provider cannot deprecate your model, change it under you, or rate-limit you |
| **Latency floor** | No network round trip. Useful when the model is inside a loop |
| **Offline** | Works on a boat, in a factory, on a phone |

And the honest counterweight: a local 8B model is not a frontier model. This
lesson measures both sides.

---

## Setup

Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3        # 4.7 GB, an 8B model
ollama serve              # usually already running
```

Everything below talks to it over plain HTTP, so there is no client library to
learn and nothing to install in Python.

```python
# no-run: needs a local Ollama server
import json, time, urllib.request

OLLAMA = "http://localhost:11434"

def generate(model, prompt, num_predict=64, **opts):
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0, **opts},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    out["wall_s"] = time.perf_counter() - t0
    return out
```

---

## An 8B model on a laptop

```python
# no-run: needs a local Ollama server
r = generate("llama3", "The capital of Egypt is", num_predict=24)
print("response:", r["response"].strip()[:120].replace("\n", " "))
print(f"prompt tokens   : {r['prompt_eval_count']}")
print(f"output tokens   : {r['eval_count']}")
print(f"load time       : {r['load_duration'] / 1e9:.2f}s")
print(f"prefill         : {r['prompt_eval_duration'] / 1e9:.3f}s")
print(f"generation      : {r['eval_duration'] / 1e9:.2f}s")
print(f"tokens/sec      : {r['eval_count'] / (r['eval_duration'] / 1e9):.1f}")
```

```text
response: The capital of Egypt is Cairo.
prompt tokens   : 15
output tokens   : 8
load time       : 3.58s
prefill         : 0.215s
generation      : 0.35s
tokens/sec      : 23.1
```

**23.1 tokens per second**, on a laptop CPU, from a 4.7 GB model. That is
roughly reading speed — fast enough for a chat interface, slow enough that you
must think about it in a loop.

The `load time` of 3.58s is the model being read into memory. It happens once;
Ollama keeps the model resident for a few minutes after the last request. **In
production you keep it warm**, because a cold 3.6-second load on the first
request of the morning is a timeout somewhere.

---

## Where the time goes — and where local differs

```python
# no-run: needs a local Ollama server
print(f"{'prompt tok':>11}{'out tok':>9}{'prefill s':>11}{'decode s':>10}{'tok/s':>8}")
for words, n in [(10, 16), (10, 64), (10, 128), (400, 16), (1200, 16)]:
    prompt = "The following is a note about an order. " * (words // 8) + "Summarise it."
    r = generate("llama3", prompt, num_predict=n)
    pre = r["prompt_eval_duration"] / 1e9
    dec = r["eval_duration"] / 1e9
    print(f"{r['prompt_eval_count']:>11}{r['eval_count']:>9}{pre:>11.2f}{dec:>10.2f}"
          f"{r['eval_count'] / dec:>8.1f}")
```

```text
 prompt tok  out tok  prefill s  decode s   tok/s
         24       16       0.18      0.72    22.3
         24       24       0.05      1.12    21.4
         24       24       0.05      1.13    21.2
        465       16       2.24      0.78    20.6
       1365       16       4.22      0.75    21.5
```

Decode is flat at about **21 tokens/sec** regardless of anything else. That part
matches the hosted case.

**Prefill does not.** A 1,365-token prompt takes **4.22 seconds** before a single
output token appears — against 0.05s for a 24-token prompt. Compare that with
LLM lesson 10, where on a hosted model 800 input tokens were *cheaper* than 100
output tokens.

The difference is batching and hardware. A hosted provider processes your prompt
inside a large batch on a GPU, so prefill is nearly free. On one machine with no
batching, prefill is real work you wait for.

**So the RAG advice inverts.** LLM lesson 06 found the accuracy ceiling at about
102 context tokens and warned that more was waste; locally, "waste" is not just
money, it is **4 seconds of the user staring at a blank screen**. Local
deployment makes short context an engineering requirement, not a cost
optimisation.

---

## Is it good enough?

```python
# no-run: needs a local Ollama server
REVIEWS = [
    ("The coffee was excellent and the staff were friendly.", "positive"),
    ("Cold food, rude waiter, never coming back.", "negative"),
    ("Best latte I have had in Cairo.", "positive"),
    ("They lost my order twice and blamed me.", "negative"),
    ("Lovely place, quiet and clean.", "positive"),
    ("Overpriced and the wifi never works.", "negative"),
    ("The cake was fresh and delicious.", "positive"),
    ("I waited forty minutes for a tea.", "negative"),
    ("Great music and comfortable chairs.", "positive"),
    ("Dirty tables and a broken air conditioner.", "negative"),
    ("Friendly barista remembered my name.", "positive"),
    ("The juice tasted like it had been sitting for days.", "negative"),
    ("Reasonable prices for the quality.", "positive"),
    ("Too loud to have a conversation.", "negative"),
    ("Fast service even when busy.", "positive"),
    ("They charged me for an extra shot I never ordered.", "negative"),
    ("Beautiful interior and good lighting.", "positive"),
    ("The pastry was stale.", "negative"),
    ("Perfect spot to work from.", "positive"),
    ("Staff ignored us for twenty minutes.", "negative"),
]
correct = 0
t0 = time.perf_counter()
for text, gold in REVIEWS:
    p = (f'Classify the sentiment of this review as exactly one word, '
         f'"positive" or "negative".\nReview: {text}\nAnswer:')
    out = generate("llama3", p, num_predict=4)["response"].strip().lower()
    pred = "positive" if "positive" in out else "negative"
    correct += pred == gold
elapsed = time.perf_counter() - t0
print(f"llama3 (8B, local)    accuracy {correct/len(REVIEWS):.2f}   "
      f"{elapsed/len(REVIEWS)*1000:>6.0f} ms/item")
print(f"gpt2-medium (355M)    accuracy 0.90      165 ms/item   (LLM lesson 04)")
print(f"bert-tiny fine-tuned  accuracy 1.00     0.07 ms/item   (LLM lesson 08)")
```

```text
llama3 (8B, local)    accuracy 1.00      216 ms/item
gpt2-medium (355M)    accuracy 0.90      165 ms/item   (LLM lesson 04)
bert-tiny fine-tuned  accuracy 1.00     0.07 ms/item   (LLM lesson 08)
```

llama3 gets every review right. So does a **4.4M-parameter fine-tuned
bert-tiny**, at **1/3000th the latency** — 0.07 ms against 216 ms.

This is LLM lesson 08's conclusion arriving with a much bigger model attached:
for a narrow, stable, high-volume task, the small fine-tune wins on every axis
except the day you spend labelling. The local 8B model earns its place on the
tasks the small model cannot do at all — open-ended generation, summarisation,
anything where the output is text a person reads.

**Do not run an 8B model to do a job a logistic regression can do.**

---

## The cost arithmetic

```python
TOK_PER_SEC = 21.0
WATTS = 60              # a laptop under sustained load, above idle
KWH_PRICE = 1.45        # EGP per kWh, Egyptian residential tier
HARDWARE = 45_000       # EGP, a machine that can run an 8B model
LIFE_HOURS = 3 * 365 * 8   # three years, eight hours a day

energy_per_1k = (1_000 / TOK_PER_SEC) / 3600 * (WATTS / 1000) * KWH_PRICE
amort_per_hour = HARDWARE / LIFE_HOURS
amort_per_1k = (1_000 / TOK_PER_SEC) / 3600 * amort_per_hour
print(f"generation speed        : {TOK_PER_SEC:.0f} tokens/sec")
print(f"1,000 output tokens take: {1_000 / TOK_PER_SEC / 60:.1f} minutes")
print(f"  electricity           : {energy_per_1k:.4f} EGP")
print(f"  hardware amortised    : {amort_per_1k:.4f} EGP")
print(f"  total                 : {energy_per_1k + amort_per_1k:.4f} EGP per 1k output tokens")
```

```text
generation speed        : 21 tokens/sec
1,000 output tokens take: 0.8 minutes
  electricity           : 0.0012 EGP
  hardware amortised    : 0.0679 EGP
  total                 : 0.0691 EGP per 1k output tokens
```

**The electricity is 1.7% of the cost.** People argue about power draw and
ignore the machine, which is fifty-five times larger.

That also means the cost per token is **a function of how busy the machine is**.
A machine running at 5% utilisation costs the same as one running flat out, so
idle local hardware is the most expensive inference in the world.

```python
USD_EGP = 48.0
for name, in_usd, out_usd in [("frontier model", 3.00, 15.00),
                              ("mid-tier model", 0.50, 1.50),
                              ("small hosted model", 0.15, 0.60)]:
    api_per_1k = out_usd / 1000 * USD_EGP
    local = energy_per_1k + amort_per_1k
    print(f"{name:<20} {api_per_1k:>8.3f} EGP/1k out tokens   "
          f"local is {api_per_1k / local:>6.1f}x cheaper")
```

```text
frontier model          0.720 EGP/1k out tokens   local is   10.4x cheaper
mid-tier model          0.072 EGP/1k out tokens   local is    1.0x cheaper
small hosted model      0.029 EGP/1k out tokens   local is    0.4x cheaper
```

Against a frontier model, local is **10.4x cheaper**. Against a mid-tier hosted
model it is **exactly a wash**. Against a small hosted model, local is **more
than twice as expensive** — and you also own the machine, the updates and the
on-call.

**"Local is cheaper" is only true against the expensive tier.** If your task
runs fine on a small hosted model, local is a privacy decision, not a cost one —
and it should be argued on those terms.

---

## The ceiling nobody checks first

```python
print(f"{'requests/day':>14}{'output tok each':>17}{'hours of compute':>19}{'feasible?':>12}")
for reqs, out in [(100, 200), (1_000, 200), (10_000, 200), (1_000, 2_000)]:
    seconds = reqs * out / TOK_PER_SEC
    hours = seconds / 3600
    print(f"{reqs:>14,}{out:>17}{hours:>19.1f}{'yes' if hours < 20 else 'NO':>12}")
print("\none machine, one request at a time, 24 hours available")
```

```text
  requests/day  output tok each   hours of compute   feasible?
           100              200                0.3         yes
         1,000              200                2.6         yes
        10,000              200               26.5          NO
         1,000             2000               26.5          NO

one machine, one request at a time, 24 hours available
```

**10,000 requests a day needs 26.5 hours of compute.** There are 24. The project
is impossible on one machine before any cost argument begins.

This is the first calculation to do, and it is usually done last. One machine at
21 tokens/sec produces about **1.8 million output tokens a day** at 100%
utilisation, which no real system reaches.

Now watch what happens when you forget it:

```python
print(f"{'daily output tokens':>21}{'API cost/month':>17}{'local cost/month':>19}")
for daily in (10_000, 100_000, 1_000_000, 5_000_000):
    api = daily * 30 * (15.00 / 1e6) * USD_EGP
    hours = daily * 30 / TOK_PER_SEC / 3600
    local = hours * (WATTS / 1000 * KWH_PRICE + amort_per_hour)
    print(f"{daily:>21,}{api:>17,.0f}{local:>19,.0f}")
print("\nlocal cost is bounded by the hardware; API cost is not")
```

```text
  daily output tokens   API cost/month   local cost/month
               10,000              216                 21
              100,000            2,160                207
            1,000,000           21,600              2,073
            5,000,000          108,000             10,365

local cost is bounded by the hardware; API cost is not
```

This table says local is ten times cheaper at every volume, and **the bottom two
rows are fiction**. One million output tokens a day is 13 hours of compute — just
possible. Five million is **66 hours a day**, which means three machines, and the
"local cost" column has quietly assumed infinite throughput.

**A cost model without a capacity model is a sales pitch.** Compute the hours
first; price only the volumes that fit.

---

## Choosing

| Situation | Choose |
|---|---|
| Sensitive data that cannot leave | **Local**, and argue it on privacy |
| Low volume, expensive model needed | Hosted — local hardware idles |
| High volume, small model sufficient | **Fine-tune something tiny** (LLM lesson 08) |
| High volume, generation needed | Local, on a GPU, with batching — or hosted |
| Offline or intermittent connectivity | **Local**, no argument needed |
| Prototyping | Hosted. Switch later; the code is the same shape |
| Unpredictable spikes | Hosted, or hybrid with local as the floor |

The hybrid deserves a mention: run local for the steady baseline load and
overflow to a hosted API at the peak. It caps the bill and removes the capacity
cliff, at the cost of two code paths and two sets of evals (LLM lesson 07 — the
two models will not agree).

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Pricing local by electricity | Electricity was 1.7% of the cost |
| No capacity calculation | 10,000 requests a day needs 26.5 hours on one machine |
| Assuming local is always cheaper | It is 0.4x against a small hosted model |
| Long RAG context on local hardware | 4.22 seconds of prefill before the first token |
| A cold model on the first request | 3.6 seconds, which is someone's timeout |
| An 8B model for a classification task | bert-tiny matched it at 1/3000th the latency |
| Comparing quality by vibes | Run LLM lesson 07's eval set on both |

---

## Exercises

1. Measure tokens/sec on your own machine for a 3B, an 8B and a 14B model.
   Plot it against model size.
2. Compute your own break-even: your volume, your hardware price, your
   electricity tariff. At which hosted tier does local win?
3. Compute your capacity ceiling in requests per day, then check whether your
   expected volume fits under it.
4. Run LLM lesson 07's evaluation set against a local model and a hosted one.
   Report both accuracies and both costs.
5. Measure the prefill time at 100, 500, 2,000 and 8,000 prompt tokens. At what
   length does your p95 latency budget break?

---

**Next:** [Lesson 15 — Edge and On-Device](15-edge-and-on-device.md)
