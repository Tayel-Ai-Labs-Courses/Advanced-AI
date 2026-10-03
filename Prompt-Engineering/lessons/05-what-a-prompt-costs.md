# Lesson 05 — What a Prompt Costs

**Goal:** treat the prompt as something you pay for on every single request.

## What you will learn

- Prompt tokens and latency, measured against the number of examples
- The arithmetic that decides how many examples you can afford
- Why the static part of a prompt is the cheap part
- The three ways a prompt gets expensive without anyone noticing

---

## The prompt is a per-request cost

A worked example costs 28 tokens once, in your editor. It costs 28 tokens
**on every request, forever**. At a million requests a month that is a line on
an invoice.

This is the thing that separates a prompt that works from a prompt that ships.
[Lesson 02](02-the-output-contract.md) found that one example takes a prompt
from 0/12 to 12/12 — worth any price. The question this lesson answers is what
the *eighth* example is worth.

---

## Setup

```python
import os, warnings, re, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
model.eval()

@torch.no_grad()
def generate(prompt, n=20):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:])
```

---

## Length, tokens, latency

```python
# skip-verify — millisecond timings are machine-specific; the shape is the lesson
SHOT = "Review: the staff were rude\nSentiment: negative\n\n"

def latency(prompt, reps=5):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids)
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); model(ids); ts.append(time.perf_counter() - t)
    return min(ts) * 1000

print(f"{'shots':>6}{'prompt tokens':>15}{'ms/call':>10}{'tokens/1k calls':>18}")
for k in [0, 1, 2, 4, 8, 16]:
    p = SHOT * k + "Review: the coffee was cold\nSentiment:"
    n = len(tok(p).input_ids)
    print(f"{k:>6}{n:>15}{latency(p):>10.1f}{n*1000:>18,}")
```

```text
 shots  prompt tokens   ms/call   tokens/1k calls
     0             10      63.2            10,000
     1             23      87.2            23,000
     2             36      96.9            36,000
     4             62     108.0            62,000
     8            114     161.3           114,000
    16            218     259.9           218,000
```

**Sixteen examples cost 21.8x the tokens of none and 4.1x the latency.**

Now put that beside what the examples bought. From
[lesson 02](02-the-output-contract.md): the **first** example took the prompt
from unusable to working. From
[LLM 04](../../LLM-and-GenAI/lessons/04-prompting.md): going past two examples
did not reliably improve anything, and 4-shot scored *worse* than 2-shot.

```text
example  1    0/12 -> 12/12 parse rate       13 extra tokens
examples 2-3  possibly a little accuracy     26 extra tokens
examples 4+   not measurable here            82 extra tokens, +150 ms
```

**Each example costs the same and they are worth wildly different amounts.**
The honest default is **one or two**, and anything beyond that has to show up
in a measurement before it stays.

Note also that latency grows more slowly than tokens — 21.8x the tokens, 4.1x
the time — because a forward pass over a short prompt is dominated by fixed
overhead. That gap closes on a GPU serving large batches, where tokens *are*
the cost ([MLOps 06](../../MLOps/lessons/06-serving.md) measures that regime).

---

## The arithmetic

Three numbers decide everything:

```text
PROMPT TOKENS      the fixed prefix, paid on every call
INPUT TOKENS       the user's text, variable
OUTPUT TOKENS      what you generate, usually priced higher
```

For a hosted model, the monthly bill is:

```python
# no-run
calls = 1_000_000
prefix, user_text, output = 114, 60, 12         # tokens
in_price, out_price = 0.30, 1.20                # per million tokens, example rates

monthly_in  = calls * (prefix + user_text) / 1e6 * in_price
monthly_out = calls * output / 1e6 * out_price
print(f"input  {monthly_in:8.2f}   of which prefix {calls*prefix/1e6*in_price:.2f}")
print(f"output {monthly_out:8.2f}")
```

Run that with your own numbers. The thing it usually reveals: **at scale, the
few-shot prefix is a large share of the input bill, and it is the part you can
delete without touching the product.** Cutting 8 examples to 2 removes 78
tokens per call — at a million calls, 78 million tokens a month, for a
difference nobody measured.

[LLM 02](../../LLM-and-GenAI/lessons/02-tokens-and-cost.md) is the companion
here, with a finding that matters for Arabic work: **the same meaning costs
2.61x more in Arabic than in English.** A prompt that is affordable in English
may not be in Arabic, and the decision is the same arithmetic with a different
multiplier.

---

## The static prefix is the cheap part

Your prompt has two halves, and they behave completely differently:

```text
STATIC     instructions, examples, schema, policy text
           identical on every request
           CACHEABLE — most hosted APIs charge a fraction for a cached prefix

DYNAMIC    the user's input, retrieved context, the current date
           different every time
           never cacheable
```

Two consequences for how you lay a prompt out:

**Put everything static first, everything dynamic last.** A cache hit needs an
exact matching prefix; a date or a user id near the top destroys it for the
whole prompt. This also happens to be what
[lesson 01](01-what-a-prompt-is.md) wants — the question last, so an answer
comes next — and what
[lesson 03](03-what-goes-in-the-context.md) wants, with the best retrieved
chunk closest to the question.

**Do not inject anything variable that you do not need.** "Today is 3 October
2026" at the top of a prompt that never reasons about dates costs you the
entire prefix cache.

---

## Three ways a prompt gets expensive quietly

| What happens | How it hides | What to do |
|---|---|---|
| Someone adds an example to fix one bad case | +13 tokens per call, forever, from a one-line commit | Prompts go through review like code ([lesson 08](08-prompts-in-production.md)) |
| Retrieved context grows as the corpus grows | The prompt template never changed | Cap the retrieved tokens, not the chunk count |
| A retry loop on parse failure | Doubles the cost of exactly the requests that were already failing | Fix the contract ([lesson 02](02-the-output-contract.md)); cap retries at one |

The third is the nastiest, because the retry rate rises exactly when something
upstream breaks — so your bill spikes during an incident, on top of the
incident.

**Log tokens per request and watch the median and the p95.**
[MLOps 07](../../MLOps/lessons/07-monitoring.md) gives the threshold arithmetic;
prompt length is just another metric to put a line under.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Keeping 8 examples because 8 felt safe | 21.8x the tokens of zero, for an unmeasured gain |
| Treating prompt length as a design detail | It is a per-request cost, paid forever |
| Measuring tokens in English only | Arabic costs 2.61x for the same meaning |
| Variable text at the top of the prompt | Destroys the prefix cache for the whole prompt |
| Unbounded retrieved context | Grows with the corpus, silently |
| Unlimited retries on parse failure | The bill spikes during an incident |
| No token metric in production | The only place this gets noticed is the invoice |

---

## Exercises

1. Run the table with your own model. Where does latency stop being overhead?
2. Compute your monthly bill at 1k, 100k and 1M calls. Which term dominates?
3. Split your prompt into static and dynamic. Is anything variable at the top?
4. Cut your examples to two and measure on a real eval set. Did anything move?
5. Measure the same prompt in English and Arabic
   ([LLM 02](../../LLM-and-GenAI/lessons/02-tokens-and-cost.md)). What is your
   multiplier?
6. Find your parse-failure rate and compute what the retries cost.

---

**Next:** [Lesson 06 — The Iteration Loop](06-the-iteration-loop.md)
