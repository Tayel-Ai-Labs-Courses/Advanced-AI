# Lesson 01 — What an LLM Actually Is

**Goal:** see, in numbers, that a language model is a next-token probability
distribution and nothing else.

## What you will learn

- One forward pass, and the table it produces
- Why generation is a loop, not an answer
- Where the intelligence is and where it is not
- Why confidence is not knowledge

---

## Everything runs locally

This whole course uses models small enough to run on a laptop CPU, so there are
**no API keys, no accounts and no bills**. `gpt2-medium` is 355M parameters and
loads in a couple of seconds.

The lessons are about the ideas, and the ideas are identical at 355M and at
400B. Where size genuinely changes the conclusion, the lesson says so.

```python
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium").eval()
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
print(f"vocabulary: {len(tok):,} tokens")
print(f"context window: {model.config.n_positions} tokens")
```

```text
parameters: 355M
vocabulary: 50,257 tokens
context window: 1024 tokens
```

---

## One forward pass

A language model takes a sequence of tokens and returns, for **every position**,
a score for each of the 50,257 possible next tokens. Generation uses only the
last position.

```python
def next_token_table(prompt, k=6):
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    probs = torch.softmax(logits, -1)
    top = torch.topk(probs, k)
    print(f"prompt: {prompt!r}")
    for p, i in zip(top.values, top.indices):
        print(f"  {tok.decode(i)!r:<16}{float(p):.4f}")
    return probs

probs = next_token_table("The capital of Egypt is")
```

```text
prompt: 'The capital of Egypt is'
  ' Cairo'        0.3742
  ' Alexandria'   0.1208
  ' the'          0.0397
  ' in'           0.0198
  ' now'          0.0177
  ' still'        0.0128
```

That table is the entire output of the model. Not a sentence, not an answer — a
**probability distribution over 50,257 tokens**, of which the top six are shown.

Three things are already visible:

- It knows the answer: `' Cairo'` at 0.3742, three times `' Alexandria'`.
- It is not certain, and it should not be — `"The capital of Egypt is the
  largest city in Africa"` is a perfectly good sentence, which is why `' the'`
  has 0.0397.
- **The leading space is part of the token.** `' Cairo'` and `'Cairo'` are
  different tokens, which is lesson 02's subject and the source of an entire
  class of prompting bug.

---

## It is only ever predicting one token

```python
for prompt in ["The capital of Egypt is Cairo, and the capital of France is",
               "2 + 2 =",
               "def add(a, b):\n    return"]:
    next_token_table(prompt, k=3)
    print()
```

```text
prompt: 'The capital of Egypt is Cairo, and the capital of France is'
  ' Paris'        0.8492
  ' Lyon'         0.0519
  ' Nice'         0.0213

prompt: '2 + 2 ='
  ' 4'            0.1974
  ' 5'            0.0958
  ' 3'            0.0942

prompt: 'def add(a, b):\n    return'
  ' a'            0.4445
  ' ('            0.0641
  ' add'          0.0336
```

The first block is the whole trick behind few-shot prompting: given a worked
example in the same format, `' Paris'` jumps to **0.8492** — more than twice the
confidence of the unassisted Cairo answer. Nothing was trained; the pattern in
the prompt did the work. Lesson 04 measures how much this is worth.

The second block deserves a poster. **`2 + 2 =` gives `' 4'` a probability of
0.1974**, with `' 5'` at 0.0958 and `' 3'` at 0.0942. The model has no
arithmetic. It has statistics about which digit tends to follow which
expression, and at this size those statistics are barely better than a coin
flip between the plausible digits. Larger models get this particular sum right,
and still fail on `478 * 39`.

**If a number matters, compute it in code, not in a language model.** That is
lesson 09's subject, and it is the single most common production mistake.

---

## Generation is a loop you write

```python
ids = tok("The capital of Egypt is", return_tensors="pt").input_ids
for step in range(6):
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    nxt = int(torch.argmax(logits))
    ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    print(f"step {step}: appended {tok.decode(nxt)!r:<12} -> {tok.decode(ids[0])!r}")
```

```text
step 0: appended ' Cairo'     -> 'The capital of Egypt is Cairo'
step 1: appended ','          -> 'The capital of Egypt is Cairo,'
step 2: appended ' the'       -> 'The capital of Egypt is Cairo, the'
step 3: appended ' capital'   -> 'The capital of Egypt is Cairo, the capital'
step 4: appended ' of'        -> 'The capital of Egypt is Cairo, the capital of'
step 5: appended ' the'       -> 'The capital of Egypt is Cairo, the capital of the'
```

Six forward passes, six tokens. `model.generate()` is this loop with better
token selection (lesson 03) and a cache so that step 5 does not recompute steps
0-4.

Two consequences that shape every product decision later in the course:

**Cost and latency are per token, and output tokens are the expensive ones.**
A 500-token answer is 500 sequential forward passes; they cannot be
parallelised, because token 6 depends on token 5.

**The model has no plan.** It did not decide to write a sentence about capitals.
At step 2 it had no intention that would be violated at step 5. Anything that
looks like reasoning is a pattern that survived 500 consecutive independent
choices — which is why long outputs drift, and why asking for structure in the
prompt works better than hoping for it.

---

## Confidence is not knowledge

The obvious way to catch a made-up answer is to look at the model's confidence.
Try it.

```python
for prompt in ["The capital of Egypt is",
               "The capital of the fictional country of Zurbia is",
               "The 47th president of Uruguay was named"]:
    ids2 = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids2).logits[0, -1]
    p = torch.softmax(logits, -1)
    top = torch.topk(p, 1)
    entropy = float(-(p * torch.log(p + 1e-12)).sum())
    print(f"{prompt[:46]:<48} top={tok.decode(top.indices[0])!r:<14} "
          f"p={float(top.values[0]):.3f}  entropy={entropy:.2f}")
```

```text
The capital of Egypt is                          top=' Cairo'       p=0.374  entropy=4.03
The capital of the fictional country of Zurbia   top=' Z'           p=0.104  entropy=6.14
The 47th president of Uruguay was named          top=' the'         p=0.152  entropy=4.60
```

The fictional country does show higher entropy — 6.14 against 4.03 — so there is
*some* signal. And then the third row ruins it.

**"The 47th president of Uruguay" does not exist**, and the model's entropy for
it is 4.60, much closer to the known fact's 4.03 than to the obvious fiction's
6.14. A threshold that catches Zurbia does not catch the fake president.

This is the honest state of hallucination detection from probabilities alone:
weakly informative, not reliable enough to gate a product on. What actually
works is **giving the model the source text and checking the answer against it**
— retrieval, lesson 06 — and **refusing to let it produce free-form facts where
a lookup would do**.

---

## What this means for building things

| Belief | Reality |
|---|---|
| "The model knows things" | It has a distribution over tokens shaped by what usually follows what |
| "It can do arithmetic" | `2 + 2 =` gave `' 4'` a probability of 0.20 |
| "It reasons about its answer" | 500 independent next-token choices, no plan |
| "High confidence means correct" | Entropy on a fake president looked like entropy on a real capital |
| "A bigger model fixes this" | It moves the boundary; it does not change the shape |
| "Temperature 0 makes it deterministic" | It makes *decoding* deterministic. The model was never the random part |

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Asking an LLM for a calculation | 0.20 probability on the right digit |
| Treating fluency as accuracy | Fluency is exactly what the objective optimised |
| Gating on model confidence | The fake-president row |
| Forgetting the leading space in a token | `' Cairo'` and `'Cairo'` are different tokens |
| Expecting long outputs to stay on plan | There is no plan to stay on |
| Budgeting input and output tokens the same | Output tokens are sequential and cost several times more |

---

## Exercises

1. Print the top 10 tokens for `"The best programming language is"`. Then for
   `"As a Python developer, the best programming language is"`. Quantify how
   much the prompt moved the distribution.
2. Compute the probability the model assigns to the exact string `" Cairo"`
   versus `"Cairo"` (no space) after `"The capital of Egypt is"`. Explain the
   gap.
3. Feed the model a 1,100-token prompt and read the error. What is the
   `n_positions` limit doing, and what do production systems do instead?
4. Extend the generation loop to 40 tokens. Where does it start repeating, and
   what does that predict about lesson 03?
5. Find a factual prompt where the model is confidently wrong: top probability
   above 0.5, answer incorrect. Then say what you would build to catch it.

---

**Next:** [Lesson 02 — Tokens, Arabic, and Cost](02-tokens-and-cost.md)
