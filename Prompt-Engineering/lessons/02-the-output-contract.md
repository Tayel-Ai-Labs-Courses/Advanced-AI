# Lesson 02 — The Output Contract

**Goal:** make the model's answer something a program can read.

## What you will learn

- Why a reasonable-sounding prompt returns nothing usable
- The parse rate, and why it comes before accuracy
- The four ways to specify a format, measured
- Why one example beats three sentences of instruction

---

## Accuracy is the second question

Before "is the answer right?" comes "**is there an answer?**" A model that
replies *"I'm not sure if I'd call it positive exactly..."* has not given you a
wrong label. It has given you a parsing problem, and parsing problems are the
ones that take down a pipeline at 2 a.m.

So the first metric for any prompt is the **parse rate**: of N inputs, how many
produced something your code could read.

---

## Setup

```python
import os, warnings, re
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

## Four ways to ask for a label

```python
REVIEWS = [
 ("the coffee was cold and the service was slow", "negative"),
 ("best breakfast I have had in Cairo", "positive"),
 ("they lost my order twice", "negative"),
 ("friendly staff and the pastries are fresh", "positive"),
 ("overpriced for what you get", "negative"),
 ("I come here every morning now", "positive"),
 ("the wifi never works", "negative"),
 ("great place to sit and work", "positive"),
 ("waited forty minutes for a sandwich", "negative"),
 ("the new menu is excellent", "positive"),
 ("dirty tables and no one cleaned them", "negative"),
 ("reasonable prices and good portions", "positive"),
]

styles = {
 "no contract": lambda t: f"Review: {t}\nWhat do you think of this review?",
 "named field": lambda t: f"Review: {t}\nSentiment:",
 "stated options": lambda t: ("Review: " + t +
      "\nSentiment (positive or negative):"),
 "worked example": lambda t: ("Review: the staff were rude\nSentiment: negative\n\n"
      f"Review: {t}\nSentiment:"),
}

WORDS = re.compile(r"\b(positive|negative)\b", re.I)
print(f"{'prompt style':<18}{'parsed':>8}{'correct':>9}{'first 24 chars of a failure':>32}")
for name, fn in styles.items():
    parsed = correct = 0
    failure = ""
    for text, label in REVIEWS:
        out = generate(fn(text), 8)
        m = WORDS.search(out)
        if m:
            parsed += 1
            correct += (m.group(1).lower() == label)
        elif not failure:
            failure = out.strip().replace("\n", " ")[:24]
    print(f"{name:<18}{parsed:>5}/{len(REVIEWS)}{correct:>9}{failure:>32}")
print(f"\n{len(REVIEWS)} reviews, greedy decoding. 'correct' counts only parsed answers.")
```

```text
prompt style        parsed  correct     first 24 chars of a failure
no contract           0/12        0             I'm not sure if I'm
named field           0/12        0             I'm not sure if I'm
stated options        0/12        0             I'm not sure if I'm
worked example       12/12       12                                

12 reviews, greedy decoding. 'correct' counts only parsed answers.
```

**Three of the four prompts produced nothing usable. Not one answer out of
twelve, three times over.** The fourth produced twelve out of twelve, and all
twelve were right.

This is the sharpest result in the course, so read what did and did not work.

**Naming the field was not enough.** `Sentiment:` looks like it specifies the
format — it is the move lesson 01 recommended — and on its own it produced zero
parseable answers here. It raises the probability of a label
([LLM 04](../../LLM-and-GenAI/lessons/04-prompting.md) measures exactly that)
without making a label the *most likely* continuation.

**Listing the options in words was not enough either.** `Sentiment (positive or
negative):` reads like an unambiguous specification to a human. The model is
not reading a specification; it is continuing a document, and in that document
a parenthetical is followed by prose.

**One worked example changed everything.** Showing a single `Review: ... /
Sentiment: negative` pair establishes the document's shape, and the model
follows the shape. Twenty-eight tokens of demonstration beat every amount of
description.

> **Show the format. Do not describe it.**
> That sentence is most of practical prompt engineering.

A caution about that `12/12`: twelve examples is a tiny eval set, one example
is 8 points, and these reviews are easy.
[Lesson 06](06-the-iteration-loop.md) computes how big an eval set has to be
before a difference means anything — but note that **the parse-rate difference
here, 0/12 against 12/12, is not the kind of result an eval set rescues.** It
is a format failure, not a noise question.

---

## Two ways to get a contract

```text
SHOW IT          one worked example, in exactly the format you want
                 cheapest, most reliable, works on every model

CONSTRAIN IT     restrict decoding to a set of allowed tokens, or score
                 the label words directly and take the argmax
                 100% parse rate by construction
```

The second is what
[LLM 04](../../LLM-and-GenAI/lessons/04-prompting.md) and
[LLM 09](../../LLM-and-GenAI/lessons/09-structured-output.md) do: if the answer
must be one of two words, do not generate freely and hope — **score both words
and pick the larger.** The parse rate becomes 100% because parsing no longer
exists.

Use constrained decoding whenever the output set is closed: labels, enums, a
JSON schema, a yes/no. Use a worked example whenever it is open: a summary, an
extracted phrase, a rewritten sentence.

---

## Designing the contract

A usable contract answers four questions **before** you write the prompt:

| Question | Bad answer | Good answer |
|---|---|---|
| What are the allowed values? | "a sentiment" | `positive` \| `negative` \| `unclear` |
| How is it delimited? | "it'll say it" | the first line, nothing else |
| What happens when the model cannot tell? | (unspecified) | `unclear`, which is a real option |
| What does your code do with a violation? | crash | log it, count it, fall back |

The third row is the one people skip. **A contract with no escape hatch forces
the model to guess**, and a forced guess is indistinguishable from a confident
answer — which is the failure
[Data-Science 06](../../Data-Science/lessons/06-evaluating-the-decision.md)
prices and [AI-Agents 02](../../AI-Agents/lessons/02-tools.md) builds around.
Give it `unclear` and you get a usable signal instead of noise.

The fourth row makes the contract real: **count your violations in production.**
A parse rate that quietly drifts from 99% to 94% is the first visible sign that
something upstream changed.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Measuring accuracy before parse rate | Three prompts here scored 0/12 and have no accuracy |
| Describing the format in words | `(positive or negative)`: 0/12 |
| Naming the field and stopping | `Sentiment:`: 0/12 on free generation |
| Free generation for a closed set | Score the options instead; parse rate 100% |
| No "unclear" option | You forced a guess and cannot tell it from an answer |
| No handler for a violation | The first malformed output takes the job down |
| Not counting parse failures in production | The drift is invisible until it is large |

---

## Exercises

1. Add a fifth style that shows **two** worked examples. Does the parse rate
   change? Does accuracy?
2. Replace the regex with an exact match on the first line. How many of the
   12/12 survive a stricter parser?
3. Add `unclear` as a third option and an ambiguous review. What happens?
4. Implement the scoring version: compare P(` positive`) and P(` negative`)
   directly. Confirm the parse rate is 100%.
5. Take a prompt you use and write its four contract answers. Which was
   unspecified?

---

**Next:** [Lesson 03 — What Goes in the Context](03-what-goes-in-the-context.md)
