# Lesson 04 — Breaking the Task Up

**Goal:** know when splitting a prompt helps, when it costs you for nothing,
and when no prompt will work.

## What you will learn

- Splitting a prompt in two, measured — and it bought nothing
- The split that doubled accuracy: move the deterministic part into code
- How to tell a prompt problem from a capability problem
- When to stop prompting

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

```python
REVIEWS = [
 ("the coffee was cold and the service was slow", "negative", "service"),
 ("best breakfast I have had in Cairo", "positive", "food"),
 ("they lost my order twice", "negative", "service"),
 ("friendly staff and the pastries are fresh", "positive", "food"),
 ("overpriced for what you get", "negative", "price"),
 ("I come here every morning now", "positive", "food"),
 ("the wifi never works", "negative", "facilities"),
 ("great place to sit and work", "positive", "facilities"),
 ("waited forty minutes for a sandwich", "negative", "service"),
 ("the new menu is excellent", "positive", "food"),
 ("dirty tables and no one cleaned them", "negative", "facilities"),
 ("reasonable prices and good portions", "positive", "price"),
]
SENT = re.compile(r"\b(positive|negative)\b", re.I)
TOPIC = re.compile(r"\b(food|service|price|facilities)\b", re.I)
```

---

## Two questions in one prompt, or two prompts

The advice everyone repeats is: one prompt, one job. Here is that advice,
measured. The task is to get both the sentiment and the topic of a review.

```python
one = lambda t: ("Review: the staff were rude\nSentiment: negative\nTopic: service\n\n"
                 f"Review: {t}\nSentiment:")
p_sent = lambda t: ("Review: the staff were rude\nSentiment: negative\n\n"
                    f"Review: {t}\nSentiment:")
p_topic = lambda t: ("Review: the staff were rude\nTopic: service\n\n"
                     f"Review: {t}\nTopic:")

s_ok = t_ok = both = 0
for text, lab, top_ in REVIEWS:
    out = generate(one(text), 14)
    ms, mt = SENT.search(out), TOPIC.search(out)
    a = bool(ms) and ms.group(1).lower() == lab
    b = bool(mt) and mt.group(1).lower() == top_
    s_ok += a; t_ok += b; both += (a and b)
print(f"{'one prompt, both fields':<28}sentiment {s_ok:>2}/12   topic {t_ok:>2}/12   both {both:>2}/12")

s_ok = t_ok = both = 0
for text, lab, top_ in REVIEWS:
    o1 = generate(p_sent(text), 6); o2 = generate(p_topic(text), 6)
    ms, mt = SENT.search(o1), TOPIC.search(o2)
    a = bool(ms) and ms.group(1).lower() == lab
    b = bool(mt) and mt.group(1).lower() == top_
    s_ok += a; t_ok += b; both += (a and b)
print(f"{'two prompts, one each':<28}sentiment {s_ok:>2}/12   topic {t_ok:>2}/12   both {both:>2}/12")

n1 = len(tok(one(REVIEWS[0][0])).input_ids)
n2 = len(tok(p_sent(REVIEWS[0][0])).input_ids) + len(tok(p_topic(REVIEWS[0][0])).input_ids)
print(f"\none prompt : {n1:>3} prompt tokens, 1 call")
print(f"two prompts: {n2:>3} prompt tokens, 2 calls  ({n2/n1:.2f}x the tokens)")
```

```text
one prompt, both fields     sentiment 12/12   topic  3/12   both  3/12
two prompts, one each       sentiment 12/12   topic  3/12   both  3/12

one prompt :  32 prompt tokens, 1 call
two prompts:  54 prompt tokens, 2 calls  (1.69x the tokens)
```

**Identical. Every single number.** Splitting the prompt in two changed
nothing, and cost 1.69x the tokens and twice the calls.

So "one prompt, one job" is not a law. It is a heuristic, and here it was
simply wrong — **the second field was not failing because it shared a prompt.
It was failing because the model cannot do it.** Sentiment is 12/12 in both
arrangements; topic is 3/12 in both, which on four classes is chance.

Before you reach for a split, check which of the two you have. The test is this
run: **if the failing part fails the same way alone, the prompt structure was
never the problem.**

---

## The split that worked

Here is a different task and a different split. Decide whether an order is
above 100 EGP.

```python
ORDERS = [("two lattes for 90 EGP", 90), ("a sandwich and juice, 145 EGP", 145),
          ("one espresso, 45 EGP", 45), ("family box for 320 EGP", 320),
          ("croissant 60 EGP", 60), ("catering order 1200 EGP", 1200),
          ("iced tea for 55 EGP", 55), ("birthday cake 480 EGP", 480),
          ("two teas 70 EGP", 70), ("office delivery 260 EGP", 260)]
THRESHOLD = 100
YES = re.compile(r"\b(yes|no)\b", re.I)
NUM = re.compile(r"(\d+)")

ask = lambda t: ("Order: a muffin for 30 EGP\nIs the total above 100 EGP? no\n\n"
                 f"Order: {t}\nIs the total above 100 EGP?")
ok = 0
for text, amt in ORDERS:
    m = YES.search(generate(ask(text), 5))
    pred = (m.group(1).lower() == "yes") if m else None
    ok += (pred == (amt > THRESHOLD))
print(f"{'model decides':<34}{ok:>3}/10")

extract = lambda t: ("Order: a muffin for 30 EGP\nTotal: 30\n\n"
                     f"Order: {t}\nTotal:")
ok = ex_ok = 0
for text, amt in ORDERS:
    m = NUM.search(generate(extract(text), 5))
    got = int(m.group(1)) if m else None
    ex_ok += (got == amt)
    ok += (got is not None and (got > THRESHOLD) == (amt > THRESHOLD))
print(f"{'model extracts, Python compares':<34}{ok:>3}/10   (extraction itself {ex_ok}/10)")
print("\nSame model, same examples. The second version never asks it to reason.")
```

```text
model decides                       5/10
model extracts, Python compares    10/10   (extraction itself 10/10)

Same model, same examples. The second version never asks it to reason.
```

**5/10 — exactly chance — against 10/10.** Same model, same single example, same
inputs. The only change is who does the comparison.

The model is excellent at the part it is good at: pulling `145` out of
`"a sandwich and juice, 145 EGP"`, ten times out of ten. It is at chance on
`145 > 100`, which is a thing a line of Python never gets wrong.

> **Split a task along the line between what must be judged and what can be
> computed.** Give the judgement to the model and the computation to code.

That is a different principle from "one prompt, one job", and it is the one
that pays. Compare the two splits in this lesson: the first divided the task
into two judgements and gained nothing; the second moved a computation out of
the model and doubled accuracy.

The same shape runs through
[AI-Agents 02](../../AI-Agents/lessons/02-tools.md) — a tool is exactly this
move, formalised — and it is why
[AI-Agents 06](../../AI-Agents/lessons/06-reliability.md) can give guarantees
that do not depend on the model behaving.

**The extraction step is also checkable**, which is the other half of the
benefit. `145` either appears in the input or it does not; you can assert that
in code, log the failures, and alarm on the rate. "Is it above 100?" is a
single opaque output with nothing to verify against.

---

## Prompt problem or capability problem?

```text
PROMPT PROBLEM                        CAPABILITY PROBLEM
the output is unparseable             the output is well-formed and wrong
one example fixes it                  more examples do nothing
it works on easy inputs               it is at chance everywhere
rephrasing changes the answer         rephrasing changes nothing
                 ↓                                     ↓
      lessons 01-03 of this course       stop prompting. See below
```

The topic task above is the second column: 3/12 on four classes, unmoved by
splitting. More prompt engineering would be a week spent on noise.

When you are in the right column, the options are not prompts:

| Option | When | Where |
|---|---|---|
| **Move it into code** | The step is deterministic | This lesson |
| **Retrieve the answer** | The model lacks the facts, not the ability | [LLM 06](../../LLM-and-GenAI/lessons/06-rag.md) |
| **Fine-tune** | You have labels and the task is narrow | [LLM 08](../../LLM-and-GenAI/lessons/08-prompt-rag-or-finetune.md) |
| **Train a small classifier** | You have labels and the task is *really* narrow | [Machine-Learning](../../Machine-Learning/) |
| **Use a bigger model** | The task is general and you can pay | [LLM 10](../../LLM-and-GenAI/lessons/10-cost-and-shipping.md) |
| **Do not build it** | The accuracy you need is not reachable | [Data-Science 02](../../Data-Science/lessons/02-framing-the-problem.md) |

The fourth row is the one AI engineers forget. A 4-class topic classifier with
500 labelled reviews is a logistic regression that takes four minutes to train
and will beat every prompt in this lesson.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Splitting a prompt because "one prompt, one job" | Cost 1.69x the tokens and changed nothing here |
| Asking the model to compute | 5/10 on `145 > 100`; Python is 10/10 |
| More prompt engineering on a capability gap | Nothing in lessons 01-03 moves a 3/12 |
| Not checking whether the failing part fails alone | That one run tells you which problem you have |
| A chain with no checkable intermediate | You have more calls and the same blindness |
| Reaching for a bigger model first | The cheap fixes are code, retrieval, and labels |
| Never considering a classical classifier | 500 labels beats every prompt here |

---

## Exercises

1. Run the topic task with 8 examples. Does 3/12 move outside noise?
2. Add an extraction step the model gets wrong sometimes. Write the assertion
   that catches it.
3. Take a prompt of yours that asks for a judgement **and** a calculation.
   Split it on that line and measure both versions.
4. Label 100 reviews by topic and train a logistic regression. Compare.
5. For one failing prompt at work, decide which column of the table it is in,
   and write down the evidence.

---

**Next:** [Lesson 05 — What a Prompt Costs](05-what-a-prompt-costs.md)
