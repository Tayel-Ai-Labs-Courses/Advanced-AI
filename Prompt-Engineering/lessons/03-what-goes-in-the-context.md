# Lesson 03 — What Goes in the Context

**Goal:** decide what to put in front of the model, and in what order.

## What you will learn

- Irrelevant text is mostly harmless; a wrong fact is not
- Why the *last* plausible answer wins
- Where a fact should sit, measured
- The context checklist

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
def next_dist(prompt):
    ids = tok(prompt, return_tensors="pt").input_ids
    return torch.softmax(model(ids).logits[0, -1], -1)

@torch.no_grad()
def generate(prompt, n=20):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:])
```

---

## One wrong fact

The context contains the answer in every row below. The only thing that changes
is what sits next to it.

```python
QUESTION = "\n\nQ: What is the wifi password?\nA: The wifi password is"
FACT = "The wifi password is mango. "
NOISE = ("The branch opened in 2019. Deliveries arrive on Tuesdays. "
         "The espresso machine is serviced every quarter. ")
DISTRACT = "The old wifi password was lemon. "
DISTRACT2 = "The staff wifi password is cedar. "

good = tok(" mango").input_ids[0]
bad1 = tok(" lemon").input_ids[0]
bad2 = tok(" cedar").input_ids[0]

def row(name, ctx):
    p = next_dist(ctx + QUESTION)
    print(f"{name:<44}{float(p[good]):>9.4f}{float(p[bad1]):>9.4f}{float(p[bad2]):>9.4f}")

print(f"{'context':<44}{'mango':>9}{'lemon':>9}{'cedar':>9}")
row("the fact alone", FACT)
row("fact + 3 irrelevant sentences", FACT + NOISE)
row("fact + 1 outdated password", FACT + DISTRACT)
row("fact + 2 other passwords", FACT + DISTRACT + DISTRACT2)
row("the 2 other passwords AFTER the fact", DISTRACT + DISTRACT2 + FACT)
print("\nP(next token). The right answer is in the context in every row.")
```

```text
context                                         mango    lemon    cedar
the fact alone                                 0.6970   0.0002   0.0002
fact + 3 irrelevant sentences                  0.9307   0.0002   0.0000
fact + 1 outdated password                     0.5147   0.0208   0.0002
fact + 2 other passwords                       0.6487   0.0124   0.0027
the 2 other passwords AFTER the fact           0.0558   0.2104   0.0178

P(next token). The right answer is in the context in every row.
```

Three findings, and the third is the one to remember.

**Irrelevant text did not hurt — it helped.** Three unrelated sentences about
deliveries and espresso machines took the right answer from 0.6970 to **0.9307**.
That is the opposite of the folklore that every token must earn its place. The
noise made the context look more like a document about a café, and the question
more like something such a document answers.

**One outdated password cost a third of the probability.** 0.6970 down to
0.5147, from a single sentence that is *true* — the old password really was
lemon. A fact that is correct and obsolete is still a distractor.

**Putting the wrong facts last flipped the answer.** Same three sentences, same
question, different order: **mango falls to 0.0558 and lemon rises to 0.2104.**
The model now answers with the wrong password, confidently.

> **Among competing candidates, recency wins.**
> The last plausible answer in your context is the one you are likely to get.

That is a retrieval instruction, not a prompting one. If you are assembling
context from a search
([LLM 06](../../LLM-and-GenAI/lessons/06-rag.md)), the ranking of the chunks
you paste in is not a presentation detail — **it is part of the answer.** Put
the best-scoring chunk closest to the question, not first.

---

## Where to put a fact in a long context

```python
FILLER = ("The branch opened in 2019. Deliveries arrive on Tuesdays. "
          "The espresso machine is serviced every quarter. Staff meetings are "
          "on Sunday mornings. The seating area holds forty people. ")
ans_id = tok(" mango").input_ids[0]

def score(context):
    return float(next_dist(context + QUESTION)[ans_id])

print(f"{'context tokens':>15}{'fact at start':>15}{'fact in middle':>16}{'fact at end':>14}")
for reps in [1, 3, 6, 10]:
    pre = FILLER * reps
    n = len(tok(pre + FACT + QUESTION).input_ids)
    start = score(FACT + pre)
    mid = score(FILLER * (reps // 2) + FACT + FILLER * (reps - reps // 2))
    end = score(pre + FACT)
    print(f"{n:>15}{start:>15.4f}{mid:>16.4f}{end:>14.4f}")
print("\nP(' mango') as the next token. Same fact, same question, same model.")
```

```text
 context tokens  fact at start  fact in middle   fact at end
             60         0.9373          0.9373        0.8208
            132         0.9418          0.8840        0.8718
            240         0.9244          0.8930        0.8817
            384         0.9034          0.9124        0.8543

P(' mango') as the next token. Same fact, same question, same model.
```

**Position barely matters here, and what effect there is runs the wrong way.**
The spread across all twelve cells is 0.82 to 0.94 — much smaller than the
0.0558-to-0.6970 swing a single distractor caused. And the *worst* column is
"fact at end", the position immediately before the question, which recency
would predict should be best.

Two honest conclusions:

**The "lost in the middle" effect you have read about is not visible at this
scale.** It is reported for contexts of tens of thousands of tokens; 384 tokens
on a 355M model is nowhere near that. **Do not assume the finding transfers to
your setup, in either direction — measure it at your own context length.**

**Position is a second-order knob and content is a first-order one.** If you
have an afternoon to spend improving a prompt, spend it removing the wrong
facts, not rearranging the right ones.

---

## The context checklist

In order of how much each is worth:

- [ ] **Is the answer actually in here?** Half of "the model got it wrong" is
      this. Check before anything else.
- [ ] **Is anything in here contradicted elsewhere in here?** One obsolete
      fact cost a third of the probability.
- [ ] **If two candidates must both be present, is the right one last?**
      0.0558 against 0.6970.
- [ ] **Is the question at the end?** So the document's next move is an answer
      ([lesson 01](01-what-a-prompt-is.md)).
- [ ] **Is anything here untrusted?** [Lesson 07](07-untrusted-text.md).
- [ ] **Is anything here just expensive?** [Lesson 05](05-what-a-prompt-costs.md).
- [ ] **Could the model tell you it does not know?** If the context lacks the
      answer and there is no escape hatch, it will invent one.

Note what is *not* on this list: trimming harmless background. It cost nothing
here and sometimes helped.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Trimming context to "reduce noise" first | Irrelevant text took 0.6970 to 0.9307 |
| Leaving an outdated-but-true fact in | It cost a third of the probability |
| Ignoring the order of retrieved chunks | Putting wrong answers last flipped the output |
| Assuming "lost in the middle" applies to you | Not visible at 384 tokens; measure yours |
| Rearranging a prompt before checking the facts | Content is first-order, position is not |
| Context with no answer and no escape hatch | You have asked for a hallucination |
| Pasting a whole document because it is easier | Every contradiction inside it is now live |

---

## Exercises

1. Add a fourth row where the distractor is *more recent and more specific*
   ("As of today the wifi password is lemon"). How far does mango fall?
2. Remove the fact entirely and look at the top tokens. What does the model
   produce when the answer is absent?
3. Repeat the position experiment with a context of 900 tokens (gpt2's limit
   is 1024). Does the pattern change?
4. Take a RAG prompt from your own work and check the order of the chunks.
5. Find one obsolete fact in a context you assemble automatically. How did it
   get there, and what removes it?

---

**Next:** [Lesson 04 — Breaking the Task Up](04-breaking-the-task-up.md)
