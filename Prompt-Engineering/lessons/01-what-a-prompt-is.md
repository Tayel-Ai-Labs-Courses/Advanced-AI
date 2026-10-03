# Lesson 01 — What a Prompt Actually Is

**Goal:** stop thinking of a prompt as a message to someone, and start seeing
what the model does with it.

## What you will learn

- A prompt is a prefix, not a question
- Why asking politely makes the answer *less* likely
- How to look at what the model is about to do
- The one habit that makes everything else in this course work

---

## The model continues text

A language model does exactly one thing: given some text, it produces a
probability for every possible next token. That is the whole mechanism
([LLM 01](../../LLM-and-GenAI/lessons/01-what-an-llm-is.md) builds it from
scratch).

So your prompt is not a request. **It is the beginning of a document, and the
model is writing the rest of it.** Everything that follows in this course comes
out of that one sentence.

The practical question is therefore never "how do I ask nicely?" It is:

> **What document is this the beginning of, and does the next thing in that
> document happen to be my answer?**

---

## Setup

```python
import os, warnings
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

def top(prompt, k=3):
    p = next_dist(prompt)
    v, i = p.topk(k)
    return [(tok.decode(j), float(x)) for x, j in zip(v, i)]
```

`gpt2-medium` is 355M parameters and runs on a laptop CPU. It is small enough
that its failures are visible, which is exactly what you want while learning.
**The mechanism is identical at 400B** — the bigger model hides these effects
behind competence, it does not escape them.

---

## The same question, four ways

```python
frames = {
    "bare statement":     "The capital of France is",
    "question":           "What is the capital of France?",
    "question + Answer:": "What is the capital of France?\nAnswer:",
    "Q/A format":         "Q: What is the capital of France?\nA:",
}
for name, p in frames.items():
    shown = "  ".join(f"{repr(w)} {x:.3f}" for w, x in top(p))
    print(f"{name:<22}{shown}")
```

```text
bare statement        ' Paris' 0.174  ' the' 0.055  ' Lyon' 0.042
question              '\n' 0.459  ' It' 0.046  ' The' 0.043
question + Answer:    ' Paris' 0.119  ' France' 0.070  ' The' 0.060
Q/A format            ' The' 0.158  ' France' 0.148  ' It' 0.091
```

Look at row two. **Asking the question properly puts 45.9% of the probability
on a newline.** Not on an answer — on the end of the line.

That is not the model being stupid. In the text it learned from, a line that
ends in a question mark is usually followed by a line break. It is continuing
the document correctly. The document just is not the one you wanted.

Now measure the thing you actually care about: how much probability is sitting
on an answer at all.

```python
CITIES = [" Paris", " paris", " PARIS"]

def mass_on(prompt, words):
    p = next_dist(prompt)
    return sum(float(p[tok(w).input_ids[0]])
               for w in words if len(tok(w).input_ids) == 1)

for name, p in frames.items():
    print(f"{name:<22}mass on a city token: {mass_on(p, CITIES):.3f}")
```

```text
bare statement        mass on a city token: 0.174
question              mass on a city token: 0.005
question + Answer:    mass on a city token: 0.119
Q/A format            mass on a city token: 0.059
```

**The bare statement is 35x more likely to answer than the polite question.**
0.174 against 0.005.

Three things follow, and they are the spine of this course.

**The phrasing that feels most like asking is the worst one here.** Everyone's
instinct is to write a clear question. The model does not want a question; it
wants the beginning of an answer.

**Adding `\nAnswer:` recovers almost all of it** — 0.005 to 0.119, a 24x
improvement from eight characters. That is not politeness or clarity. It is
making the document one where an answer comes next.

**`Q:`/`A:` scored worse than plain `Answer:`** (0.059 vs 0.119), which nobody
would have predicted. The formats people repeat as rules are not rules. They
are guesses that happen to have survived retelling, and the only way to know is
to measure — [lesson 06](06-the-iteration-loop.md) is how.

---

## Instruction-tuned models, honestly

You will object that ChatGPT or Claude answers the polite question perfectly.
True, and worth being precise about why.

Those models were **fine-tuned on conversations**, so for them "a user turn
ending in a question mark" really is a document where an answer comes next.
The mechanism did not change; the training distribution did.

| | Base model (`gpt2-medium`) | Instruction-tuned model |
|---|---|---|
| What it continues | Any text | A conversation |
| Best prompt shape | The start of the answer | A clear instruction |
| What this lesson shows | Directly | Hidden, but still there |

**The effects do not disappear, they shrink.** The ordering sensitivity in
[lesson 03](03-what-goes-in-the-context.md), the output-contract failures in
[lesson 02](02-the-output-contract.md), and the injection in
[lesson 07](07-untrusted-text.md) all reproduce on frontier models. They are
properties of "predict the next token", not of model size.

---

## The habit

Before you change a prompt, ask the question that this lesson is built on:

```text
1. What document am I starting?
2. In that document, what comes immediately after my text?
3. Is that the thing I want?
4. If not, what prefix would make it so?
```

Step 4 is usually one of three moves, and the rest of this course measures all
three: **name the field** (`Sentiment:`), **show one example**, or **put the
answer's first words in the prompt** (`The wifi password is`).

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Writing the prompt as a message to a person | The model is finishing a document, not replying |
| A polite question with no answer cue | 0.005 of the mass on the answer |
| Copying a prompt format because it is popular | `Q:/A:` scored half of plain `Answer:` here |
| Judging a prompt by how it reads | Read the output distribution, not the prompt |
| Assuming an instruction-tuned model is immune | The effects shrink, they do not vanish |
| Changing three things at once | You learn nothing from the result |

---

## Exercises

1. Run the four framings on a different question. Does the ordering hold?
2. Add a fifth framing that puts the first words of the answer in the prompt.
   Where does it land?
3. Find a framing where the top token is punctuation. What document is the
   model writing?
4. Take a prompt you use at work and write down what document it starts.
5. Try the same four framings against an instruction-tuned model if you have
   one locally ([Optimization 14](../../Optimization/lessons/14-running-models-locally.md)).
   How much of the gap closes?

---

**Next:** [Lesson 02 — The Output Contract](02-the-output-contract.md)
