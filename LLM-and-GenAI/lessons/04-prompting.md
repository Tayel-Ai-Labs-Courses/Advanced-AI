# Lesson 04 — Prompting, Measured

**Goal:** stop guessing whether a prompt is better, and find out how little of
"prompt engineering" survives measurement.

## What you will learn

- Turning a prompt into a scored classifier
- Six prompt formats, with accuracy numbers
- Why example order changes the answer
- The label bias that makes a prompt agree with everything

---

## A prompt you can score

Prompting is untestable until the output is constrained. Here the model chooses
between two label words, so "which prompt is better" becomes an accuracy number.

```python
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium").eval()

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
SHOTS = [
    ("The sandwich was tasty.", "positive"),
    ("My drink arrived cold.", "negative"),
    ("Wonderful atmosphere.", "positive"),
    ("The waiter was dismissive.", "negative"),
    ("Everything was fresh.", "positive"),
    ("It smelled bad inside.", "negative"),
    ("Quick and polite service.", "positive"),
    ("They got my order wrong.", "negative"),
]

POS = tok(" positive").input_ids[0]
NEG = tok(" negative").input_ids[0]

def classify(prompt):
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    return "positive" if logits[POS] > logits[NEG] else "negative"

def accuracy(build_prompt):
    ok = sum(classify(build_prompt(t)) == lab for t, lab in REVIEWS)
    return ok / len(REVIEWS)

templates = {
    "bare": lambda t: f"{t}\nSentiment:",
    "instruction": lambda t: ("Classify the review sentiment as positive or negative.\n"
                              f"Review: {t}\nSentiment:"),
    "2-shot": lambda t: ("".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in SHOTS[:2])
                         + f"Review: {t}\nSentiment:"),
    "4-shot": lambda t: ("".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in SHOTS[:4])
                         + f"Review: {t}\nSentiment:"),
    "8-shot": lambda t: ("".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in SHOTS)
                         + f"Review: {t}\nSentiment:"),
    "8-shot + instruction": lambda t: (
        "Classify the review sentiment as positive or negative.\n\n"
        + "".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in SHOTS)
        + f"Review: {t}\nSentiment:"),
}

print(f"{len(REVIEWS)} labelled reviews, {len(SHOTS)} available examples")
```

```text
20 labelled reviews, 8 available examples
```

`classify` never generates text. It runs **one forward pass** and compares the
logit of `" positive"` against `" negative"`. That is the cheapest way to use a
language model for classification: one token of compute, no parsing, no
possibility of the model answering "I think this review is quite..." instead of
a label.

---

## Six prompts

```python
templates = {
    "bare": lambda t: f"{t}\nSentiment:",
    "instruction": lambda t: ("Classify the review sentiment as positive or negative.\n"
                              f"Review: {t}\nSentiment:"),
    "2-shot": lambda t: ("".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in SHOTS[:2])
                         + f"Review: {t}\nSentiment:"),
    "4-shot": lambda t: ("".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in SHOTS[:4])
                         + f"Review: {t}\nSentiment:"),
    "8-shot": lambda t: ("".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in SHOTS)
                         + f"Review: {t}\nSentiment:"),
    "8-shot + instruction": lambda t: (
        "Classify the review sentiment as positive or negative.\n\n"
        + "".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in SHOTS)
        + f"Review: {t}\nSentiment:"),
}
print(f"{'prompt':<24}{'accuracy':>10}{'prompt tokens':>15}")
for name, fn in templates.items():
    n_tok = len(tok(fn(REVIEWS[0][0])).input_ids)
    print(f"{name:<24}{accuracy(fn):>10.2f}{n_tok:>15}")
print(f"{'majority class':<24}{0.50:>10.2f}{0:>15}")
```

```text
prompt                    accuracy  prompt tokens
bare                          0.90             14
instruction                   0.80             27
2-shot                        1.00             44
4-shot                        0.50             70
8-shot                        0.80            126
8-shot + instruction          0.65            138
majority class                0.50              0
```

Before drawing any conclusion: **the eval set is 20 examples, so one example is
0.05.** Everything in that column has a standard error of roughly 0.10. Hold
that thought; it is most of the lesson.

Now read the table.

**Adding the instruction made it worse** — 0.90 down to 0.80. **Four examples
scored 0.50**, exactly chance, worse than no examples at all. **Eight examples
plus an instruction, the "most engineered" prompt in the table, scored 0.65**
and cost ten times the tokens of the bare prompt.

The ordering is not monotone in anything: not in examples, not in instruction,
not in tokens. If you had tried these six in the order most people try them —
bare, then instruction, then more shots — you would have concluded after step 2
that instructions hurt, and after step 3 that few-shot helps enormously, and
both conclusions would have been drawn from single-example differences.

**This is the real state of prompt engineering on a small eval set: the
differences are mostly noise, and the noise is large enough to support whatever
story you want.**

---

## The same prompt, reordered

```python
import itertools, random
accs = []
for seed in range(6):
    rnd = random.Random(seed)
    shots = SHOTS[:]
    rnd.shuffle(shots)
    fn = lambda t, sh=shots: ("".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in sh)
                              + f"Review: {t}\nSentiment:")
    accs.append(accuracy(fn))
print("8-shot accuracy with 6 different example orders:",
      " ".join(f"{a:.2f}" for a in accs))
print(f"range {min(accs):.2f} to {max(accs):.2f}")
```

```text
8-shot accuracy with 6 different example orders: 0.85 0.80 0.95 0.90 0.95 0.90
range 0.80 to 0.95
```

Identical examples, identical instruction, identical model. **Only the order
changed, and accuracy moved by 15 points** — a swing as large as any difference
in the previous table.

Two things follow.

**A prompt is not one configuration.** Reporting "the 8-shot prompt scores 0.80"
is like lesson 09 of the RL course reporting one seed. Run several orders and
report the median and range.

**Order effects are a known, exploitable property.** Models are sensitive to
recency — the last example carries more weight — and to the label of the final
example in particular. If your examples are sorted by label, you have built a
bias into the prompt.

---

## The bias that makes a prompt agree with everything

```python
pos_only = [s for s in SHOTS if s[1] == "positive"]
fn = lambda t: ("".join(f"Review: {s}\nSentiment: {l}\n\n" for s, l in pos_only)
                + f"Review: {t}\nSentiment:")
preds = [classify(fn(t)) for t, _ in REVIEWS]
print(f"accuracy {accuracy(fn):.2f}, predicted positive "
      f"{preds.count('positive')}/{len(preds)} times")
```

```text
accuracy 0.50, predicted positive 20/20 times
```

Show the model eight positive examples and it predicts **positive for every
single input**, including "Dirty tables and a broken air conditioner."

It is not reading the review. It has inferred, correctly, that in this prompt
the answer after `Sentiment:` is `positive`, and it is completing the pattern —
which is exactly what lesson 01 said it does.

This failure is easy to create by accident:

- Examples copied from the top of a sorted spreadsheet
- A class that is rare in production, so few examples of it exist
- Examples chosen by hand, and humans pick clear cases, which skew

**Balance the labels in your examples, and check the prediction distribution,
not only accuracy.** `predicted positive 20/20` is visible in one line of code
and invisible in an accuracy number when the eval set is balanced differently.

---

## What actually works

Ranked by how reliably it survives measurement:

| Technique | Worth it? |
|---|---|
| **Constrain the output** (label words, JSON schema, enum) | Always. Removes parsing and most failure modes |
| **Balance the few-shot labels** | Always. Fixes the 20/20 failure above |
| **Put the task before the data** | Usually, for long inputs — the instruction stays in attention range |
| **Report median over several example orders** | Always. Otherwise you are reporting one seed |
| A longer, more detailed instruction | Sometimes. It cost 10 points here |
| More examples | Sometimes. 4-shot scored worse than 2-shot here |
| "You are an expert..." personas | Rarely measurable |
| Politeness, threats, tips | No |

And the technique that beats all of them: **a bigger eval set**. With 20
examples you cannot distinguish 0.80 from 0.90. With 300 you can, and then the
whole exercise stops being folklore.

```text
n = 20   ->  standard error ~0.10   ->  can detect: nothing below 20 points
n = 100  ->  standard error ~0.04   ->  can detect: ~10 points
n = 500  ->  standard error ~0.02   ->  can detect: ~5 points
```

Build the eval set first. It is the asset; the prompt is disposable.

**The craft side of this** — how to get a parseable answer at all, what to put
in the context, what each example costs per call, and what happens when the
text in your prompt was written by an attacker — is
[`Prompt-Engineering`](../../Prompt-Engineering/). Its
[lesson 02](../../Prompt-Engineering/lessons/02-the-output-contract.md) is the
mirror of this one: free generation from three reasonable prompts produced
**0 parseable answers out of 12**, while scoring the label words as above
cannot produce an unparseable answer at all. That is the case for constraining
the output, from the other direction.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Comparing prompts on 20 examples | One example is 5 points; every difference here is noise |
| Reporting one example order | The same prompt ranged 0.80 to 0.95 |
| Unbalanced few-shot labels | Predicted positive 20 times out of 20 |
| Assuming more examples is better | 4-shot scored 0.50, chance |
| Assuming an instruction helps | It cost 10 points |
| Free-text output for a classification task | Now you have a parsing problem too |
| Keeping a prompt because it worked once | That is the definition of overfitting to your eval set |

---

## Exercises

1. Expand `REVIEWS` to 100 examples and rerun the first table. How many of the
   six differences survive?
2. Sort `SHOTS` so all negatives come last. Predict what happens before running
   it, then run it.
3. Replace the label words with `" good"` / `" bad"`, then with `" 1"` / `" 0"`.
   Does the choice of label word change accuracy more than the prompt format
   did?
4. Compute a bootstrap interval on the 8-shot accuracy across the 6 orders.
   Report the prompt's performance as one sentence containing a range.
5. Write the smallest eval set that would let you detect a 5-point improvement,
   and say how long it would take you to label it.

---

**Next:** [Lesson 05 — Embeddings and Search](05-embeddings-and-search.md)
