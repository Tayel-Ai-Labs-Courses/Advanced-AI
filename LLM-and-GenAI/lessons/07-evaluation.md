# Lesson 07 — Evaluating LLM Output

**Goal:** build a number you can trust about a system whose output is text —
and see every automatic metric rank a wrong system above a right one.

## What you will learn

- Four metrics, and how each one lies
- Why an embedding judge prefers a confidently wrong answer
- The length bias in similarity scoring
- How big the eval set has to be

---

## Setup

```python
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

import numpy as np, torch, re
from transformers import AutoTokenizer, AutoModel

GOLD = [
    ("What is the refund window?", "14 days"),
    ("How much is shipping?", "40 EGP"),
    ("When is free shipping?", "orders above 600 EGP"),
    ("How long do points last?", "12 months"),
    ("What is the cash on delivery fee?", "15 EGP"),
    ("How many points for a free drink?", "100 points"),
    ("How long are items kept?", "14 days"),
    ("What notice for catering?", "48 hours"),
]
# three candidate systems, hand-written to isolate the metrics
ANSWERS = {
    "terse": ["14 days", "40 EGP", "600 EGP", "12 months", "15 EGP",
              "100 points", "14 days", "48 hours"],
    "verbose": ["The refund window is 14 days from purchase.",
                "Shipping costs 40 EGP for standard delivery.",
                "Shipping is free on orders above 600 EGP.",
                "Loyalty points expire after 12 months.",
                "Cash on delivery adds a fee of 15 EGP.",
                "You need 100 points for a free drink.",
                "Lost items are kept for 14 days at the counter.",
                "Catering requires 48 hours of notice."],
    "confident_wrong": ["30 days", "50 EGP", "500 EGP", "6 months", "20 EGP",
                        "150 points", "7 days", "24 hours"],
}

def exact(pred, gold): return pred.strip().lower() == gold.strip().lower()
def contains(pred, gold): return gold.strip().lower() in pred.strip().lower()

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(MODEL); enc = AutoModel.from_pretrained(MODEL).eval()
def embed(texts):
    b = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        h = enc(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    v = (h * m).sum(1) / m.sum(1)
    return torch.nn.functional.normalize(v, dim=1).numpy()

def emb_sim(preds, golds, threshold=0.6):
    P, G = embed(preds), embed(golds)
    sims = (P * G).sum(1)
    return sims

print(f"{len(GOLD)} questions, {len(ANSWERS)} candidate systems")
```

```text
8 questions, 3 candidate systems
```

Three deliberately different systems: one that answers tersely and correctly,
one that answers in full sentences and correctly, and one that is fluent,
well-formatted and **wrong about every number**.

A human would rank them: verbose = terse >> confident_wrong. Watch what the
metrics do.

---

## Four metrics

```python
golds = [g for _, g in GOLD]
print(f"{'system':<18}{'exact':>8}{'contains':>10}{'emb>0.6':>10}{'mean sim':>10}")
for name, preds in ANSWERS.items():
    e = np.mean([exact(p, g) for p, g in zip(preds, golds)])
    c = np.mean([contains(p, g) for p, g in zip(preds, golds)])
    sims = emb_sim(preds, golds)
    print(f"{name:<18}{e:>8.2f}{c:>10.2f}{np.mean(sims > 0.6):>10.2f}{np.mean(sims):>10.3f}")
```

```text
system               exact  contains   emb>0.6  mean sim
terse                 0.88      0.88      1.00     0.973
verbose               0.00      1.00      0.50     0.579
confident_wrong       0.00      0.00      1.00     0.824
```

Every column contains a wrong answer about which system is better.

**Exact match gives the correct verbose system 0.00.** "The refund window is 14
days from purchase." is a perfect answer, scored as a total failure, because it
is not string-identical to `"14 days"`. Exact match measures formatting.

**Embedding similarity ranks the wrong system above the right one.**
`confident_wrong` scores **1.00** on `emb>0.6` and **0.824** mean similarity;
the correct verbose system scores **0.50** and **0.579**.

Read that again, because it is the most important sentence in this lesson.
*"30 days"* is semantically very close to *"14 days"* — both are short spans of
time, in the same units, in the same context. Embeddings measure **topic**, not
**truth**. For an answer whose content is a number, a date or a name, semantic
similarity is close to useless and actively harmful: the more confidently wrong
the system, the better it scores.

**`contains` is the only column that ranks all three correctly** here — 0.88,
1.00, 0.00 — because the task has a short, checkable answer string. That is not
a general victory for `contains`; it is a reminder that **the metric has to
match the shape of the answer.**

---

## The verbose system is correct

```python
for p, g in list(zip(ANSWERS["verbose"], golds))[:3]:
    print(f"  gold {g!r:<24} pred {p!r}")
print("  exact match scores it 0.00; a human scores it 1.00")
```

```text
  gold '14 days'                pred 'The refund window is 14 days from purchase.'
  gold '40 EGP'                 pred 'Shipping costs 40 EGP for standard delivery.'
  gold 'orders above 600 EGP'   pred 'Shipping is free on orders above 600 EGP.'
  exact match scores it 0.00; a human scores it 1.00
```

If you had shipped exact match as your regression test, this system would have
failed CI at 0.00 and been rejected. Teams do this, decide the model got worse,
and roll back an improvement.

---

## The length bias

The other thing an embedding judge does is punish completeness.

```python
short = "14 days"
padded = ["14 days",
          "The refund window is 14 days.",
          "According to our published policy, the refund window is 14 days from the date of purchase.",
          "Thank you for your question. According to our published refund policy, which applies to all customers, the refund window is 14 days from the date of purchase, and refunds are issued to the original payment method."]
sims = emb_sim(padded, [short] * 4)
for p, s in zip(padded, sims):
    print(f"  sim {s:.3f}  ({len(p.split()):>3} words)  {p[:62]}")
```

```text
  sim 1.000  (  2 words)  14 days
  sim 0.548  (  6 words)  The refund window is 14 days.
  sim 0.411  ( 16 words)  According to our published policy, the refund window is 14 day
  sim 0.351  ( 36 words)  Thank you for your question. According to our published refund
```

**All four contain the correct fact.** Similarity falls from 1.000 to 0.351 as
the answer becomes more polite and more complete, because averaging over more
tokens dilutes the two that carry the answer.

So a similarity-based metric rewards terseness and punishes the customer-service
register that your product probably requires. If you tune prompts against that
metric, you will tune your assistant into a machine that answers "14 days" and
nothing else — which is optimising the measurement, not the product.

**LLM-as-judge has the same failure class**, with different specifics: judges
show position bias (the first option wins more often), length bias (longer
answers are rated better — the opposite direction from the embedding judge), and
self-preference (a model rates its own outputs higher). If you use a judge,
**randomise the order, and validate it against human labels on at least 100
examples** before trusting a single number it produces.

---

## How big does the eval set have to be?

```python
rng = np.random.default_rng(0)
true_rate = 0.80
print(f"{'n':>6}{'95% interval on an 0.80 system':>34}{'width':>8}")
for n in (8, 20, 50, 100, 300, 1000):
    draws = rng.random((4000, n)) < true_rate
    means = draws.mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"{n:>6}{f'[{lo:.2f}, {hi:.2f}]':>34}{hi - lo:>8.2f}")
```

```text
     n    95% interval on an 0.80 system   width
     8                      [0.50, 1.00]    0.50
    20                      [0.60, 0.95]    0.35
    50                      [0.68, 0.90]    0.22
   100                      [0.72, 0.87]    0.15
   300                      [0.75, 0.84]    0.09
  1000                      [0.78, 0.82]    0.05
```

A system that is truly 80% correct will, on an **8-question** eval set, report
anywhere between **0.50 and 1.00**. That is not a measurement; it is a mood.

This table is the answer to "our new prompt improved accuracy from 0.78 to
0.85". On 100 examples, the interval is +/-0.075 — that improvement is inside
the noise. To detect a 5-point change you need roughly 300 examples, and to
detect a 2-point change, around 1,000.

**The eval set is the asset. The prompt, the model and the framework are all
replaceable; the labelled examples are what let you tell whether a replacement
helped.** Build it first, grow it every time production surprises you, and
never tune on the set you report.

---

## A working evaluation

```text
1. Split the eval set     dev (tune on this) and test (report this), never mixed
2. Choose metrics by shape
     short factual answer  -> contains / exact on the normalised span
     classification        -> accuracy, per-class recall
     structured output     -> schema validity, then field-level accuracy
     long free text        -> human review on a sample, with a written rubric
3. Always report an interval, from the table above
4. Always include a "should refuse" subset
     questions with no answer in the source; measure refusal rate
5. Freeze regressions
     every production bug becomes a permanent eval case
6. Review 20 outputs by hand, every time
     the metric will not tell you what it is missing
```

Step 4 is the one teams forget. A system that answers everything scores well on
a set where everything is answerable, and hallucinated confidently the first
time a user asks something out of scope.

Step 6 is not optional. Every failure in this lesson was invisible in the
numbers and obvious in the text.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Exact match on free-form answers | Scored a fully correct system 0.00 |
| Embedding similarity for factual answers | Ranked the all-wrong system above the correct one |
| Tuning against a similarity metric | You will optimise for terseness |
| Trusting an LLM judge unvalidated | Position, length and self-preference bias |
| An 8-question eval set | The interval is [0.50, 1.00] |
| Reporting a 7-point gain on 100 examples | Inside the noise |
| No unanswerable questions | You never measure the failure that matters most |
| Tuning and reporting on the same split | The oldest mistake in this whole track |

---

## Exercises

1. Add a fourth system that answers correctly but in Arabic. Which metrics
   break, and what does that tell you about evaluating multilingual output?
2. Write a normaliser (lowercase, strip punctuation, unify "EGP"/"pounds") and
   rerun exact match. How much of the verbose system's 0.00 does it recover?
3. Build a 20-question "should refuse" set and measure refusal rate for a RAG
   pipeline from lesson 06.
4. Use the model as a judge on these three systems. Does it rank
   `confident_wrong` correctly? Then swap the presentation order and check
   whether its verdict changes.
5. Compute how many eval examples you would need to detect a 3-point
   improvement with 95% confidence, and estimate the hours to label them.

---

**Next:** [Lesson 08 — Prompt, RAG, or Fine-tune](08-prompt-rag-or-finetune.md)
