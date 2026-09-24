# Lesson 07 — Sequence Labelling and NER

**Goal:** tag every token, and score the result the way the task requires.

## What you will learn

- The BIO scheme
- Token features, and a working tagger
- Entity-level metrics — and why token accuracy lies
- Handling subword alignment

---

## The task

Classification labels a document. Sequence labelling labels **every token**.

```text
Adam    Tayel   visited  Cairo   on   Monday
B-PER   I-PER   O        B-LOC   O    B-DATE
```

| Tag | Means |
|---|---|
| `B-X` | Beginning of an entity of type X |
| `I-X` | Inside — continues the previous entity |
| `O` | Outside any entity |

The B/I distinction is not decoration: it is the only thing separating two
adjacent entities from one long one.

```python
def decode_entities(tokens, tags):
    """Turn BIO tags into (text, type, start, end) spans."""
    entities, current = [], None
    for index, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current:
                entities.append(current)
            current = [token, tag[2:], index, index]
        elif tag.startswith("I-") and current and current[1] == tag[2:]:
            current[0] += " " + token
            current[3] = index
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    return [(text, kind, start, end) for text, kind, start, end in entities]

tokens = ["Adam", "Tayel", "visited", "Cairo", "and", "Giza", "on", "Monday"]
tags = ["B-PER", "I-PER", "O", "B-LOC", "O", "B-LOC", "O", "B-DATE"]

for entity in decode_entities(tokens, tags):
    print(entity)
```

```text
('Adam Tayel', 'PER', 0, 1)
('Cairo', 'LOC', 3, 3)
('Giza', 'LOC', 5, 5)
('Monday', 'DATE', 7, 7)
```

`Cairo` and `Giza` stayed separate because each starts with `B-`. Tag `Giza`
as `I-LOC` and they merge into one entity called "Cairo and Giza" — a
different, wrong answer.

---

## A working tagger

You do not need a transformer to start. Token features plus logistic
regression is a real baseline.

```python
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sentences = [
    (["Adam", "visited", "Cairo", "on", "Monday"], ["B-PER", "O", "B-LOC", "O", "B-DATE"]),
    (["Sara", "lives", "in", "Giza"], ["B-PER", "O", "O", "B-LOC"]),
    (["We", "met", "Omar", "in", "Alexandria", "on", "Sunday"],
     ["O", "O", "B-PER", "O", "B-LOC", "O", "B-DATE"]),
    (["Youssef", "travelled", "to", "Aswan", "on", "Friday"],
     ["B-PER", "O", "O", "B-LOC", "O", "B-DATE"]),
    (["Nour", "works", "in", "Luxor"], ["B-PER", "O", "O", "B-LOC"]),
]

def features(tokens, i):
    """Features for one token: itself, its shape, and its neighbours."""
    token = tokens[i]
    return {
        "lower": token.lower(),
        "is_title": token.istitle(),
        "is_upper": token.isupper(),
        "suffix3": token[-3:].lower(),
        "prev": tokens[i - 1].lower() if i > 0 else "<s>",
        "next": tokens[i + 1].lower() if i < len(tokens) - 1 else "</s>",
        "position": i,
    }

X = [features(tokens, i) for tokens, _ in sentences for i in range(len(tokens))]
y = [tag for _, tags in sentences for tag in tags]

model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X, y)

test_tokens = ["Hala", "visited", "Cairo", "on", "Sunday"]
predicted = model.predict([features(test_tokens, i) for i in range(len(test_tokens))])

for token, tag in zip(test_tokens, predicted):
    print(f"{token:<10} {tag}")
```

```text
Hala       B-PER
visited    O
Cairo      B-LOC
on         O
Sunday     B-DATE
```

Five training sentences, and it tagged a sentence it had never seen. The
features that did the work: capitalisation, the previous token ("on" before a
date, "in" before a place), and position.

For production you would use a CRF (which models tag transitions) or fine-tune
a transformer (lesson 09). The features above are what those models learn
automatically.

---

## Token accuracy lies

```python
def token_accuracy(true_tags, predicted_tags):
    correct = sum(t == p for t, p in zip(true_tags, predicted_tags))
    return correct / len(true_tags)

true = ["B-PER", "I-PER", "O", "B-LOC", "O", "O", "O", "B-DATE", "I-DATE", "O"]
predicted = ["O"] * 10                      # predict nothing at all

print(f"token accuracy of a model that finds nothing: {token_accuracy(true, predicted):.3f}")
print("entities found: 0 of 3")
```

```text
token accuracy of a model that finds nothing: 0.500
entities found: 0 of 3
```

Fifty per cent accuracy, zero entities — and on a realistic document, where
entities are far sparser than in this toy sentence, the same do-nothing model
scores **95%**. Most tokens are `O`, so token accuracy
measures the class imbalance — the same trap as ML lesson 06, in a new costume.

**Score entities, not tokens.** An entity counts as correct only if its span
**and** its type both match exactly.

```python
def entity_scores(true_tags, predicted_tags, tokens):
    """Precision, recall and F1 at the entity level (exact span and type)."""
    true_entities = {(kind, start, end) for _, kind, start, end
                     in decode_entities(tokens, true_tags)}
    predicted_entities = {(kind, start, end) for _, kind, start, end
                          in decode_entities(tokens, predicted_tags)}

    correct = len(true_entities & predicted_entities)
    precision = correct / len(predicted_entities) if predicted_entities else 0.0
    recall = correct / len(true_entities) if true_entities else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "correct": correct,
            "predicted": len(predicted_entities), "true": len(true_entities)}

tokens = ["Adam", "Tayel", "visited", "Cairo", "last", "week", "with", "Sara", "Ali", "again"]
true = ["B-PER", "I-PER", "O", "B-LOC", "O", "O", "O", "B-PER", "I-PER", "O"]

partial = ["B-PER", "O", "O", "B-LOC", "O", "O", "O", "B-PER", "I-PER", "O"]
wrong_type = ["B-LOC", "I-LOC", "O", "B-LOC", "O", "O", "O", "B-PER", "I-PER", "O"]

print("perfect:    ", entity_scores(true, true, tokens))
print("half a name:", entity_scores(true, partial, tokens))
print("wrong type: ", entity_scores(true, wrong_type, tokens))
```

```text
perfect:     {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'correct': 3, 'predicted': 3, 'true': 3}
half a name: {'precision': 0.667, 'recall': 0.667, 'f1': 0.667, 'correct': 2, 'predicted': 3, 'true': 3}
wrong type:  {'precision': 0.667, 'recall': 0.667, 'f1': 0.667, 'correct': 2, 'predicted': 3, 'true': 3}
```

Tagging "Adam" but missing "Tayel" scores **zero** for that entity — 9 of 10
tokens right, and one of three entities lost. That severity is correct: half a
person's name is not a usable extraction.

In production use the `seqeval` library, which implements exactly this and
handles the variant tagging schemes.

---

## Subword alignment

Transformers tokenise into subwords, but your labels are per word. They must
be realigned, and getting it wrong silently shifts every tag.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

words = ["youssef", "visited", "heliopolis"]
word_labels = ["B-PER", "O", "B-LOC"]

encoded = tokenizer(words, is_split_into_words=True)
subwords = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
word_ids = encoded.word_ids()

aligned = []
previous = None
for word_id in word_ids:
    if word_id is None:
        aligned.append(-100)                      # special token: ignored by the loss
    elif word_id != previous:
        aligned.append(word_labels[word_id])      # first subword keeps the label
    else:
        aligned.append(-100)                      # continuation: ignored
    previous = word_id

for subword, word_id, label in zip(subwords, word_ids, aligned):
    print(f"{subword:<14} word_id={str(word_id):<5} label={label}")
```

```text
[CLS]          word_id=None  label=-100
you            word_id=0     label=B-PER
##sse          word_id=0     label=-100
##f            word_id=0     label=-100
visited        word_id=1     label=O
he             word_id=2     label=B-LOC
##lio          word_id=2     label=-100
##polis        word_id=2     label=-100
[SEP]          word_id=None  label=-100
```

Three words became eight tokens. `word_ids()` maps each subword back to its
word — the three pieces of "youssef" all report `word_id=0`. The convention is
to label the **first** subword and mark the rest `-100`, which PyTorch's
`CrossEntropyLoss` ignores.

Get this wrong — label every subword, or forget the specials — and each tag
lands on the wrong token. The model still trains, the loss still falls, and
the output is nonsense.

When a word splits into several pieces, the alternative convention labels
continuations `I-X`. Either works; mixing them does not.

---

## Where NER goes wrong

| Problem | Example |
|---|---|
| Nested entities | "Bank of Alexandria" — one ORG, containing a LOC |
| Ambiguity | "Washington" — person, city or state |
| New entities | A company founded last week is in no training set |
| Boundaries | Does the title belong in the name? "Dr Adam Tayel" |
| Arabic | No capitalisation to signal names at all |

That last row matters for this course: **English NER leans heavily on
capitalisation**, and Arabic has none. Models trained on English transfer
poorly for exactly this reason — lesson 12.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Reporting token accuracy | 60% for a model that finds nothing |
| Ignoring the B/I distinction | Adjacent entities merge |
| Mislabelled subwords | Every tag shifts by one position |
| Evaluating with partial credit | You ship half-extracted names |
| Assuming capitalisation | Breaks on Arabic, lowercase text and social media |
| No `-100` on special tokens | The model learns to predict `[CLS]` tags |

---

## Exercises

1. Write `decode_entities` yourself and test it on adjacent entities.
2. Train the feature-based tagger on 20 sentences of your own.
3. Show that token accuracy and entity F1 disagree on a real prediction.
4. Align word labels to subwords for a sentence where a word splits into four
   pieces.
