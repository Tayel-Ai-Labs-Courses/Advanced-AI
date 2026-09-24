# Lesson 02 — Text Preprocessing

**Goal:** clean text without deleting the signal.

## What you will learn

- Normalisation, and what it costs
- Stopwords, stemming, lemmatisation — and when each is wrong
- Unicode, and why it bites
- Building a preprocessing function you can defend

---

## Cleaning is a modelling decision

Every transformation throws information away. Sometimes that is the point;
sometimes you deleted the answer.

```python
import re

raw = "The COFFEE was NOT good!!! Visit https://example.com :) 😞 #disappointed"

lowered = raw.lower()
no_urls = re.sub(r"https?://\S+", " ", lowered)
no_punctuation = re.sub(r"[^\w\s]", " ", no_urls)
collapsed = re.sub(r"\s+", " ", no_punctuation).strip()

print("raw:       ", raw)
print("lowered:   ", lowered)
print("no urls:   ", no_urls)
print("no punct:  ", no_punctuation)
print("collapsed: ", collapsed)
```

```text
raw:        The COFFEE was NOT good!!! Visit https://example.com :) 😞 #disappointed
lowered:    the coffee was not good!!! visit https://example.com :) 😞 #disappointed
no urls:    the coffee was not good!!! visit   :) 😞 #disappointed
no punct:   the coffee was not good    visit         disappointed
collapsed:  the coffee was not good visit disappointed
```

Look at what the last two steps removed: `!!!` (intensity), `:)` (sentiment),
the `#` that marked a hashtag — **and the 😞 emoji**, which `[^\w\s]` swallowed
along with the punctuation. For a sentiment model those were the four most
predictive things in the sentence.

| Step | Removes | Cost |
|---|---|---|
| Lowercasing | Case | "US" → "us", "Apple" → "apple" |
| Stripping punctuation | `!`, `?`, emoticons | Intensity and sentiment |
| Removing digits | Numbers | Prices, dates, quantities |
| Removing stopwords | "not", "no", "very" | **Negation** — often fatal |
| Stemming | Word endings | "university" → "univers" |

**The rule: strip only what your task does not use.** Then measure with and
without, because your intuition will be wrong at least once.

---

## Stopwords and the negation trap

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["the coffee was good", "the coffee was not good"]

with_stopwords = CountVectorizer()
without_stopwords = CountVectorizer(stop_words="english")

a = with_stopwords.fit_transform(texts).toarray()
b = without_stopwords.fit_transform(texts).toarray()

print("keeping stopwords:", with_stopwords.get_feature_names_out().tolist())
print("  vectors:", a.tolist())
print("removing stopwords:", without_stopwords.get_feature_names_out().tolist())
print("  vectors:", b.tolist())
print("identical after removal:", (b[0] == b[1]).all())
```

```text
keeping stopwords: ['coffee', 'good', 'not', 'the', 'was']
  vectors: [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]
removing stopwords: ['coffee', 'good']
  vectors: [[1, 1], [1, 1]]
identical after removal: True
```

Two sentences with **opposite meanings** became the same vector. `not` is in
scikit-learn's English stopword list, and removing it deleted the entire
signal.

Stopword removal made sense when memory was scarce. With TF-IDF weighting and
modern models, it is usually unnecessary and occasionally catastrophic. If you
use it, use a list you wrote for your task — and never one that contains
negations.

---

## Stemming and lemmatisation

```text
stemming:       chop the ending with rules     studies -> studi, better -> better
lemmatisation:  map to the dictionary form     studies -> study, better -> good
```

Stemming is fast, crude and language-specific. Lemmatisation is slower,
correct, and needs a dictionary plus part-of-speech information.

```python
def naive_stem(word):
    """A toy Porter-like stemmer, to show the idea and the damage."""
    for suffix in ("ational", "ing", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word

for word in ["studies", "studying", "cats", "caresses", "bus", "operational"]:
    print(f"{word:<12} -> {naive_stem(word)}")
```

```text
studies      -> stud
studying     -> study
cats         -> cat
caresses     -> caress
bus          -> bus
operational  -> oper
```

`studies` and `studying` reduced to different stems — the normalisation
failed at exactly the job it exists for. This is why real stemmers are
hundreds of rules, and why for most modern work you should skip both:
**subword tokenisation (lesson 03) handles morphology better than either.**

Where they still help: TF-IDF on small datasets, search indexes, and
morphologically rich languages — including Arabic, where lesson 12 shows the
specific tooling.

---

## Unicode

```python
import unicodedata

café_nfc = "café"                    # é as one code point
café_nfd = "café"                   # e + combining accent

print("look identical:", café_nfc, café_nfd)
print("equal:", café_nfc == café_nfd)
print("lengths:", len(café_nfc), len(café_nfd))

normalised_a = unicodedata.normalize("NFC", café_nfc)
normalised_b = unicodedata.normalize("NFC", café_nfd)
print("equal after NFC:", normalised_a == normalised_b)
```

```text
look identical: café café
equal: False
lengths: 4 5
```

Two strings that render identically, compare unequal, and hash differently —
so your vocabulary contains both, your deduplication misses them, and your
lookup fails.

**Normalise to NFC at the point of ingestion**, before anything else touches
the text. The same applies to invisible characters:

```python
messy = "coffee​  is﻿ good"       # zero-width, nbsp, BOM
cleaned = "".join(c for c in messy if unicodedata.category(c) != "Cf")
cleaned = cleaned.replace(" ", " ")

print("raw repr:    ", repr(messy))
print("cleaned repr:", repr(" ".join(cleaned.split())))
```

```text
raw repr:     'coffee​ \xa0is﻿ good'
cleaned repr: 'coffee is good'
```

Zero-width spaces and non-breaking spaces come from copy-paste, PDFs and web
scraping. They are invisible in your terminal and they split words in your
tokeniser.

---

## A preprocessing function

```python
import re
import unicodedata

URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\S+@\S+\.\S+")
WHITESPACE = re.compile(r"\s+")

def clean(text, lowercase=True, keep_emoji=True, replace_entities=True):
    """Normalise text for modelling. Every step is optional and documented."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")

    if replace_entities:
        text = URL.sub(" <url> ", text)          # a token, not deletion
        text = EMAIL.sub(" <email> ", text)

    if not keep_emoji:
        text = "".join(c for c in text if not unicodedata.category(c).startswith("So"))

    if lowercase:
        text = text.lower()

    return WHITESPACE.sub(" ", text).strip()

samples = [
    "Visit https://shop.example.com NOW!!! 😞",
    "Contact  us​ at sales@example.com",
    "Café   CLOSED   today",
]
for sample in samples:
    print(f"{sample!r}\n  -> {clean(sample)!r}")
```

```text
'Visit https://shop.example.com NOW!!! 😞'
  -> 'visit <url> now!!! 😞'
'Contact  us​ at sales@example.com'
  -> 'contact us at <email>'
'Café   CLOSED   today'
  -> 'café closed today'
```

Three decisions in that function worth copying:

- **Replace, do not delete.** `<url>` keeps the fact that a link was there,
  which is often predictive (spam, marketing).
- **Every step is a flag.** You will need to measure with and without.
- **Normalisation first**, before any matching or lowercasing.

---

## Measure it

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

categories = ["rec.sport.hockey", "sci.space", "talk.politics.mideast", "comp.graphics"]
train = fetch_20newsgroups(subset="train", categories=categories,
                           remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", categories=categories,
                          remove=("headers", "footers", "quotes"))

settings = {
    "raw": {},
    "lowercase only": {"lowercase": True},
    "english stopwords": {"stop_words": "english"},
    "min_df=5": {"min_df": 5},
}
for name, options in settings.items():
    model = make_pipeline(TfidfVectorizer(**options), LogisticRegression(max_iter=1000))
    model.fit(train.data, train.target)
    accuracy = accuracy_score(test.target, model.predict(test.data))
    vocabulary = len(model.named_steps["tfidfvectorizer"].vocabulary_)
    print(f"{name:<20} accuracy {accuracy:.4f}   vocabulary {vocabulary:,}")
```

```text
raw                  accuracy 0.8832   vocabulary 32,288
lowercase only       accuracy 0.8832   vocabulary 32,288
english stopwords    accuracy 0.9127   vocabulary 31,986
min_df=5             accuracy 0.8870   vocabulary 7,105
```

Four configurations, and two surprises:

- Lowercasing is already `TfidfVectorizer`'s default — the two rows are
  identical, and passing the flag changed nothing.
- **Stopword removal gained 2.9 points here** (0.8832 → 0.9127), the largest
  single effect in the table.
- `min_df=5` cut the vocabulary by 78% for +0.4 points. Much smaller, slightly
  better.

Now read that second bullet against the negation example earlier in this
lesson, where removing stopwords destroyed the signal completely. Both are
true, and the difference is the **task**:

- **Topic classification** — "the", "was", "is" appear equally in every class,
  so they are pure noise and removing them helps.
- **Sentiment, stance, intent** — "not", "no", "never" carry the label, and
  removing them is fatal.

There is no universally correct preprocessing. There is only preprocessing
that suits your task, and the only way to know is the table above, run on your
own data.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Removing stopwords containing negations | Opposite meanings become identical |
| Stripping punctuation for sentiment | You delete `!!!` and `:)` |
| No Unicode normalisation | Duplicate vocabulary entries, failed matching |
| Aggressive stemming | Different words collapse, related words do not |
| Cleaning the test set differently | Train/serve skew |
| Preprocessing without measuring | Effort spent making the model worse |

---

## Exercises

1. Take 20 real documents and clean them; read the before and after by hand.
2. Show the negation trap on your own data with `CountVectorizer`.
3. Find a Unicode issue in a real corpus — normalisation or invisible
   characters.
4. Measure four preprocessing settings, as above. Which wins on your data?
