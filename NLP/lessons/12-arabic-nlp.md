# Lesson 12 — Arabic NLP

**Goal:** handle Arabic as Arabic, not as English with different letters.

## What you will learn

- What actually differs: script, morphology, dialect
- Normalisation that helps and normalisation that destroys
- Tokenisation cost, measured
- Which models to use

---

## Four real differences

| | English | Arabic |
|---|---|---|
| Script | Left to right, spaced | Right to left, letters join, optional diacritics |
| Capitalisation | Marks names | **Does not exist** — NER loses its strongest feature |
| Morphology | Light | Rich: one word can be a whole clause |
| Standard form | One written standard | MSA for writing, many dialects for speech and social media |

```python
words = {
    "وكتبناها": "and we wrote it",
    "أفسيقونيها": "will you water it for us",
    "بالمدرسة": "in the school",
}
for arabic, english in words.items():
    print(f"{arabic:<12} {len(arabic):>2} characters, 1 word  =  "
          f"{len(english.split())} English words: {english}")
```

```text
وكتبناها      8 characters, 1 word  =  4 English words: and we wrote it
أفسيقونيها   10 characters, 1 word  =  6 English words: will you water it for us
بالمدرسة      8 characters, 1 word  =  3 English words: in the school
```

One Arabic word carries a conjunction, a preposition, a verb, a subject and an
object. Anything that splits on whitespace and treats a token as a unit of
meaning is already wrong.

---

## Normalisation

Arabic has several ways to write the same thing. Unify them — carefully.

```python
import re
import unicodedata

ALEF = re.compile("[إأآا]")
YEH = re.compile("ى")
TEH_MARBUTA = re.compile("ة")
TATWEEL = re.compile("ـ+")
DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰۖ-ۭ]")

def normalise_arabic(text, strip_diacritics=True, unify_alef=True,
                     unify_yeh=True, unify_teh=False):
    """Unicode-normalise and optionally unify Arabic letter variants."""
    text = unicodedata.normalize("NFKC", text)
    text = TATWEEL.sub("", text)
    if strip_diacritics:
        text = DIACRITICS.sub("", text)
    if unify_alef:
        text = ALEF.sub("ا", text)
    if unify_yeh:
        text = YEH.sub("ي", text)
    if unify_teh:
        text = TEH_MARBUTA.sub("ه", text)
    return re.sub(r"\s+", " ", text).strip()

samples = ["إستقبال", "استقبال", "أحمد", "احمد", "مـــرحبا", "مَدْرَسَة", "مدرسة"]
for sample in samples:
    print(f"{sample:<12} -> {normalise_arabic(sample)}")

print("\nvariants that collapse to one form:")
print(len({normalise_arabic(s) for s in ["إستقبال", "استقبال", "اِستقبال"]}), "distinct")
```

```text
إستقبال      -> استقبال
استقبال      -> استقبال
أحمد         -> احمد
احمد         -> احمد
مـــرحبا     -> مرحبا
مَدْرَسَة    -> مدرسة
مدرسة        -> مدرسة
variants that collapse to one form:
1 distinct
```

Three spellings of the same word became one. Without this, your vocabulary
holds each variant separately, your TF-IDF splits their weight, and exact
matching fails.

**What to be careful with:**

- **Diacritics** usually carry no information in modern writing — strip them.
  In classical or religious text they are meaning, and stripping is
  destructive.
- **Alef forms** (`إ أ آ ا`) are written inconsistently; unifying helps
  almost always.
- **`ة` → `ه`** is common in dialect writing but changes the word. Off by
  default, above, deliberately.
- **`ى` → `ي`** helps for search, and loses a real distinction in some words.

Each of these is a hypothesis — measure it, as lesson 02 did.

---

## Tokenisation costs more

```python
from transformers import AutoTokenizer

english_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
multilingual = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

pairs = [
    ("the coffee was excellent and the service was fast",
     "كانت القهوة ممتازة والخدمة سريعة"),
    ("i want to return this order",
     "أريد أن أرجع هذا الطلب"),
]

print(f"{'language':<10}{'chars':>7}{'tokens':>8}{'chars/token':>13}")
for english, arabic in pairs:
    for label, text in [("english", english), ("arabic", arabic)]:
        tokens = english_tokenizer.tokenize(text)
        print(f"{label:<10}{len(text):>7}{len(tokens):>8}{len(text) / len(tokens):>13.2f}")
    print()
```

```text
language    chars  tokens  chars/token
english        49       9         5.44
arabic         32      27         1.19

english        27       6         4.50
arabic         22      11         2.00
```

An English tokeniser gets **5.44 characters per token** on English and
**1.19** on Arabic — close to one token per character. Twenty-seven tokens for
a five-word sentence. The consequences are not subtle:

- **Three to five times the tokens per word**, so three to five times the API
  cost for the same content.
- **Sequences hit the length limit sooner**, so more truncation.
- The model sees letters, not morphemes, and learns far less per token.

```python
print(english_tokenizer.tokenize("القهوة"))
```

```text
['ا', '##ل', '##ق', '##ه', '##و', '##ة']
```

That is the word for "the coffee", shattered into six characterless pieces.
An Arabic-aware tokeniser encodes it in one or two.

**Check this before choosing a model.** Tokenise fifty sentences of your real
text and compare characters per token across candidates; the difference
between 1.0 and 4.0 is your cost and your quality.

---

## Dialects

MSA is what news and books are written in. Almost nothing on social media is
MSA.

```text
MSA:       أريد أن أذهب إلى المطعم
Egyptian:  عايز أروح المطعم
Gulf:      أبغى أروح المطعم
Levantine: بدي روح عالمطعم
```

Four ways to say "I want to go to the restaurant", sharing very little
surface form. A model trained on MSA news text handles the first well and the
other three poorly — and your users write the other three.

Practical consequences:

- **Label data in the dialect you will serve.** MSA training data does not
  transfer to Egyptian Arabic for free.
- **Expect code-switching**: Arabic and English in one sentence, plus
  Franco-Arabic (`3ayez arou7`) written in Latin letters with digits.
- **Franco-Arabic needs its own handling** — it is invisible to every Arabic
  model, because it is not Arabic script.

---

## Models to use

| Model | Notes |
|---|---|
| `aubmindlab/bert-base-arabertv2` | The standard Arabic BERT; strong on MSA |
| `CAMeL-Lab/bert-base-arabic-camelbert-mix` | MSA, dialect and classical mixed |
| `UBC-NLP/MARBERT` | Trained on tweets — the best starting point for dialect |
| `xlm-roberta-base` | Multilingual; a reasonable fallback, weaker than the above |
| `intfloat/multilingual-e5-base` | Multilingual retrieval, for search |
| `camel-tools` | Not a model: morphology, dialect ID, normalisation |

Do not assume an English model transfers. **Measure on your own Arabic data**,
with an Arabic baseline (TF-IDF over normalised text) to compare against — the
baseline is often closer than expected, because Arabic morphology makes the
bag of words surprisingly informative once normalised.

---

## A checklist for an Arabic project

- [ ] Unicode NFKC normalisation at ingestion
- [ ] Tatweel removed, alef forms unified, diacritics decided deliberately
- [ ] Tokenisation cost measured on your text, across candidate models
- [ ] Dialect identified — and training data in that dialect
- [ ] Code-switching and Franco-Arabic handled or explicitly out of scope
- [ ] NER does not rely on capitalisation
- [ ] An Arabic TF-IDF baseline, for comparison
- [ ] Right-to-left rendering checked in any interface you build

---

## Common mistakes

| Mistake | What happens |
|---|---|
| An English tokeniser on Arabic | 3–5× the tokens, far worse quality |
| No letter normalisation | Three spellings, three vocabulary entries |
| Stripping diacritics from classical text | Meaning destroyed |
| MSA training data for dialect users | Quietly poor accuracy in production |
| English NER heuristics | No capitalisation to lean on |
| Ignoring Franco-Arabic | A whole segment of users unhandled |

---

## Exercises

1. Normalise 50 real Arabic sentences; count how much the vocabulary shrinks.
2. Compare characters per token across three tokenisers on your own text.
3. Collect ten dialect sentences and ten MSA; test a model on both.
4. Build an Arabic TF-IDF baseline and compare it with AraBERT.
