# Lesson 04 — Bag of Words and TF-IDF

**Goal:** the baseline that beats a transformer more often than anyone admits.

## What you will learn

- Counting words into a matrix
- TF-IDF, and what the IDF part does
- N-grams
- Reading a linear model's coefficients

---

## Bag of words

Count each word. Throw away the order. It works far better than it has any
right to.

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "the coffee is good",
    "the coffee is bad",
    "good coffee good service",
]

vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(texts)

print("vocabulary:", vectorizer.get_feature_names_out().tolist())
print("matrix:")
print(matrix.toarray())
print("shape:", matrix.shape, "| stored values:", matrix.nnz)
```

```text
vocabulary: ['bad', 'coffee', 'good', 'is', 'service', 'the']
matrix:
[[0 1 1 1 0 1]
 [1 1 0 1 0 1]
 [0 1 2 0 1 0]]
shape: (3, 6) | stored values: 11
```

Each row is a document, each column a vocabulary word. The matrix is **sparse**
— scikit-learn stores only the 11 non-zero values, not 18. On a real corpus
with a 30,000-word vocabulary and 20 words per document, 99.9% of the cells
are zero, and sparse storage is the only reason this is practical.

The cost is stated in the name: "the coffee is good" and "good is coffee the"
produce identical rows.

---

## TF-IDF

Raw counts over-reward common words. TF-IDF scales each count by how rare the
word is across the corpus.

```text
tf(t, d)   = count of t in d, normalised
idf(t)     = log(N / documents containing t) + 1
tfidf      = tf × idf
```

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "the coffee is good",
    "the coffee is bad",
    "the coffee is cold",
    "the espresso is excellent",
]

vectorizer = TfidfVectorizer()
vectorizer.fit(texts)

idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_.round(3)))
for word in sorted(idf, key=idf.get):
    print(f"{word:<12} idf {idf[word]}")
```

```text
is           idf 1.0
the          idf 1.0
coffee       idf 1.223
bad          idf 1.916
cold         idf 1.916
espresso     idf 1.916
excellent    idf 1.916
good         idf 1.916
```

`the` and `is` appear in every document, so their IDF is the minimum — they
are automatically discounted without a stopword list. `espresso` appears once
and scores nearly double.

**This is why TF-IDF reduces the need for a stopword list**: the weighting
discounts ubiquitous words automatically, per corpus, rather than from a fixed
list someone else wrote.

It does not make stopword removal useless — lesson 02 measured a 2.9-point
gain from it on this same dataset, because dropping those columns entirely
also removes the noise they add to every distance calculation. It does mean
you should treat the list as a hyperparameter, not a ritual.

---

## N-grams put some order back

```python
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["the coffee was not good", "the coffee was good"]

unigrams = TfidfVectorizer()
bigrams = TfidfVectorizer(ngram_range=(1, 2))

unigrams.fit(texts)
bigrams.fit(texts)

print("unigrams:", unigrams.get_feature_names_out().tolist())
print("with bigrams:", bigrams.get_feature_names_out().tolist())
print(f"vocabulary size: {len(unigrams.vocabulary_)} -> {len(bigrams.vocabulary_)}")
```

```text
unigrams: ['coffee', 'good', 'not', 'the', 'was']
with bigrams: ['coffee', 'coffee was', 'good', 'not', 'not good', 'the', 'the coffee', 'was', 'was good', 'was not']
vocabulary size: 5 -> 10
```

`not good` is now a feature in its own right — the negation that unigrams
could not represent. The cost is a vocabulary that grows fast; control it with
`min_df` and `max_features`.

`ngram_range=(1, 2)` is the usual setting. Trigrams rarely pay for themselves
on anything but large corpora.

---

## The full baseline, measured

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import time

categories = ["rec.sport.hockey", "sci.space", "talk.politics.mideast", "comp.graphics"]
train = fetch_20newsgroups(subset="train", categories=categories,
                           remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", categories=categories,
                          remove=("headers", "footers", "quotes"))

configurations = [
    ("counts + naive bayes", CountVectorizer(), MultinomialNB()),
    ("tfidf + naive bayes", TfidfVectorizer(), MultinomialNB()),
    ("tfidf + logistic", TfidfVectorizer(), LogisticRegression(max_iter=1000)),
    ("tfidf + linear svm", TfidfVectorizer(), LinearSVC()),
    ("tfidf 1-2gram + svm", TfidfVectorizer(ngram_range=(1, 2), min_df=2), LinearSVC()),
]

for name, vectorizer, classifier in configurations:
    model = make_pipeline(vectorizer, classifier)
    start = time.perf_counter()
    model.fit(train.data, train.target)
    elapsed = time.perf_counter() - start
    accuracy = accuracy_score(test.target, model.predict(test.data))
    print(f"{name:<22} accuracy {accuracy:.4f}   trained in {elapsed:.2f}s")
```

```text
counts + naive bayes   accuracy 0.8864   trained in 0.15s
tfidf + naive bayes    accuracy 0.9050   trained in 0.15s
tfidf + logistic       accuracy 0.8832   trained in 0.51s
tfidf + linear svm     accuracy 0.9134   trained in 0.18s
tfidf 1-2gram + svm    accuracy 0.8999   trained in 0.47s
```

**91.3% in 0.18 seconds of training**, on a laptop, with no GPU and no
pretrained model. Remember this number when a fine-tuned transformer costs an
afternoon to reach 93%.

Three observations worth carrying:

- **The linear SVM is both the most accurate and among the fastest.** For
  sparse text features, `LinearSVC` is usually the best classical choice —
  logistic regression is three times slower here and three points worse.
- TF-IDF helped naive Bayes by 1.9 points, and hurt nothing.
- **Bigrams cost 1.4 points** (0.9134 → 0.8999) while quadrupling the feature
  count. On 2,341 documents there is not enough data to estimate that many
  features, and the extra ones are noise. N-grams help on large corpora and
  hurt on small ones — which you only learn by running both rows.

---

## Read what it learned

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

categories = ["rec.sport.hockey", "sci.space", "talk.politics.mideast", "comp.graphics"]
train = fetch_20newsgroups(subset="train", categories=categories,
                           remove=("headers", "footers", "quotes"))

model = make_pipeline(TfidfVectorizer(min_df=2), LinearSVC()).fit(train.data, train.target)
words = model.named_steps["tfidfvectorizer"].get_feature_names_out()
weights = model.named_steps["linearsvc"].coef_

for index, category in enumerate(train.target_names):
    top = np.argsort(weights[index])[-6:][::-1]
    print(f"{category:<24} {[words[i] for i in top]}")
```

```text
comp.graphics            ['graphics', 'computer', 'image', 'file', '3d', 'hi']
rec.sport.hockey         ['hockey', 'game', 'team', 'games', 'nhl', 'season']
sci.space                ['space', 'orbit', 'nasa', 'moon', 'launch', 'spacecraft']
talk.politics.mideast    ['israel', 'israeli', 'jews', 'jewish', 'loser', 'mr']
```

Every prediction can be traced to words a human can read. Most of these make
obvious sense — and two do not: `hi` for computer graphics, and `loser` and
`mr` for the politics group.

Those are the interesting ones. They are stylistic artefacts of who posts in
each group, not the topic, and they will not transfer to any other corpus.
Reading this table is the fastest leakage check there is: when a class's top
features are an email domain, a timestamp or a posting habit, your model has
found a shortcut rather than the signal.

---

## Where this baseline wins

| Situation | TF-IDF or transformer? |
|---|---|
| Clear topical vocabulary | **TF-IDF** — often within 2 points, 100× cheaper |
| You must explain each prediction | **TF-IDF** |
| Under ~1,000 labelled examples | **TF-IDF** — a transformer overfits |
| Meaning depends on order or context | Transformer |
| Sarcasm, negation, subtle sentiment | Transformer |
| Arabic dialect, heavy morphology | Transformer (lesson 12) |
| CPU only, tight latency | **TF-IDF** |

---

## Common mistakes

| Mistake | What happens |
|---|---|
| `fit_transform` on the test set | Leakage — use `transform` |
| No `min_df` | Vocabulary full of typos appearing once |
| TF-IDF with `MultinomialNB` | Worse than plain counts, as above |
| Dense conversion (`.toarray()`) on a real corpus | Gigabytes, then a crash |
| Trigrams by default | Vocabulary explosion, no gain |
| Skipping this baseline | You cannot tell whether the transformer helped |

---

## Exercises

1. Build a TF-IDF + `LinearSVC` baseline on any labelled text of your own.
2. Compare `ngram_range` of (1,1), (1,2) and (1,3); note accuracy and
   vocabulary size.
3. Print the top ten features per class and read them. Any leakage?
4. Inspect ten misclassified documents. What do they have in common?
