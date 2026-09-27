# Lesson 05 — Embeddings and Search

**Goal:** turn text into vectors, and measure whether that beats keyword search
on your own data.

## What you will learn

- What an embedding is, and mean pooling
- Recall@k, the metric that matters for retrieval
- Keywords against embeddings against hybrid, measured
- Where each one fails

---

## A knowledge base and a query set

Retrieval cannot be evaluated without **queries with known correct answers**.
This is the part teams skip, and it is why they cannot tell whether their
retrieval works.

```python
from pathlib import Path

KB = r'''
DOCS = [
 ("refund-window", "Customers may request a refund within 14 days of purchase. Refunds are issued to the original payment method within 5 business days."),
 ("refund-exceptions", "Digital gift cards and personalised items cannot be refunded once delivered."),
 ("shipping-times", "Standard shipping inside Cairo takes 1 to 2 working days. Other governorates take 3 to 5 working days."),
 ("shipping-cost", "Shipping costs 40 EGP and is free on orders above 600 EGP."),
 ("payment-methods", "We accept Visa, Mastercard, Meeza, and cash on delivery. Cash on delivery adds a 15 EGP fee."),
 ("payment-failed", "If a card payment fails, the order is held for 24 hours before being cancelled automatically."),
 ("account-delete", "To delete your account, email support. Deletion removes your order history permanently after 30 days."),
 ("account-password", "Password resets are sent by email and the link expires after 60 minutes."),
 ("loyalty-points", "Members earn 1 point per 10 EGP spent. 100 points can be exchanged for a free drink."),
 ("loyalty-expiry", "Loyalty points expire 12 months after they were earned."),
 ("opening-hours", "Branches open at 8:00 and close at 23:00, except the Maadi branch which closes at 01:00."),
 ("wifi", "Free wifi is available in all branches. The password is printed on the receipt."),
 ("allergens", "All drinks containing nuts are marked on the menu. Please tell staff about allergies before ordering."),
 ("vegan-options", "Oat and almond milk are available at no extra cost. Three cakes are fully vegan."),
 ("catering", "Catering orders require 48 hours notice and a minimum of 20 people."),
 ("complaints", "Complaints are answered within 2 working days. Escalations go to the branch manager."),
 ("lost-items", "Items left in a branch are kept for 14 days at the counter."),
 ("job-applications", "Send your CV to careers@example.com. We review applications every two weeks."),
 ("franchise", "Franchise enquiries require a business plan and proof of 2 million EGP capital."),
 ("gift-cards", "Gift cards are valid for 12 months and can be topped up in any branch."),
 ("subscriptions", "The monthly coffee subscription costs 400 EGP for 20 drinks and can be paused twice a year."),
 ("subscription-cancel", "Subscriptions can be cancelled any time; the current month is not refunded."),
 ("parking", "The Zamalek and Heliopolis branches have parking. The Downtown branch does not."),
 ("groups", "Tables for more than 6 people should be booked by phone at least one day ahead."),
 ("bean-origin", "Our house blend is 70% Ethiopian and 30% Brazilian arabica, roasted weekly in Cairo."),
 ("decaf", "Decaf is available for all espresso drinks at no extra charge."),
 ("delivery-apps", "We are on Talabat and Elmenus. Prices there include a 10% platform markup."),
 ("invoices", "VAT invoices can be requested at the counter or by email within 30 days of purchase."),
 ("pets", "Well-behaved dogs are welcome in the outdoor seating areas only."),
 ("noise-policy", "The Downtown branch has a quiet zone upstairs with no music after 18:00."),
]

QUERIES = [
 ("How long do I have to get my money back?", {"refund-window"}),
 ("Can I return a gift card?", {"refund-exceptions", "gift-cards"}),
 ("When will my order arrive in Alexandria?", {"shipping-times"}),
 ("Is delivery free?", {"shipping-cost"}),
 ("Do you take cash?", {"payment-methods"}),
 ("My visa was declined, what happens to the order?", {"payment-failed"}),
 ("I want to close my account", {"account-delete"}),
 ("How many points for a free coffee?", {"loyalty-points"}),
 ("Do my points ever run out?", {"loyalty-expiry"}),
 ("What time do you close in Maadi?", {"opening-hours"}),
 ("Is there internet in the cafe?", {"wifi"}),
 ("I am allergic to peanuts", {"allergens"}),
 ("Do you have plant milk?", {"vegan-options"}),
 ("I need coffee for an office event of 40 people", {"catering"}),
 ("Where can I leave a complaint?", {"complaints"}),
 ("I forgot my bag at your shop", {"lost-items"}),
 ("How do I apply for a job?", {"job-applications"}),
 ("I want to open a branch of your brand", {"franchise"}),
 ("Can I stop my monthly plan?", {"subscription-cancel", "subscriptions"}),
 ("Is there somewhere to leave the car?", {"parking"}),
 ("Where do your beans come from?", {"bean-origin"}),
 ("Can I bring my dog?", {"pets"}),
 ("I need a tax invoice", {"invoices"}),
 ("Somewhere quiet to work in the evening", {"noise-policy"}),
]
'''

Path("/tmp/kb.py").write_text(KB)
import sys
sys.path.insert(0, "/tmp")
from kb import DOCS, QUERIES
print(f"{len(DOCS)} documents, {len(QUERIES)} labelled queries")
```

```text
30 documents, 24 labelled queries
```

24 queries, each with the set of documents that genuinely answer it. Thirty
minutes of work, and every number in this lesson depends on it.

---

## Embedding text

```python
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(MODEL)
enc = AutoModel.from_pretrained(MODEL).eval()

def embed(texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i+batch], padding=True, truncation=True,
                max_length=256, return_tensors="pt")
        with torch.no_grad():
            h = enc(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1)          # mean pooling
        out.append(torch.nn.functional.normalize(v, dim=1))
    return torch.cat(out).numpy()

ids = [d[0] for d in DOCS]
texts = [d[1] for d in DOCS]
V = embed(texts)
print(f"corpus: {len(DOCS)} documents, embedding dim {V.shape[1]}")
print(f"a normalised vector's norm: {np.linalg.norm(V[0]):.3f}")
```

```text
corpus: 30 documents, embedding dim 384
a normalised vector's norm: 1.000
```

Three details in `embed` that people get wrong:

**Mean pooling over the attention mask.** The model returns one vector per
token; you need one per document. Averaging over *all* positions including
padding drags every short document towards the padding vector. The mask makes
the average cover real tokens only.

**Normalising to unit length.** After normalisation, the dot product *is* the
cosine similarity, so search becomes one matrix multiply.

**`truncation=True, max_length=256`.** Anything past 256 tokens is silently
discarded. A 2,000-word document embedded whole is mostly *not* embedded — which
is why lesson 06 is about chunking.

---

## Nearest neighbours

```python
q = "How long do I have to get my money back?"
qv = embed([q])[0]
sims = V @ qv
order = np.argsort(sims)[::-1][:4]
print(f"query: {q!r}")
for i in order:
    print(f"  {sims[i]:.3f}  {ids[i]:<20}{texts[i][:56]}")
```

```text
query: 'How long do I have to get my money back?'
  0.668  refund-window       Customers may request a refund within 14 days of purchas
  0.364  payment-failed      If a card payment fails, the order is held for 24 hours
  0.323  subscription-cancel Subscriptions can be cancelled any time; the current mon
  0.322  invoices            VAT invoices can be requested at the counter or by email
```

The query and the correct document **share no content words at all**: "money
back" against "refund", "how long" against "within 14 days". Cosine similarity
0.668, and the next candidate is far behind at 0.364.

Note the absolute numbers. 0.668 is a *good* match here, and 0.322 is
effectively unrelated. **These scales are model-specific and not calibrated** —
0.7 does not mean 70% relevant, and a threshold tuned on one embedding model is
meaningless on another. Tune the threshold on your own labelled queries, or
avoid thresholds and use top-k.

---

## Keywords against embeddings

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english").fit(texts)
D = tfidf.transform(texts)

def rank_tfidf(query):
    qv = tfidf.transform([query])
    return np.asarray((D @ qv.T).todense()).ravel()

def rank_dense(query):
    return V @ embed([query])[0]

def recall_at_k(ranker, k):
    hits = 0
    for query, relevant in QUERIES:
        scores = ranker(query)
        top = [ids[i] for i in np.argsort(scores)[::-1][:k]]
        hits += any(t in relevant for t in top)
    return hits / len(QUERIES)

def rank_hybrid(query, alpha=0.5):
    a, b = rank_tfidf(query), rank_dense(query)
    norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    return alpha * norm(a) + (1 - alpha) * norm(b)

print(f"{'method':<22}{'recall@1':>10}{'recall@3':>10}{'recall@5':>10}")
for name, r in [("TF-IDF (keywords)", rank_tfidf), ("MiniLM (embeddings)", rank_dense),
                ("hybrid 50/50", rank_hybrid)]:
    print(f"{name:<22}{recall_at_k(r, 1):>10.2f}{recall_at_k(r, 3):>10.2f}"
          f"{recall_at_k(r, 5):>10.2f}")
```

```text
method                  recall@1  recall@3  recall@5
TF-IDF (keywords)           0.33      0.50      0.54
MiniLM (embeddings)         0.79      0.92      1.00
hybrid 50/50                0.71      0.92      0.96
```

**Embeddings more than double TF-IDF's recall@1**, 0.79 against 0.33, and reach
1.00 by k=5 where TF-IDF plateaus at 0.54.

The gap is this large because real user questions and written documentation use
**different words for the same thing**. TF-IDF cannot match "money back" to
"refund"; it has no notion that they are related.

And the honest surprise: **the hybrid is worse than pure embeddings** at k=1
(0.71 against 0.79) and at k=5 (0.96 against 1.00). The usual advice is that
hybrid always wins. It wins when queries contain **exact tokens that must
match** — product codes, error numbers, names, SKUs — because embeddings blur
those. This query set has none, so the keyword half only adds noise.

**Which means: measure it on your queries.** If your users paste error codes,
hybrid will win. If they ask questions in their own words, it may not.

---

## Where keywords fail

```python
for query, relevant in QUERIES:
    t1 = [ids[i] for i in np.argsort(rank_tfidf(query))[::-1][:1]]
    d1 = [ids[i] for i in np.argsort(rank_dense(query))[::-1][:1]]
    if (t1[0] in relevant) != (d1[0] in relevant):
        winner = "embeddings" if d1[0] in relevant else "TF-IDF"
        print(f"  {query[:44]:<46} tfidf->{t1[0]:<20} dense->{d1[0]:<20} {winner}")
```

```text
  How long do I have to get my money back?       tfidf->noise-policy         dense->refund-window        embeddings
  Can I return a gift card?                      tfidf->payment-failed       dense->refund-exceptions    embeddings
  When will my order arrive in Alexandria?       tfidf->payment-failed       dense->shipping-times       embeddings
  Is delivery free?                              tfidf->payment-methods      dense->shipping-cost        embeddings
  I am allergic to peanuts                       tfidf->noise-policy         dense->allergens            embeddings
  Where can I leave a complaint?                 tfidf->noise-policy         dense->complaints           embeddings
  How do I apply for a job?                      tfidf->noise-policy         dense->job-applications     embeddings
  Can I stop my monthly plan?                    tfidf->franchise            dense->subscription-cancel  embeddings
  Is there somewhere to leave the car?           tfidf->noise-policy         dense->parking              embeddings
  Can I bring my dog?                            tfidf->noise-policy         dense->pets                 embeddings
  I need a tax invoice                           tfidf->noise-policy         dense->invoices             embeddings
```

Eleven queries where embeddings win and TF-IDF loses. **Zero the other way.**

Look at how often TF-IDF answers `noise-policy`. That document is about a quiet
zone upstairs — it has nothing to do with dogs, parking, jobs or allergies. It
wins by default because **none of the query's words appear anywhere in the
corpus**, so every document scores near zero and the ranking is essentially
arbitrary.

That is the failure mode to watch for in production: a keyword system with no
match does not say "I don't know", it returns its arbitrary favourite. Which is
why lesson 06 measures whether the *retrieved context actually contains the
answer*, instead of trusting that retrieval worked.

---

## Choosing a setup

| Corpus size | Index |
|---|---|
| Under ~10,000 chunks | A NumPy matrix and a dot product. This lesson |
| 10k - 1M | `faiss` flat or HNSW, in memory |
| Over 1M | A vector database, and a serious look at whether you need one |

| You need | Do this |
|---|---|
| Exact identifiers matched | Hybrid, or keyword filter then rerank |
| Multiple languages | A multilingual embedding model — test it on your language |
| Best possible accuracy | Retrieve 20 with embeddings, rerank with a cross-encoder |
| Lowest latency | Precompute and cache document vectors; only the query is live |

The reranking row is the highest-value upgrade not shown here: a cross-encoder
scores each (query, document) pair jointly instead of comparing two independent
vectors. It is far slower, which is why it runs on the top 20 rather than the
whole corpus.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| No labelled query set | Every number in this lesson would be unavailable |
| Mean pooling without the attention mask | Short documents drift towards the padding vector |
| Embedding documents longer than the model's limit | Silently truncated at 256 tokens |
| Comparing similarity scores across models | Uncalibrated and model-specific |
| Assuming hybrid always wins | It lost here, at k=1 and k=5 |
| Assuming keyword search is a safe fallback | With no match it returns an arbitrary document, confidently |
| Reaching for a vector database at 30 documents | A 30x384 matrix is a dot product |

---

## Exercises

1. Add 10 queries containing exact strings — an order number, a branch name, a
   price. Rerun the comparison. Does hybrid now beat pure embeddings?
2. Measure MRR (mean reciprocal rank) alongside recall@k. Which metric would
   you report to an engineer, and which to a product owner?
3. Truncate every document to its first 5 words and rerun. How much recall
   survives, and what does that tell you about where the signal lives?
4. Swap MiniLM for a multilingual model and evaluate on Arabic translations of
   the queries. Report recall@3 for both languages.
5. Build the "no good match" detector: find a similarity threshold below which
   you return "I don't know" instead of the top document. Report how many of
   the 24 queries it wrongly refuses.

---

**Next:** [Lesson 06 — RAG](06-rag.md)
