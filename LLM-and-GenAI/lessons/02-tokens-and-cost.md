# Lesson 02 — Tokens, Arabic, and Cost

**Goal:** understand the unit you are billed in, and why the same sentence costs
2.6 times more in Arabic.

## What you will learn

- What a token is, by looking at them
- The Arabic tax, measured
- Why two tokenisers disagree about the same string
- The arithmetic that decides whether a feature ships

---

## A token is not a word

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

for text in ["Hello world", "unbelievable", "antidisestablishmentarianism",
             "print('hi')", "3.14159", "  leading spaces"]:
    ids = enc.encode(text)
    print(f"{text!r:<34}{len(ids):>3} tokens  {[enc.decode([i]) for i in ids]}")
```

```text
'Hello world'                       2 tokens  ['Hello', ' world']
'unbelievable'                      3 tokens  ['un', 'belie', 'vable']
'antidisestablishmentarianism'      6 tokens  ['ant', 'idis', 'establish', 'ment', 'arian', 'ism']
"print('hi')"                       4 tokens  ['print', "('", 'hi', "')"]
'3.14159'                           4 tokens  ['3', '.', '141', '59']
'  leading spaces'                  3 tokens  [' ', ' leading', ' spaces']
```

A tokeniser is a compression scheme learned from a training corpus. Common
sequences become single tokens; rare ones get chopped. The consequences are not
cosmetic:

- **`3.14159` is four tokens: `3`, `.`, `141`, `59`.** The model does not see a
  number. It sees fragments whose boundaries depend on which digit strings were
  frequent in the corpus. This is a large part of why lesson 01's `2 + 2` went
  badly, and why `478 * 39` is hopeless.
- **`unbelievable` splits as `un` + `belie` + `vable`** — not into meaningful
  morphemes. The model has to reassemble meaning from the pieces the compressor
  happened to choose.
- **Whitespace is data.** `'  leading spaces'` spends a whole token on the first
  space. Sloppy prompt indentation costs real money at volume.

---

## The Arabic tax

```python
pairs = [
    ("Hello, how are you?", "مرحبا، كيف حالك؟"),
    ("The weather is nice today.", "الجو جميل اليوم."),
    ("Artificial intelligence changes everything.", "الذكاء الاصطناعي يغير كل شيء."),
]
print(f"{'':<46}{'chars':>7}{'tokens':>8}{'tok/char':>10}")
for en, ar in pairs:
    for label, text in (("EN", en), ("AR", ar)):
        ids = enc.encode(text)
        print(f"{label} {text[:42]:<43}{len(text):>7}{len(ids):>8}"
              f"{len(ids) / len(text):>10.2f}")
    print()

en_total = sum(len(enc.encode(en)) for en, _ in pairs)
ar_total = sum(len(enc.encode(ar)) for _, ar in pairs)
print(f"total English tokens {en_total}, total Arabic tokens {ar_total} "
      f"-> Arabic costs {ar_total / en_total:.2f}x")
```

```text
                                                chars  tokens  tok/char
EN Hello, how are you?                             19       6      0.32
AR مرحبا، كيف حالك؟                                16      14      0.88

EN The weather is nice today.                      26       6      0.23
AR الجو جميل اليوم.                                16      11      0.69

EN Artificial intelligence changes everything      43       6      0.14
AR الذكاء الاصطناعي يغير كل شيء.                   29      22      0.76

total English tokens 18, total Arabic tokens 47 -> Arabic costs 2.61x
```

**The same meaning costs 2.61 times more in Arabic.** Look at the third pair:
43 English characters become 6 tokens; 29 Arabic characters become **22**.

The reason is the training corpus. `cl100k_base` was learned mostly from
English text, so English words are single tokens while Arabic words are split
into two, three or four pieces each. The tokeniser is not neutral: it encodes
which languages its authors' data contained.

Four practical consequences, and they all hit Arabic-language products:

| Consequence | Detail |
|---|---|
| **Cost** | Every Arabic request bills roughly 2.6x for the same content |
| **Context window** | An 8k window holds ~3k tokens' worth of Arabic meaning |
| **Latency** | Output is per token; an Arabic answer is 2.6x as many sequential steps |
| **Quality** | More tokens per word means longer dependencies for the same sentence |

So a per-user cost estimate built on English test prompts will be **wrong by
more than a factor of two** for an Arabic user base. Measure on your own text,
in your own language, before quoting a number to anyone.

---

## Tokenisers disagree

```python
from transformers import AutoTokenizer
gpt2 = AutoTokenizer.from_pretrained("gpt2-medium")
bert = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

samples = ["Artificial intelligence changes everything.",
           "الذكاء الاصطناعي يغير كل شيء.",
           "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"]
print(f"{'text':<46}{'cl100k':>8}{'gpt2':>7}{'bert':>7}")
for s in samples:
    print(f"{s[:44]:<46}{len(enc.encode(s)):>8}"
          f"{len(gpt2.encode(s)):>7}{len(bert.encode(s)):>7}")
```

```text
text                                            cl100k   gpt2   bert
Artificial intelligence changes everything.          6      6      7
الذكاء الاصطناعي يغير كل شيء.                       22     31     27
def fib(n): return n if n < 2 else fib(n-1)         23     25     31
```

English: 6, 6, 7 — near-identical. Arabic: **22, 31, 27** — a 41% spread. Code:
23 to 31.

So "how many tokens is this?" has no answer without naming the model. A token
budget computed with `gpt2` and spent on a `cl100k` model is off by a third on
Arabic text, and always in the direction that surprises you in the invoice.

**Use the tokeniser of the model you are actually calling.**

---

## The arithmetic that decides whether a feature ships

```python
PRICE_IN, PRICE_OUT = 3.00 / 1e6, 15.00 / 1e6     # example USD per token

def cost(prompt_tokens, output_tokens, calls_per_day):
    daily = calls_per_day * (prompt_tokens * PRICE_IN + output_tokens * PRICE_OUT)
    return daily, daily * 30

system_prompt = "You are a helpful assistant for a coffee shop in Cairo. " * 10
sys_tokens = len(enc.encode(system_prompt))
print(f"system prompt: {sys_tokens} tokens, resent on EVERY call")
print(f"{'calls/day':>10}{'prompt tok':>12}{'out tok':>9}{'per day':>10}{'per month':>12}")
for calls in (100, 1_000, 10_000):
    for extra, out in ((50, 100), (2_000, 100)):
        d, m = cost(sys_tokens + extra, out, calls)
        print(f"{calls:>10}{sys_tokens + extra:>12}{out:>9}{'$' + format(d, '.2f'):>10}"
              f"{'$' + format(m, '.2f'):>12}")
```

```text
system prompt: 121 tokens, resent on EVERY call
 calls/day  prompt tok  out tok   per day   per month
       100         171      100     $0.20       $6.04
       100        2121      100     $0.79      $23.59
      1000         171      100     $2.01      $60.39
      1000        2121      100     $7.86     $235.89
     10000         171      100    $20.13     $603.90
     10000        2121      100    $78.63    $2358.90
```

The two rows per volume are the same feature **with and without 2,000 tokens of
retrieved context** (lesson 06). At 10,000 calls a day that difference is
**$604 against $2,359 a month** — retrieval nearly quadrupled the bill.

Read the table before you build, not after. Three levers, in order of how much
they usually save:

1. **Shorten the context.** Lesson 06 shows retrieval hitting its accuracy
   ceiling at ~102 context tokens on a task where 306 was the naive choice.
2. **Cache.** The system prompt is 121 identical tokens on every call. Prompt
   caching bills those at a fraction of the price; without it you are paying
   full rate to re-send the same paragraph a million times.
3. **Use a smaller model for the easy cases.** Route by difficulty; most
   requests do not need the expensive model.

And one anti-lever: **do not shorten the output to save money** before checking
whether the short answer still works. Output tokens cost 5x input here, so the
temptation is strong, and a truncated answer that sends the user to a human
costs more than the tokens saved.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Estimating cost from English prompts for an Arabic product | Off by 2.6x |
| Counting tokens with the wrong tokeniser | 41% spread on Arabic between cl100k and gpt2 |
| `len(text.split())` as a token estimate | Wrong in both directions, badly, for Arabic and code |
| Asking a model to do arithmetic on numbers | `3.14159` is four tokens: `3`, `.`, `141`, `59` |
| Resending a static system prompt uncached | You pay for the same 121 tokens forever |
| Ignoring whitespace in prompts | Indentation is billable |
| Planning context length in characters | A 4k window is not 4k characters of Arabic |

---

## Exercises

1. Compute the Arabic-to-English token ratio for 20 sentences of your own
   product's copy. Report the median and the worst case.
2. Tokenise `"١٢٣"` (Arabic-Indic digits) and `"123"`. How many tokens each, and
   what does that imply about numeric input in Arabic forms?
3. Take your longest system prompt and cut it by half without losing meaning.
   Report the monthly saving at 5,000 calls a day.
4. Find the token count at which your context exceeds the model's window, then
   write the truncation rule you would use — and say which information it drops.
5. Price the same feature on a model at $0.15/$0.60 per million instead of
   $3/$15. At what volume does the cheaper model's lower quality stop being
   worth the saving? State the assumption that makes your answer possible.

---

**Next:** [Lesson 03 — Decoding](03-decoding.md)
