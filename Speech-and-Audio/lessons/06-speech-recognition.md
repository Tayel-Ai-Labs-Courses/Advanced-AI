# Lesson 06 — Speech Recognition

**Goal:** evaluate a recogniser by what it costs you, not by its word error
rate.

## What you will learn

- How modern ASR works, in one page
- WER, and the four errors it scores identically
- Two systems with the same WER and different outcomes
- Arabic, dialect, and code-switching

---

## How ASR works now

Three generations, and you will meet all three in the wild:

```text
HMM-GMM        phonemes, a pronunciation dictionary, a language model
(1990s-2010s)  many components, each separately trained and tuned

CTC / HYBRID   a neural acoustic model emits characters per frame
(2015-2020)    an alignment-free loss; still often a separate language model

ENCODER-DECODER  audio in, text out, one model, trained on huge data
(2020-)        Whisper and its relatives. Handles punctuation and language id
```

What they share is the input: **a log-mel spectrogram**, exactly the one you
built in [lesson 03](03-features.md). Everything in lessons 01-05 is upstream
of all three, which is why this course spends five lessons there and one here.

What changed with the third generation is where the difficulty moved. The
acoustic modelling is largely solved for clear English; the remaining problems
are **your audio pipeline, your domain vocabulary, your accent coverage, and
your evaluation** — which are the subjects of the rest of this lesson.

---

## Word error rate

```python
import numpy as np

def wer(ref, hyp):
    """Edit distance between word sequences, divided by reference length."""
    r, h = ref.split(), hyp.split()
    D = np.zeros((len(r)+1, len(h)+1), dtype=int)
    D[:, 0] = np.arange(len(r)+1); D[0, :] = np.arange(len(h)+1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            D[i, j] = min(D[i-1, j]+1,                      # deletion
                          D[i, j-1]+1,                      # insertion
                          D[i-1, j-1] + (r[i-1] != h[j-1])) # substitution
    return D[len(r), len(h)] / len(r)

REF = "cancel the order for table four"
cases = [
 ("perfect",                   "cancel the order for table four"),
 ("one filler word added",     "cancel the uh order for table four"),
 ("'the' dropped",             "cancel order for table four"),
 ("'four' heard as 'fourteen'","cancel the order for table fourteen"),
 ("'cancel' heard as 'confirm'","confirm the order for table four"),
 ("two small words wrong",     "cancel an order for a table four"),
]
print(f'reference: "{REF}"\n')
print(f"{'what went wrong':<30}{'WER':>7}  transcript")
for name, hyp in cases:
    print(f"{name:<30}{wer(REF, hyp):>7.3f}  {hyp}")
```

```text
reference: "cancel the order for table four"

what went wrong                   WER  transcript
perfect                         0.000  cancel the order for table four
one filler word added           0.167  cancel the uh order for table four
'the' dropped                   0.167  cancel order for table four
'four' heard as 'fourteen'      0.167  cancel the order for table fourteen
'cancel' heard as 'confirm'     0.167  confirm the order for table four
two small words wrong           0.333  cancel an order for a table four
```

**Four completely different outcomes all score 0.167.**

- An `uh` was inserted. **Nothing happened.**
- `the` was dropped. **Nothing happened.**
- Table four became table fourteen. **The wrong customer's order is cancelled.**
- `cancel` became `confirm`. **The action is inverted.**

And the last row — two harmless function words wrong — scores **twice as badly**
as inverting the action.

WER treats every word as equally important because it has no idea what any of
them mean. It is a useful engineering metric for comparing acoustic models on
the same data, and it is **not a measure of whether your product works.**

---

## The same WER, different outcomes

```python
ref_set = [("cancel the order for table four", "cancel", 4),
           ("confirm the order for table nine", "confirm", 9),
           ("cancel the order for table twelve", "cancel", 12)]
hyps = {
 "system A": ["cancel the order for table four",
              "confirm an order for table nine",
              "cancel the order for table twelve"],
 "system B": ["cancel the order for table fourteen",
              "confirm the order for table nine",
              "cancel the order for table twelve"],
}
NUM = {"four":4,"nine":9,"twelve":12,"fourteen":14}
print(f"{'system':<22}{'WER':>7}{'orders cancelled correctly':>28}")
for name, hs in hyps.items():
    w = np.mean([wer(r[0], h) for r, h in zip(ref_set, hs)])
    ok = 0
    for (rtext, verb, table), h in zip(ref_set, hs):
        toks = h.split()
        got_tab = next((NUM[t] for t in toks if t in NUM), None)
        ok += (toks[0] == verb and got_tab == table)
    print(f"{name:<22}{w:>7.3f}{f'{ok}/3':>28}")
print("\nidentical WER. One of them cancels the wrong table.")
```

```text
system                    WER  orders cancelled correctly
system A                0.056                         3/3
system B                0.056                         2/3

identical WER. One of them cancels the wrong table.
```

**Identical WER to three decimals. One system is correct on every order; the
other cancels a stranger's lunch.**

System A's error was `the` → `an`. System B's was `four` → `fourteen`. WER
cannot tell them apart; your customers can.

So the metric to report is **task accuracy on the thing the transcript feeds**:

```text
IF THE TRANSCRIPT FEEDS       MEASURE
an intent classifier          intent accuracy, end to end
a slot filler (table, amount) slot accuracy, per slot
a search index                retrieval quality on real queries
a human reading it            readability, and WER is a fair proxy
a compliance archive          WER, and keep the audio
```

Build a **small set of entity-weighted test utterances** — the table numbers,
the product names, the amounts, the words that trigger actions — and track
accuracy on those beside WER. Fifty such utterances will tell you more than a
thousand generic ones, and this is the same move
[Prompt-Engineering 06](../../Prompt-Engineering/lessons/06-the-iteration-loop.md)
makes with an eval set: the asset is the test data, not the model.

---

## Arabic, dialect, and code-switching

Four problems that are not edge cases in Egypt, and are badly served by
English-centric systems:

**Diacritics are usually absent.** Written Arabic drops short vowels, so the
same written form has several pronunciations and meanings. A WER computed
against undiacritised reference text is measuring something looser than it
looks.

**Morphology inflates WER.** Arabic attaches prepositions, articles and
pronouns to the word. One wrong affix makes the whole token wrong, so Arabic
WER is systematically higher than English WER **for the same quality of
recognition**. Comparing the two numbers directly is meaningless; consider
character error rate (CER) alongside it.

**Dialect is not Modern Standard Arabic.** Models trained mostly on MSA degrade
sharply on Egyptian, Gulf or Levantine speech. If your users speak Egyptian,
your evaluation set must be Egyptian — a model's published Arabic WER tells you
almost nothing about your deployment.

**Code-switching is normal.** Real speech mixes Arabic and English mid-sentence
("اعمل cancel للاوردر"). Systems with a single language id per utterance handle
this badly, and your test set must contain it in the proportion your users
produce it.

[NLP 12](../../NLP/lessons/12-arabic-nlp.md) covers the text side of these,
and [LLM 02](../../LLM-and-GenAI/lessons/02-tokens-and-cost.md) measured the
cost consequence: the same meaning costs **2.61x more tokens in Arabic**, which
applies to everything downstream of the transcript.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| WER as the product metric | Four different outcomes scored 0.167 |
| Comparing two systems on WER alone | Identical WER, 3/3 against 2/3 |
| Comparing Arabic WER to English WER | Morphology inflates it structurally |
| Evaluating on MSA for dialect users | A published WER says nothing about your users |
| No code-switched utterances in the test set | Your users produce them daily |
| A generic test set | 50 entity-weighted utterances beat 1,000 generic ones |
| Reporting WER without VAD recall | [Lesson 05](05-finding-the-speech.md): 66.2% never arrived |
| No domain vocabulary or biasing | Product names are exactly the words that matter |

---

## Exercises

1. Compute WER and CER for the six cases. Does CER rank them differently?
2. Build 50 entity-weighted test utterances for your domain.
3. Measure end-to-end task accuracy beside WER for one week of real traffic.
4. Collect ten code-switched utterances and transcribe them by hand.
5. Write the error that would be most expensive for your product, and check
   whether anything in your evaluation would catch it.

---

**Next:** [Lesson 07 — Generating Audio](07-generating-audio.md)
