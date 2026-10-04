# Lesson 08 — Speech in Production

**Goal:** ship a speech system that is fast enough, cheap enough and legal.

## What you will learn

- Streaming latency, computed — where the time actually goes
- The cost of an hour of audio, four ways
- Why the VAD is the cheapest optimisation you have
- Audio is biometric data, and what that changes

---

## Streaming latency

A batch system waits for the recording to end. A streaming one must answer
while the person is still talking, and the arithmetic is unforgiving.

```python
RTF = 0.25          # real-time factor: 0.25 s of compute per 1 s of audio
NET = 60            # ms round trip

print(f"{'chunk':>8}{'lookahead':>11}{'compute':>10}{'network':>10}{'latency to text':>18}")
for chunk_ms in [100, 250, 500, 1000, 2000, 5000]:
    look = chunk_ms * 0.5
    compute = (chunk_ms + look) * RTF
    total = chunk_ms + look + compute + NET
    print(f"{chunk_ms:>6} ms{look:>9.0f} ms{compute:>8.0f} ms{NET:>8} ms{total:>15.0f} ms")
print("\nthe chunk itself must be recorded before anything can process it.")
```

```text
   chunk  lookahead   compute   network   latency to text
   100 ms       50 ms      38 ms      60 ms            248 ms
   250 ms      125 ms      94 ms      60 ms            529 ms
   500 ms      250 ms     188 ms      60 ms            998 ms
  1000 ms      500 ms     375 ms      60 ms           1935 ms
  2000 ms     1000 ms     750 ms      60 ms           3810 ms
  5000 ms     2500 ms    1875 ms      60 ms           9435 ms

the chunk itself must be recorded before anything can process it.
```

**The dominant term is not the model. It is the chunk.** At a 1-second chunk
you have already spent 1,000 ms before a single multiply happens, and the
compute is 375 ms of a 1,935 ms total.

Three consequences:

**Optimising the model is the wrong first move.** Halving RTF from 0.25 to
0.125 takes the 1-second row from 1,935 ms to 1,748 ms — a 10% improvement.
Halving the chunk takes it to 998 ms, a 48% improvement, for free. This is the
same shape as [MLOps 06](../../MLOps/lessons/06-serving.md)'s finding that a
40 ms model answers in 598 ms because the queue, not the model, is the latency.

**Lookahead is a real cost.** A streaming recogniser needs some future context
to decide what it just heard — the row assumes half a chunk. More lookahead is
more accurate and slower, and it is a parameter, not a constant.

**Below ~250 ms the user perceives it as instant; above ~1 s they notice.**
Pick the largest chunk that stays under your target, because bigger chunks are
more accurate and cheaper per second.

The usual production answer is **partial results**: emit a provisional
transcript every 100-250 ms and correct it as more audio arrives. The user sees
words appearing immediately, and the final text is the accurate one. It costs
more compute and is almost always worth it.

---

## What an hour of audio costs

```python
print(f"{'option':<34}{'EGP/hour':>10}{'1,000 h/month':>16}")
for name, egp in [("hosted ASR API, typical rate", 18.0),
                  ("self-hosted GPU, 24/7 at 60% use", 7.5),
                  ("self-hosted CPU, small model", 2.2),
                  ("on-device (phone), streaming", 0.0)]:
    print(f"{name:<34}{egp:>10.2f}{egp*1000:>15,.0f}")
print("\nplus storage: 115 MB/hour raw (lesson 01), and retention is a policy, not a default.")
```

```text
option                              EGP/hour   1,000 h/month
hosted ASR API, typical rate           18.00         18,000
self-hosted GPU, 24/7 at 60% use        7.50          7,500
self-hosted CPU, small model            2.20          2,200
on-device (phone), streaming            0.00              0

plus storage: 115 MB/hour raw (lesson 01), and retention is a policy, not a default.
```

Those rates are illustrative — **put your own in** — but the ordering is
stable and so is the lesson: the spread between the cheapest and the most
expensive option is more than an order of magnitude, and the choice is
usually made by whoever wrote the first prototype.

Read it with [HPC 05](../../HPC-and-Cloud/lessons/05-renting-hardware.md)'s
rule: a self-hosted GPU only beats an API above a volume threshold, because
it costs the same at 3 a.m. as at noon. Compute your crossover before
migrating.

On-device is the interesting row. It is free, it is private, it has no network
latency, and the model is smaller and less accurate —
[Optimization 15](../../Optimization/lessons/15-edge-and-on-device.md) is how
to decide whether that accuracy is acceptable.

---

## The cheapest optimisation

```python
for speech_frac in [0.9, 0.6, 0.35, 0.2]:
    print(f"  speech is {speech_frac:>4.0%} of the audio -> "
          f"{18.0*speech_frac:>5.2f} EGP/hour instead of 18.00  "
          f"({(1-speech_frac)*18*1000:>7,.0f} EGP/month saved at 1,000 h)")
```

```text
  speech is  90% of the audio -> 16.20 EGP/hour instead of 18.00  (  1,800 EGP/month saved at 1,000 h)
  speech is  60% of the audio -> 10.80 EGP/hour instead of 18.00  (  7,200 EGP/month saved at 1,000 h)
  speech is  35% of the audio ->  6.30 EGP/hour instead of 18.00  ( 11,700 EGP/month saved at 1,000 h)
  speech is  20% of the audio ->  3.60 EGP/hour instead of 18.00  ( 14,400 EGP/month saved at 1,000 h)
```

**A meeting recording that is 35% speech costs 6.30 EGP/hour instead of 18.00
if you run the VAD first** — 11,700 EGP a month at a thousand hours, from the
ten-line function in [lesson 05](05-finding-the-speech.md).

But hold that beside lesson 05's other number: at -5 dB the same VAD **drops
66.2% of the speech**. The saving and the damage come from the same dial.
Measure VAD recall on your own audio before you bank the saving, and set the
threshold low ([lesson 05](05-finding-the-speech.md): precision barely moved,
recall collapsed).

---

## Audio is biometric data

This is the part that is a legal question rather than an engineering one, and
it is the part that ends projects.

```text
A VOICE IDENTIFIES A PERSON. Under GDPR and most comparable regimes, voice
used for identification is a SPECIAL CATEGORY of personal data, with
stricter rules than a name or an email address.
```

What that means in practice:

| Requirement | What it looks like |
|---|---|
| **Consent** | Explicit, specific, before recording — not buried in terms |
| **Notice** | Everyone on the call, not only your user. The other party has rights too |
| **Purpose limitation** | Recorded for support quality ≠ may be used to train a model |
| **Retention** | A policy with a number and an automated deletion job |
| **Minimisation** | Keep the transcript, delete the audio, if the audio is not needed |
| **Access and erasure** | You must be able to find and delete one person's audio |
| **Cross-border** | Sending audio to an API may move it to another jurisdiction |

Three engineering decisions follow directly:

**Default to deleting the audio and keeping the transcript.** Most products
need the text. The audio is the liability, it is 115 MB/hour
([lesson 01](01-sound-as-numbers.md)), and deleting it removes a whole class
of risk.

**Redact in the transcript, not later.** Card numbers, national IDs and
addresses spoken aloud end up in your logs and your training data otherwise —
and [Data-Security-for-AI 03](../../Data-Security-for-AI/lessons/03-memorisation.md)
measured how readily a model leaks its training data back out.

**"We will use recordings to improve our service" is not consent to train a
model on them**, and a speaker embedding derived from that audio is itself
biometric data.

Treat all of this as [MLOps 08](../../MLOps/lessons/08-team-practices.md)'s
model card plus a retention policy with an owner's name on it.

---

## The production checklist

- [ ] Sample rate validated and normalised at ingestion ([lesson 01](01-sound-as-numbers.md))
- [ ] Resampling is filtered, never `x[::n]`
- [ ] VAD runs first, threshold set low, **recall measured**
- [ ] Feature code is **one function**, imported by training and serving
- [ ] Chunk size chosen from the latency table, with partial results
- [ ] WER **and** end-to-end task accuracy tracked ([lesson 06](06-speech-recognition.md))
- [ ] Per-accent and per-dialect accuracy reported separately
- [ ] Audio retention policy, with an automated deletion job
- [ ] Consent recorded, and purpose limited
- [ ] Cost per hour measured, including storage
- [ ] A fallback when the recogniser is down ([MLOps 06](../../MLOps/lessons/06-serving.md))

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Optimising the model for latency first | The chunk is 1,000 ms of a 1,935 ms total |
| No partial results | The user stares at nothing for a second |
| No VAD | Up to 14,400 EGP/month at 1,000 hours |
| Banking the VAD saving without measuring recall | The same dial drops 66.2% at -5 dB |
| Keeping raw audio by default | 115 MB/hour of special-category data |
| "To improve our service" treated as training consent | It is not |
| Not redacting spoken card numbers | They are in your logs and your training set |
| One accuracy number across all accents | The worst dialect is invisible |
| Choosing hosted vs self-hosted by habit | A 10x spread, decided by the prototype |

---

## Where to go next

| Next | Why |
|---|---|
| [Project 25](../Project-25/) | Build one, end to end, including the bill |
| [NLP 12](../../NLP/lessons/12-arabic-nlp.md) | The text side of Arabic and dialect |
| [MLOps](../../MLOps/) | Serving, monitoring, the gate and the rollback |
| [Optimization 15](../../Optimization/lessons/15-edge-and-on-device.md) | Getting the model onto the phone |
| [Data-Security-for-AI](../../Data-Security-for-AI/) | What leaks, and how to measure it |
| [AI-Governance](../../AI-Governance/) | Consent, retention and the obligations in writing |
