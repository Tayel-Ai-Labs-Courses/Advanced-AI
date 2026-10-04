# Project 25 — One Audio System

Build one working audio system end to end: ingestion, VAD, features, model,
evaluation, cost and a retention policy.

Pick **one** task:

| Task | Good when |
|---|---|
| **Sound event detection** | You have recordings with events worth spotting |
| **Spoken command recognition** | A small fixed vocabulary, e.g. 10 commands |
| **Transcription + an action** | You have a recogniser and need the task accuracy |
| **Speaker or language id** | You have labelled recordings from several sources |

Record your own audio if you can — twenty minutes on a phone is enough, and it
will teach you more about sample rates and clipping than any dataset. If you
cannot, generate it with [lesson 02](../lessons/02-the-spectrogram.md)'s
`clip()` and change the parameters so your numbers are not these numbers.

---

## What you deliver

```text
audio/
  ingest.py          sample-rate validation, resampling, the audit function
  vad.py             with its threshold chosen from a measured sweep
  features.py        ONE function, imported by training and serving
  train.py
  evaluate.py        per-class, per-SNR, per-speaker, and the VAD recall
  serve.py           streaming or batch, with a fallback
  RETENTION.md       what you keep, for how long, and who deletes it
  REPORT.md          the numbers below
```

---

## The eight requirements

| # | Requirement | The check | Lesson |
|---|---|---|---|
| 1 | **Ingestion audit** — sample rate, clipping, silence, DC, channels | Run it over every file; report what it found | 01 |
| 2 | **No `x[::n]`** anywhere | Resampling is filtered | 01 |
| 3 | **A window size you justify** | One sentence saying what you need to resolve | 02 |
| 4 | **Three feature sets compared**, including crude waveform statistics | A table with dims and accuracy | 03 |
| 5 | **An SNR grid**: train at 3 levels, test at 4 | Find your cliff | 04 |
| 6 | **A VAD threshold sweep**, with **recall reported** | Precision, recall, F1, and the audio kept | 05 |
| 7 | **Task accuracy, not just the model metric** | What the output actually feeds | 06 |
| 8 | **Cost per hour and a retention policy with a name on it** | EGP/hour including storage; a deletion job | 08 |

Requirement 6 is the one that separates this from a notebook. **Report what
your VAD throws away**, and report your headline accuracy beside it — a model
that is 95% accurate on the 60% of audio it saw is not a 95% system.

---

## The report

```text
1. THE AUDIO        hours, sample rates found, clipped files, silent files,
                    what the audit rejected
2. FEATURES         three sets, dims, accuracy. Did the crude one get close?
3. THE MODEL        accuracy, and per-class accuracy. Which class is worst?
4. NOISE            the SNR grid. Where is the cliff?
5. VAD              the threshold sweep, your chosen threshold, and the
                    recall you are accepting
6. TASK             end-to-end accuracy on what the output feeds, beside
                    the model metric
7. LATENCY          your chunk size, and the four terms of the total
8. COST             EGP per hour including storage, at your volume, with
                    and without the VAD
9. GOVERNANCE       consent, retention period, deletion job, who owns it
10. VERDICT         ship it, or not, and what it is worth per month
```

Section 9 is not paperwork. Voice is special-category personal data; a system
with no retention policy is a system that cannot be deployed —
[AI-Governance](../../AI-Governance/) is the long version.

---

## Rules

- **Every number is from your own run.** None copied from these lessons.
- **Report accuracy and VAD recall together, always.**
- **Per-class, per-speaker, per-accent** — never one aggregate.
- **A contradicting result is better than an agreeing one.** If training on
  clean audio costs you nothing on your data, show the grid and say why.
- **Listen to twenty errors.** Write what you heard. This is a requirement,
  not a suggestion.

---

## Scoring yourself

| | |
|---|---|
| **Not done** | A model with an accuracy number |
| **Done** | Eight requirements, ten report sections |
| **Done well** | A stage you **fixed upstream** of the model — resampling, VAD threshold, augmentation — with the before and after, plus an honest statement of what your system cannot hear |

The third row is the course. The model is the easy part.

---

## Prerequisites

All eight [Speech-and-Audio lessons](../lessons/), and twenty minutes of audio.
