# Lesson 03 — Features

**Goal:** compress a spectrogram to something a model can learn from, and find
out how little you need.

## What you will learn

- The mel scale, and why it is not an arbitrary choice
- MFCCs in four lines
- 257 numbers against 26, measured
- When the compression actually pays — and it is not where you expect

---

## The mel scale

A 512-point spectrogram gives 257 frequency bins, evenly spaced 31.2 Hz apart.
Hearing does not work that way: the difference between 200 and 400 Hz is
obvious, the difference between 7,000 and 7,200 Hz is inaudible.

The **mel scale** spaces bins the way hearing does — narrow and many at low
frequencies, wide and few at high ones:

```text
257 linear bins  →  [mel filterbank]  →  40 mel bands  →  [DCT]  →  13 MFCCs
   31.2 Hz each      triangular,            perceptual      decorrelate,
                     log-spaced             resolution      keep the shape
```

The DCT at the end deserves a word. Adjacent mel bands are highly correlated —
energy at 500 Hz implies energy at 520 Hz — and the DCT turns those correlated
bands into mostly-uncorrelated coefficients, with the useful information
concentrated in the first dozen. Coefficient 0 is overall loudness; low
coefficients describe the broad shape of the spectrum; high ones are fine
detail that is usually noise.

**Keeping 13 is convention, not law.** Sweep it.

---

---

## Setup

```python
import numpy as np
SR = 16000

def clip(kind, rng, dur=1.0):
    """Four sounds you might hear in a cafe, synthesised from scratch."""
    n = int(SR * dur); t = np.arange(n) / SR
    if kind == "speech":                       # pitch + three formants + syllables
        f0 = rng.uniform(90, 200)
        x = sum(np.sin(2*np.pi*f0*h*t)/h for h in range(1, 12))
        for f, bw in [(rng.uniform(500,800),100), (rng.uniform(1200,1800),150),
                      (rng.uniform(2300,3000),200)]:
            x += 0.6*np.sin(2*np.pi*f*t) * np.exp(-((t % 0.25)-0.1)**2/(2*(bw/8000)**2))
        x *= 0.5 + 0.5*np.sin(2*np.pi*rng.uniform(3,6)*t)
    elif kind == "machine":                    # grinder: broadband + harmonics
        x = rng.normal(size=n)
        b = rng.uniform(55, 70)
        x = x*0.6 + sum(np.sin(2*np.pi*b*h*t) for h in range(1, 6))
    elif kind == "bell":                       # door chime: two decaying tones
        f = rng.uniform(900, 1400)
        x = (np.sin(2*np.pi*f*t) + 0.7*np.sin(2*np.pi*f*1.5*t)) * np.exp(-4*t)
    elif kind == "room":                       # room tone: low-pass noise
        x = np.convolve(rng.normal(size=n), np.ones(80)/80, mode="same")
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)

CLASSES = ["speech", "machine", "bell", "room"]

def dataset(n_per=120, seed=0, dur=1.0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per):
            X.append(clip(c, rng, dur)); y.append(ci)
    return np.array(X), np.array(y)

def add_noise(x, snr_db, rng):
    """Mix in white noise at a given signal-to-noise ratio."""
    noise = rng.normal(size=x.shape)
    ps = (x**2).mean(); pn = (noise**2).mean()
    return (x + np.sqrt(ps / (pn * 10**(snr_db/10))) * noise).astype(np.float32)
```

```python
def frames(x, n_fft, hop):
    idx = np.arange(0, len(x)-n_fft+1, hop)
    return np.stack([x[i:i+n_fft] for i in idx]) * np.hanning(n_fft)

def spectrogram(x, n_fft=512, hop=160):
    return np.abs(np.fft.rfft(frames(x, n_fft, hop), axis=1))

def mel_bank(n_mels, n_fft, sr=SR):
    def hz2mel(f): return 2595*np.log10(1+f/700)
    def mel2hz(m): return 700*(10**(m/2595)-1)
    pts = mel2hz(np.linspace(hz2mel(50), hz2mel(sr/2), n_mels+2))
    bins = np.floor((n_fft+1)*pts/sr).astype(int)
    B = np.zeros((n_mels, n_fft//2+1))
    for m in range(n_mels):
        l, c, r = bins[m], bins[m+1], bins[m+2]
        for k in range(l, c): B[m, k] = (k-l)/max(c-l, 1)
        for k in range(c, r): B[m, k] = (r-k)/max(r-c, 1)
    return B

def dct(X, n):
    M = X.shape[1]
    D = np.cos(np.pi/M*(np.arange(M)+0.5)[None, :]*np.arange(n)[:, None])
    return X @ D.T

MB40 = mel_bank(40, 512)

def feats(x, kind):
    S = spectrogram(x)
    if kind == "raw waveform stats":
        return np.array([x.mean(), x.std(), np.abs(x).max(),
                         np.mean(np.abs(np.diff(np.sign(x)))>0)])
    if kind == "spectrum (257 bins)":
        return np.log(S.mean(0)+1e-9)
    M = np.log(S @ MB40.T + 1e-9)
    if kind == "mel (40)":
        return M.mean(0)
    if kind == "MFCC (13)":
        return np.concatenate([dct(M, 13).mean(0), dct(M, 13).std(0)])
```

---

## How much do you actually need?

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = dataset(120, dur=0.25)
rngn = np.random.default_rng(7)
X = np.stack([add_noise(x, 0, rngn) for x in X])      # 0 dB: noise as loud as signal

print(f"{'features':<24}{'dims':>6}{'accuracy':>10}")
for kind in ["raw waveform stats", "spectrum (257 bins)", "mel (40)", "MFCC (13)"]:
    F = np.stack([feats(x, kind) for x in X])
    Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    print(f"{kind:<24}{F.shape[1]:>6}{m.score(sc.transform(Xte), yte):>10.4f}")
print(f"\n{len(X)} clips of 0.25s at 0 dB SNR, 4 classes, logistic regression, 30% held out.")
```

```text
features                  dims  accuracy
raw waveform stats           4    0.8056
spectrum (257 bins)        257    1.0000
mel (40)                    40    1.0000
MFCC (13)                   26    1.0000

480 clips of 0.25s at 0 dB SNR, 4 classes, logistic regression, 30% held out.
```

**26 numbers match 257.** A 10x compression with no measurable loss, on audio
where the noise is as loud as the signal.

Two honest readings of that table, and you need both:

**The compression is free.** This is the real argument for MFCCs: not that
they are more accurate, but that they are **as accurate at a tenth the size**,
which makes everything downstream cheaper, faster and easier to regularise.

**This task is too easy to separate the top three.** Three rows tie at 1.0000,
so the table cannot rank them. A result at the ceiling is not a result — it is
a sign the experiment needs to be harder, and the honest move is to make it
harder rather than to report the tie as a finding.

Even the four crude waveform statistics reach 0.8056 — four numbers per clip,
one line of code, and 81% of a four-class problem at 0 dB. **Compute the stupid
features first.** They are not competitive here, but they are the baseline that
tells you the other 253 numbers bought you 19 points, and
[Time-Series 01](../../Time-Series-and-Forecasting/lessons/01-baselines.md)
keeps finding that a baseline nobody measured was closer than anyone expected.

---

## Where the compression actually pays

Make the experiment harder by taking away data, which is the condition you are
actually in:

```python
print(f"{'clips per class':>16}" + "".join(f"{k:>22}" for k in
      ["spectrum (257)", "mel (40)", "MFCC (26)"]))
for n_per in [6, 12, 25, 60, 120]:
    X, y = dataset(n_per, dur=0.25)
    rngn = np.random.default_rng(7)
    X = np.stack([add_noise(x, 0, rngn) for x in X])
    row = f"{n_per:>16}"
    for kind in ["spectrum (257 bins)", "mel (40)", "MFCC (13)"]:
        F = np.stack([feats(x, kind) for x in X])
        Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=0.4,
                                              random_state=0, stratify=y)
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
        row += f"{m.score(sc.transform(Xte), yte):>22.4f}"
    print(row)
print("\n4 classes, 0 dB SNR, 0.25s clips, 40% held out.")
```

```text
 clips per class        spectrum (257)              mel (40)             MFCC (26)
               6                0.4000                0.7000                0.8000
              12                1.0000                1.0000                1.0000
              25                1.0000                0.9750                1.0000
              60                1.0000                1.0000                1.0000
             120                1.0000                1.0000                1.0000

4 classes, 0 dB SNR, 0.25s clips, 40% held out.
```

**At six clips per class the raw spectrum scores 0.40 and MFCCs score 0.80.**
Twice the accuracy from throwing 90% of the numbers away.

By twelve clips per class everything ties, and stays tied.

So the rule is not "MFCCs are better". It is:

> **Feature compression buys you accuracy exactly when you are short of
> labelled data, and nothing when you are not.**

With 257 dimensions and 14 training examples, the model has more parameters
than data and fits noise — the plain dimensionality argument from
[Foundations 03](../../Foundations/lessons/03-decomposition.md), appearing in
audio form. With thousands of hours, this is why end-to-end neural systems
learn their own features from the raw spectrogram and beat MFCCs: they have
enough data for the extra dimensions to be information rather than noise.

**Which regime are you in?** Count your labelled minutes. Under a few hours,
use MFCCs and a simple classifier. Over a few hundred, consider learning the
features.

---

## Features in practice

```text
ALWAYS       log-mel spectrogram            the standard input
             per-utterance mean normalisation (CMN)
                                            removes the channel and the mic
OFTEN        deltas and delta-deltas        how features change over time
             MFCC mean AND std per clip     the std carries the dynamics
SOMETIMES    pitch, voicing, energy         for prosody or emotion
RARELY       the raw waveform               only with a lot of data
```

**Cepstral mean normalisation is the cheapest robustness you will ever buy.**
Subtracting each recording's own mean feature vector removes the constant
colouration of the microphone and the room — a convolution in time is an
addition in the log-spectral domain, so a constant channel effect is a constant
offset, and subtracting the mean deletes it. Two lines, and it is most of the
difference between a model that transfers to a new device and one that does
not.

Note what `feats` does for MFCCs above: it keeps **mean and standard
deviation** across frames. The mean says what the sound is; the std says how
much it moves, which is what separates steady machine noise from syllabic
speech. A mean-only feature vector throws that away.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Starting with learned features on 20 minutes of audio | 0.40 against 0.80 at six clips per class |
| Never trying the four crude statistics | They got 0.8056 here, in one line |
| Reporting a table that ties at 1.0000 | Not a result; make the task harder |
| Keeping 13 coefficients without checking | It is a convention, not a law |
| Skipping cepstral mean normalisation | The model learns the microphone |
| Mean-only features | The std is what distinguishes steady from syllabic |
| Different feature code in training and serving | [MLOps 07](../../MLOps/lessons/07-monitoring.md): one function, imported twice |

---

## Exercises

1. Sweep the number of MFCCs from 4 to 40. Where does accuracy stop improving?
2. Add deltas. Does anything change?
3. Implement cepstral mean normalisation, then add a fixed filter to the test
   audio to simulate a different microphone. Measure with and without CMN.
4. Make the task harder until the three feature sets stop tying.
5. Count the labelled minutes you actually have. Which regime are you in?

---

**Next:** [Lesson 04 — Classifying Sound](04-classifying-sound.md)
