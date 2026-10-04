# Lesson 04 — Classifying Sound

**Goal:** build an audio classifier that survives the room it is deployed in.

## What you will learn

- The train/test mismatch that halves accuracy
- Why training on noisy audio costs nothing
- Augmentation that is worth doing
- Evaluating an audio model honestly

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

## The mismatch

You record training audio in a quiet office. The system runs in a café with a
grinder, a till and forty people.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = dataset(60, dur=0.25, seed=0)
Xt, yt = dataset(40, dur=0.25, seed=99)        # a different recording session
SNRS = [20, 10, 0, -5]

def fit(train_snr, mixed=False):
    rng = np.random.default_rng(1)
    if mixed:
        Xa = np.stack([add_noise(x, rng.choice(SNRS), rng) for x in X])
    else:
        Xa = np.stack([add_noise(x, train_snr, rng) for x in X])
    F = np.stack([feats(x, "MFCC (13)") for x in Xa])
    sc = StandardScaler().fit(F)
    return sc, LogisticRegression(max_iter=3000).fit(sc.transform(F), y)

def score(sc, m, snr):
    rng = np.random.default_rng(2)
    Xa = np.stack([add_noise(x, snr, rng) for x in Xt])
    F = np.stack([feats(x, "MFCC (13)") for x in Xa])
    return m.score(sc.transform(F), yt)

print(f"{'trained at':<22}" + "".join(f"{f'test {s} dB':>13}" for s in SNRS))
for tr_snr in SNRS:
    sc, m = fit(tr_snr)
    print(f"{f'{tr_snr} dB SNR':<22}" + "".join(f"{score(sc,m,s):>13.4f}" for s in SNRS))
sc, m = fit(None, mixed=True)
print(f"{'mixed 20/10/0/-5 dB':<22}" + "".join(f"{score(sc,m,s):>13.4f}" for s in SNRS))
print("\nsame model, same features. Only the noise the training data had.")
```

```text
trained at               test 20 dB   test 10 dB    test 0 dB   test -5 dB
20 dB SNR                    1.0000       1.0000       0.5563       0.4938
10 dB SNR                    1.0000       1.0000       0.9938       0.6562
0 dB SNR                     1.0000       1.0000       1.0000       1.0000
-5 dB SNR                    1.0000       1.0000       1.0000       1.0000
mixed 20/10/0/-5 dB          1.0000       1.0000       1.0000       0.9938

same model, same features. Only the noise the training data had.
```

Read the first row. **A model trained on clean audio scores 1.0000 in the lab
and 0.4938 in a loud room** — barely better than guessing one of four classes.
Nothing about the model is wrong. It has simply never seen what it is now
being asked to classify.

Now read the first column. **Every model scores 1.0000 on clean audio,
including the ones trained at -5 dB.** Training on heavily degraded audio cost
nothing at all on clean test data.

That asymmetry is the whole lesson:

> **Training on noisy audio is close to free. Training on clean audio is
> expensive, and the bill arrives in production.**

The row trained at 0 dB is perfect everywhere. The mixed-SNR row is essentially
as good and is the safer default in reality, where you do not know the
deployment SNR in advance and a single training SNR is a bet.

Note also that the degradation is not gradual. 20 dB → 10 dB costs nothing;
10 dB → 0 dB costs 44 points. **There is a cliff, and offline evaluation on
clean audio cannot see where it is.** Sweep the test SNR and find yours.

---

## Augmentation worth doing

In rough order of value for speech and audio events:

```text
NOISE          mix in noise at a range of SNRs       the finding above
GAIN           scale the waveform up and down        microphone distance
TIME SHIFT     roll the clip by a random offset      events are not centred
SPEED          resample by 0.9-1.1x                  speaking rate, cheap and strong
REVERB         convolve with a room impulse response closest to a real new room
SPEC AUGMENT   mask random time and frequency bands  strong for neural models
```

Two rules about all of them:

**Augment the training set only.** Augmenting the test set measures robustness,
which is useful, but it is a separate experiment — and augmenting both at once
measures nothing.

**Augment with the distortions you will actually meet.** Reverb matters if the
device is across a room and not at all for a headset. Adding distortions your
deployment will never produce costs capacity for nothing.

---

## Evaluating honestly

The audio-specific traps, beyond the usual:

- [ ] **Split by recording session or speaker, never by clip.** Two clips from
      one recording share the microphone, the room and the background. A random
      clip split lets the model memorise the session and reports a number you
      will never see again. This is
      [Data-Science 03](../../Data-Science/lessons/03-the-data-you-have.md)'s
      leakage in its most common audio form.
- [ ] **Report per-class accuracy.** A four-class aggregate can hide one class
      at zero ([MLOps 04](../../MLOps/lessons/04-ci-gate.md)).
- [ ] **Sweep the test SNR.** One number at one SNR tells you nothing about
      the cliff.
- [ ] **Test on a different session**, as the code above does with
      `seed=99` — same generator, different draws.
- [ ] **Check clip length.** If one class is systematically longer, a model can
      score well by learning duration.
- [ ] **Listen to twenty errors.** Audio is the one domain where the errors are
      immediately interpretable by a human, and almost nobody does it.

The last is the highest-value hour in any audio project.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Training only on clean audio | 1.0000 in the lab, 0.4938 in the room |
| Reporting accuracy at one SNR | The cliff was between 10 dB and 0 dB |
| Splitting by clip, not by session | The model memorises the microphone |
| Augmenting the test set too | Now you are measuring nothing |
| Augmenting with distortions you will never meet | Spent capacity, no benefit |
| Classes with different durations | Duration becomes the feature |
| Never listening to the errors | The cheapest diagnostic in the field |

---

## Exercises

1. Reproduce the SNR grid on your own data. Where is your cliff?
2. Add reverb augmentation and re-measure at each SNR.
3. Split by session and by clip, and report both numbers. How big is the gap?
4. Report per-class accuracy for the worst row of the grid. Which class dies
   first?
5. Listen to twenty errors and group them. What is the biggest group?

---

**Next:** [Lesson 05 — Finding the Speech](05-finding-the-speech.md)
