# Lesson 02 — The Spectrogram

**Goal:** turn a waveform into the picture every audio model actually sees.

## What you will learn

- Why nobody models the raw waveform directly
- Framing, windowing, and the FFT
- The time/frequency trade-off, measured
- Choosing a window size on purpose

---

## Why not the waveform?

One second of 16 kHz audio is 16,000 numbers, and the thing you care about —
*is that speech or a coffee grinder* — is not visible in any one of them. It is
in **how the frequency content changes over time**.

So the standard move is:

```text
1. CUT the signal into short overlapping frames        ~25 ms each
2. WINDOW each frame                                   taper the edges
3. FFT each frame                                      frequencies in that frame
4. STACK the results                                   a 2-D image: time x frequency
```

That image is the spectrogram, and **it is the input to essentially every
audio model**, classical or neural.

The 25 ms frame length is not arbitrary. Speech changes meaningfully every
10-30 ms; inside one frame the signal is roughly stationary, which is the
assumption the FFT needs.

---

## The data

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

---

## Framing and the FFT

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

Three details in that code that matter more than they look:

**The Hann window (`np.hanning`).** Cutting a frame out of a signal creates
artificial discontinuities at both edges, and the FFT reports those as
frequencies that are not there (spectral leakage). Tapering the frame to zero
at the edges removes most of it. Never frame without a window.

**The hop (160 samples = 10 ms).** Frames overlap. With a 512-sample window
and a 160-sample hop, each sample appears in about three frames, so a sound
that starts mid-frame is still captured cleanly by the next one.

**`np.abs` throws away the phase.** The spectrogram keeps only magnitude. For
recognition that is fine and standard; for anything that must *reconstruct*
audio — enhancement, separation, generation — phase is exactly what you need
and discarding it is the hard problem.

---

## The trade-off

```python
print(f"{'window':>9}{'ms':>7}{'freq bins':>11}{'Hz per bin':>12}{'frames in 1s':>14}")
for n_fft in [64, 128, 256, 512, 1024, 2048]:
    print(f"{n_fft:>9}{1000*n_fft/SR:>7.1f}{n_fft//2+1:>11}{SR/n_fft:>12.1f}"
          f"{len(range(0, SR-n_fft+1, 160)):>14}")
print("\nyou cannot have fine time and fine frequency at once. Pick one.")
```

```text
   window     ms  freq bins  Hz per bin  frames in 1s
       64    4.0         33       250.0           100
      128    8.0         65       125.0           100
      256   16.0        129        62.5            99
      512   32.0        257        31.2            97
     1024   64.0        513        15.6            94
     2048  128.0       1025         7.8            88

you cannot have fine time and fine frequency at once. Pick one.
```

**A 4 ms window resolves time beautifully and cannot distinguish 250 Hz from
500 Hz. A 128 ms window resolves 7.8 Hz and smears everything inside an eighth
of a second into one column.**

This is not an engineering limitation to be overcome; it is a mathematical
fact about what a short signal can tell you. Choose by what you need to see:

| You need to detect | Window | Because |
|---|---|---|
| Speech phonemes | 20-32 ms (320-512) | The standard compromise |
| A drum hit, a click, an onset | 4-8 ms (64-128) | Time precision |
| A low hum, a musical note, a pitch | 64-128 ms (1024-2048) | Frequency precision |
| A machine fault at 57 Hz | 128 ms+ | 7.8 Hz bins can separate 57 from 64 |

**512 samples at 16 kHz is the default for speech** for exactly the reason in
that table, and `n_fft=512, hop=160` is the configuration you will see in
nearly every speech system.

---

## Log, always

One more step that is not optional:

```text
S  = |FFT|            linear magnitude    range spans 6 orders of magnitude
log(S + eps)          log magnitude       what you actually use
```

Hearing is roughly logarithmic — a sound ten times more powerful is perceived
as moderately louder, not ten times louder. A linear spectrogram is dominated
entirely by the loudest component, and every quiet-but-informative detail is
numerically invisible.

The `+ eps` is not a detail either: `log(0)` is `-inf`, and one silent frame
will propagate `nan` through your whole training run.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Framing without a window function | Spectral leakage invents frequencies |
| Non-overlapping frames | Events falling on a boundary are smeared |
| A window copied from a tutorial | 4 ms cannot see pitch; 128 ms cannot see onsets |
| Linear magnitude instead of log | The loudest component drowns everything |
| `log(S)` with no epsilon | One silent frame gives `nan` |
| Expecting to reconstruct audio from a spectrogram | You threw away the phase |
| Different `n_fft` in training and serving | [MLOps 07](../../MLOps/lessons/07-monitoring.md): the same features, computed twice |

---

## Exercises

1. Plot spectrograms of all four classes at `n_fft` 128 and 1024. Which window
   shows the bell's decay? Which shows the machine's harmonics?
2. Remove the Hann window and compare the spectrum of a pure tone.
3. Compute the spectrogram of a 2-second clip at hop 160 and hop 512. How many
   frames each, and what is lost?
4. Take the log out and plot. What can you still see?
5. For a task you care about, state the window size you need and why.

---

**Next:** [Lesson 03 — Features](03-features.md)
