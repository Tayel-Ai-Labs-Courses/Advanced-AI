# Lesson 01 — Sound as Numbers

**Goal:** understand what an audio file is, and the one mistake that silently
corrupts it.

## What you will learn

- Sample rate, bit depth, and what an hour of audio costs
- Nyquist, and why 16 kHz is the speech standard
- Aliasing, measured — a 7,800 Hz tone that becomes 200 Hz
- The five things to check about any audio you are given

---

## An audio file is a list of numbers

A microphone measures air pressure. An audio file is that measurement, taken
many thousands of times a second, stored as integers.

```text
sample rate   how many measurements per second        16,000 Hz
bit depth     how precisely each one is stored        16 bits
channels      how many microphones                    1 (mono) or 2
duration      samples / sample rate                   16,000 samples = 1 s
```

That is the whole format. Everything else — MP3, WAV, Opus — is a way of
storing or compressing that list.

```python
import numpy as np

print(f"{'sample rate':>12}{'use':>26}{'samples/s':>11}{'MB per hour (16-bit)':>22}")
for sr, use in [(8000,"phone call"),(16000,"speech recognition"),
                (22050,"old web audio"),(44100,"CD / music"),(48000,"video")]:
    print(f"{sr:>12,}{use:>26}{sr:>11,}{sr*2*3600/1e6:>22.1f}")
```

```text
 sample rate                       use  samples/s  MB per hour (16-bit)
       8,000                phone call      8,000                  57.6
      16,000        speech recognition     16,000                 115.2
      22,050             old web audio     22,050                 158.8
      44,100                CD / music     44,100                 317.5
      48,000                     video     48,000                 345.6
```

**An hour of 16 kHz speech is 115 MB raw.** A call centre taking a thousand
hours a month is storing 115 GB a month of uncompressed audio, which is a data
engineering problem before it is a machine learning one
([Data-Engineering 03](../../Data-Engineering/lessons/03-storage-and-formats.md)).

Note that speech recognition settled on **16 kHz, not CD quality**. Doubling to
44.1 kHz would triple your storage and compute and add almost nothing, because
of the rule in the next section.

---

## Nyquist

> **A sample rate of `sr` can represent frequencies up to `sr/2`, and nothing
> above it.**

That half-the-sample-rate limit is the Nyquist frequency. At 16 kHz you get
8 kHz of bandwidth, and human speech carries nearly all of its information
below 8 kHz — which is exactly why 16 kHz is the standard.

The part people skip is what happens to the frequencies above the limit. They
do not vanish.

---

## Aliasing

```python
DUR = 0.5
print(f"{'true tone':>11}{'sampled at':>12}{'Nyquist':>9}{'what you measure':>19}")
for f_true, sr in [(400, 8000), (3000, 8000), (3000, 4000), (5000, 8000),
                   (7800, 8000), (440, 1000)]:
    t = np.arange(int(sr*DUR)) / sr
    x = np.sin(2*np.pi*f_true*t)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    measured = freqs[np.argmax(spec)]
    flag = "" if abs(measured - f_true) < 5 else "   <-- wrong"
    print(f"{f_true:>9} Hz{sr:>10} Hz{sr//2:>7} Hz{measured:>14.0f} Hz{flag}")
print("\nabove Nyquist the frequency does not disappear. It comes back as a lie.")
```

```text
  true tone  sampled at  Nyquist   what you measure
      400 Hz      8000 Hz   4000 Hz           400 Hz
     3000 Hz      8000 Hz   4000 Hz          3000 Hz
     3000 Hz      4000 Hz   2000 Hz          1000 Hz   <-- wrong
     5000 Hz      8000 Hz   4000 Hz          3000 Hz   <-- wrong
     7800 Hz      8000 Hz   4000 Hz           200 Hz   <-- wrong
      440 Hz      1000 Hz    500 Hz           440 Hz

above Nyquist the frequency does not disappear. It comes back as a lie.
```

**A 7,800 Hz tone sampled at 8 kHz measures as 200 Hz.** Not as noise, not as
silence — as a clean, confident, completely wrong low tone.

Three things follow, and the third is the one that bites in production.

**The error is silent.** There is no exception, no warning, no obvious
artefact in the array. The data looks fine and is wrong.

**It folds, it does not clip.** A frequency `f` above Nyquist `N` comes back at
`|f - 2N|`: 5,000 Hz at 8 kHz folds to 3,000; 7,800 folds to 200. The further
above the limit, the lower the lie.

**It happens when you downsample.** Taking 44.1 kHz audio to 16 kHz by keeping
every third sample aliases everything between 8 kHz and 22 kHz down into your
speech band. **You must low-pass filter before you downsample** — every proper
resampling function does this, and `x[::3]` does not.

```python
# no-run
from scipy.signal import resample_poly
y = resample_poly(x, 16000, 44100)     # correct: filters, then decimates
y = x[::3]                             # WRONG: aliases, silently
```

---

## What to check about any audio you are given

- [ ] **Sample rate**, and whether every file has the same one
- [ ] **How it was resampled**, if it was. `x[::n]` is a bug
- [ ] **Clipping**: what fraction of samples are at the maximum value?
- [ ] **Silence**: is the file empty, or near-empty?
- [ ] **Channels**: mono or stereo, and if stereo, is one channel dead?
- [ ] **DC offset**: does the mean differ from zero?

```python
# no-run
def audit(x, sr):
    return {
        "seconds": len(x) / sr,
        "clipped": float((np.abs(x) >= 0.999).mean()),
        "rms": float(np.sqrt((x ** 2).mean())),
        "dc_offset": float(x.mean()),
        "silent": bool(np.sqrt((x ** 2).mean()) < 1e-4),
    }
```

This is the audio version of
[Data-Engineering 07](../../Data-Engineering/lessons/07-data-quality.md)'s
validation rules, and it belongs in the same place: at ingestion, before
anything trains on it.

**Mixed sample rates in one dataset is the single most common bug** in audio
projects. Half the files at 8 kHz and half at 16 kHz means your features mean
different things for different rows, and your model learns the recording
device rather than the content.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Downsampling with `x[::n]` | 7,800 Hz becomes 200 Hz, silently |
| Mixing sample rates in one dataset | The model learns the device, not the content |
| Recording speech at 44.1 kHz | 3x the storage and compute for nothing |
| Not checking for clipping | Clipped audio has harmonics that were never spoken |
| Ignoring DC offset | It wastes dynamic range and shifts every energy measure |
| Assuming stereo means two useful channels | One is often dead or identical |
| Treating audio as "just an array" | It has a sample rate, and that is half its meaning |

---

## Exercises

1. Take a 44.1 kHz file. Downsample it both ways and compare the spectra.
2. Generate a 9 kHz tone at 16 kHz. Where does it appear?
3. Write the `audit` function and run it over a folder. What did it find?
4. Compute the storage for your own expected hourly volume, for a year.
5. Find the sample rates in a dataset you have. Are they all the same?

---

**Next:** [Lesson 02 — The Spectrogram](02-the-spectrogram.md)
