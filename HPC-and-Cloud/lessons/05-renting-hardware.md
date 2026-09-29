# Lesson 05 — Renting Hardware

**Goal:** choose an instance without overpaying by a factor of ten.

## What you will learn

- What GPUs cost, and in Egyptian pounds
- Spot instances, and what they really cost
- Choosing a provider
- The checklist before you press start

---

## The price list

```python
PRICES = {"T4": 0.35, "A10G": 1.00, "A100-40": 3.00, "A100-80": 4.00, "H100": 8.00}
USD_EGP = 48
print(f"{'gpu':<10}{'$/hr':>7}{'8h run':>10}{'8h in EGP':>12}{'spot (~30%)':>13}")
for g, p in PRICES.items():
    print(f"{g:<10}{p:>7.2f}{p*8:>10.2f}{p*8*USD_EGP:>12,.0f}{p*8*0.3*USD_EGP:>13,.0f}")
```

```text
gpu          $/hr    8h run   8h in EGP  spot (~30%)
T4           0.35      2.80         134           40
A10G         1.00      8.00         384          115
A100-40      3.00     24.00       1,152          346
A100-80      4.00     32.00       1,536          461
H100         8.00     64.00       3,072          922
```

Indicative on-demand prices; yours will differ by provider and region, and the
**ratios** are the stable part.

**An 8-hour run is 134 EGP on a T4 and 3,072 on an H100** — a 23x spread for the
same wall-clock time. So lesson 01's question ("does LoRA fit on a T4?") is
worth more than any optimisation in this course.

Spot pricing is roughly 30% of on-demand, which turns that H100 run into 922
EGP. What it costs in reliability is the next section.

---

## Choosing a GPU

```mermaid
flowchart TD
    Adoes it fit after<br/>lesson 03's list? -->|"yes, on 16GB"| T["<b>T4 / L4</b><br/>cheapest per hour"]
    A -->|"yes, on 24GB"| B["<b>A10G / L4</b>"]
    A -->|"needs 40-80GB"| C["<b>A100</b>"]
    A -->|"no single GPU"| D["multi-GPU, lesson 04<br/>fill one machine first"]
    T --> Eis it fast enough?
    E -->|no| F["measure throughput per EGP,<br/>not per hour"]
```

The box that matters is the last one. **Compare throughput per unit of money,
not price per hour.** An H100 at 8x the price of a T4 is often 15-20x the
throughput on a large transformer, which makes it *cheaper* per sample — and on
a small model that does not saturate it, it is 8x the price for 2x the speed.

The only way to know is to run 100 steps on each. That measurement costs about
a dollar and routinely saves hundreds.

---

## Spot instances

The provider can take the machine back with a couple of minutes' notice. In
exchange, it costs about a third.

| | On-demand | Spot |
|---|---|---|
| Price | 1x | **~0.3x** |
| Interrupted | no | yes, with warning |
| Good for | Short runs, deadlines, inference | **Training with checkpoints** |
| Bad for | — | Anything that cannot resume |

**Spot is almost always right for training and almost always wrong for serving.**
Training is restartable if you checkpoint; a scoring API that vanishes is an
outage.

The requirement it imposes is the subject of lesson 06, and it is not optional:
**a spot instance with no checkpointing is a machine that deletes your work at
random.**

---

## Providers

| Kind | Examples | Good | Watch out |
|---|---|---|---|
| **Hyperscalers** | AWS, GCP, Azure | Everything integrates; regions everywhere | Most expensive; quota requests; egress fees |
| **GPU specialists** | Lambda, RunPod, Vast, CoreWeave | 2-5x cheaper; instant availability | Fewer regions, variable reliability |
| **Notebook services** | Colab, Kaggle | Free tier; zero setup | Time limits; the session dies |
| **Your own hardware** | A workstation | No hourly cost | Optimization lesson 14's arithmetic: idle hardware is expensive |

For learning and for most fine-tuning, **a GPU specialist is the sensible
default** — often a third of hyperscaler pricing for an identical card.

Three things that cost real money and are not on the price page:

1. **Egress.** Moving data *out* is billed, and it is not cheap. Keep data and
   compute in the same region and the same provider.
2. **Storage while idle.** A 500 GB volume attached to a stopped instance still
   bills. Snapshot and delete.
3. **The forgotten instance.** The single largest cloud bill surprise in this
   field. Lesson 07 is about preventing it.

For an Egyptian team there is a fourth: **where the data is allowed to live.**
Training on customer records in a foreign region is a Data-Security lesson 08
question before it is an engineering one.

---

## Before you press start

- [ ] Lesson 01's memory arithmetic, done. You know it fits
- [ ] Lesson 03's list applied. You know it fits on the **cheapest** card
- [ ] The code ran end to end on **1% of the data, on a laptop or a free tier**
- [ ] Checkpointing works, and you have restarted from one on purpose
- [ ] The data is in the same region as the GPU
- [ ] A billing alert exists at a number you chose
- [ ] The instance has a shutdown timer, or a script that stops it
- [ ] You know the expected wall-clock time, so a hang is visible
- [ ] Run records are being written (Data-Science lesson 07)

The third item is the one that saves the most money. **Never debug on a rented
GPU.** A shape mismatch found in minute two of an H100 rental costs the same as
one found on your laptop, and the laptop does not bill by the hour.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Comparing price per hour | Compare throughput per EGP |
| Debugging on a rented GPU | The same bug, at $8 an hour |
| Spot for serving | An outage with a discount |
| Spot without checkpointing | Work deleted at random |
| Data in a different region from the GPU | Egress fees and a slow loader |
| No billing alert | The bill is the alert |
| No shutdown timer | The weekend costs more than the run |
| A hyperscaler by default | Often 3x a specialist for the same card |

---

## Exercises

1. Price your intended run on three providers for the same GPU. What is the
   spread?
2. Run 100 training steps on two GPU types and compute throughput per EGP.
   Which is cheaper per sample?
3. Compute your egress cost for moving the trained model and logs out.
4. Set a billing alert and a shutdown timer, then verify both fire.
5. Run your whole pipeline on 1% of the data on a free tier. How many bugs did
   it find?

---

**Next:** [Lesson 06 — Interruptions and Checkpoints](06-interruptions.md)
