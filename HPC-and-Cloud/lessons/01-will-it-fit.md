# Lesson 01 — Will It Fit?

**Goal:** answer, in arithmetic and before renting anything, whether a model can
be trained on a given GPU.

## What you will learn

- The four things that occupy GPU memory
- The table that decides your hardware
- Why LoRA changes the answer and quantisation changes it more
- The question to ask before any cloud decision

---

## Four consumers of memory

Training holds four things at once:

```text
1. WEIGHTS          params x bytes_per_param
2. GRADIENTS        one per trainable parameter, same precision
3. OPTIMIZER STATE  Adam keeps TWO fp32 values per trainable parameter
4. ACTIVATIONS      scales with batch size x sequence length
```

The third is the one people forget, and it is usually the largest.

```python
def memory_gb(params_b, bytes_per_param=4, optimizer="adam", trainable_fraction=1.0):
    """Training memory, excluding activations. params_b is in BILLIONS."""
    p = params_b * 1e9
    weights = p * bytes_per_param
    trainable = p * trainable_fraction
    grads = trainable * bytes_per_param
    states = {"adam": 2, "sgd_momentum": 1, "sgd": 0}[optimizer] * trainable * 4
    return {"weights": weights / 1e9, "grads": grads / 1e9,
            "optimizer": states / 1e9,
            "total": (weights + grads + states) / 1e9}

print(f"{'model':<12}{'precision':<12}{'weights':>9}{'grads':>8}{'optim':>8}{'TOTAL GB':>10}")
for name, b in [("0.5B", 0.5), ("7B", 7.0), ("13B", 13.0), ("70B", 70.0)]:
    for prec, nbytes in (("fp32", 4), ("bf16 mixed", 2)):
        m = memory_gb(b, nbytes)
        print(f"{name:<12}{prec:<12}{m['weights']:>9.1f}{m['grads']:>8.1f}"
              f"{m['optimizer']:>8.1f}{m['total']:>10.1f}")
print("\n(activations are extra, and scale with batch size x sequence length)")
```

```text
model       precision     weights   grads   optim  TOTAL GB
0.5B        fp32              2.0     2.0     4.0       8.0
0.5B        bf16 mixed        1.0     1.0     4.0       6.0
7B          fp32             28.0    28.0    56.0     112.0
7B          bf16 mixed       14.0    14.0    56.0      84.0
13B         fp32             52.0    52.0   104.0     208.0
13B         bf16 mixed       26.0    26.0   104.0     156.0
70B         fp32            280.0   280.0   560.0    1120.0
70B         bf16 mixed      140.0   140.0   560.0     840.0

(activations are extra, and scale with batch size x sequence length)
```

**A 7B model needs 84 GB to fine-tune in mixed precision** — and 56 of those 84
are Adam's state, not the model.

Note what mixed precision does and does not do. Going from fp32 to bf16 halves
weights and gradients (28+28 becomes 14+14) and **leaves the optimizer state
untouched at 56 GB**, because Adam keeps its moments in fp32 for numerical
stability. Mixed precision took 112 GB to 84, a 25% saving, not the 50% people
expect.

---

## What fits on what

```python
GPUS = {"T4": 16, "A10G": 24, "L4": 24, "A100-40": 40, "A100-80": 80, "H100": 80}
print(f"{'model':<8}{'full fp32':>12}{'full bf16':>12}{'LoRA bf16':>12}{'inference bf16':>16}")
for name, b in [("0.5B", 0.5), ("7B", 7.0), ("13B", 13.0), ("70B", 70.0)]:
    full32 = memory_gb(b, 4)["total"]
    full16 = memory_gb(b, 2)["total"]
    lora = memory_gb(b, 2, trainable_fraction=0.01)["total"]
    infer = b * 2
    def fits(gb):
        ok = [g for g, v in GPUS.items() if v >= gb]
        return ok[0] if ok else f"{gb:.0f}GB: none"
    print(f"{name:<8}{fits(full32):>12}{fits(full16):>12}{fits(lora):>12}{fits(infer):>16}")
print("\ncheapest single GPU that fits, from T4(16) A10G(24) A100(40/80) H100(80)")
```

```text
model      full fp32   full bf16   LoRA bf16  inference bf16
0.5B              T4          T4          T4              T4
7B       112GB: none  84GB: none          T4              T4
13B      208GB: none 156GB: none     A100-40         A100-40
70B     1120GB: none 840GB: none 147GB: none     140GB: none

cheapest single GPU that fits, from T4(16) A10G(24) A100(40/80) H100(80)
```

Read the 7B row. **Full fine-tuning does not fit on any single GPU in that
list. LoRA fits on a T4** — the cheapest GPU there is, at about $0.35 an hour.

That is not a small optimisation. It is the difference between "we need a
multi-GPU cluster" and "we need the cheapest instance available", and it is
decided entirely by the optimizer-state column: LoRA trains ~1% of the
parameters, so grads and Adam state shrink by 100x while the frozen weights stay.

The mechanics are in [Optimization lesson 07](../../Optimization/lessons/07-parameter-efficient-finetuning.md);
this table is why you reach for it.

The 70B row is the other lesson: **nothing fits, including inference.** 140 GB
of weights alone. That model is a multi-GPU deployment or a quantised one —
4-bit takes those 140 GB to about 35, which is one A100-40
([Optimization lesson 08](../../Optimization/lessons/08-quantisation.md)).

---

## Activations, the part this table omits

Activations scale with **batch size x sequence length x layers**, and for
transformers at long context they can exceed the weights.

Two controls:

| Control | Effect | Cost |
|---|---|---|
| **Smaller micro-batch** + gradient accumulation | Linear reduction | Throughput (lesson 02) |
| **Gradient checkpointing** | ~sqrt(layers) instead of layers | ~30% more compute |
| Shorter sequences | Quadratic for attention | Task-dependent |

The practical loop: compute the table above, subtract from your GPU's memory,
and the remainder is your activation budget. If it is negative, no batch size
saves you and the answer is LoRA, quantisation, or more GPUs.

---

## The order to ask the questions

```mermaid
flowchart TD
    Q1does inference fit?<br/>params x 2 bytes -->|no| Q1N["quantise, or shard<br/>across GPUs"]
    Q1 -->|yes| Q2does LoRA fit?<br/>weights + 1% state
    Q2 -->|yes| L["<b>LoRA on the cheapest GPU</b><br/>start here"]
    Q2 -->|no| Q3does full fine-tuning fit?
    Q3 -->|yes| F["full fine-tune"]
    Q3 -->|no| M["multi-GPU: ZeRO / FSDP<br/>lesson 04"]
```

**Start at the bottom of that tree, not the top.** The common failure is
renting an 8xA100 node because the model is "big", when a single T4 running LoRA
would have answered the question — and Data-Science lesson 05's baseline might
have answered it without a GPU at all.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Counting only the weights | Adam's state was 56 GB of the 7B model's 84 |
| Expecting bf16 to halve memory | It halves two of four terms: 112 GB to 84, not 56 |
| Ignoring activations | They can exceed the weights at long context |
| Renting a big node before checking LoRA | 7B: "no single GPU" becomes "a T4" |
| Assuming inference memory equals training memory | 14 GB against 84 for a 7B model |
| Sizing from a blog post | Your sequence length and batch size are not theirs |

---

## Exercises

1. Compute the table for the exact model you intend to train, including
   activations at your batch size and sequence length.
2. Find the largest model you can LoRA-fine-tune on a 24 GB GPU.
3. Compute the memory for SGD with momentum instead of Adam. How much does the
   optimizer choice buy?
4. Take a model that does not fit and find the combination — LoRA, 4-bit,
   gradient checkpointing, shorter sequences — that makes it fit on one GPU.
5. For a model you already train, measure the real peak memory and compare it
   with this arithmetic. Where is the gap?

---

**Next:** [Lesson 02 — Where the Time Goes](02-where-the-time-goes.md)
