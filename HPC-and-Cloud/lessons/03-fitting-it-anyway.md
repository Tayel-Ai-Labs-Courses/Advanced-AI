# Lesson 03 — Fitting It Anyway

**Goal:** train a model that does not fit, using the four techniques in the
order they should be tried.

## What you will learn

- Mixed precision, and what it does not halve
- Gradient checkpointing, priced
- ZeRO and FSDP, in one paragraph each
- The order to try them

---

## The order

Lesson 01 gave you a number that is too big for your GPU. Work down this list
and stop at the first thing that makes it fit.

```text
1. Mixed precision (bf16)        -25% memory, FASTER. Do this always
2. A smaller micro-batch         linear, costs throughput (lesson 02)
3. Gradient checkpointing        large saving, ~30% more compute
4. LoRA / PEFT                   -99% of optimizer state. Usually the answer
5. 8-bit optimizer               -75% of optimizer state
6. Quantised base (QLoRA)        -75% of weights
7. ZeRO / FSDP across GPUs       shards everything, needs an interconnect
8. A bigger GPU                  the expensive answer, and often unnecessary
```

Most teams jump from 1 to 8. Steps 4 and 6 between them turn "we need a
multi-GPU node" into "a single 24 GB card", which is a 10-20x cost difference.

---

## Mixed precision

Weights and activations in bf16, the optimizer's accumulations in fp32.

| | fp32 | bf16 mixed |
|---|---|---|
| Weights, 7B | 28 GB | **14 GB** |
| Gradients | 28 GB | **14 GB** |
| Adam state | 56 GB | 56 GB — **unchanged** |
| Total | 112 GB | 84 GB |
| Speed on modern GPUs | 1x | **2-3x** |

**It is faster and smaller, so there is no trade to think about** — this is the
one item on the list that costs nothing. Use `bf16` rather than `fp16` where the
hardware supports it: same memory, far fewer overflow problems, and no loss
scaler to tune.

What it does not do is touch the optimizer state, which lesson 01 showed is the
largest term. That is what steps 4 and 5 are for.

---

## Gradient checkpointing

Instead of keeping every layer's activations for the backward pass, keep a few
and recompute the rest.

```text
memory:   O(layers)  ->  O(sqrt(layers))
compute:  one forward -> roughly 1.3 forwards
```

**Trade ~30% more time for a large activation saving.** On a long-context
transformer this is often what makes a batch size of 4 possible instead of 1 —
and since lesson 02 measured that batch 1 is catastrophic for throughput, the
30% is frequently *repaid* by the larger batch.

Measure it both ways. It is one flag:

```python
# no-run
model.gradient_checkpointing_enable()
```

---

## 8-bit optimizers and QLoRA

**An 8-bit Adam** stores its two moments in int8 with block-wise quantisation.
The 56 GB of state for a 7B model becomes about 14 GB, for a quality difference
that is usually unmeasurable. It is another one-line change, and it is
under-used.

**QLoRA** quantises the frozen base weights to 4-bit and trains LoRA adapters in
bf16 on top. For the 7B model: weights 28 GB (fp32) become about 3.5 GB, and the
optimizer state is 1% of a small number. The whole thing fits comfortably on a
16 GB card, which is the cheapest GPU any provider rents.

The quality cost is real but small, and — this is the point —
**measurable with the eval set from LLM lesson 07 before you commit.**

---

## ZeRO and FSDP, when one GPU is not enough

Both shard the training state across GPUs instead of replicating it.

| Stage | Shards | Memory per GPU |
|---|---|---|
| Plain data parallel | nothing; every GPU has a full copy | 1x |
| **ZeRO-1** | optimizer state | ~0.5x |
| **ZeRO-2** | + gradients | ~0.3x |
| **ZeRO-3** / FSDP | + weights | ~1/N |

ZeRO-3 and PyTorch's FSDP are the same idea: each GPU holds 1/N of everything
and gathers what it needs for each layer, then throws it away. Memory falls
almost linearly with the number of GPUs; **communication rises**, which is
lesson 04's subject and the reason the interconnect decides whether this works.

The practical guidance: **ZeRO-2 is usually the sweet spot** on commodity
interconnects. ZeRO-3 needs fast networking or it spends its time waiting.

---

## Worked example

A 7B model, one 24 GB GPU, and lesson 01's 84 GB problem:

```text
start                              84 GB   does not fit
+ bf16 mixed precision             84 GB   (already counted)
+ 8-bit Adam                       42 GB   still does not fit
+ LoRA (1% trainable)            14.6 GB   FITS - and this is usually enough
+ 4-bit base weights (QLoRA)      4.1 GB   comfortable, room for a real batch
```

**Two one-line changes took an impossible job to a 16 GB card.** That is the
whole lesson, and it is worth an hour before any cloud quote.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Renting more GPUs first | Steps 4 and 6 are one line each |
| fp16 instead of bf16 on modern hardware | Overflow problems and a loss scaler to tune |
| Expecting mixed precision to halve memory | It leaves the largest term untouched |
| Skipping gradient checkpointing because of the 30% | It buys back more through a larger batch |
| ZeRO-3 on slow networking | It waits more than it computes |
| Not measuring quality after quantising | The saving might have cost accuracy you needed |
| Full fine-tuning by default | LoRA is usually as good and 20x cheaper |

---

## Exercises

1. Apply the list to your own model and record the memory after each step.
2. Measure gradient checkpointing: memory, step time, and the batch size it
   unlocks. Does the larger batch repay the 30%?
3. Run the same fine-tune with Adam and with 8-bit Adam. Compare final eval
   score and peak memory.
4. Compare LoRA against a full fine-tune on your eval set. What does the
   difference cost per point?
5. Compute the cheapest GPU that fits your job after applying steps 1-6.

---

**Next:** [Lesson 04 — More Than One GPU](04-more-than-one-gpu.md)
