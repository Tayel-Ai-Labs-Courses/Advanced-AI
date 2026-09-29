# Lesson 04 — More Than One GPU

**Goal:** know what N GPUs actually buy, which is never N times the speed.

## What you will learn

- Data, model and pipeline parallelism, distinguished
- Scaling efficiency, measured against the interconnect
- Why 64 GPUs gave 22x
- What the speed costs

---

## Three kinds of parallel

| Kind | Split | Use when |
|---|---|---|
| **Data parallel** | The batch. Every GPU has the whole model | The model fits on one GPU. **90% of cases** |
| **Tensor / model parallel** | Individual layers, across GPUs | One layer does not fit. Needs very fast links |
| **Pipeline parallel** | Layers into stages, GPU 1 does 1-8, GPU 2 does 9-16 | The model does not fit, links are slower |
| **FSDP / ZeRO-3** | Everything, gathered per layer | Data parallel that also does not fit (lesson 03) |

Start with data parallel. The others exist because the model does not fit, not
because they are faster.

---

## What data parallel costs

Each step, every GPU computes gradients on its shard of the batch, and then
**all of them must agree**. That agreement is an all-reduce over the network,
and it is the whole story.

```python
def scaling(n_gpus, compute_s=0.100, params_b=0.5, bandwidth_gbps=25, overlap=0.7):
    """One step: compute, then all-reduce the gradients over the network."""
    bytes_ = params_b * 1e9 * 2                    # bf16 gradients
    # ring all-reduce moves 2*(n-1)/n * bytes per GPU
    comm_s = 0.0 if n_gpus == 1 else (2 * (n_gpus - 1) / n_gpus) * bytes_ / (bandwidth_gbps * 1e9 / 8)
    step = compute_s + comm_s * (1 - overlap)
    throughput = n_gpus / step
    return step, throughput, comm_s

base = scaling(1)[1]
print(f"0.5B model, 100 ms of compute per step, {25} Gbps interconnect, 70% overlap")
print(f"{'GPUs':>6}{'step ms':>10}{'comm ms':>10}{'throughput':>13}{'speedup':>10}{'efficiency':>12}")
for n in (1, 2, 4, 8, 16, 32, 64):
    step, thr, comm = scaling(n)
    print(f"{n:>6}{step*1000:>10.1f}{comm*1000:>10.1f}{thr:>13.1f}{thr/base:>9.1f}x{thr/base/n:>11.0%}")
```

```text
0.5B model, 100 ms of compute per step, 25 Gbps interconnect, 70% overlap
  GPUs   step ms   comm ms   throughput   speedup  efficiency
     1     100.0       0.0         10.0      1.0x       100%
     2     196.0     320.0         10.2      1.0x        51%
     4     244.0     480.0         16.4      1.6x        41%
     8     268.0     560.0         29.9      3.0x        37%
    16     280.0     600.0         57.1      5.7x        36%
    32     286.0     620.0        111.9     11.2x        35%
    64     289.0     630.0        221.5     22.1x        35%
```

**64 GPUs give 22.1x the throughput.** You paid for 64 and received 22, an
efficiency of 35%.

And look at the second row: **two GPUs are barely faster than one** (10.2
against 10.0). The communication cost appears in full as soon as there is
anything to communicate, and a 0.5B model's gradients are 1 GB to move every
step.

Two structural facts in that table:

- **Efficiency stabilises**, it does not keep collapsing. Ring all-reduce moves
  `2(N-1)/N` of the data per GPU, which approaches a constant. Going from 32 to
  64 GPUs really does roughly double throughput — at a fixed 35% efficiency.
- **The first doubling is the worst.** Going 1 to 2 buys nothing here.

---

## The interconnect decides

```python
print(f"{'interconnect':<22}{'8 GPUs':>10}{'32 GPUs':>10}{'64 GPUs':>10}")
for name, gbps in [("1 Gbps ethernet", 1), ("10 Gbps", 10),
                   ("25 Gbps", 25), ("200 Gbps InfiniBand", 200),
                   ("NVLink (~900)", 900)]:
    row = ""
    for n in (8, 32, 64):
        thr = scaling(n, bandwidth_gbps=gbps)[1]
        row += f"{thr/base/n:>10.0%}"
    print(f"{name:<22}{row}")
print("\ncells are scaling efficiency: 100% means n GPUs do n times the work")
```

```text
interconnect              8 GPUs   32 GPUs   64 GPUs
1 Gbps ethernet               2%        2%        2%
10 Gbps                      19%       18%       17%
25 Gbps                      37%       35%       35%
200 Gbps InfiniBand          83%       81%       81%
NVLink (~900)                96%       95%       95%

cells are scaling efficiency: 100% means n GPUs do n times the work
```

**On 1 Gbps ethernet, 64 GPUs do the work of 1.3.** You would be paying 64 times
the price for a 30% improvement.

This is the number that should decide your instance type, and it is the number
buried deepest in the documentation. Multi-GPU *inside one machine* (NVLink or
PCIe) is a completely different proposition from multi-GPU *across machines*.

**Practical rule: fill one machine before adding a second.** 8 GPUs in one box
on NVLink beat 8 GPUs in 8 boxes on ethernet by a factor of forty in efficiency,
and cost the same to rent.

---

## What the speed costs

```python
PRICE_PER_GPU_HR = 3.00
WORK = 1000.0          # arbitrary units; 1 GPU does `base` units/sec
for n in (1, 8, 32, 64):
    thr = scaling(n)[1]
    hours = WORK / thr / 3600 * 1000
    cost = hours * n * PRICE_PER_GPU_HR
    print(f"{n:>3} GPUs: {hours:>7.2f} h wall clock, {hours*n:>8.2f} GPU-hours, "
          f"${cost:>8.2f}")
print("\nmore GPUs finish sooner and cost more. That is the trade, priced.")
```

```text
  1 GPUs:   27.78 h wall clock,    27.78 GPU-hours, $   83.33
  8 GPUs:    9.31 h wall clock,    74.44 GPU-hours, $  223.33
 32 GPUs:    2.48 h wall clock,    79.44 GPU-hours, $  238.33
 64 GPUs:    1.25 h wall clock,    80.28 GPU-hours, $  240.83

more GPUs finish sooner and cost more. That is the trade, priced.
```

**64 GPUs finish 22 times sooner and cost 2.9 times as much.**

That is not automatically a bad trade. Whether it is depends on a question that
is not technical:

| If... | Then |
|---|---|
| You are iterating, and each experiment informs the next | **Buy the speed.** A 28-hour loop means one experiment a day |
| It is a final production run | **Buy the cheap.** One GPU overnight |
| You are exploring hyperparameters | Run **many single-GPU jobs in parallel** — 100% efficiency, no communication |
| The deadline is real | Speed |

The third row is the one people miss. Eight independent single-GPU runs have
**perfect** scaling efficiency, because they never communicate. If your goal is
"try eight configurations", do not run one job on eight GPUs.

---

## Practical notes

**The learning rate.** Data parallel multiplies the effective batch size by N.
The linear scaling rule — multiply the learning rate by N, with a warmup — is
the usual starting point, and it breaks at large N. See
[Optimization lesson 04](../../Optimization/lessons/04-learning-rate-schedules.md).

**Determinism.** Multi-GPU reductions change the order of floating-point
additions, so runs are not bit-identical. Data-Science lesson 07's seeding
advice still applies; bit-exactness does not.

**Debug on one GPU.** Always. A bug that only appears on 8 GPUs is usually a
bug that also exists on 1 and is merely easier to see with more noise.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Assuming N GPUs give N times the speed | 64 gave 22.1 |
| Not checking the interconnect | 2% efficiency on 1 Gbps |
| Spreading across machines before filling one | 40x worse efficiency for the same money |
| One big job for a hyperparameter search | Eight separate jobs scale perfectly |
| Forgetting to scale the learning rate | The effective batch grew by N |
| Debugging at scale | The same bug, more expensively |
| Reporting GPU-hours as if they were wall-clock savings | 2.9x the money for 22x the speed |

---

## Exercises

1. Find your instance type's actual inter-GPU and inter-node bandwidth. Compute
   your expected efficiency at 2, 4 and 8 GPUs.
2. Measure real scaling on 1 and 2 GPUs. How close is it to the model?
3. Compute the crossover: at what deadline does 8 GPUs beat 1 on value?
4. Run four hyperparameter configurations as four single-GPU jobs and compare
   total wall clock with one 4-GPU job.
5. Compute the largest number of GPUs worth using for your model, defining
   "worth" as efficiency above 50%.

---

**Next:** [Lesson 05 — Renting Hardware](05-renting-hardware.md)
