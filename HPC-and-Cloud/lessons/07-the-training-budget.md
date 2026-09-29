# Lesson 07 — The Training Budget

**Goal:** know what a project will cost before it starts, and stop the bill that
arrives anyway.

## What you will learn

- Estimating a run before you launch it
- The experiments budget, which is the real cost
- The four controls that stop a surprise bill
- What to do when the estimate says no

---

## Estimating a run

Four numbers, and none of them require a cloud account.

```python
def estimate(samples, epochs, samples_per_sec, gpu_price_usd_hr, n_gpus=1,
             efficiency=1.0, usd_egp=48.0):
    total = samples * epochs
    gpu_seconds = total / (samples_per_sec * n_gpus * efficiency)
    hours = gpu_seconds / 3600
    return {
        "wall_clock_h": round(hours, 2),
        "gpu_hours": round(hours * n_gpus, 2),
        "usd": round(hours * n_gpus * gpu_price_usd_hr, 2),
        "egp": round(hours * n_gpus * gpu_price_usd_hr * usd_egp, 0),
    }

print(f"{'setup':<34}{'wall h':>9}{'gpu-h':>9}{'USD':>9}{'EGP':>10}")
for label, kw in [
    ("1x T4, 500 samples/s", dict(samples_per_sec=500, gpu_price_usd_hr=0.35)),
    ("1x A10G, 1,800 samples/s", dict(samples_per_sec=1800, gpu_price_usd_hr=1.00)),
    ("1x A100, 5,000 samples/s", dict(samples_per_sec=5000, gpu_price_usd_hr=3.00)),
    ("4x A100, 37% efficiency", dict(samples_per_sec=5000, gpu_price_usd_hr=3.00,
                                     n_gpus=4, efficiency=0.37)),
]:
    r = estimate(samples=1_200_000, epochs=3, **kw)
    print(f"{label:<34}{r['wall_clock_h']:>9.2f}{r['gpu_hours']:>9.2f}"
          f"{r['usd']:>9.2f}{r['egp']:>10,.0f}")
```

```text
setup                                wall h    gpu-h      USD       EGP
1x T4, 500 samples/s                   2.00     2.00     0.70        34
1x A10G, 1,800 samples/s               0.56     0.56     0.56        27
1x A100, 5,000 samples/s               0.20     0.20     0.60        29
4x A100, 37% efficiency                0.14     0.54     1.62        78
```

**Three of these cost about the same money and differ by 10x in wall-clock
time.** The A10G is both the cheapest and ten times faster than the T4, because
throughput scaled better than price did — which is lesson 05's "compare
throughput per EGP" made concrete.

And the fourth row is lesson 04's tax: four A100s finish 1.4x sooner than one
and cost 2.7x as much.

**Get `samples_per_sec` by running 100 steps.** That one measurement turns every
number here from a guess into an estimate.

---

## The real cost is the experiments

A single run is rarely the bill. The bill is the search.

```text
one run                      0.56 h      27 EGP
x 12 hyperparameter configs  6.7 h      324 EGP
x 3 seeds each (RL 09, DS 05) 20 h       972 EGP
x 4 rounds of "let's try..."  80 h     3,888 EGP
+ 2 failed runs at 80%        12 h       583 EGP
+ the week the instance
  was left running            168 h    8,064 EGP   <- the largest line
```

The last line is not a joke. **The most expensive item in most ML cloud bills
is an idle instance nobody turned off.** The second most expensive is a
hyperparameter search that Data-Science lesson 05 would have shown was
pointless — `C` across three orders of magnitude moved AUC by 0.0001.

So the budget conversation has three parts, in this order:

1. **How many experiments, and why that many?** Twelve configurations at three
   seeds is a decision, not a default.
2. **What would make us stop early?** A result inside the noise means stop.
3. **Who turns it off?**

---

## The four controls

| Control | Stops | Effort |
|---|---|---|
| **A billing alert** at a number you chose | The surprise | 5 minutes |
| **A hard budget / quota** on the account | The catastrophe | 15 minutes |
| **An auto-shutdown timer** on every instance | The forgotten machine | one flag |
| **A cost line in every run record** | The slow creep | Data-Science lesson 07 |

The third is the one that pays for itself fastest. Every provider supports
either an idle-shutdown setting or a `shutdown -h +480` in the startup script.
**Set it to your estimated wall clock plus 50%**, and a hung job costs you that
instead of a weekend.

The fourth turns cost into something you can review. Add `gpu_hours` and
`cost_egp` to the run record beside the metrics, and a month later you can sort
your experiments by cost per point of improvement — which is the number that
tells you whether the search was worth running.

---

## When the estimate says no

The estimate comes back at 40,000 EGP and the project has 5,000. In order:

| Move | Typical saving |
|---|---|
| **Use a smaller model** and measure the gap first | 5-20x |
| **Subsample the data.** Learning curves flatten (LLM lesson 08: 50 examples) | 2-10x |
| **LoRA instead of full fine-tuning** (lesson 03) | 5-20x |
| **Spot instances** with checkpointing (lesson 06) | 3x |
| **A cheaper GPU** that fits after lesson 03 | 2-20x |
| Fewer configurations, chosen by the lesson-05 cheap grid | 3-10x |
| **A different provider** | 2-5x |
| **Do not train.** Prompt or use an existing model | ∞ |

The last row is a real answer. LLM lesson 08 measured a case where four
examples in a prompt matched a fine-tune — the training budget there was zero.

And the second row deserves its own habit: **plot the learning curve on 10% of
the data first.** If accuracy has flattened by 50%, the full run buys nothing,
and you found that out for a tenth of the price.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Estimating from a blog post | Your throughput is not theirs; measure 100 steps |
| Budgeting one run | The search is 50x the run |
| No auto-shutdown | The largest line in the bill |
| No billing alert | The invoice is the alert |
| A hyperparameter search with no stopping rule | Money spent inside the noise |
| Picking by price per hour | The A10G was cheaper *and* 10x faster |
| Training before checking a smaller model | 5-20x, for free |
| No cost in the run record | You cannot review what you did not record |

---

## Exercises

1. Measure `samples_per_sec` for your model and produce the four-row table for
   your own candidates.
2. Write your full experiment budget: runs x configs x seeds, with a total.
3. Set a billing alert, a hard quota and an auto-shutdown, and verify each.
4. Add `gpu_hours` and `cost_egp` to your run records. Sort last month's
   experiments by cost per point of improvement.
5. Train on 10% of your data and plot the learning curve. Does the full run
   still look necessary?

---

**Next:** [Lesson 08 — From Laptop to Cloud](08-laptop-to-cloud.md)
