# Lesson 06 — Interruptions and Checkpoints

**Goal:** make a long run survive being killed, and choose how often to save.

## What you will learn

- The checkpoint frequency optimum, computed
- What a checkpoint must contain
- Resuming correctly, which is harder than saving
- The spot-instance workflow

---

## How often should you checkpoint?

Too rarely and an interruption costs hours. Too often and you spend the run
writing files. There is an optimum, and it is arithmetic.

```python
import numpy as np
def expected_waste(run_hours, mtbf_hours, checkpoint_every_h, checkpoint_cost_min=2):
    """Expected extra hours from interruptions + checkpoint overhead."""
    n_ckpt = run_hours / checkpoint_every_h
    overhead = n_ckpt * checkpoint_cost_min / 60
    failures = run_hours / mtbf_hours
    lost_per_failure = checkpoint_every_h / 2          # on average, half an interval
    return overhead + failures * lost_per_failure

RUN, MTBF = 24.0, 6.0
print(f"a {RUN:.0f}h run on spot instances interrupted every ~{MTBF:.0f}h")
print(f"{'checkpoint every':>18}{'overhead h':>12}{'lost to restarts h':>21}{'total waste h':>15}")
for every in (0.25, 0.5, 1, 2, 4, 8, 24):
    n = RUN / every
    over = n * 2 / 60
    lost = (RUN / MTBF) * (every / 2)
    print(f"{every:>15.2f}h{over:>12.1f}{lost:>21.1f}{over + lost:>15.1f}")
best = min(((every, RUN/every*2/60 + (RUN/MTBF)*(every/2)) for every in
            np.arange(0.1, 8, 0.05)), key=lambda t: t[1])
print(f"\noptimum near every {best[0]:.2f}h, wasting {best[1]:.1f}h of {RUN:.0f}")
```

```text
a 24h run on spot instances interrupted every ~6h
  checkpoint every  overhead h   lost to restarts h  total waste h
           0.25h         3.2                  0.5            3.7
           0.50h         1.6                  1.0            2.6
           1.00h         0.8                  2.0            2.8
           2.00h         0.4                  4.0            4.4
           4.00h         0.2                  8.0            8.2
           8.00h         0.1                 16.0           16.1
          24.00h         0.0                 48.0           48.0

optimum near every 0.65h, wasting 2.5h of 24
```

**Checkpointing only at the end wastes 48 hours on a 24-hour run** — which is
the arithmetic way of saying it never finishes. Every interruption restarts from
zero, and at one interruption every six hours it never gets more than six hours
in.

The optimum is around **every 40 minutes**, costing 2.5 hours of a 24-hour
run — about 10%, which is the price of using spot instances at a third of the
cost. That trade is overwhelmingly worth it.

The shape generalises: **checkpoint interval ~ sqrt(2 x checkpoint_cost x
MTBF)**. Cheaper checkpoints mean more frequent ones; a more reliable machine
means fewer.

Two adjustments in practice:

- **Measure your own checkpoint cost.** A 7B model's optimizer state is tens of
  gigabytes, and writing it to network storage is minutes, not the two assumed
  above. Write to local disk, then upload asynchronously.
- **Know your MTBF.** Spot interruption rates vary hugely by instance type and
  region; providers publish them, and choosing a less popular type can change
  6 hours to 24.

---

## What a checkpoint must contain

This is where resumed runs go quietly wrong. A checkpoint that holds only the
weights will resume, train, and produce a worse model than an uninterrupted run
— with nothing in the logs to say why.

```text
model weights                    obviously
optimizer state                  Adam's moments. Without these, the first
                                 steps after resume are effectively a reset
lr scheduler state               or the schedule restarts from step 0
epoch AND step within the epoch  so the data order resumes correctly
RNG state                        torch, numpy, python, and the dataloader's
                                 worker seeds - or augmentation repeats
the config                       so you know what this checkpoint IS
the metric so far                to compare against, and to pick the best
the data version / hash          Data-Science lesson 07
```

```python
# no-run
torch.save({
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "scheduler": sched.state_dict(),
    "epoch": epoch,
    "global_step": step,
    "rng": {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate()},
    "config": cfg,
    "best_metric": best,
    "data_sha": data_hash,
}, "/local/ckpt-step-{step}.pt")
```

**The scheduler line is the one most often missing**, and its failure mode is
the nastiest: the learning rate jumps back to its warmup value mid-training, the
loss spikes, and it looks like a data problem.

---

## Resuming is harder than saving

| Trap | Symptom | Fix |
|---|---|---|
| Scheduler not restored | Loss spikes after resume | Save and restore it |
| Optimizer not restored | Slow recovery, worse final model | Save and restore it |
| Data order restarts | Some samples seen twice, some never | Save `global_step`, skip forward |
| RNG not restored | Augmentation repeats exactly | Save all four RNG states |
| Checkpoint written during a crash | A truncated, unloadable file | **Write to a temp name, then rename** |
| Only the latest kept | It was corrupt, and it is your only copy | Keep the last N **and** the best |
| Resume never tested | It fails at 3 a.m. on the real run | Test it on purpose, day one |

The last row is the requirement. **Kill your own training at step 500 and resume
it**, and confirm the loss curve is continuous across the join. A resume path
that has never been exercised does not work; that is not pessimism, it is the
base rate.

The rename trick deserves its own line: `torch.save` to `ckpt.tmp`, then
`os.replace("ckpt.tmp", "ckpt.pt")`. Rename is atomic, so an interruption
mid-write leaves the *previous* good checkpoint intact instead of a corrupt new
one.

---

## The spot workflow

```mermaid
flowchart TD
    S["start / resume<br/>from the latest checkpoint"] --> T["train"]
    T --> Ccheckpoint<br/>interval?
    C -->|yes| W["write locally, rename atomically,<br/>upload async"]
    W --> T
    C -->|no| Pinterruption<br/>notice?
    P -->|yes| E["<b>checkpoint immediately</b><br/>~2 min of warning"]
    E --> X["exit cleanly"]
    X -.->|"orchestrator restarts"| S
    P -->|no| T
```

Three requirements this diagram imposes:

1. **Handle the interruption signal.** Providers give ~2 minutes of warning.
   Catching it and checkpointing turns a lost interval into a lost minute.
2. **Checkpoints live off the instance.** Local disk dies with the machine.
   Write locally for speed, upload asynchronously for survival.
3. **Something restarts the job.** A job that needs a human to restart it is an
   on-demand job wearing a spot price.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Checkpointing only at the end | 48 hours of waste on a 24-hour run |
| Checkpointing every few minutes | 3.2 hours of overhead, for nothing |
| Saving only the weights | The resumed run is quietly worse |
| Forgetting the scheduler | The LR jumps; it looks like a data bug |
| Writing directly to the final filename | An interruption corrupts your only copy |
| Keeping only the latest | It was the corrupt one |
| Never testing resume | It fails on the run that mattered |
| Checkpoints on the instance's local disk only | They die with the machine |

---

## Exercises

1. Measure how long one checkpoint takes to write for your model, locally and
   to object storage.
2. Compute your optimum interval from that cost and your provider's published
   interruption rate.
3. Kill a run at step 500, resume, and plot the loss across the join. Is it
   continuous?
4. Remove the scheduler from your checkpoint deliberately and observe the
   failure so you recognise it later.
5. Implement the interruption handler and trigger it manually.

---

**Next:** [Lesson 07 — The Training Budget](07-the-training-budget.md)
