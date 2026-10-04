# Lesson 05 — Human Oversight

**Goal:** build oversight that actually changes outcomes, and measure whether
it does.

## What you will learn

- The three levels of oversight, and which one your obligation means
- Why a reviewer who overrides at random makes the system worse — measured
- The reviewer who catches 60% of errors and still makes it worse
- The four numbers that prove oversight is real

---

## Three levels

```text
HUMAN IN COMMAND   the system advises; the person decides every case
HUMAN IN THE LOOP  the system proposes; a person approves before it acts
HUMAN ON THE LOOP  the system acts; a person monitors and can intervene
```

Most regulation of high-risk systems means one of the first two. Most products
implement the third and call it the second.

The distinction is not semantic: **in the loop means nothing happens until a
person acts.** If your "approval" step has a default-approve timeout, you are
on the loop, and you should say so.

---

## Oversight that is not oversight

```python
import numpy as np

rng = np.random.default_rng(1)
N = 20000
truth = rng.random(N) < 0.08                               # 8% should be refused
model = np.where(rng.random(N) < 0.92, truth, ~truth)      # a 92% accurate model
wrong = model != truth
print("model alone: error rate", f"{wrong.mean():.4f}")
print()
print(f"{'reviewer':<34}{'overrides':>11}{'final error':>13}{'change':>10}")
base = wrong.mean()
rows = [
 ("rubber stamp (never overrides)", 0.0, 0.0),
 ("random: overrides 5% at random", None, None),
 ("sees 30% of the model's errors", 0.30, 0.01),
 ("sees 60% of the model's errors", 0.60, 0.01),
 ("sees 60%, doubts 10% of good calls", 0.60, 0.10),
 ("sees 90%, doubts 2% of good calls", 0.90, 0.02),
]
for name, catch, fp in rows:
    if catch is None:
        ov = rng.random(N) < 0.05
    else:
        ov = np.where(wrong, rng.random(N) < catch, rng.random(N) < fp)
    final = np.where(ov, ~model, model)
    e = (final != truth).mean()
    print(f"{name:<34}{ov.mean():>10.1%}{e:>13.4f}{e-base:>+10.4f}")
print("\noversight is a control only when the reviewer knows something the model does not.")
```

```text
model alone: error rate 0.0769

reviewer                            overrides  final error    change
rubber stamp (never overrides)          0.0%       0.0769   +0.0000
random: overrides 5% at random          5.0%       0.1196   +0.0427
sees 30% of the model's errors          3.4%       0.0613   -0.0156
sees 60% of the model's errors          5.5%       0.0400   -0.0369
sees 60%, doubts 10% of good calls     14.0%       0.1267   +0.0498
sees 90%, doubts 2% of good calls       8.9%       0.0267   -0.0502

oversight is a control only when the reviewer knows something the model does not.
```

Four readings, and the fifth row is the one worth the whole lesson.

**The rubber stamp changes nothing.** 0.0769 before, 0.0769 after. It satisfies
a process diagram and provides no protection whatsoever. This is the most
common real-world implementation of "human oversight".

**Random overriding makes it worse.** 0.0769 → 0.1196, a 56% increase in
errors, from a reviewer who overrides 5% of the time with no insight. A
reviewer under time pressure, with no information the model lacks, is
*actively harmful* — and still ticks the compliance box.

**An informed reviewer helps substantially.** Catching 60% of the model's
errors takes the error rate to 0.0400, nearly halving it.

**But catching 60% of errors while doubting 10% of correct decisions makes it
worse: 0.1267.** Look at why. Errors are 7.7% of cases, so 60% of them is 4.6%
of all cases caught. Correct decisions are 92.3% of cases, so doubting 10% of
them is 9.2% of all cases wrongly flipped — **twice as much damage as the
benefit.**

> **Because the model is mostly right, a reviewer's false overrides are
> multiplied by a much larger base than their catches.**

That asymmetry is the core of oversight design, and it is almost never stated.
A good reviewer is not one who is suspicious; it is one who is **selectively**
suspicious — the last row catches 90% of errors with a 2% false-override rate
and halves the error rate.

---

## What makes a reviewer informed

The reviewer must have **information the model does not have**. If they see
only the same features and the model's output, they can only agree or guess,
and guessing is the second row.

```text
GIVE THE REVIEWER          the model's confidence, and what it means
                           the 2-3 features that drove this case
                           cases the model flags as near the threshold
                           the context the model cannot see: a phone call,
                             a document, a note, their own knowledge
                           what happened the last 10 times they overrode

DO NOT GIVE THEM           every case, at a rate that forces 8 seconds each
                           the raw feature vector
                           a queue with a default-approve timeout
                           a target that rewards agreement
```

The last item on the "do not" list is the one that quietly destroys oversight:
if a reviewer is measured on throughput, or if overriding requires a written
justification while approving requires one click, **the system is designed to
produce a rubber stamp**, and it will.

And route by uncertainty, not uniformly. Reviewing every case at scale forces
seconds per case; reviewing the 5% nearest the threshold gives the reviewer
minutes on the cases where their judgement can actually change the outcome.

---

## The four numbers

To demonstrate oversight is real — to an auditor, or to yourself — log and
report these every month:

```text
1. OVERRIDE RATE          how often a human changes the outcome
                          0% means a rubber stamp. 40% means a broken model
2. OVERRIDE ACCURACY      when they override, were they right?
                          this is the number that separates rows 2 and 6
3. TIME PER REVIEW        under ~30 seconds on a high-tier decision is a
                          rubber stamp with extra steps
4. OUTCOME DELTA          final error rate with oversight vs model alone
                          the only number that proves the control works
```

Number 2 requires outcome data on overridden cases, which means you must
**follow up on overrides** rather than closing them. That is the cost of
knowing whether your oversight works, and it is the reason most teams cannot
answer.

Number 1 is a two-sided alarm. An override rate near zero means nobody is
really reviewing. An override rate of 40% means the reviewers do not trust the
model, and the model — not the oversight — is the problem to fix.

---

## The appeal route

Oversight before the decision is not the same as a route after it. For
anything in the high tier you owe both:

- [ ] A person can **learn a system was involved** ([lesson 06](06-transparency.md))
- [ ] They can **ask for a reason**, in language they understand
- [ ] They can **contest it to a human with authority to change it**
- [ ] That person can see the decision, the inputs and the model version
      ([lesson 03](03-the-record.md))
- [ ] The outcome of appeals is **logged and reviewed**, because a 90% appeal
      success rate is a finding about the model
- [ ] There is a **time limit** on the appeal, published

The fourth item is the one that fails in practice: the appeal reaches a person
who cannot see what the model saw, and so can only re-run the same decision by
hand. Test it as a customer, as in
[lesson 02](02-classifying-risk.md)'s exercise 4.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A rubber stamp counted as oversight | 0.0769 before, 0.0769 after |
| A reviewer with no extra information | Random overriding: 0.0769 → 0.1196 |
| Assuming any override is an improvement | 60% catch with 10% false overrides: worse |
| Reviewing every case | Seconds per case; route by uncertainty instead |
| Default-approve timeouts | That is on the loop, not in it |
| Measuring reviewers on throughput | You have designed a rubber stamp |
| Asymmetric friction (override needs an essay) | Same result, more slowly |
| Not following up overrides | You cannot compute override accuracy |
| An appeal route that cannot see the decision | The human re-guesses by hand |

---

## Exercises

1. Compute your override rate. Is it near 0% or above 30%?
2. Measure time per review on your highest-tier system.
3. Follow up 50 overrides and compute override accuracy.
4. Compute the outcome delta: error rate with and without oversight.
5. List what your reviewer sees that the model does not. If the list is empty,
   fix that first.
6. Appeal one of your own system's decisions as a customer would.

---

**Next:** [Lesson 06 — Transparency](06-transparency.md)
