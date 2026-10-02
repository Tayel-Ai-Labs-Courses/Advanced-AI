# Lesson 05 — Optimisation and Conditioning

**Goal:** understand why the same model trains in 64 steps or 76,746, depending
on something that has nothing to do with the model.

## What you will learn

- The condition number, and what it costs
- Why feature scaling is not cosmetic
- Step size, and the edge of stability
- What momentum and Adam are actually fixing

---

## The measurement

```python
import numpy as np

def steps_to_converge(cond, tol=1e-6, max_steps=2_000_000):
    """Gradient descent on f(x) = x0^2 + cond*x1^2, at the largest stable step."""
    a, b = 1.0, float(cond)
    lr = 0.9 / b                       # largest step that is stable in x1
    x = np.array([1.0, 1.0])
    for i in range(1, max_steps + 1):
        x = x - lr * np.array([2 * a * x[0], 2 * b * x[1]])
        if np.linalg.norm(x) < tol:
            return i
    return max_steps
print(f"{'condition number':>18}{'steps to converge':>20}")
for cond in (1, 10, 100, 1000, 10000):
    print(f"{cond:>18}{steps_to_converge(cond):>20,}")
print("\ncondition number = ratio of feature scales. This is why StandardScaler exists.")
```

```text
  condition number   steps to converge
                 1                  64
                10                  70
               100                 761
              1000               7,669
             10000              76,746

condition number = ratio of feature scales. This is why StandardScaler exists.
```

**Identical objective, identical algorithm, 1,200 times the work.** The only
thing that changed is the ratio between the two directions' curvature — which,
for a real model, is the ratio between your features' scales.

A dataset with `tenure_days` in the hundreds and `soil_moisture` in [0, 1] has a
condition number in the thousands before you have written a line of modelling
code.

**This is the entire justification for `StandardScaler`**, and it is why
[Data-Science lesson 04](../../Data-Science/lessons/04-features-and-pipelines.md)
puts it inside the pipeline. The lesson there measured that scaling barely moved
the score on a well-conditioned dataset; this lesson is why it sometimes moves
everything.

---

## Why it happens

Gradient descent takes the same step size in every direction. The largest
curvature sets the **maximum stable step**:

```text
stable requires    lr < 1 / largest curvature
progress along the flat direction is then governed by the SMALLEST curvature
so total steps scale with   largest / smallest  =  the condition number
```

You are forced to take tiny steps because of one steep direction, and then you
must take many of them because another direction is flat.

```python
for cond in (1, 100, 10000):
    lr = 0.9 / cond
    decay_flat = abs(1 - 2 * lr * 1.0)
    print(f"cond {cond:>6}: max stable lr {lr:.2e}, "
          f"flat direction shrinks by {decay_flat:.6f} per step")
```

```text
cond      1: max stable lr 9.00e-01, flat direction shrinks by 0.800000 per step
cond    100: max stable lr 9.00e-03, flat direction shrinks by 0.982000 per step
cond  10000: max stable lr 9.00e-05, flat direction shrinks by 0.999820 per step
```

At condition number 10,000 the flat direction shrinks by a factor of 0.99982 per
step. To reduce it by 1e-6 you need about 77,000 steps — which is exactly the
measured number above.

---

## What the optimisers are fixing

| Method | What it does about conditioning |
|---|---|
| **Plain SGD** | Nothing. You pay the full condition number |
| **Momentum** | Averages recent gradients, which damps the oscillation in the steep direction and accelerates the flat one |
| **Adam / RMSProp** | Divides each coordinate by its own recent gradient magnitude — **an approximate per-feature rescaling, learned on the fly** |
| **Newton / L-BFGS** | Uses the curvature directly. Condition number stops mattering, at the cost of the Hessian |
| **Batch norm / layer norm** | Keeps activations well-conditioned *between* layers, which is conditioning applied inside the network |

**Adam is popular largely because it is robust to bad conditioning.** It
approximates what scaling would have done, per parameter, continuously. That is
why it often "just works" on raw features where SGD needs careful tuning — and
why [Optimization lesson 03](../../Optimization/lessons/03-optimisers.md)
measures SGD sometimes winning once the data is scaled properly.

**Scaling your features is still worth doing.** It helps every optimiser, costs
one line inside the pipeline, and removes a variable from every later debugging
session.

---

## Step size, and the edge

```python
for lr in (0.05, 0.4, 0.9, 1.05):
    x = 1.0
    hist = []
    for _ in range(6):
        x = x - lr * 2 * x          # f(x) = x^2
        hist.append(round(x, 4))
    print(f"lr={lr:<5} {hist}")
```

```text
lr=0.05  [0.9, 0.81, 0.729, 0.6561, 0.5905, 0.5314]
lr=0.4   [0.2, 0.04, 0.008, 0.0016, 0.0003, 0.0001]
lr=0.9   [-0.8, 0.64, -0.512, 0.4096, -0.3277, 0.2621]
lr=1.05  [-1.1, 1.21, -1.331, 1.4641, -1.6105, 1.7716]
```

Four regimes, visible in the numbers:

- **0.05 — too small.** Converging, slowly. Wasting compute.
- **0.4 — right.** Fast, monotone.
- **0.9 — oscillating but still converging.** The edge of stability, and where
  many well-tuned runs actually sit.
- **1.05 — diverging.** The loss goes up, and it will become `NaN`.

The diagnostic worth memorising: **a loss that oscillates and grows means the
learning rate is too high, not that the model is wrong.** Halve it before
changing anything else.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Not scaling features | 1,200x the steps, from nothing to do with the model |
| Assuming Adam removes the need to scale | It approximates it; scaling still helps everywhere else |
| Debugging a diverging loss as a data problem | It is the learning rate |
| A learning rate copied from another dataset | It depends on your conditioning |
| Tiny learning rate "to be safe" | Converging slowly is also failing |
| Ignoring that batch norm is conditioning | It explains why removing it breaks training |

---

## Exercises

1. Compute the condition number of your own feature matrix, before and after
   scaling.
2. Train the same model with and without scaling, and count epochs to the same
   loss.
3. Find your learning rate's edge of stability by doubling until it diverges.
   Use half of that.
4. Compare SGD, momentum and Adam on unscaled data, then on scaled. Which gap
   closes?
5. Remove batch norm from a network that uses it and watch what happens to the
   usable learning rate.

---

**Next:** [Lesson 06 — Probability and Base Rates](06-probability.md)
