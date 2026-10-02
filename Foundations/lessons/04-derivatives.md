# Lesson 04 — Derivatives and Gradients

**Goal:** understand the one operation every model in this track is trained by.

## What you will learn

- A gradient as a direction
- Checking a derivative numerically, which finds real bugs
- The chain rule, which *is* backpropagation
- Where derivatives do not exist, and what that costs

---

## A gradient points uphill

```python
import numpy as np

def f(x):
    return (x[0] - 3) ** 2 + 2 * (x[1] + 1) ** 2

def grad_analytic(x):
    return np.array([2 * (x[0] - 3), 4 * (x[1] + 1)])

def grad_numeric(f, x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g

x = np.array([0.0, 0.0])
a, n = grad_analytic(x), grad_numeric(f, x)
print(f"analytic gradient at (0,0): {a}")
print(f"numeric  gradient at (0,0): {np.round(n, 6)}")
print(f"max absolute difference   : {np.abs(a - n).max():.2e}")
print("\nif these disagree by more than ~1e-5, your derivative is wrong.")
```

```text
analytic gradient at (0,0): [-6.  4.]
numeric  gradient at (0,0): [-6.  4.]
max absolute difference   : 3.93e-11

if these disagree by more than ~1e-5, your derivative is wrong.
```

The gradient `[-6, 4]` says: *increasing `x0` decreases `f` fastest, increasing
`x1` increases it.* Gradient **descent** therefore steps in `-gradient`.

And the second half of that block is a tool, not a demonstration. **The
numerical gradient check is how you find a wrong derivative**, and it applies to
any custom loss or layer you write:

```text
|analytic - numeric| / (|analytic| + |numeric|)   should be < 1e-6
```

If you ever write a custom loss and the model "trains but badly", run this
first. It takes two minutes and it has a very high hit rate.

---

## Descent, by hand

```python
x = np.array([0.0, 0.0])
print(f"{'step':>5}{'x0':>9}{'x1':>9}{'f(x)':>10}")
for step in range(9):
    if step in (0, 1, 2, 4, 8):
        print(f"{step:>5}{x[0]:>9.4f}{x[1]:>9.4f}{f(x):>10.4f}")
    x = x - 0.15 * grad_analytic(x)
print("minimum is at (3, -1), where f = 0")
```

```text
 step       x0       x1      f(x)
    0   0.0000   0.0000   11.0000
    1   0.9000  -0.6000    4.7300
    2   1.5300  -0.8400    2.2121
    4   2.2797  -0.9744    0.5201
    8   2.8271  -0.9993    0.0299
minimum is at (3, -1), where f = 0
```

Eight steps, from 11.0 to 0.03. Note that `x1` arrived almost immediately while
`x0` is still crawling — because the curvature differs by a factor of 2 between
the directions. That asymmetry is the whole subject of lesson 05.

---

## The chain rule is backpropagation

```python
# a two-layer network on one example, differentiated by hand
w1, w2, xin, target = 0.5, -1.2, 2.0, 1.0
h = w1 * xin              # layer 1
yhat = w2 * h             # layer 2
loss = (yhat - target) ** 2
dloss_dyhat = 2 * (yhat - target)
dyhat_dh = w2
dh_dw1 = xin
dloss_dw1 = dloss_dyhat * dyhat_dh * dh_dw1     # chain rule
dloss_dw2 = dloss_dyhat * h
print(f"forward : h = {h}, yhat = {yhat}, loss = {loss}")
print(f"backward: dloss/dw2 = {dloss_dw2}")
print(f"          dloss/dw1 = {dloss_dyhat} * {dyhat_dh} * {dh_dw1} = {dloss_dw1}")
eps = 1e-6
num = (((w2 * ((w1 + eps) * xin)) - target) ** 2 - loss) / eps
print(f"numeric dloss/dw1       = {num:.6f}")
```

```text
forward : h = 1.0, yhat = -1.2, loss = 4.840000000000001
backward: dloss/dw2 = -4.4
          dloss/dw1 = -4.4 * -1.2 * 2.0 = 10.56
numeric dloss/dw1       = 10.560006
```

That is backpropagation, complete, for a two-layer network. The gradient with
respect to an early weight is **a product of local derivatives along the path**:

```text
dloss/dw1 = dloss/dyhat  x  dyhat/dh  x  dh/dw1
              (-4.4)          (-1.2)      (2.0)   =  10.56
```

PyTorch does exactly this, automatically, for millions of parameters. Knowing
it is a product of terms explains two things you will meet:

- **Vanishing gradients.** Multiply fifty numbers smaller than 1 and you get
  approximately zero, so early layers stop learning. Residual connections exist
  to give the product a path of 1s.
- **Exploding gradients.** Multiply fifty numbers larger than 1 and you get
  infinity. Gradient clipping is a cap on that product.

Both are consequences of one multiplication, not mysterious deep-learning
phenomena.

---

## Where the derivative does not exist

| Function | Differentiable? | Consequence |
|---|---|---|
| `ReLU` at 0 | Not at one point | Nobody cares; a subgradient of 0 works |
| `abs` at 0 | Same | Same |
| Accuracy, F1, precision@k | **No** — they are step functions | **You cannot train on them** |
| A discrete choice (which tool, which bin) | No | Needs RL, or a relaxation |
| Tree splits | No | Trees use a different algorithm entirely |

Row three is the one that shapes practice. **Accuracy has a gradient of zero
almost everywhere**, so you train on cross-entropy — a smooth surrogate — and
*evaluate* on accuracy. That gap between the loss you minimise and the metric
you care about is why
[Data-Science lesson 06](../../Data-Science/lessons/06-evaluating-the-decision.md)
exists: the threshold is chosen afterwards, because the training could not
choose it.

And row four is why
[Reinforcement-Learning](../../Reinforcement-Learning) is a separate course: no
gradient through "which action", so you need a different estimator entirely.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Writing a custom loss without a gradient check | It trains, badly, and you blame the data |
| Expecting to optimise accuracy directly | Zero gradient almost everywhere |
| Treating vanishing gradients as mysterious | It is a product of small numbers |
| Forgetting the minus in descent | The model maximises the loss |
| A step size chosen without reference to curvature | Lesson 05 |

---

## Exercises

1. Write a custom loss, derive its gradient by hand, and check it numerically.
2. Compute `dloss/dw1` for a three-layer version of the network above. How many
   terms in the product?
3. Multiply fifty numbers drawn around 0.9, then around 1.1. This is vanishing
   and exploding gradients in two lines.
4. Explain, in one sentence each, why you cannot train on precision@1000 and
   what you do instead.
5. Take a loss from your own work and check its gradient. Did you find anything?

---

**Next:** [Lesson 05 — Optimisation and Conditioning](05-optimisation.md)
