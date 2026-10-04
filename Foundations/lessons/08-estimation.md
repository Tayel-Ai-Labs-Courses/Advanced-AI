# Lesson 08 — Bias, Variance and Estimation

**Goal:** separate the two reasons a model is wrong, because they have opposite
cures.

## What you will learn

- Bias and variance, measured separately
- The decomposition, and where the minimum sits
- Which cure applies to which problem
- Why this is the whole of model selection

---

## The decomposition, measured

```python
import numpy as np

def fit_poly(degree, n=20, noise=0.35, seed=0):
    r = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    y_true = np.sin(2.5 * x)
    y = y_true + r.normal(0, noise, n)
    coef = np.polyfit(x, y, degree)
    xt = np.linspace(-1, 1, 200)
    return np.polyval(coef, xt), np.sin(2.5 * xt)

print(f"{'degree':>8}{'bias^2':>10}{'variance':>11}{'total error':>13}")
xt = np.linspace(-1, 1, 200)
truth = np.sin(2.5 * xt)
for d in (0, 1, 3, 9, 15):
    preds = np.array([fit_poly(d, seed=s)[0] for s in range(60)])
    mean_pred = preds.mean(0)
    bias2 = ((mean_pred - truth) ** 2).mean()
    var = preds.var(0).mean()
    print(f"{d:>8}{bias2:>10.4f}{var:>11.4f}{bias2 + var:>13.4f}")
print("\nunderfit: high bias. Overfit: high variance. The sum is what you feel.")
```

```text
  degree    bias^2   variance  total error
       0    0.5947     0.0048       0.5995
       1    0.0809     0.0104       0.0912
       3    0.0011     0.0226       0.0236
       9    0.0006     0.0571       0.0577
      15    0.0140     1.3123       1.3263

underfit: high bias. Overfit: high variance. The sum is what you feel.
```

Sixty datasets, drawn from the same truth with different noise, fitted with
polynomials of increasing degree. Then, at every point:

- **bias²** — how far the *average* fit is from the truth
- **variance** — how much the fits differ *from each other*

Read the two columns in opposite directions:

**Degree 0** (a horizontal line): bias² **0.5947**, variance 0.0048. It is
consistently, reliably wrong. Every dataset gives nearly the same bad answer.

**Degree 15**: bias² 0.0140, variance **1.3123**. On average it is close to the
truth — but any individual fit is wild, because it chased its own noise. Its
variance is **270 times** the degree-3 model's.

**Degree 3 is the minimum**, at 0.0236 total. Not because its bias is lowest
(degree 9 has lower) but because the sum is.

---

## The shape

```python
for d in (0, 1, 2, 3, 5, 9, 15):
    preds = np.array([fit_poly(d, seed=s)[0] for s in range(60)])
    b2 = ((preds.mean(0) - truth) ** 2).mean()
    v = preds.var(0).mean()
    bar = "#" * int(40 * min(b2 + v, 0.6) / 0.6)
    print(f"degree {d:>2}  bias2 {b2:>7.4f}  var {v:>7.4f}  total {b2+v:>7.4f} {bar}")
```

```text
degree  0  bias2  0.5947  var  0.0048  total  0.5995 #######################################
degree  1  bias2  0.0809  var  0.0104  total  0.0912 ######
degree  2  bias2  0.0812  var  0.0169  total  0.0981 ######
degree  3  bias2  0.0011  var  0.0226  total  0.0236 #
degree  5  bias2  0.0005  var  0.0327  total  0.0331 ##
degree  9  bias2  0.0006  var  0.0571  total  0.0577 ###
degree 15  bias2  0.0140  var  1.3123  total  1.3263 ########################################
```

The classic U. Bias falls fast — with a bump at degree 2, where an even
polynomial fits an odd function badly — variance rises slowly and then explodes,
and the minimum sits at **degree 3, total 0.0236**. Degree 5 has *lower* bias
(0.0005 against 0.0011) and a worse total, because its variance grew faster than
its bias fell.

---

## Which cure

The two failures look identical from a single training run — both give a bad
score — and have **opposite** remedies.

| Symptom | Diagnosis | Cure |
|---|---|---|
| Train error high, test error similar | **High bias** (underfit) | A bigger model, more features, longer training, less regularisation |
| Train error low, test error much higher | **High variance** (overfit) | **More data**, regularisation, a simpler model, early stopping, augmentation |
| Both high, and unstable across seeds | High variance, small data | More data first; everything else second |

**The diagnostic is the gap between train and test.** One number, and it tells
you which direction to move — which is why
[Data-Science lesson 05](../../Data-Science/lessons/05-baselines-and-selection.md)
reports the fold spread beside the mean, and why
[Data-Security lesson 03](../../Data-Security-for-AI/lessons/03-memorisation.md)
uses the same gap as a **privacy** measurement: a memorising model is both
overfitted and leaky.

**More data only fixes variance.** It does nothing for bias — a linear model on
a curved truth stays wrong with a billion rows. That asymmetry is why "get more
data" is sometimes the right answer and sometimes an expensive way to avoid the
real problem.

---

## Where this is the whole game

Every model-selection decision in this track is this trade-off:

| Decision | Bias side | Variance side |
|---|---|---|
| Polynomial degree, tree depth | Too shallow | Too deep |
| Regularisation strength `C`, `alpha` | Too strong | Too weak |
| Number of features | Too few | Too many |
| LoRA rank | Too low | Too high |
| `k` in k-NN | Too large | Too small |
| Training time | Stopped too early | Stopped too late |
| Ensemble size | — | Reduces variance, not bias |

And the row that explains bagging: **averaging many high-variance models reduces
variance without touching bias.** That is the whole idea of a random forest —
and it is also why
[Data-Science lesson 05](../../Data-Science/lessons/05-baselines-and-selection.md)
measured an unconstrained random forest *losing* to logistic regression: with
1,929 positives, 300 memorising trees still share the same overfit, and
averaging does not cure a bias every member has.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Treating every bad score as overfitting | Half the time it is underfitting; the cures are opposite |
| More data for a bias problem | Expensive, and it does not help |
| A more complex model for a variance problem | Makes it worse |
| Not comparing train with test error | The one number that diagnoses it |
| Choosing the lowest-bias model | Degree 9 had lower bias and twice the error |
| Assuming ensembles fix everything | They reduce variance only |

---

## Exercises

1. Plot train and test error against complexity for your own model. Where is the
   minimum?
2. Diagnose your current model from the train/test gap. Which cure applies?
3. Double your training data and re-measure. Did the gap close? What does that
   tell you?
4. Take an overfitting model and fix it three ways — regularisation, simpler
   model, more data. Which was cheapest?
5. Run the membership-inference attack from Data-Security lesson 03 on your
   high-variance and low-variance models. Compare the AUCs.

---

**Done with the lessons.** Next: [Project 19](../Project-19/) — prove you can
use this, on a model you already have.

---

## Where to go next

| Next | Why |
|---|---|
| [This course's project](../Project-19/) | It is the assessment, and it is not optional |
| [Python](../../Python/) | Build something with the maths behind you |
| [Machine-Learning](../../Machine-Learning/) | Where every idea here gets used |
| [Optimization 01-07](../../Optimization/) | Gradients, conditioning and schedules, applied |
