# Lesson 09 — Evaluating RL Agents

**Goal:** report a number about an agent that somebody else could reproduce.

## What you will learn

- Why one seed is not a result
- Training reward against greedy evaluation
- How many seeds you actually need
- Estimating a policy's value from logged data, and when that breaks

---

## Setup

```python
import sys
sys.path.insert(0, "/tmp")            # cartpole.py, from lesson 07
import numpy as np
from cartpole import CartPole

BINS = [np.linspace(-2.4, 2.4, 4), np.linspace(-3, 3, 6),
        np.linspace(-0.21, 0.21, 12), np.linspace(-3, 3, 12)]

def discretise(s):
    return tuple(int(np.digitize(v, b)) for v, b in zip(s, BINS))

def train(episodes=1_500, alpha=0.1, gamma=0.99, seed=0):
    env = CartPole(seed=seed)
    rng = np.random.default_rng(seed)
    Q, lengths = {}, []
    for ep in range(episodes):
        eps = max(0.02, 1.0 - ep / (episodes * 0.5))
        s = discretise(env.reset())
        steps = 0
        for _ in range(500):
            q = Q.setdefault(s, np.zeros(2))
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(q))
            nxt_raw, r, done = env.step(a)
            nxt = discretise(nxt_raw)
            qn = Q.setdefault(nxt, np.zeros(2))
            Q[s][a] += alpha * (r + gamma * (0.0 if done else qn.max()) - Q[s][a])
            s = nxt
            steps += 1
            if done:
                break
        lengths.append(steps)
    return Q, np.array(lengths)

def evaluate(Q, episodes=100, seed=1234):
    """Greedy, no exploration, on held-out environment seeds."""
    env = CartPole(seed=seed)
    lengths = []
    for _ in range(episodes):
        s = discretise(env.reset())
        steps = 0
        for _ in range(500):
            a = int(np.argmax(Q.get(s, np.zeros(2))))
            nxt, r, done = env.step(a)
            s = discretise(nxt)
            steps += 1
            if done:
                break
        lengths.append(steps)
    return np.array(lengths)

print("defined")
```

```text
defined
```

---

## One seed is not a result

```python
agents = [train(seed=s) for s in range(20)]
final_train = np.array([a[1][-100:].mean() for a in agents])
final_eval = np.array([evaluate(a[0]).mean() for a in agents])

print("20 agents, identical code, different seeds")
print(f"  training reward, last 100 : min {final_train.min():.0f}  "
      f"median {np.median(final_train):.0f}  max {final_train.max():.0f}")
print(f"  greedy evaluation         : min {final_eval.min():.0f}  "
      f"median {np.median(final_eval):.0f}  max {final_eval.max():.0f}")
print(f"  best/median ratio (eval)  : {final_eval.max() / np.median(final_eval):.2f}x")
```

```text
20 agents, identical code, different seeds
  training reward, last 100 : min 107  median 135  max 227
  greedy evaluation         : min 37  median 142  max 346
  best/median ratio (eval)  : 2.44x
```

Twenty runs of the same program. The best scores **346**, the median **142**,
the worst **37** — a range of nearly ten to one, with no difference in the code
at all.

A paper, a blog post or a stand-up update reporting "our agent reaches 346
steps" is not lying about the number. It is reporting the maximum of a
distribution and calling it the agent. **Best-of-N is not a measurement, it is
a selection**, and repeating it on a new seed will not reproduce.

Report the median and the range. If you must report one number, report the
median — the mean is dragged around by the lucky run.

---

## Training reward is not performance

```python
print(f"{'':<22}{'training':>12}{'greedy eval':>14}{'gap':>10}")
for label, idx in [("median agent", int(np.argsort(final_eval)[10])),
                   ("best agent", int(np.argmax(final_eval))),
                   ("worst agent", int(np.argmin(final_eval)))]:
    print(f"{label:<22}{final_train[idx]:>12.1f}{final_eval[idx]:>14.1f}"
          f"{final_eval[idx] - final_train[idx]:>+10.1f}")
```

```text
                          training   greedy eval       gap
median agent                 134.4         147.7     +13.3
best agent                   227.3         345.5    +118.2
worst agent                  131.6          36.8     -94.8
```

Look at the worst agent. Its **training reward is 131.6 — indistinguishable
from the median agent's 134.4** — and its actual greedy performance is **36.8**,
a quarter as good. Watching the training curve would have given no warning at
all.

The two numbers measure different things. Training reward includes exploratory
actions (2% random here) and is collected on the same environment seeds the
agent learned on. Greedy evaluation turns exploration off and uses held-out
seeds. The gap can go either way: exploration usually costs you reward (the best
agent gains 118 when it stops exploring), but an agent that only works on the
states it happened to train on loses.

**Evaluate separately, greedily, on held-out environment seeds.** It costs a
hundred episodes and it is the difference between a number and a guess.

---

## How many seeds?

```python
rng = np.random.default_rng(0)
for n in (1, 3, 5, 10, 20):
    means = [np.mean(rng.choice(final_eval, n, replace=True)) for _ in range(2_000)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"{n:>3} seeds: reported mean would land in [{lo:.0f}, {hi:.0f}]"
          f"   width {hi - lo:.0f}")
```

```text
  1 seeds: reported mean would land in [37, 346]   width 309
  3 seeds: reported mean would land in [87, 237]   width 151
  5 seeds: reported mean would land in [105, 215]   width 110
 10 seeds: reported mean would land in [118, 196]   width 78
 20 seeds: reported mean would land in [128, 185]   width 57
```

This is a bootstrap over the 20 agents: if you ran `n` seeds and reported the
mean, where would that number land?

**With one seed, anywhere between 37 and 346.** With three — the number most
often used — still a 151-wide window. You need about **ten seeds before the
interval is narrower than the effect sizes people usually claim**.

The practical rule: before comparing two RL methods, run each five to ten times
and compute this interval. If the intervals overlap, you have not measured a
difference — exactly the discipline from Data-Science lesson 05, where the
paired fold interval did the same job.

---

## Estimating a policy you did not run

Sometimes you cannot deploy a candidate policy to measure it — the actions cost
money. You have logs from the policy currently running, and you want the value
of a different one. **Importance sampling** reweights the logged rewards by how
much more likely the new policy was to take each action.

```python
TRUE = np.array([0.30, 0.55, 0.45])

def collect(behaviour, n, seed):
    rng = np.random.default_rng(seed)
    a = rng.choice(3, size=n, p=behaviour)
    r = (rng.random(n) < TRUE[a]).astype(float)
    return a, r

def ips(a, r, behaviour, target):
    w = target[a] / behaviour[a]
    return float((w * r).mean()), float((w * r).sum() / w.sum())

target = np.array([0.0, 1.0, 0.0])          # "always show offer 1"
print(f"true value of the target policy: {TRUE @ target:.3f}\n")
print(f"{'behaviour policy':<28}{'IPS':>10}{'self-norm':>12}{'ESS':>8}{'error':>9}")
for name, behaviour in [("uniform  [.33 .33 .33]", np.array([1/3, 1/3, 1/3])),
                        ("skewed   [.80 .10 .10]", np.array([0.8, 0.1, 0.1])),
                        ("near-blind [.98 .01 .01]", np.array([0.98, 0.01, 0.01]))]:
    est1, est2, ess = [], [], []
    for seed in range(200):
        a, r = collect(behaviour, 2_000, seed)
        i, s = ips(a, r, behaviour, target)
        w = target[a] / behaviour[a]
        ess.append(w.sum() ** 2 / (w ** 2).sum())
        est1.append(i)
        est2.append(s)
    print(f"{name:<28}{np.mean(est1):>10.3f}{np.mean(est2):>12.3f}"
          f"{np.mean(ess):>8.0f}{np.std(est2):>9.3f}")
```

```text
true value of the target policy: 0.550

behaviour policy                   IPS   self-norm     ESS    error
uniform  [.33 .33 .33]           0.549       0.550     666    0.018
skewed   [.80 .10 .10]           0.548       0.548     200    0.034
near-blind [.98 .01 .01]         0.561       0.565      20    0.122
```

The estimator is **unbiased in all three rows** — every mean is close to the
true 0.550. What changes is how much you can trust any single estimate.

**ESS** is the effective sample size: how many independent observations those
2,000 logged rows are really worth for this question. Under the near-blind
logging policy it is **20**. The logs contain 2,000 rows and answer the question
as well as twenty would, because the target action was only shown 1% of the
time.

And the error column follows: 0.018 with uniform logging, **0.122** with
near-blind logging — nearly seven times wider. An estimate of "0.565 give or
take 0.12" cannot distinguish offer 1 from offer 2.

This is lesson 01's logged-data problem stated precisely. Two consequences for
production:

1. **Always log the propensity.** Store the probability the behaviour policy
   assigned to the action it took. Without it, `behaviour[a]` is unknown and
   none of this is computable.
2. **Keep a floor on exploration.** A policy that shows one arm 98% of the time
   makes every future counterfactual question unanswerable, however much data
   you accumulate.

Use the **self-normalised** estimator (the second column) by default: it is
very slightly biased and much better behaved when the weights are large.

---

## An evaluation protocol

```text
1. Fix and record seeds        training seeds, evaluation seeds, separate
2. Train N >= 10 agents        identical code, different seeds
3. Evaluate greedily           100 episodes each, HELD-OUT environment seeds
4. Report median and range     not the max, not one run
5. Bootstrap the interval      before claiming a difference
6. State the budget            episodes, environment steps, wall clock
7. Log the environment         version, physics constants, reward function
```

Step 6 is the one that is missing from most comparisons. "Method A beats method
B" means nothing without the sample budget: lesson 08's REINFORCE reached 478
steps in 600 episodes, and lesson 07's DQN reached 86 in 300. Comparing those
two numbers as if they answered the same question would be wrong, and it is the
most common error in RL write-ups.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Reporting one seed | The range was 37 to 346 for identical code |
| Reporting the best of N | A selection presented as a measurement |
| Using the training curve as performance | The worst agent's training reward looked normal |
| Evaluating on the training environment seeds | Measures memorisation |
| Three seeds | A 151-wide interval on a 142 median |
| Comparing methods at different budgets | 600 episodes against 300 is not a comparison |
| Off-policy evaluation without propensities | `behaviour[a]` is unknown; nothing is computable |
| Trusting an IPS estimate without ESS | 2,000 rows behaving like 20 |

---

## Exercises

1. Compute the bootstrap interval for the **median** instead of the mean. Is it
   narrower at n=5? Why might the median be the better statistic to report?
2. Evaluate the 20 agents on the *training* environment seed instead of the
   held-out one. How much does the reported performance rise?
3. Build the IPS table for a target policy of `[0.1, 0.8, 0.1]` instead of a
   deterministic one. What happens to ESS, and why is a stochastic target
   easier to evaluate?
4. Add a propensity column to lesson 01's logged-data experiment and estimate
   the value of "always show offer 2" from it. Compare with the truth of 0.45.
5. Take any two agents from the 20 and run a paired comparison across the 100
   evaluation episodes. How large does the gap have to be before the interval
   excludes zero?

---

**Next:** [Lesson 10 — RL in the Real World](10-rl-in-the-real-world.md)
