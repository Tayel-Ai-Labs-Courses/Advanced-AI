# Lesson 02 — Bandits and the Explore/Exploit Trade-off

**Goal:** spend a fixed budget of decisions well, and measure "well" with
regret instead of reward.

## What you will learn

- Regret, and why it is the right metric
- Four strategies, measured on the same problem
- How regret grows, which is what separates them
- The tuning constant that matters more than the algorithm

---

## Regret

Reward alone cannot tell you whether an agent is good, because it depends on
how generous the problem is. **Regret** is what you lost against always
choosing the best action:

```text
regret(T) = T * p_best  -  sum of expected rewards actually collected
```

A perfect agent has regret 0. An agent that never learns has regret growing
linearly with T. Everything interesting is in between.

```python
import numpy as np

class Bandit:
    def __init__(self, probabilities, seed=0):
        self.p = np.asarray(probabilities, float)
        self.rng = np.random.default_rng(seed)
        self.best = self.p.max()

    def step(self, a):
        return float(self.rng.random() < self.p[a])

def epsilon_greedy(eps):
    def policy(t, counts, values, rng):
        if rng.random() < eps:
            return int(rng.integers(len(values)))
        return int(np.argmax(values))
    return policy

def decaying_epsilon(c=1.0):
    def policy(t, counts, values, rng):
        if rng.random() < min(1.0, c / (t + 1) ** 0.5):
            return int(rng.integers(len(values)))
        return int(np.argmax(values))
    return policy

def ucb(c=2.0):
    def policy(t, counts, values, rng):
        if (counts == 0).any():
            return int(np.argmin(counts))          # try everything once
        return int(np.argmax(values + c * np.sqrt(np.log(t + 1) / counts)))
    return policy

def run(policy, probabilities, steps, seed):
    env = Bandit(probabilities, seed=seed)
    rng = np.random.default_rng(seed + 99)
    k = len(probabilities)
    counts, values = np.zeros(k), np.zeros(k)
    total = regret = 0.0
    for t in range(steps):
        a = policy(t, counts, values, rng)
        r = env.step(a)
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]    # incremental mean
        total += r
        regret += env.best - env.p[a]
    return total, regret, counts

def run_thompson(probabilities, steps, seed):
    env = Bandit(probabilities, seed=seed)
    rng = np.random.default_rng(seed + 99)
    k = len(probabilities)
    alpha, beta = np.ones(k), np.ones(k)            # Beta(1,1) = uniform prior
    counts = np.zeros(k)
    total = regret = 0.0
    for t in range(steps):
        a = int(np.argmax(rng.beta(alpha, beta)))   # sample, then act greedily
        r = env.step(a)
        alpha[a] += r
        beta[a] += 1 - r
        counts[a] += 1
        total += r
        regret += env.best - env.p[a]
    return total, regret, counts
```

The update line is worth pausing on — `values[a] += (r - values[a]) / counts[a]`.

That is the running mean, written as **old estimate + step_size x (target -
old estimate)**. Every algorithm in this course is that shape. Replace
`1/counts[a]` with a constant and you get an agent that tracks a changing
world instead of averaging over all history.

---

## Six strategies on the same problem

```python
P = [0.30, 0.55, 0.45]
STEPS, SEEDS = 1_000, 300

strategies = {
    "greedy (eps=0)": epsilon_greedy(0.0),
    "eps-greedy 0.01": epsilon_greedy(0.01),
    "eps-greedy 0.10": epsilon_greedy(0.10),
    "decaying eps": decaying_epsilon(1.0),
    "UCB (c=2)": ucb(2.0),
}

print(f"{'strategy':<18}{'reward':>9}{'regret':>9}{'best-arm %':>12}")
for name, policy in strategies.items():
    rows = [run(policy, P, STEPS, s) for s in range(SEEDS)]
    rewards = np.array([r[0] for r in rows])
    regrets = np.array([r[1] for r in rows])
    share = np.array([r[2][1] / STEPS for r in rows])
    print(f"{name:<18}{rewards.mean():>9.1f}{regrets.mean():>9.1f}{share.mean():>12.1%}")

rows = [run_thompson(P, STEPS, s) for s in range(SEEDS)]
print(f"{'Thompson':<18}{np.mean([r[0] for r in rows]):>9.1f}"
      f"{np.mean([r[1] for r in rows]):>9.1f}"
      f"{np.mean([r[2][1] / STEPS for r in rows]):>12.1%}")
print(f"{'oracle':<18}{STEPS * max(P):>9.1f}{0.0:>9.1f}{1.0:>12.1%}")
```

```text
strategy             reward   regret  best-arm %
greedy (eps=0)        299.7    250.0        0.0%
eps-greedy 0.01       429.8    120.1       40.4%
eps-greedy 0.10       514.9     34.9       76.7%
decaying eps          521.5     28.3       77.0%
UCB (c=2)             489.1     60.3       59.1%
Thompson              529.9     19.2       85.0%
oracle                550.0      0.0      100.0%
```

**Pure greedy scores 0.0% on the best arm.** Not "rarely" — never, in 300 runs.
All estimates start at 0, `argmax` of a tie picks index 0, arm 0 pays out
sometimes, so its estimate rises above 0 and the agent never tries anything
else. It converges to the *worst* arm with perfect consistency.

This is not a contrived bug. It is what happens to any system that acts on its
current best estimate with no mechanism for doubt, and it is why "just deploy
the argmax" fails in the real world.

**eps=0.01 is not cautious, it is broken differently.** 120.1 regret, best arm
40% of the time. One percent exploration on 1,000 steps is ten exploratory
pulls spread across three arms — not enough to distinguish 0.55 from 0.45.

**Thompson sampling wins**, and it has no exploration parameter at all. It
keeps a Beta posterior per arm, draws one sample from each, and plays the
winner. Arms it is unsure about have wide posteriors and sometimes draw high,
so they get tried; arms it has ruled out have narrow posteriors and stop
appearing. Exploration falls out of the uncertainty instead of being scheduled.

**And UCB is fourth**, behind plain eps-greedy. Which brings us to the number
that actually decided this table.

---

## The constant matters more than the algorithm

```python
print(f"{'':<12}{'regret @1k':>12}{'regret @10k':>13}")
for c in (0.25, 0.5, 1.0, 2.0):
    a = np.mean([run(ucb(c), P, 1_000, s)[1] for s in range(60)])
    b = np.mean([run(ucb(c), P, 10_000, s)[1] for s in range(60)])
    print(f"UCB c={c:<6}{a:>12.1f}{b:>13.1f}")
```

```text
              regret @1k  regret @10k
UCB c=0.25          22.2        104.2
UCB c=0.5           16.6         28.1
UCB c=1.0           33.0         81.1
UCB c=2.0           62.5        228.3
```

**c=0.5 has one eighth the regret of c=2.0** at 10,000 steps — and c=2.0 is
the value printed in most tutorials, including the first version of the table
above. Tuned properly, UCB (28.1) is competitive with Thompson (34.0 at the
same horizon); left at the default, it loses to a one-line eps-greedy.

The reason is a units mismatch. The bonus `c * sqrt(log t / n)` is added to a
value estimate living in [0, 1]. With c=2 the bonus dwarfs the real difference
between arms for a long time, so the agent explores far past the point of
usefulness. c=2 is a sensible default when rewards span a wider range.

Note c=0.25 is also bad, and worse at 10k than at 1k — too little exploration
means it can settle on the wrong arm and never revisit. **The cost is not
monotone in either direction**, which is why this table is a measurement and
not a rule.

---

## How regret grows

The mean at one horizon hides the property that matters.

```python
print(f"{'steps':>8}{'eps-greedy 0.10':>18}{'UCB c=0.5':>12}{'Thompson':>11}")
for steps in (100, 1_000, 10_000):
    print(f"{steps:>8}"
          f"{np.mean([run(epsilon_greedy(0.10), P, steps, s)[1] for s in range(60)]):>18.1f}"
          f"{np.mean([run(ucb(0.5), P, steps, s)[1] for s in range(60)]):>12.1f}"
          f"{np.mean([run_thompson(P, steps, s)[1] for s in range(60)]):>11.1f}")
```

```text
   steps   eps-greedy 0.10   UCB c=0.5   Thompson
     100              11.6         5.5        6.4
    1000              35.5        16.6       19.5
   10000             145.4        28.1       34.0
```

Read the columns down, not across.

**eps-greedy grows linearly**: 11.6, 35.5, 145.4 — roughly ten times the steps,
ten times the regret. It must, because it explores at a fixed 10% forever. At
step nine million it is still throwing away one decision in ten on arms it
settled years ago.

**UCB and Thompson grow logarithmically**: 5.5 to 16.6 to 28.1 while the
horizon grows a hundredfold. They stop exploring when the evidence is in.

At 100 steps the three are nearly identical, so a short experiment cannot tell
them apart. **Run your comparison at the horizon you will actually deploy at.**

| Strategy | Regret | Tuning | Use when |
|---|---|---|---|
| Greedy | Linear, terrible | none | Never |
| eps-greedy | Linear | eps | A baseline, or a genuinely changing world |
| Decaying eps | Sublinear | decay rate | Simple and good enough |
| UCB | Logarithmic | **c, and it matters** | You want a deterministic rule |
| **Thompson** | Logarithmic | none | **Default choice** |

---

## When the arms are close

```python
print(f"{'':<18}{'eps-greedy':>12}{'UCB c=0.5':>12}{'Thompson':>11}")
for probs, label in [([0.30, 0.55, 0.45], "easy  gap 0.10"),
                     ([0.50, 0.52, 0.51], "hard  gap 0.01")]:
    print(f"{label:<18}"
          f"{np.mean([run(epsilon_greedy(0.10), probs, 1_000, s)[1] for s in range(200)]):>12.2f}"
          f"{np.mean([run(ucb(0.5), probs, 1_000, s)[1] for s in range(200)]):>12.2f}"
          f"{np.mean([run_thompson(probs, 1_000, s)[1] for s in range(200)]):>11.2f}")
```

```text
                    eps-greedy   UCB c=0.5   Thompson
easy  gap 0.10           35.68       16.55      20.19
hard  gap 0.01            9.85        8.19       8.18
```

On the hard problem all three are within 1.7 of each other, and **every one of
them has lower regret than on the easy problem**. That looks backwards until
you see what regret measures: when the arms are nearly identical, choosing
wrongly barely costs anything. Total regret is small because the *mistakes* are
cheap, not because the agent is smart.

The practical conclusion is the one nobody wants: **if your variants differ by
1%, the choice of bandit algorithm is irrelevant.** What you need is more
traffic or a bigger idea. This is the same arithmetic as lesson 03 of the
Data-Analysis Advanced track — detecting a small effect is expensive no matter
which method you use.

---

## Contextual bandits, in one paragraph

Real problems come with a customer attached. A **contextual bandit** picks the
action given a state, but still assumes the action has no lasting effect: show
offer B, observe the click, the next customer arrives unaffected. The standard
recipe is one model per arm predicting reward from context, plus Thompson or
UCB over the models' uncertainty. If your action *does* change what happens
next — the customer's next visit, the next page they see — you have left bandit
territory, and lesson 03 begins.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Deploying the greedy argmax | Locked onto the worst arm in 100% of runs |
| A tiny exploration rate as "being careful" | eps=0.01 had 6x the regret of eps=0.10 |
| Copying `c=2` from a tutorial | 8x the regret of `c=0.5` at 10k steps |
| Comparing algorithms at 100 steps | All three are identical there |
| Reporting reward instead of regret | Reward depends on the problem's generosity |
| Constant eps forever | Linear regret; you pay 10% tax on every future decision |
| One run | Every row here is 60-300 runs; one run tells you nothing (lesson 09) |

---

## Exercises

1. Make the world change: after 500 steps, swap the probabilities to
   `[0.55, 0.30, 0.45]`. Which strategy adapts, and which never recovers?
   Then fix the loser by replacing `1/counts[a]` with a constant step size.
2. Add 10 arms instead of 3 and rerun the table. Which strategies degrade most,
   and why does the `(counts == 0).any()` line in UCB start to hurt?
3. Implement optimistic initialisation: set `values = np.ones(k)` with pure
   greedy. Compare it to eps-greedy. Why does it work, and when does it stop
   working?
4. Give Thompson a wrong prior — `Beta(1, 20)` on every arm — and report the
   regret. How many steps does it take to recover?

---

**Next:** [Lesson 03 — Markov Decision Processes](03-mdps.md)
