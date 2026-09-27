# Lesson 06 — Reward Design and Exploration

**Goal:** write a reward the agent cannot cheat, and get it to find a reward
that is far away.

## What you will learn

- Sparse rewards, and why they are hard
- Potential-based shaping, which is safe
- Naive shaping, which the agent will farm
- Three exploration strategies on a problem eps-greedy cannot solve

---

## Setup

```python
import sys
sys.path.insert(0, "/tmp")            # gridworld.py, from lesson 03
import numpy as np
from gridworld import GridWorld, ACTIONS

def sample_step(grid, s, a, rng):
    outs = grid.transitions(s, a)
    i = rng.choice(len(outs), p=[p for p, _, _ in outs])
    _, nxt, r = outs[i]
    return nxt, r

def q_learn(grid, episodes, alpha=0.5, eps=0.1, gamma=0.99, seed=0,
            shape=None, bonus=0.0):
    rng = np.random.default_rng(seed)
    Q = {(s, a): 0.0 for s in grid.states for a in ACTIONS}
    counts = {(s, a): 0 for s in grid.states for a in ACTIONS}
    steps_per_ep = []
    for _ in range(episodes):
        s = grid.start
        for step in range(200):
            if s in grid.terminals:
                break
            if rng.random() < eps:
                a = ACTIONS[rng.integers(4)]
            else:
                scores = [Q[(s, b)] + bonus / np.sqrt(counts[(s, b)] + 1)
                          for b in ACTIONS]
                a = ACTIONS[int(np.argmax(scores))]
            nxt, r = sample_step(grid, s, a, rng)
            counts[(s, a)] += 1
            if shape is not None:
                r = r + gamma * shape(nxt) - shape(s)     # potential-based
            target = r + gamma * (0.0 if nxt in grid.terminals
                                  else max(Q[(nxt, b)] for b in ACTIONS))
            Q[(s, a)] += alpha * (target - Q[(s, a)])
            s = nxt
        steps_per_ep.append(step + 1)
    return Q, np.array(steps_per_ep)

def greedy_policy(grid, Q):
    return {s: ACTIONS[int(np.argmax([Q[(s, b)] for b in ACTIONS]))]
            for s in grid.states}

def score(policy, env, episodes=200, limit=100, seed=0):
    """Run a greedy policy in the REAL environment. Shaping does not count."""
    rng = np.random.default_rng(seed)
    rets, wins = [], 0
    for _ in range(episodes):
        s = env.start
        total = 0.0
        for _ in range(limit):
            if s in env.terminals:
                wins += env.terminals[s] > 0
                break
            nxt, r = sample_step(env, s, policy[s], rng)
            total += r
            s = nxt
        rets.append(total)
    return np.mean(rets), wins / episodes

print("ready")
```

```text
ready
```

The important line is in `score`: the learned policy is always evaluated in the
**real** environment, with the real reward. Any shaping is a training aid, and
measuring an agent with its own training bonus included is how people convince
themselves a broken agent works.

---

## Sparse rewards

A **sparse** reward is zero almost everywhere. Take the step cost away and the
grid gives no feedback at all until the agent stumbles into a terminal.

```python
def manhattan_to_goal(s):
    """Potential: higher is closer to the +1 at (0, 3)."""
    return -(abs(s[0] - 0) + abs(s[1] - 3))

def q_learn_naive_bonus(grid, episodes, seed=0, alpha=0.5, eps=0.1,
                        gamma=0.99, scale=0.1):
    """A bonus for progress, written the way it first occurs to everyone."""
    rng = np.random.default_rng(seed)
    Q = {(s, a): 0.0 for s in grid.states for a in ACTIONS}
    for _ in range(episodes):
        s = grid.start
        for _ in range(200):
            if s in grid.terminals:
                break
            a = (ACTIONS[rng.integers(4)] if rng.random() < eps
                 else ACTIONS[int(np.argmax([Q[(s, b)] for b in ACTIONS]))])
            nxt, r = sample_step(grid, s, a, rng)
            r = r + scale * (manhattan_to_goal(nxt) - manhattan_to_goal(s)) + scale
            target = r + gamma * (0.0 if nxt in grid.terminals
                                  else max(Q[(nxt, b)] for b in ACTIONS))
            Q[(s, a)] += alpha * (target - Q[(s, a)])
            s = nxt
    return Q

dense = GridWorld(step_reward=-0.04)
sparse = GridWorld(step_reward=0.0)

print(f"{'setup':<26}{'mean return':>13}{'reached goal':>14}")
for name, g, kw in [("dense (-0.04/step)", dense, {}),
                    ("sparse (0/step)", sparse, {}),
                    ("sparse + potential shaping", sparse,
                     {"shape": manhattan_to_goal})]:
    r, w = [], []
    for seed in range(20):
        Q, _ = q_learn(g, 300, seed=seed, **kw)
        a, b = score(greedy_policy(g, Q), dense, seed=seed)
        r.append(a)
        w.append(b)
    print(f"{name:<26}{np.mean(r):>13.3f}{np.mean(w):>14.1%}")

r, w = [], []
for seed in range(20):
    Qn = q_learn_naive_bonus(sparse, 300, seed=seed)
    a, b = score(greedy_policy(sparse, Qn), dense, seed=seed)
    r.append(a)
    w.append(b)
print(f"{'sparse + naive bonus':<26}{np.mean(r):>13.3f}{np.mean(w):>14.1%}")
print(f"{'optimal policy':<26}{0.745:>13.3f}{'~100%':>14}")
```

```text
setup                       mean return  reached goal
dense (-0.04/step)                0.393         94.1%
sparse (0/step)                   0.162         89.4%
sparse + potential shaping        0.522         95.3%
sparse + naive bonus             -3.972          0.0%
optimal policy                    0.745         ~100%
```

Four rows, and the fourth is the one to remember.

**Sparse is worse than dense** (0.162 against 0.393). With no step cost the
agent has no reason to hurry, so it learns a policy that eventually arrives —
89.4% of the time within 100 steps — and wanders on the way.

**Potential-based shaping is best** (0.522). It gave the agent a hint about
direction and cost it nothing in correctness.

**The naive bonus destroyed the agent: 0.0% ever reach the goal.**

---

## Why the naive bonus fails and the other one does not

The difference is one line.

```text
potential-based   r + gamma * PHI(s') - PHI(s)
naive             r + scale * (PHI(s') - PHI(s)) + scale
```

Potential-based shaping is a telescoping sum: over any trajectory the added
terms cancel except at the ends, so **the ranking of policies is unchanged**.
That is a theorem (Ng, Harada & Russell, 1999), and it is the only shaping you
should write without thinking hard.

The naive version adds `+ scale` on every step — a small reward for existing.
The agent found that 0.1 per step forever beats +1 once, and the optimal
behaviour under the reward *as written* is to never finish. That is exactly the
`+0.05` agent from lesson 04, arrived at by accident instead of on purpose,
which is how it happens in real projects.

Note the mean return of **-3.972**: negative, because in the real environment
those 100 wandering steps each cost -0.04. The agent's own training reward was
strongly positive the whole time. **If we had reported the training reward, this
would have been the best agent in the table.**

| Shaping style | Safe? | Note |
|---|---|---|
| `gamma * PHI(s') - PHI(s)` | **Yes** | Policy-invariant, proven |
| A bonus per step | No | Pays for existing; agent stops finishing |
| A bonus for progress, not differenced | No | Farms the oscillation |
| A bonus for a subgoal | Usually no | Agent visits the subgoal repeatedly |
| Penalty for time | Yes | It is just a step cost |

The habit: **write the reward, then ask what the laziest possible exploit
is.** Then run it and look at the *behaviour*, not the score.

---

## Exploration when the reward is far away

eps-greedy explores by taking random actions. In a grid that is enough. Here is
a problem where it is not.

```python
class Lock:
    """A chain of N rooms. In each room one of two doors advances; the other
    sends you back to the start. Reward +1 only in the final room."""
    def __init__(self, n=10, seed=0):
        self.n = n
        rng = np.random.default_rng(seed)
        self.correct = rng.integers(0, 2, n)      # which door advances, per room

    def step(self, s, a):
        if a == self.correct[s]:
            nxt = s + 1
            if nxt == self.n:
                return nxt, 1.0, True
            return nxt, 0.0, False
        return 0, 0.0, False                       # one wrong door: back to room 0
```

To reach the reward by luck you must pick correctly **ten times in a row**:
one chance in 1,024, and any mistake resets you. This is the structure of every
hard exploration problem — a long sequence where partial progress earns nothing.

```python
def train_lock(kind, n=10, steps=200_000, alpha=0.5, eps=0.1, gamma=0.99,
               bonus=1.0, seed=0):
    env = Lock(n, seed=0)
    rng = np.random.default_rng(seed)
    Q = np.zeros((n + 1, 2))
    if kind == "optimistic":
        Q[:] = 1.0                                 # assume everything is great
    counts = np.zeros((n + 1, 2))
    s, solved_at, successes = 0, None, 0
    for t in range(steps):
        if kind == "eps":
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[s]))
        elif kind == "optimistic":
            a = int(np.argmax(Q[s]))
        else:                                      # count-based curiosity
            a = int(np.argmax(Q[s] + bonus / np.sqrt(counts[s] + 1)))
        nxt, r, done = env.step(s, a)
        counts[s, a] += 1
        Q[s, a] += alpha * (r + gamma * (0.0 if done else Q[nxt].max()) - Q[s, a])
        if done:
            successes += 1
            solved_at = t if solved_at is None else solved_at
            s = 0
        else:
            s = nxt
    return successes, solved_at

print(f"{'strategy':<14}{'first success':>15}{'successes':>12}{'runs solved':>14}")
for kind in ("eps", "optimistic", "count-based"):
    firsts, totals = [], []
    for seed in range(10):
        succ, first = train_lock(kind, seed=seed)
        firsts.append(first if first is not None else np.nan)
        totals.append(succ)
    reached = int(np.sum(~np.isnan(firsts)))
    label = "never" if reached == 0 else f"{np.nanmean(firsts):,.0f} steps"
    print(f"{kind:<14}{label:>15}{np.mean(totals):>12.0f}{f'{reached}/10':>14}")
```

```text
strategy        first success   successes   runs solved
eps              93,809 steps        1506          4/10
optimistic           92 steps       19984         10/10
count-based       1,043 steps       19698         10/10
```

**eps-greedy solved it in 4 runs out of 10**, and when it did, it took 94,000
steps to get there. In the other six runs, 200,000 steps produced nothing at
all. Random exploration does not find a reward behind ten correct decisions; it
is a random walk that gets reset every time it errs.

**Optimistic initialisation solved every run in 92 steps** — a thousand times
faster than eps-greedy, from one line: `Q[:] = 1.0`. Initialising the values
above anything achievable makes every untried action look attractive, so the
agent systematically tries everything once and only stops when reality has
disappointed it. Exploration becomes *directed* rather than random.

**Count-based curiosity also solved every run**, in 1,043 steps. The bonus
`1/sqrt(n(s,a))` explicitly rewards novelty, which is the idea behind the deep
RL exploration literature — and here it is eleven times slower than the
one-liner.

This is the third time in this course that the simplest method has won
(logistic-style simplicity in lesson 02's Thompson, value iteration in 04, and
now optimistic init). Try the one-liner before the paper.

| Strategy | Cost | Works when |
|---|---|---|
| eps-greedy | One line | The reward is nearby. **Not** behind a sequence |
| Decaying eps | One line | Same, plus you want to stop paying the tax |
| **Optimistic init** | One line | Rewards are bounded and you know the bound |
| Count-based bonus | A counter per state-action | Large state spaces, where optimism is hard to set |
| Curiosity / prediction error | A whole extra model | Continuous states, images |

Optimistic initialisation has a catch worth knowing: it needs an upper bound on
the return. Set it too low and it does nothing; set it enormous and the agent
explores for a very long time. Here +1 is the real maximum, so `1.0` is exactly
right — which is a luxury a shaped or unbounded reward takes away from you.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Reporting the shaped reward | The broken agent had the best training score in the table |
| A per-step "alive" bonus | 0.0% of episodes ever finished |
| Shaping that is not a potential difference | The agent farms the oscillation |
| eps-greedy on a sequential-reward problem | 6 of 10 runs never saw the reward in 200,000 steps |
| Reaching for curiosity modules first | Optimistic init was 11x faster and one line |
| Optimistic init with no known bound | Either useless or endless |
| Judging by score instead of watching the policy | Both failures here are invisible in the number you optimise |

---

## Exercises

1. Set `scale=0.001` in the naive bonus. Does it still break the agent? Find
   the scale at which the goal becomes worth reaching, and explain why that
   number is fragile.
2. Rewrite the naive bonus as a proper potential difference and confirm you
   recover the shaped row's performance.
3. Lengthen the lock to `n=15`. Does count-based exploration still solve it in
   200,000 steps? Does optimistic init?
4. Set the optimistic value to `Q[:] = 100.0`. How many steps until the first
   success now, and what is the agent doing for all of them?
5. Add a decoy: `+0.3` in room 5, terminal. Which of the three strategies gets
   stuck on it, and which still finds the +1?

---

**Next:** [Lesson 07 — Function Approximation](07-function-approximation.md)
