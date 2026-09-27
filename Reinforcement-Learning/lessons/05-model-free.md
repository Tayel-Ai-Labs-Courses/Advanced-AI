# Lesson 05 — Learning Without a Model

**Goal:** learn from experience when nobody hands you `transitions()` — which
is every real problem.

## What you will learn

- Sampling episodes instead of summing over probabilities
- Monte Carlo and TD(0), measured against the exact answer
- Q-learning and SARSA
- The cliff: why the algorithm that learns the optimal path earns less

---

## No model, only experience

Lesson 04's algorithms needed `grid.transitions(s, a)` — the probability of
every outcome. You will not have that for a customer, a warehouse or a game.
What you can do is **act, and see what happens**.

```python
import sys
sys.path.insert(0, "/tmp")
import numpy as np
from gridworld import GridWorld, ACTIONS, show

grid = GridWorld()

def sample_step(grid, s, a, rng):
    """One sampled transition. The agent never sees the probabilities."""
    outcomes = grid.transitions(s, a)
    i = rng.choice(len(outcomes), p=[p for p, _, _ in outcomes])
    _, nxt, r = outcomes[i]
    return nxt, r

manual = {s: "right" for s in grid.states}
manual[(2, 0)] = "up"
manual[(1, 0)] = "up"
manual[(0, 0)] = "right"
manual[(2, 1)] = "left"
manual[(2, 2)] = "up"
manual[(2, 3)] = "left"
manual[(1, 2)] = "up"

def episode(grid, policy, rng, limit=200):
    s = grid.start
    trace = []
    for _ in range(limit):
        if s in grid.terminals:
            break
        a = policy[s]
        nxt, r = sample_step(grid, s, a, rng)
        trace.append((s, a, r))
        s = nxt
    return trace

rng = np.random.default_rng(3)
tr = episode(grid, manual, rng)
print(f"length {len(tr)} steps, return {sum(r for _, _, r in tr):+.2f}")
print("first 5:", [(s, a, round(r, 2)) for s, a, r in tr[:5]])

returns = [sum(r for _, _, r in episode(grid, manual, np.random.default_rng(s)))
           for s in range(500)]
lengths = [len(episode(grid, manual, np.random.default_rng(s))) for s in range(500)]
print(f"over 500 episodes: mean return {np.mean(returns):+.3f}  "
      f"std {np.std(returns):.3f}  mean length {np.mean(lengths):.1f}")
print("DP answer from lesson 03: +0.745")
```

```text
length 6 steps, return +0.80
first 5: [((2, 0), 'up', -0.04), ((1, 0), 'up', -0.04), ((0, 0), 'right', -0.04), ((0, 0), 'right', -0.04), ((0, 1), 'right', -0.04)]
over 500 episodes: mean return +0.735  std 0.275  mean length 6.8
DP answer from lesson 03: +0.745
```

Two things to notice. The third and fourth steps are both `(0, 0) -> right` —
the first one slipped and went nowhere. And the standard deviation of the
return is **0.275**, seven times the mean's distance from the true value. One
episode tells you almost nothing; five hundred get you to +0.735 against a true
+0.745.

**This is the tax you pay for not having a model**: lesson 04 computed +0.745
exactly, in 23 sweeps, with no randomness at all.

---

## Monte Carlo: average the returns

```python
def monte_carlo(grid, policy, episodes, seed=0, gamma=1.0):
    rng = np.random.default_rng(seed)
    returns = {s: [] for s in grid.states}
    V = {s: 0.0 for s in grid.states}
    for _ in range(episodes):
        tr = episode(grid, policy, rng)
        G = 0.0
        first_visit = {}
        for s, a, r in reversed(tr):
            G = r + gamma * G
            first_visit[s] = G      # overwritten until the EARLIEST visit wins
        for s, g in first_visit.items():
            returns[s].append(g)
            V[s] = float(np.mean(returns[s]))
    return V

for n in (10, 100, 1_000, 10_000):
    V = monte_carlo(grid, manual, n, seed=1)
    print(f"{n:>6} episodes: V(start) = {V[grid.start]:+.4f}   "
          f"error {abs(V[grid.start] - 0.745):.4f}")
```

```text
    10 episodes: V(start) = +0.5760   error 0.1690
   100 episodes: V(start) = +0.7364   error 0.0086
  1000 episodes: V(start) = +0.7350   error 0.0100
 10000 episodes: V(start) = +0.7445   error 0.0005
```

The comment on the `first_visit` line is load-bearing. Walking the trace
backwards and writing `first_visit[s] = G` every time leaves the value from the
*earliest* visit, because it is written last. The obvious alternative — skip
states already seen while going backwards — keeps the **last** visit instead,
and that is biased: a state reached again later has fewer remaining step costs,
so its return looks better than it is. That bug produced +0.763 instead of
+0.745 here, and it is invisible unless you have an exact answer to check
against.

---

## TD(0): bootstrap instead of waiting

Monte Carlo waits for the episode to end. **Temporal difference** learning
updates after every single step, using its own current estimate of where it
landed:

```text
V(s) <- V(s) + alpha * [ r + gamma * V(s') - V(s) ]
```

The bracket is the **TD error** — the surprise. Everything from here to deep RL
is a variation on it.

```python
def td_zero(grid, policy, episodes, alpha=0.1, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    V = {s: 0.0 for s in grid.states}
    for _ in range(episodes):
        s = grid.start
        for _ in range(200):
            if s in grid.terminals:
                break
            nxt, r = sample_step(grid, s, policy[s], rng)
            target = r + gamma * (0.0 if nxt in grid.terminals else V[nxt])
            V[s] += alpha * (target - V[s])
            s = nxt
    return V

print(f"{'episodes':>9}{'MC mean':>10}{'MC std':>9}{'TD mean':>10}{'TD std':>9}")
for n in (10, 100, 1_000):
    mc = [monte_carlo(grid, manual, n, seed=s)[grid.start] for s in range(20)]
    td = [td_zero(grid, manual, n, seed=s)[grid.start] for s in range(20)]
    print(f"{n:>9}{np.mean(mc):>+10.3f}{np.std(mc):>9.3f}"
          f"{np.mean(td):>+10.3f}{np.std(td):>9.3f}")
print(f"{'DP truth':>9}{0.745:>+10.3f}{0.0:>9.3f}{0.745:>+10.3f}{0.0:>9.3f}")
```

```text
 episodes   MC mean   MC std   TD mean   TD std
       10    +0.740    0.071    -0.048    0.004
      100    +0.750    0.021    +0.645    0.042
     1000    +0.746    0.007    +0.746    0.023
 DP truth    +0.745    0.000    +0.745    0.000
```

**Monte Carlo wins this one**, which is not what the textbook ordering leads
you to expect. At 10 episodes MC is already at +0.740 while TD is at -0.048; at
1,000 both hit the right mean but MC's spread is a third of TD's.

The reason is visible in the earlier output: **episodes here are 6.8 steps
long.** A short episode makes a Monte Carlo return a low-variance quantity,
while TD(0) with a constant `alpha=0.1` has to propagate information backwards
one cell per visit and then keeps jittering forever, because a constant step
size never stops responding to noise.

TD's advantages appear when MC's assumptions break:

| | Monte Carlo | TD(0) |
|---|---|---|
| Needs the episode to end | **Yes** | No |
| Long or endless episodes | Unusable | Fine |
| Bias | None | Biased early (bootstraps off wrong values) |
| Variance per update | High (whole return) | Low (one step) |
| This 7-step grid | **Better** | Slower |
| A 10,000-step game | Hopeless | The only option |

Pick by episode length, not by reputation.

---

## Control: Q-learning and SARSA

Prediction evaluates a fixed policy. **Control** improves one. Learn `Q(s, a)`
instead of `V(s)`, and act greedily on it — with some exploration, because
lesson 02 happened.

The two algorithms differ by a single expression:

```text
SARSA        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s', a') - Q(s,a) ]
Q-learning   Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_b Q(s',b) - Q(s,a) ]
```

SARSA uses the action it **actually took next**, including exploratory
mistakes. Q-learning uses the **best** action available, whether or not it will
take it. SARSA is *on-policy*: it learns the value of the policy it is
following, exploration and all. Q-learning is *off-policy*: it learns the value
of the greedy policy while behaving differently.

That sounds like a technicality. Here is what it costs.

---

## The cliff

A 4x12 grid. Start bottom-left, goal bottom-right, and the ten cells between
them are a cliff: step in and you get **-100** and go back to the start. Every
other step costs -1. The optimal path walks right along the cliff edge — 13
steps — and the safe path goes up and over.

```python
from pathlib import Path

CLIFF = r'''
import numpy as np

ACTIONS = ["up", "down", "left", "right"]
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

class CliffWalk:
    """4x12 grid. Bottom row between start and goal is a cliff: -100 and back to start."""
    def __init__(self):
        self.rows, self.cols = 4, 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = {(3, c) for c in range(1, 11)}
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def step(self, state, action):
        dr, dc = DELTA[action]
        nxt = (min(max(state[0] + dr, 0), self.rows - 1),
               min(max(state[1] + dc, 0), self.cols - 1))
        if nxt in self.cliff:
            return self.start, -100.0, False
        if nxt == self.goal:
            return nxt, -1.0, True
        return nxt, -1.0, False

def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return ACTIONS[rng.integers(4)]
    values = [Q[(s, a)] for a in ACTIONS]
    return ACTIONS[int(np.argmax(values))]

def train(kind, episodes=500, alpha=0.5, eps=0.1, gamma=1.0, seed=0):
    env = CliffWalk()
    rng = np.random.default_rng(seed)
    Q = {(s, a): 0.0 for s in env.states for a in ACTIONS}
    rewards = []
    for _ in range(episodes):
        s = env.start
        a = epsilon_greedy(Q, s, eps, rng)
        total = 0.0
        for _ in range(500):
            nxt, r, done = env.step(s, a)
            total += r
            nxt_a = epsilon_greedy(Q, nxt, eps, rng)
            if kind == "sarsa":
                target = r + gamma * (0.0 if done else Q[(nxt, nxt_a)])
            else:
                target = r + gamma * (0.0 if done else max(Q[(nxt, b)] for b in ACTIONS))
            Q[(s, a)] += alpha * (target - Q[(s, a)])
            s, a = nxt, nxt_a
            if done:
                break
        rewards.append(total)
    return Q, np.array(rewards)

def greedy_path(Q, env, limit=60):
    s = env.start
    path = [s]
    for _ in range(limit):
        a = ACTIONS[int(np.argmax([Q[(s, b)] for b in ACTIONS]))]
        s, r, done = env.step(s, a)
        path.append(s)
        if done or s == env.start and len(path) > 2:
            break
    return path

def draw(env, Q):
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    out = []
    for r in range(env.rows):
        row = ""
        for c in range(env.cols):
            s = (r, c)
            if s in env.cliff:
                row += "C"
            elif s == env.goal:
                row += "G"
            elif s == env.start:
                row += "S"
            else:
                row += arrows[ACTIONS[int(np.argmax([Q[(s, b)] for b in ACTIONS]))]]
        out.append(row)
    return "\n".join(out)
'''

Path("/tmp/cliff.py").write_text(CLIFF)
from cliff import CliffWalk, train, greedy_path, draw
print("written")
```

```text
written
```

Both agents use epsilon-greedy with eps=0.1, so **one step in ten is random**.

```python
env = CliffWalk()
print(f"{'algorithm':<12}{'first 100':>11}{'last 100':>11}"
      f"{'greedy path':>13}{'cliff falls':>13}")
for kind in ("sarsa", "qlearning"):
    first, last, lens, falls = [], [], [], []
    for seed in range(30):
        Q, rew = train(kind, episodes=2_000, seed=seed)
        first.append(rew[:100].mean())
        last.append(rew[-100:].mean())
        path = greedy_path(Q, env)
        lens.append(len(path) - 1 if path[-1] == env.goal else np.nan)
        falls.append(np.mean(rew[-100:] < -50))
    print(f"{kind:<12}{np.mean(first):>11.1f}{np.mean(last):>11.1f}"
          f"{np.nanmean(lens):>13.1f}{np.mean(falls):>13.1%}")
```

```text
algorithm     first 100   last 100  greedy path  cliff falls
sarsa             -71.8      -26.3         17.1         5.6%
qlearning         -83.0      -52.9         13.0        26.4%
```

```python
for kind in ("sarsa", "qlearning"):
    Q, _ = train(kind, episodes=2_000, seed=0)
    print(f"\n--- {kind} greedy policy ---")
    print(draw(env, Q))
```

```text

--- sarsa greedy policy ---
>>>>>>>>>>vv
>^^>^>>^^<>v
^<^>^^^^<^>v
SCCCCCCCCCCG

--- qlearning greedy policy ---
>^>>>v>>vvvv
vvvvvvvvvvvv
>>>>>>>>>>>v
SCCCCCCCCCCG
```

**Q-learning found the optimal path and earned half as much.**

Its greedy policy is 13 steps along the bottom row — provably optimal. Its
actual reward while learning is **-52.9**, against SARSA's -26.3, and in the
last hundred episodes it falls off the cliff **26.4% of the time**.

The reason is the one expression that differs. Q-learning's target assumes it
will act greedily next step. It will not: one time in ten it moves at random,
and next to a cliff a random move costs -100. It is learning the value of a
policy it is not following.

SARSA's target uses the action it will actually take, so the -100s from
exploratory steps flow back into the values of cells near the edge. It learns
that *given how I actually behave*, the edge is dangerous, and routes 17 steps
around it — falling in only 5.6% of the time.

**Neither is wrong.** They answer different questions:

| Question | Algorithm |
|---|---|
| "What is the best policy, assuming I will execute it perfectly?" | Q-learning |
| "What is the best policy, given that I explore / act imperfectly?" | SARSA |
| Training in a simulator, deploying greedily | Q-learning |
| Learning online, where every fall costs real money | **SARSA** |
| Learning from someone else's logged data | Q-learning (off-policy) |

Q-learning's off-policy property is why it became the basis of DQN and
everything after it: it can learn from a replay buffer full of old, bad actions.
It just should not be the thing acting next to a cliff.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Last-visit instead of first-visit MC | A quiet +0.018 bias you cannot see without ground truth |
| Monte Carlo on long episodes | Nothing updates until the episode ends |
| Constant alpha and expecting convergence | TD(0)'s spread never shrinks to zero |
| Assuming TD beats MC | MC was better here by a factor of three |
| Q-learning online near a real cliff | 26% of episodes ended in a -100 |
| Judging by the greedy policy alone | Q-learning's policy is optimal and its behaviour is expensive |
| Reporting one seed | Every number in the cliff table is 30 runs |

---

## Exercises

1. Decay epsilon from 1.0 to 0.01 over the 2,000 episodes. Does Q-learning's
   online reward reach SARSA's? Explain what that says about the difference
   between them.
2. Set `alpha` to `1/count(s, a)` in `td_zero` and rerun the comparison table.
   Does TD's spread at 1,000 episodes come down to MC's?
3. Remove exploration entirely (`eps=0`) from the cliff experiment. What
   happens to both algorithms, and why does lesson 02 predict it?
4. Implement Expected SARSA: use the expectation over the epsilon-greedy policy
   instead of `max` or the sampled `a'`. Where does it land between the two?
5. Make the cliff -10 instead of -100. At what penalty do SARSA and Q-learning
   choose the same path?

---

**Next:** [Lesson 06 — Reward Design and Exploration](06-reward-and-exploration.md)
