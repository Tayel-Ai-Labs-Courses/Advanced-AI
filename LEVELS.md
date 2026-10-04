# Who Should Take What — Levels and Coverage

Twenty-seven courses is not a queue. This page says where each group starts, what
they can skip, and what fraction of the skills an AI engineer needs the track
now covers.

---

## Coverage: what the track has, and what it does not

**27 courses · 26 projects · 3 capstones.** See
[`CURRICULUM.md`](CURRICULUM.md) for the full ordering, and
[the unified roadmap](README.md#the-unified-roadmap) for one diagram covering
this repository and the eight applied repositories around it.

Judged against the skills an AI engineer is actually asked for in a job:

| Area | Covered | Where |
|---|---|---|
| Python, data structures, algorithms | **yes** | Python |
| pandas / NumPy / sklearn fluency | **yes** | Python, Machine-Learning |
| Classical ML, validation, leakage | **yes** | Machine-Learning, Data-Science |
| Deep learning, PyTorch | **yes** | Deep-Learning |
| Model efficiency, quantisation, PEFT, derivative-free search | **yes** | Optimization |
| NLP | **yes** | NLP |
| Computer vision | **yes** | Computer-Vision |
| Pipelines, storage, orchestration, streaming | **yes** | Data-Engineering |
| Analysis, statistics, experiments | **yes** | Data-Analysis |
| Problem framing, thresholds, shipping, monitoring | **yes** | Data-Science |
| One full reliable system end to end | **yes** | Advanced-Practical-AI |
| Sequential decisions, bandits, RL | **yes** | Reinforcement-Learning |
| **Recommenders: ranking, cold start, exposure, feedback loops** | **yes** | Recommender-Systems |
| **Speech and audio: sampling, spectrograms, VAD, ASR evaluation** | **yes** | Speech-and-Audio |
| **Governance: risk tiers, oversight, consent, retention, audits** | **yes** | AI-Governance |
| **Writing prompts: contracts, context, cost, injection** | **yes** | Prompt-Engineering |
| LLMs, prompting, RAG, evals, fine-tune vs prompt | **yes** | LLM-and-GenAI |
| Agents, tools, automation in Python | **yes** | AI-Agents |
| Privacy, poisoning, extraction, adversarial, DP | **yes** | Data-Security-for-AI |
| Writing, charts, docs, artifacts, presenting | **yes** | Communication-and-Documentation |
| **Maths foundations** (linear algebra, gradients, probability, statistics) | **yes** | Foundations |
| **SQL, schema design, window functions, indexes** | **yes** | Databases-and-SQL |
| **HPC, GPUs, distributed training, cloud cost** | **yes** | HPC-and-Cloud |
| **Research, reproduction, review papers** | **yes** | Research-and-Review |
| **System design, API contracts for AI services** | **yes** | AI-System-Design |
| **Time-series forecasting, backtesting, horizons, intervals** | **yes** | Time-Series-and-Forecasting |
| **MLOps: packaging, CI gates, deployment, serving, monitoring** | **yes** | MLOps |
| Kubernetes, Terraform, cloud consoles | **no, by choice** | [`Cloud-Computing`](https://github.com/Tayel-Ai-Labs-Courses/Cloud-Computing), a separate repository |

**Everything on that list is now covered by a course**, with one row marked
*no, by choice*: specific cloud tooling changes faster than a curriculum can,
and the transferable part — what a GPU-hour costs and whether the model fits —
is in HPC-and-Cloud. The rest is in
[`Cloud-Computing`](https://github.com/Tayel-Ai-Labs-Courses/Cloud-Computing).

For the applied surfaces — **maths in depth, cloud, embedded, AI for
cybersecurity, prompting, frontend, UI/UX, tooling** — see
[Block 9 of `CURRICULUM.md`](CURRICULUM.md#block-9--applied-repositories).
Eight separate repositories in the same organisation, each with its prerequisite
from here and its overlap stated plainly.

---

## Beginner — has written some Python, has not shipped anything

**Goal: produce one thing that works and one thing that is measured.**

```text
0. Foundations         one lesson per symptom           optional, run it alongside
1. Python              both tracks, Projects 1-2        the foundation, no shortcuts
2. Databases-and-SQL   + Project 20                     every dataset comes from one
3. Machine-Learning    + Project 3                      first real model
4. Data-Analysis       Basic track only, + Project 9    how to not fool yourself
5. Data-Science        lessons 01-06                    what the job actually is
6. Communication       lessons 01-04, + Project 15      make it readable
7. Prompt-Engineering  + Project 23                     if an LLM is in the job
```

**Skip for now:** Deep-Learning, Optimization, RL, Agents, Security, HPC,
System-Design, Research.

**On Prompt-Engineering:** it is in Block 5 but needs only Python, so a
beginner already being asked to "use AI" at work can take it now. Its
[lesson 06](Prompt-Engineering/lessons/06-the-iteration-loop.md) is the one
that matters — a 16-point improvement is still noise on 50 examples, which is
the regime nearly all prompting happens in.

**On Foundations:** do not work through all eight lessons before starting. Use
its [lesson 01](Foundations/lessons/01-what-you-need.md) symptom table and read
the one lesson that matches whatever you are currently doing by trial and
error.

Why this order: a beginner who starts with deep learning learns to run a
training loop without learning whether the result means anything. Data-Analysis
Basic and Data-Science 01-06 are what turn "I trained a model" into "I know what
it is worth".

**Done when** they can take a CSV, frame a question, build a baseline, beat it,
say by how much with an interval, and write one page a manager can act on.

---

## Intermediate — has trained models, has not shipped one that survived

**Goal: ship something that still works next month.**

```text
1. Data-Science        all 10 + Project 10              the spine of this level
2. Deep-Learning       + Project 4                      if the domain needs it
3. Databases-and-SQL   + Project 20                     if you skipped it
4. Data-Engineering    + Project 8                      the usual missing piece
5. Data-Analysis       Advanced track + Project 9
6. AI-System-Design    lessons 01-03                    before anyone integrates
7. Prompt-Engineering  + Project 23                     before LLM, if the work is text
8. LLM-and-GenAI       + Project 12                     if the work is text
9. MLOps               lessons 01-05 + Project 22       the gate is what makes it survive
10. Communication      all 8 + Project 15
11. Advanced-Practical-AI                               one system, end to end
```

**Then one domain course** — NLP, Computer-Vision,
**Time-Series-and-Forecasting**, **Recommender-Systems**,
**Speech-and-Audio**, or Optimization — depending on what they are
paid to do. Take Time-Series if anyone has ever asked you for a forecast:
[its lesson 02](Time-Series-and-Forecasting/lessons/02-evaluating.md) will
probably invalidate a result you already believe.

Why: this level's failure is a model that works in a notebook and dies in
contact with a schedule, a new plan value, or a stakeholder. Data-Science 07-09,
Data-Engineering and MLOps 01-05 are the cure — and
[MLOps 07](MLOps/lessons/07-monitoring.md) is where you learn that the most
likely cause of "it got worse in production" is not the model at all.

**Done when** they have something running unattended, with monitoring, a
holdout, and a written verdict.

---

## Senior — ships models, is now responsible for other people's

**Goal: judgement, and the things that go wrong at scale.**

```text
1. AI-Governance         + Project 26      you now sign off other people's systems
2. Data-Security-for-AI  + Project 14      the gap most seniors have
3. Research-and-Review   lesson 03         read this one even if you skip the rest
4. AI-System-Design      + Project 16      you now review other people's designs
5. AI-Agents             + Project 13      if the team is building agents
6. Optimization          + Project 5       cost and latency become yours
7. HPC-and-Cloud         + Project 17      when models outgrow one machine
8. MLOps                 all 8 + Project 22  you now price controls, not just add them
9. Reinforcement-Learning + Project 11     if the problem is sequential
10. Communication        lessons 06-08     artifacts, handover, presenting
11. Final Projects                         the cross-course capstones
```

**Read, do not work through:** the "Common mistakes" table and the
counterintuitive results at the end of every course README. That is the review
checklist for other people's work, and it is most of the value of this track for
someone at this level.

**Done when** they can look at a colleague's project and name, in ten minutes,
which of these failures it has — and, on the MLOps side, say which control they
would *not* add, with the price of adding it and the price of the risk.
[MLOps 08](MLOps/lessons/08-team-practices.md) is explicit that more process is
not monotonically better: the most cautious deployment strategy in
[lesson 05](MLOps/lessons/05-deployment.md) was also the most expensive.

---

## The parts everyone does, at every level

Three courses are not a stage, they are a habit:

- **Communication-and-Documentation** — start lessons 01-04 in week one, whatever
  level. Everything else is worth less without it.
- **Data-Science lesson 01** — the single most useful hour in the track. Do it
  early even if the rest is out of reach.
- **The projects.** A course without its project is a course someone watched.

---

## If you only have four weeks

| Week | Do |
|---|---|
| 1 | Python Basic (skim what you know), Databases-and-SQL 01-04 |
| 2 | Machine-Learning 01-06, Data-Science 01-06 |
| 3 | Communication 01-04, Prompt-Engineering 01-02 if you will use an LLM, and start a real project |
| 4 | Finish the project. Data-Science 07-10 while it runs |

Four weeks does not make an AI engineer. It makes someone who can frame a
problem, beat a baseline, and say what the result is worth — which is more than
most people with a year of tutorials can do.

---

## What is still missing, and who should mind

The two long-standing gaps — time-series forecasting and MLOps — **are now
courses**. What remains is out of scope on purpose:

| Not here | Matters most to | Where it is |
|---|---|---|
| Kubernetes, Terraform, cloud consoles | Whoever owns the infrastructure | [`Cloud-Computing`](https://github.com/Tayel-Ai-Labs-Courses/Cloud-Computing); the cost model is in HPC-and-Cloud |
| Hardware, boards, embedded toolchains | Whoever ships to a device | [`Embedded-AI`](https://github.com/Tayel-Ai-Labs-Courses/Embedded-AI); the measurements are in Optimization 15 |
| A catalogue of prompt techniques | Anyone doing heavy LLM work | [`Advanced-Prompt-Engineering`](https://github.com/Tayel-Ai-Labs-Courses/Advanced-Prompt-Engineering); the two depths before it are Prompt-Engineering and LLM 04 |
| Graph machine learning | Anyone with relational data | Nothing here — a genuine, niche gap |
| Maths beyond the decision-oriented version | Anyone going into research | [`Mathematics-for-AI`](https://github.com/Tayel-Ai-Labs-Courses/Mathematics-for-AI); Foundations is the short route |
| Models *for* security operations | Security engineers | [`Cyber-Ai`](https://github.com/Tayel-Ai-Labs-Courses/Cyber-Ai) — not the same thing as Data-Security-for-AI, which is the security *of* a model |
| Feature stores as products | Teams past ~20 models | [MLOps 07](MLOps/lessons/07-monitoring.md) covers the problem they solve |

Everything listed as a gap in earlier versions of this page — maths, SQL, HPC,
research, system design, forecasting, MLOps — now has its own course or a named
repository.
