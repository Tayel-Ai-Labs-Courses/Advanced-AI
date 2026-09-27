# Who Should Take What — Levels and Coverage

Sixteen courses is not a queue. This page says where each group starts, what
they can skip, and what fraction of the skills an AI engineer needs the track
now covers.

---

## Coverage: what the track has, and what it does not

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
| LLMs, prompting, RAG, evals, fine-tune vs prompt | **yes** | LLM-and-GenAI |
| Agents, tools, automation in Python | **yes** | AI-Agents |
| Privacy, poisoning, extraction, adversarial, DP | **yes** | Data-Security-for-AI |
| Writing, charts, docs, artifacts, presenting | **yes** | Communication-and-Documentation |
| **SQL and warehouse modelling** | partial | Data-Engineering 02-03 touch it; no dedicated course |
| **Maths foundations** (linear algebra, calculus, probability) | **no** | Assumed throughout, taught nowhere |
| **HPC, GPUs, cloud training** | **no** | Planned |
| **Research and review papers** | **no** | Planned |
| **Time-series forecasting** | partial | Data-Analysis Advanced 04 analyses; no forecasting models |
| **MLOps tooling** (Docker, K8s, CI/CD for models) | partial | Data-Science 08-09 and CI here; no dedicated course |

**Roughly 80% of what the job asks for is covered today.** The four honest gaps
are maths foundations, HPC/cloud, research skills, and SQL — the first two
matter most for a junior, the third for anyone going further than applying known
methods.

---

## Beginner — has written some Python, has not shipped anything

**Goal: produce one thing that works and one thing that is measured.**

```text
1. Python              both tracks, Projects 1-2        the foundation, no shortcuts
2. Machine-Learning    + Project 3                      first real model
3. Data-Analysis       Basic track only, + Project 9    how to not fool yourself
4. Data-Science        lessons 01-06                    what the job actually is
5. Communication       lessons 01-04, + Project 15      make it readable
```

**Skip for now:** Deep-Learning, Optimization, RL, Agents, Security.

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
3. Data-Engineering    + Project 8                      the usual missing piece
4. Data-Analysis       Advanced track + Project 9
5. LLM-and-GenAI       + Project 12                     if the work is text
6. Communication       all 8 + Project 15
7. Advanced-Practical-AI                                one system, end to end
```

**Then one domain course** — NLP, Computer-Vision, or Optimization — depending
on what they are paid to do.

Why: this level's failure is a model that works in a notebook and dies in
contact with a schedule, a new plan value, or a stakeholder. Data-Science 07-09
and Data-Engineering are the cure.

**Done when** they have something running unattended, with monitoring, a
holdout, and a written verdict.

---

## Senior — ships models, is now responsible for other people's

**Goal: judgement, and the things that go wrong at scale.**

```text
1. Data-Security-for-AI  + Project 14      the gap most seniors have
2. AI-Agents             + Project 13      if the team is building agents
3. Optimization          + Project 5       cost and latency become yours
4. Reinforcement-Learning + Project 11     if the problem is sequential
5. Communication         lessons 06-08     artifacts, handover, presenting
6. Final Projects                          the cross-course capstones
```

**Read, do not work through:** the "Common mistakes" table and the
counterintuitive results at the end of every course README. That is the review
checklist for other people's work, and it is most of the value of this track for
someone at this level.

**Done when** they can look at a colleague's project and name, in ten minutes,
which of these failures it has.

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
| 1 | Python Basic (skim what you know), Machine-Learning 01-06 |
| 2 | Data-Science 01-06, Data-Analysis Basic 01-04 |
| 3 | Communication 01-04, and start a real project |
| 4 | Finish the project. Data-Science 07-10 while it runs |

Four weeks does not make an AI engineer. It makes someone who can frame a
problem, beat a baseline, and say what the result is worth — which is more than
most people with a year of tutorials can do.

---

## What is still missing, and who should mind

| Gap | Matters most to | Interim advice |
|---|---|---|
| **Maths foundations** | Beginners | Do 3Blue1Brown's linear algebra series alongside Deep-Learning |
| **HPC and cloud training** | Intermediate, when models outgrow a laptop | Read Optimization 06 and 12; rent one GPU hour and measure |
| **Research and review papers** | Senior, and anyone going past applied work | Read one paper a week with the Data-Science 05 comparison checklist in hand |
| **SQL** | Everyone, immediately | Data-Engineering 02-03, then any SQL exercise site |
| **Time-series forecasting** | Whoever is asked for a forecast | Data-Analysis Advanced 04, then lag features with Machine-Learning's validation rules |

These four are planned as courses. Until they exist, the rows above are the
honest workaround, not a substitute.
