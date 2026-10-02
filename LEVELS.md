# Who Should Take What — Levels and Coverage

Sixteen courses is not a queue. This page says where each group starts, what
they can skip, and what fraction of the skills an AI engineer needs the track
now covers.

---

## Coverage: what the track has, and what it does not

**21 courses · 20 projects · 3 capstones.** See
[`CURRICULUM.md`](CURRICULUM.md) for the full ordering and the specialisation
tracks.

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
| **Maths foundations** (linear algebra, gradients, probability, statistics) | **yes** | Foundations |
| **SQL, schema design, window functions, indexes** | **yes** | Databases-and-SQL |
| **HPC, GPUs, distributed training, cloud cost** | **yes** | HPC-and-Cloud |
| **Research, reproduction, review papers** | **yes** | Research-and-Review |
| **System design, API contracts for AI services** | **yes** | AI-System-Design |
| **Time-series forecasting** | partial | Data-Analysis Advanced 04 analyses; no forecasting models |
| **MLOps tooling** (Docker, K8s, CI/CD for models) | partial | Data-Science 08-09 and CI here; no dedicated course |

**Roughly 95% of what the job asks for is covered today.** The two remaining
partials are time-series forecasting and MLOps tooling (Docker, Kubernetes,
CI/CD for models) — the principles of the second are in Data-Science 08-11 and
HPC 08, but there is no dedicated course.

For the applied surfaces — **AI in UI/UX, AI in Frontend, Frontend
Engineering** — see the Specialisations section of
[`CURRICULUM.md`](CURRICULUM.md). Those are separate repositories in the same
organisation, deliberately outside this progression.

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
```

**Skip for now:** Deep-Learning, Optimization, RL, Agents, Security, HPC,
System-Design, Research.

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
7. LLM-and-GenAI       + Project 12                     if the work is text
8. Communication       all 8 + Project 15
9. Advanced-Practical-AI                                one system, end to end
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
2. Research-and-Review   lesson 03         read this one even if you skip the rest
3. AI-System-Design      + Project 16      you now review other people's designs
4. AI-Agents             + Project 13      if the team is building agents
5. Optimization          + Project 5       cost and latency become yours
6. HPC-and-Cloud         + Project 17      when models outgrow one machine
7. Reinforcement-Learning + Project 11     if the problem is sequential
8. Communication         lessons 06-08     artifacts, handover, presenting
9. Final Projects                          the cross-course capstones
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
| 1 | Python Basic (skim what you know), Databases-and-SQL 01-04 |
| 2 | Machine-Learning 01-06, Data-Science 01-06 |
| 3 | Communication 01-04, and start a real project |
| 4 | Finish the project. Data-Science 07-10 while it runs |

Four weeks does not make an AI engineer. It makes someone who can frame a
problem, beat a baseline, and say what the result is worth — which is more than
most people with a year of tutorials can do.

---

## What is still missing, and who should mind

| Gap | Matters most to | Interim advice |
|---|---|---|
| **Time-series forecasting** | Whoever is asked for a forecast | Data-Analysis Advanced 04, then lag features with Machine-Learning's validation rules and Data-Science 03's time split |
| **MLOps tooling** (Docker, K8s, CI/CD for models) | Intermediate and senior | The principles are in Data-Science 08-11 and HPC 08; containerise one project using HPC 08's pinning checklist |

Both are partial rather than absent. Everything else listed as a gap in earlier
versions of this page — maths, SQL, HPC, research, system design — now has its
own course.
