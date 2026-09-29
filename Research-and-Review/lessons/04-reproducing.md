# Lesson 04 — Reproducing a Result

**Goal:** turn a paper's claim into a number on your own data, and know what to
conclude when it does not match.

## What you will learn

- What "reproduce" means, in four increasing degrees
- The order to try things
- What a gap actually tells you
- When to stop

---

## Four degrees

| Degree | You do | Tells you |
|---|---|---|
| **1. Re-run** | Their code, their data, their config | Whether the artefact works at all |
| **2. Reproduce** | Your code, their data, their config | Whether the method is described correctly |
| **3. Replicate** | Their method, **your data** | Whether it applies to you |
| **4. Generalise** | Their method, your data, your scale, your baseline | **Whether to use it** |

Papers are usually discussed at degree 1 and matter at degree 4.

**Degree 3 is the one worth your day.** Degree 1 tells you the authors packaged
their work well; degree 3 tells you whether the method helps on the problem you
are paid to solve — and that answer is very often "no", for reasons that are
nobody's fault.

---

## The order

```text
1. Read the code before running it            30 min
   - does the eval split match the paper's?
   - is the baseline in the repo tuned?
   - are the reported numbers produced BY this code, or pasted?

2. Run their code on their data, one seed     1-2 h
   - do you get their number, within their reported variance?
   - if they report no variance, run 3 seeds yourself

3. Run YOUR baseline on THEIR data            1 h
   - lesson 03: tune it as hard as they tuned the method
   - this is where most claimed improvements evaporate

4. Run their method on YOUR data              2-4 h
   - with your baseline, tuned equally
   - this is the number that decides anything

5. Write down the gap and what explains it
```

**Step 3 is the step everyone skips.** It is also the cheapest way to find out
that the paper's baseline was undertuned, which lesson 03 showed produces a
+0.0226 "improvement" between identical methods.

---

## What a gap means

You got 0.79 where the paper reports 0.84. Work down this list before concluding
anything:

| Cause | How to check |
|---|---|
| **Different data version** | Benchmarks get revised. Check the exact release |
| **Different split** | Standard splits differ between papers. Check row counts |
| **Different preprocessing** | Tokeniser, normalisation, image resizing |
| **Different metric definition** | Macro vs micro F1; accuracy on what population |
| **Fewer seeds** | Lesson 03: they may have reported the best of N |
| **Less tuning** | Did you give it their search budget? |
| **Different compute / batch size** | Large-batch results often do not survive small-batch |
| **A bug in your implementation** | The most likely cause, and check it last, not first |
| **The result does not generalise** | The interesting answer, and the rarest |

**Check the boring causes first.** In practice the order of frequency is:
split, preprocessing, metric definition, tuning, your bug, and only then "the
paper is wrong".

And note what a gap does **not** mean. A 5-point gap on your data is not
evidence of misconduct; it is usually evidence that the method is sensitive to
something the paper did not vary. That sensitivity is itself the most useful
thing you learned, and it belongs in your note's `DOUBT` line.

---

## When you cannot reproduce it

Reality: for many papers, degree 1 is not achievable. No code, missing
hyperparameters, a dataset you cannot obtain, or compute you cannot afford.

That is not a dead end — it is information:

```text
no code released                  -> treat the claim as unverified
code released but does not run    -> the result is one person's laptop
hyperparameters not stated        -> lesson 03's search budget is unknown
data not obtainable               -> you can only test degree 3, on your data
compute out of reach              -> test at your scale, and say so
```

**Testing at your scale and reporting that is a valid result.** "We could not
reproduce the reported 3-point gain at 1/50th the compute; at our scale the
methods are within noise" is a genuinely useful sentence, and a rare one.

---

## Reproducing your own work

The same four degrees apply inside your team, and the failures are identical.

Data-Science lesson 07 built the machinery: seeds, data fingerprints, run
records, environment capture. The test is simple and worth running on yourself:

```text
Can a colleague, from your repository alone, produce a number
within your reported variance, on a different machine, in a day?
```

If not, your work has the same problem as the papers in this lesson.

And the habit that prevents it: **every number in a report or a slide carries
its run id.** Communication lesson 06 makes this an artifact; this lesson is why
it matters.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Skipping the code read | Twenty minutes that saves a day |
| Not running your own baseline on their data | Where most claimed gains evaporate |
| Assuming a gap means the paper is wrong | Split and preprocessing are far likelier |
| Assuming a gap means you made a mistake | Also often wrong; check the boring causes |
| Only reproducing degree 1 | Tells you about packaging, not usefulness |
| Not writing the reproduction down | You will re-do it in a year |
| Treating "could not reproduce" as failure | It is a finding, and a publishable one |

---

## Exercises

1. Read the code of a paper you rely on, without running it. What did you learn
   about its evaluation?
2. Run your own tuned baseline on a paper's dataset. Does the gap survive?
3. Reproduce one result at degree 3, on your data, and write the gap and its
   explanation.
4. Ask the reproducibility question of your own last project, and try it with a
   colleague.
5. Write one paragraph you could publish about a result you could not reproduce.

---

**Next:** [Lesson 05 — Searching the Literature](05-searching.md)
