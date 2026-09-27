# Lesson 04 — Charts That Carry the Finding

**Goal:** make a chart that says one thing, and measure how badly a common
default distorts it.

## What you will learn

- The truncated axis, quantified
- One chart, one finding
- Choosing the chart type from the question
- The checklist before a chart leaves your machine

---

## The truncated axis, measured

Everyone knows not to truncate a bar chart's y-axis. Here is the size of the
lie.

```python
values = {"Branch A": 62.0, "Branch B": 58.0, "Branch C": 61.0}
true_ratio = max(values.values()) / min(values.values())
print(f"real values: {values}")
print(f"A is {true_ratio:.2f}x B in reality  ({max(values.values()) - min(values.values()):.0f} points)\n")
print(f"{'y-axis starts at':>18}{'visual bar ratio':>19}{'exaggeration':>14}")
for base in (0, 40, 50, 55, 57):
    heights = {k: v - base for k, v in values.items()}
    visual = max(heights.values()) / min(heights.values())
    print(f"{base:>18}{visual:>19.2f}{visual / true_ratio:>14.1f}x")
print("\nthe bars are the same data; only the axis floor changed")
```

```text
real values: {'Branch A': 62.0, 'Branch B': 58.0, 'Branch C': 61.0}
A is 1.07x B in reality  (4 points)

  y-axis starts at   visual bar ratio  exaggeration
                 0               1.07           1.0x
                40               1.22           1.1x
                50               1.50           1.4x
                55               2.33           2.2x
                57               5.00           4.7x

the bars are the same data; only the axis floor changed
```

**A 7% difference can be drawn to look like a 5x difference.** Starting the axis
at 57 — which many plotting defaults will do automatically to "use the space" —
exaggerates the gap by **4.7 times**.

The reader is not doing arithmetic. They are comparing **areas**, which is the
whole point of a bar chart, and the area is a function of where you put the
floor.

**Bar charts start at zero. Always.** If the differences are too small to see at
zero, that is a finding — say it in words: *"the three branches are within 4
points of each other"* — not a problem to be fixed with the axis.

Line charts are different: they show change over time, the reader reads the
slope rather than the area, and a non-zero axis is acceptable *when it is
labelled clearly*. The distinction is whether the mark's size encodes the value.

---

## One chart, one finding

Before drawing anything, write the **title as a sentence**. If you cannot, you
do not yet know what the chart is for.

| Title | Verdict |
|---|---|
| "Churn by plan" | Names the axes. Says nothing |
| "Revenue over time" | Same |
| "Basic-plan subscribers churn 3x more than pro" | **A finding** |
| "Churn rose 4 points after the May price change" | **A finding** |

The title is the most-read text on any chart, and captioning it with the column
names wastes it.

And if two findings are in one chart, make two charts. A chart that needs a
paragraph to explain is carrying too much.

---

## Choosing the type

```mermaid
flowchart TD
    Qwhat is the question? --> A["how much, compared?"]
    Q --> B["how has it changed?"]
    Q --> C["how is it distributed?"]
    Q --> D["how do two things relate?"]
    Q --> E["what share of a whole?"]
    A --> A1["<b>bar</b>, from zero, sorted by value"]
    B --> B1["<b>line</b>, time on x, no partial periods"]
    C --> C1["<b>histogram</b> or box - never just the mean"]
    D --> D1["<b>scatter</b>, with n stated"]
    E --> E1["<b>stacked bar</b> - not a pie, above 3 slices"]
```

Two additions from the Data-Analysis course that matter more than the type:

- **Sort bars by value, not alphabetically.** The reader is looking for the
  biggest one; make it first.
- **Never plot a mean alone.** Lesson 02 of Data-Analysis Basic exists because
  a mean without a distribution hides everything interesting.

---

## The checklist

Before a chart leaves your machine:

- [ ] The **title states the finding**, as a sentence
- [ ] Both axes labelled, **with units**
- [ ] Bars start at **zero**
- [ ] **No partial periods** at the edge of a time series — the last incomplete
      month looks like a collapse
- [ ] **n is shown** wherever a rate is plotted
- [ ] Sorted by value, unless the order is meaningful (time, size categories)
- [ ] Readable in **greyscale** — colour is decoration, not encoding
- [ ] Legend only if there is more than one series; otherwise put it in the title
- [ ] The data behind it is one command away and reproducible
- [ ] Colour is not the only way to distinguish series (about 1 in 12 men cannot
      separate red from green)

The partial-period item is the most common real error in business reporting. A
dashboard drawn on the 5th of the month shows this month at a fifth of its
value, and somebody panics every single month.

---

## Tables are often better

A chart is for **shape**: a trend, a comparison of many things, a distribution.
For fewer than about six numbers, a table is clearer, denser and easier to check.

```text
Branch A   62.0
Branch C   61.0
Branch B   58.0
```

Three numbers. No chart improves on that, and no axis choice can distort it.

The test: **would a reader want to read the exact values off it?** Then it is a
table.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A truncated bar axis | A 7% difference drawn as 5x |
| The title naming the columns | Wastes the most-read text on the chart |
| Two findings in one chart | Neither lands |
| Alphabetical bars | The reader hunts for the biggest |
| The last partial month included | Monthly panic, monthly explanation |
| A rate without n | 100% of two people |
| A pie chart with eight slices | Nobody can compare angles |
| Colour as the only encoding | Invisible to some readers and in print |
| A chart for three numbers | A table is better |

---

## Exercises

1. Take a chart from your own work and rewrite its title as a finding.
2. Redraw a truncated-axis chart from zero and compute the exaggeration factor
   it had.
3. Find a time series in your dashboards that includes the current partial
   period. Fix it, and see whether anyone notices the "drop" disappearing.
4. Convert one chart to a table. Is it worse?
5. Print one of your charts in greyscale. How many series can you still tell
   apart?

---

**Next:** [Lesson 05 — Code Documentation](05-code-documentation.md)
