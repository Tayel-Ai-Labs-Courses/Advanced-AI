# Lesson 07 — Many Series at Once

**Goal:** forecast a thousand series without training a thousand models you
cannot maintain.

## What you will learn

- Local against global models
- Hierarchies, and making the totals agree
- Cold starts and intermittent demand
- Choosing per series, automatically

---

## One model or a thousand?

```text
LOCAL    one model per series.  1,000 SKUs = 1,000 models
GLOBAL   one model, trained on all series, with the series id as a feature
```

| | Local | Global |
|---|---|---|
| Captures per-series quirks | **Yes** | Only through features |
| Works for a new series | **No** — no history | **Yes** |
| Short series | Poor | **Borrows strength from the others** |
| Models to retrain and monitor | 1,000 | **1** |
| Training cost | Low each, high total | One pass |
| Needs normalisation across scales | No | **Yes** — a SKU selling 3/day and one selling 30,000 |
| Debuggability | Easy per series | Harder |

**Global is the default at scale**, and the reason is operational before it is
statistical: one model is one thing to monitor, retrain, roll back and explain
([Data-Science 09](../../Data-Science/lessons/09-monitoring-and-drift.md)
applies once rather than a thousand times).

The statistical argument is also real: a global model learns the *shape* of a
weekly cycle from all series together, so a SKU with six weeks of history
inherits what the others know.

**The practical pattern:**

```text
1. A global model over all series, with series-level features
     (category, region, price band, age of series)
2. Normalise each series before training — divide by its own mean or use log
3. Local models ONLY for the handful of high-value series that justify the
   maintenance
4. Fall back to seasonal naive for anything with too little history
```

---

## Hierarchies must add up

Most real forecasts live in a hierarchy:

```text
                    total
           /          |          \
      Cairo         Giza      Alexandria
      /    \        /   \       /     \
   store  store  store store  store  store
```

Forecast each level independently and **the children will not sum to the
parent** — which is the first thing a finance team notices.

| Approach | How | Trade |
|---|---|---|
| **Bottom-up** | Forecast the leaves, sum upward | Totals are consistent; leaf noise accumulates |
| **Top-down** | Forecast the total, split by historical share | A stable total; leaves miss their own dynamics |
| **Middle-out** | Forecast a middle level, go both ways | A compromise |
| **Reconciliation** (MinT, OLS) | Forecast every level, then project onto the consistent space | **Usually the most accurate**, and it needs a library |

Start **bottom-up** — it is one line and it is always consistent. Move to
reconciliation when the top-level number is the one the business acts on and the
leaves are noisy.

**State which you used.** "Our regional forecasts sum to the national one
because we forecast bottom-up" is a sentence that prevents a long meeting.

---

## Cold starts and intermittent demand

Two cases that break ordinary forecasting:

**A new series with no history.** Nothing to lag. Options, in order:

1. The **global model**, using only series-level features — this is its main
   advantage
2. The **average of similar series**, by category or region
3. A business estimate, flagged as such

**Intermittent demand** — mostly zeros, occasional spikes. A SKU that sells 0
units on most days and 7 on a few.

| Why normal methods fail | What to do |
|---|---|
| MAPE is undefined on zeros (lesson 03) | Use MAE or a count metric |
| A mean forecast of 0.3 units is never correct | Forecast **two things**: *will there be demand?* and *how much, given demand?* |
| Smoothing flattens the spikes away | Croston's method, or a hurdle model |
| The decision is usually "how much to stock" | Forecast a **quantile**, not a mean (lesson 06) |

The two-part framing is the useful one: a classifier for "any demand this
week?" and a regressor for the size. It also makes the output honest — "70%
chance of 0, otherwise about 6 units" is a better planning input than "1.8".

---

## Choosing per series

With a thousand series you cannot choose a method by hand. Automate it with the
backtest you already have (lesson 02):

```text
for each series:
    if history < 2 seasonal cycles      -> seasonal naive
    elif mostly zeros                   -> intermittent method
    else                                -> backtest {global, local, seasonal naive}
                                           and pick the lowest MASE
    record the choice and the MASE
```

Then **monitor the distribution of MASE across series**, not the average. The
question "how many series is our model worse than seasonal naive on?" is the one
that matters, and the answer is rarely zero.

```text
series where MASE < 0.9   the model clearly helps
0.9 <= MASE <= 1.1        indistinguishable — use the baseline, it is free
MASE > 1.1                the model is harmful. Fall back
```

Shipping the baseline for the middle band is not defeat; it removes a thousand
things that can break for no measurable gain.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A thousand local models | A thousand things to monitor and retrain |
| A global model without normalising scale | The large series dominate the loss |
| Forecasting levels independently | The children do not sum to the parent |
| Not stating your reconciliation method | Finance finds the mismatch first |
| MAPE on intermittent demand | Undefined, or infinite |
| A mean forecast for a stocking decision | You need a quantile |
| Reporting average MASE across series | It hides the series where you made things worse |

---

## Exercises

1. Train a global model on all your series and compare, per series, with local
   models. Which wins, and for which kinds of series?
2. Check whether your regional forecasts sum to your national one.
3. Count the series in your portfolio with MASE > 1.1. What are you doing for
   them?
4. Find your most intermittent series. What does your current method predict?
5. Write the per-series selection rule for your own portfolio.

---

**Next:** [Lesson 08 — Forecasting in Production](08-production.md)
