# Lesson 05 — Comparing Groups

**Goal:** compare fairly, and notice when the comparison is rigged.

## What you will learn

- Like for like
- Rates, not counts
- Simpson's paradox
- Small groups and noise

---

## Compare rates, not counts

```python
import pandas as pd

campaign = pd.DataFrame({
    "group": ["email", "sms", "push"],
    "sent": [50_000, 8_000, 120_000],
    "conversions": [1_500, 400, 2_400],
})
campaign["rate"] = (campaign["conversions"] / campaign["sent"] * 100).round(2)

print(campaign.sort_values("conversions", ascending=False).to_string(index=False))
print("\nranked by conversions:", campaign.sort_values("conversions", ascending=False)["group"].tolist())
print("ranked by rate:       ", campaign.sort_values("rate", ascending=False)["group"].tolist())
```

```text
group   sent  conversions  rate
 push 120000         2400   2.0
email  50000         1500   3.0
  sms   8000          400   5.0

ranked by conversions: ['push', 'email', 'sms']
ranked by rate:        ['sms', 'email', 'push']
```

**The two rankings are exact opposites.** Push produced the most conversions
and converts worst; SMS produced the fewest and converts two and a half times
better.

Which is right depends on the question. "Where did conversions come from?" —
push. "Which channel should we invest in?" — SMS, subject to whether it
scales. Presenting only the counts answers the first question while appearing
to answer the second.

---

## Like for like

```python
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet")

first_half = orders[orders["ordered_at"] < "2026-04-01"]
second_half = orders[orders["ordered_at"] >= "2026-04-01"]

print(f"Jan-Mar: {len(first_half):>5} orders, {first_half['amount'].sum():>9,.0f} EGP, "
      f"{(first_half['ordered_at'].max() - first_half['ordered_at'].min()).days + 1} days")
print(f"Apr-Jun: {len(second_half):>5} orders, {second_half['amount'].sum():>9,.0f} EGP, "
      f"{(second_half['ordered_at'].max() - second_half['ordered_at'].min()).days + 1} days")

first_days = (first_half["ordered_at"].max() - first_half["ordered_at"].min()).days + 1
second_days = (second_half["ordered_at"].max() - second_half["ordered_at"].min()).days + 1
print(f"\nper day:  {first_half['amount'].sum() / first_days:,.0f} vs "
      f"{second_half['amount'].sum() / second_days:,.0f} EGP")
print(f"change:   {(second_half['amount'].sum() / second_days) / (first_half['amount'].sum() / first_days) - 1:+.1%}")
```

```text
Jan-Mar:  2510 orders,   266,365 EGP, 90 days
Apr-Jun:  2475 orders,   262,980 EGP, 90 days

per day:  2,960 vs 2,922 EGP
change:   -1.3%
```

Ninety days each, so the comparison is fair and the answer is **−1.3%** —
which, on 2,500 orders a half, is almost certainly noise rather than a trend.

Had the periods been 90 and 60 days, the raw totals would have shown a 33%
"collapse" that was entirely calendar.

The checklist before any comparison:

| Check | Why |
|---|---|
| Same length of period | Otherwise you compare calendars |
| Same number of trading days | A month with two extra weekends differs |
| Same population | New branches opening mid-period distort everything |
| Same definition | "Active customer" must mean one thing |
| Same data quality | A source that started reporting in March |

---

## Simpson's paradox

The most important trap in this lesson: **a pattern that holds in every group
can reverse when the groups are combined.**

```python
import pandas as pd

data = pd.DataFrame({
    "branch":   ["Zamalek", "Zamalek", "Giza", "Giza"],
    "channel":  ["app", "walk-in", "app", "walk-in"],
    "orders":   [900, 100, 100, 900],
    "complaints": [45, 9, 4, 72],
})
data["rate"] = (data["complaints"] / data["orders"] * 100).round(1)
print(data.to_string(index=False))

combined = data.groupby("branch").agg(orders=("orders", "sum"),
                                      complaints=("complaints", "sum"))
combined["rate"] = (combined["complaints"] / combined["orders"] * 100).round(1)
print("\ncombined:")
print(combined.to_string())
```

```text
 branch channel  orders  complaints  rate
Zamalek     app     900          45   5.0
Zamalek walk-in     100           9   9.0
   Giza     app     100           4   4.0
   Giza walk-in     900          72   8.0

combined:
         orders  complaints  rate
branch                           
Giza       1000          76   7.6
Zamalek    1000          54   5.4
```

Read the rates carefully. **Giza is better in app (4.0 against 5.0) and better
in walk-in (8.0 against 9.0) — and worse overall (7.6 against 5.4).**

Giza wins both segments and loses the total. Neither number is miscalculated.

The cause is the mix: walk-in generates twice the complaints of app, and 90%
of Giza's orders are walk-in while 90% of Zamalek's are app. The combined
number is measuring the **channel mix**, not the branch.

How to avoid being fooled:

- **Always break a comparison down by the obvious confounders** — channel,
  segment, product, time.
- If the direction flips when you do, the aggregate is measuring the mix.
- Report the segmented table, not the headline.
- Ask "what else differs between these groups?" before presenting any
  difference.

---

## Small groups are mostly noise

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
true_rate = 0.05

rows = []
for size in [20, 100, 1_000, 10_000]:
    observed = [rng.binomial(size, true_rate) / size for _ in range(1_000)]
    rows.append({
        "sample_size": size,
        "mean_observed": round(float(np.mean(observed)), 4),
        "p5": round(float(np.percentile(observed, 5)), 4),
        "p95": round(float(np.percentile(observed, 95)), 4),
    })
frame = pd.DataFrame(rows)
frame["spread"] = (frame["p95"] - frame["p5"]).round(4)
print(frame.to_string(index=False))
```

```text
 sample_size  mean_observed     p5    p95  spread
          20         0.0488 0.0000 0.1500  0.1500
         100         0.0497 0.0200 0.0900  0.0700
        1000         0.0498 0.0390 0.0610  0.0220
       10000         0.0501 0.0466 0.0538  0.0072
```

Every row has the **same true rate of 5%**. With 20 observations, 90% of the
outcomes fall between **0% and 15%** — a branch could look perfect, or three
times worse than another, purely by chance.

At 10,000 the same 5% is pinned between 4.66% and 5.38%. The mean is right in
every row; only the *spread* changes, and the spread is what decides whether a
difference you observe means anything.

A useful rule of thumb: the uncertainty in a rate shrinks with the square root
of the sample, so **quadrupling the sample halves the spread.** Practically:

| Group size | Treat a difference as |
|---|---|
| Under 30 | Almost certainly noise |
| 30–100 | Suggestive at best |
| 100–1,000 | Worth investigating |
| Over 1,000 | Reportable, with an interval |

Advanced lesson 01 turns this into confidence intervals. In this track, the
discipline is simpler: **always print the group size beside the rate**, and
never rank groups whose sizes differ by an order of magnitude without saying
so.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Comparing counts across unequal populations | The biggest group always "wins" |
| Unequal periods | A calendar artefact reported as a trend |
| No breakdown by confounders | Simpson's paradox goes unnoticed |
| Ranking tiny groups | You chase noise |
| Rate without the denominator | "50% better" on four observations |
| Changed definitions mid-period | An artificial jump |

---

## Exercises

1. Take a comparison from a report and add the denominators. Does the ranking
   hold?
2. Construct a Simpson's paradox from your own data by splitting on a
   confounder.
3. Print group sizes beside every rate in one of your analyses.
4. Simulate a 5% rate at n = 25 a thousand times; how often does it exceed 10%?
