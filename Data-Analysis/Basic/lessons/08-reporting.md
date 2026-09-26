# Lesson 08 — Reporting

**Goal:** write it so a decision follows.

## What you will learn

- The structure that works
- Writing the finding, not the process
- Quantifying the recommendation
- Limitations, and why they add credibility

---

## The structure

```text
1. The answer            one sentence, with the number
2. What it means         the decision it supports
3. The evidence          three to five findings, each with a number and a chart
4. Method                what you did, briefly, so it can be reproduced
5. Limitations           what this does NOT show
6. Appendix              detail, cleaning log, code
```

**Most people write 4, 3, 1.** The decision-maker reads the first paragraph
and, at best, skims the charts. Put the answer first.

```python
report = {
    "answer": ("Stop stocking juice at Giza and Heliopolis: it is 11% of "
               "revenue there against 12.6% chain-wide, and the lowest-volume "
               "line in both branches."),
    "decision": "Remove juice from the Q3 stocking list for two branches; keep it in Zamalek and Maadi.",
    "value": "Frees shelf space; ~9,000 EGP of quarterly revenue at risk, against 21,000 EGP of holding cost.",
    "confidence": "Medium — six months of data, no seasonal cycle observed yet.",
}
for key, value in report.items():
    print(f"{key.upper():<12}{value}")
```

```text
ANSWER      Stop stocking juice at Giza and Heliopolis: it is 11% of revenue there against 12.6% chain-wide, and the lowest-volume line in both branches.
DECISION    Remove juice from the Q3 stocking list for two branches; keep it in Zamalek and Maadi.
VALUE       Frees shelf space; ~9,000 EGP of quarterly revenue at risk, against 21,000 EGP of holding cost.
CONFIDENCE  Medium — six months of data, no seasonal cycle observed yet.
```

Four lines, and a manager can act. Everything else in the document is support.

---

## Write findings, not activities

| Activity (weak) | Finding (strong) |
|---|---|
| "I analysed revenue by branch" | "Zamalek generates 36% of revenue from 30% of orders" |
| "I cleaned the data" | "12% of rows were duplicates or impossible; revenue was overstated by 58%" |
| "I built a model" | "Orders in the last 30 days predict churn with 0.72 recall" |
| "The data shows a correlation" | "Branches with app orders have 3.4 fewer complaints per 100 orders" |

Every finding contains **a number and a subject**. If a sentence in your
report has neither, delete it.

```python
def is_finding(sentence):
    """A crude check: does this sentence carry a number?"""
    return any(character.isdigit() for character in sentence)

sentences = [
    "Revenue has been analysed across all branches.",
    "Zamalek generates 36.3% of revenue from 36.2% of orders.",
    "The data was cleaned and prepared for analysis.",
    "Removing 55 duplicate and impossible rows cut reported revenue by 58%.",
]
for sentence in sentences:
    print(f"{'FINDING ' if is_finding(sentence) else 'ACTIVITY'}  {sentence}")
```

```text
ACTIVITY  Revenue has been analysed across all branches.
FINDING   Zamalek generates 36.3% of revenue from 36.2% of orders.
ACTIVITY  The data was cleaned and prepared for analysis.
FINDING   Removing 55 duplicate and impossible rows cut reported revenue by 58%.
```

---

## Quantify the recommendation

```python
def recommendation_value(current, expected_change, confidence, cost):
    """Expected value of a recommendation, with its downside."""
    upside = current * expected_change
    expected = upside * confidence - cost
    return {
        "current_annual": f"{current:,.0f} EGP",
        "if_it_works": f"{upside:+,.0f} EGP",
        "probability": f"{confidence:.0%}",
        "cost_to_do": f"{cost:,.0f} EGP",
        "expected_value": f"{expected:+,.0f} EGP",
        "verdict": "do it" if expected > 0 else "not worth it",
    }

for key, value in recommendation_value(
        current=1_058_690, expected_change=0.03,
        confidence=0.6, cost=12_000).items():
    print(f"{key:<16}{value}")
```

```text
current_annual  1,058,690 EGP
if_it_works     +31,761 EGP
probability     60%
cost_to_do      12,000 EGP
expected_value  +7,056 EGP
verdict         do it
```

"We recommend X" is an opinion. "We recommend X; it is worth about 7,000 EGP
in expectation, costs 12,000 to implement, and we are 60% confident" is an
argument — and it invites the stakeholder to argue with the **assumptions**
rather than with you.

State the assumptions explicitly. Someone who disagrees with the 60% can
substitute their own number and see what happens.

---

## Limitations

```python
limitations = [
    ("Six months of data", "no seasonal cycle can be detected; juice may sell in summer"),
    ("Observational, not experimental", "we cannot say removing juice causes the space to be used better"),
    ("Revenue only", "margin per product was not available; a low-revenue line may be high-margin"),
    ("Excludes 55 rows (1.1%)", "duplicates and impossible amounts; see the cleaning log"),
    ("No customer-level view", "we cannot tell whether juice buyers also buy other items"),
]
for limitation, consequence in limitations:
    print(f"- {limitation}: {consequence}")
```

```text
- Six months of data: no seasonal cycle can be detected; juice may sell in summer
- Observational, not experimental: we cannot say removing juice causes the space to be used better
- Revenue only: margin per product was not available; a low-revenue line may be high-margin
- Excludes 55 rows (1.1%): duplicates and impossible amounts; see the cleaning log
- No customer-level view: we cannot tell whether juice buyers also buy other items
```

The third limitation is the one that could reverse the recommendation: **a
low-revenue product may be the highest-margin one.** Naming it yourself is
worth more than any chart in the document — it shows you looked for the reason
you might be wrong.

Stakeholders trust analysts who state limits. The alternative is having one
found in the meeting, by someone else.

---

## A one-page template

```markdown
# Should we stop stocking juice?

**Answer.** Yes, in Giza and Heliopolis. Juice is 11.3% and 12.6% of revenue
there, the lowest line in both, and it occupies the same shelf space as
espresso at half the turnover.

**Decision.** Remove juice from the Q3 stocking list in two branches.
Owner: operations. Review: end of Q3.

## Evidence
1. Juice is the smallest line chain-wide: 60,055 EGP of 529,345 (11.3%).
2. The product mix is near-identical across branches (28-30% latte everywhere),
   so this is a chain-level decision, not a local preference. [chart]
3. Revenue per unit is flat across products (41.9-42.9 EGP), so the difference
   is volume, not price. [table]

## Method
Six months of order lines (Jan-Jun 2026), 4,985 rows after cleaning.
Duplicates and rows failing `amount = quantity x unit_price` removed (55 rows,
1.1%). Aggregated by branch and product. Code: `analysis/juice.ipynb`.

## Limitations
- Six months; no seasonality observed yet
- Revenue only; margin data unavailable
- Observational; no experiment was run

## Next
Run a four-week removal trial in one branch and measure total branch revenue,
not just juice revenue.
```

One page. The answer in the first two lines, three numbered findings, the
method in four sentences, the limits, and a concrete next step.

---

## Presenting it

- **Lead with the answer.** Not the agenda, not the data sources.
- **One message per slide**, and the message is the title.
- **Stop talking after the recommendation.** Silence invites questions; filling
  it invites doubt.
- **Bring the detail, do not show it.** Appendix slides exist for the question
  you hope is asked.
- **Say "I don't know"** when you do not. It is the cheapest credibility
  available.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Method first, answer last | The reader stops before the answer |
| Activities instead of findings | "We analysed" tells nobody anything |
| Recommendation without a number | An opinion, argued as an opinion |
| No limitations | Someone finds one in the meeting |
| Every chart you made | The three that matter get lost |
| No named owner or date | Nothing happens |

---

## Exercises

1. Take a report you wrote and move the answer to the first sentence.
2. Rewrite three activity sentences as findings with numbers.
3. Quantify one recommendation, including the probability and the cost.
4. Write five limitations for your last analysis. Which could reverse it?
