# Lesson 02 — Classifying Risk

**Goal:** decide how much governance a system actually needs, before building
any of it.

## What you will learn

- Why one standard for all systems fails in both directions
- A four-tier classification you can apply in ten minutes
- The questions that decide the tier
- What each tier costs

---

## Proportionality

Two failure modes, equally common:

```text
TOO LITTLE   a model that decides who gets a loan, with no subgroup metrics,
             no override path and no owner

TOO MUCH     a 40-page impact assessment for a model that reorders a list of
             help articles
```

The second is not harmless. A governance process that demands the same of both
gets ignored for both, and the ignoring starts with the trivial case and
spreads.

So the first governance decision is always: **how much does this system
need?**

---

## Four tiers

| Tier | What it does | You owe |
|---|---|---|
| **Minimal** | No decision about a person. Spam filter, image tagging, search ranking | Inventory entry, owner, basic eval |
| **Limited** | Affects a person's experience, reversible, low stakes. Recommendations, routing, chatbots | + disclosure, subgroup metrics, a feedback path |
| **High** | Affects access to something that matters: credit, employment, housing, education, health, benefits, policing | + impact assessment, human oversight, explanation, logging, appeal, external review |
| **Unacceptable** | Social scoring, covert manipulation, biometric categorisation by protected traits, real-time remote biometric identification in public | **Do not build it.** This is the tier where the answer is no |

The exact boundaries vary by regime; the shape does not. Note that the fourth
tier exists: **some systems are not a compliance problem to be solved, and the
engineering answer is to decline**, which is the same judgement
[Data-Science 02](../../Data-Science/lessons/02-framing-the-problem.md) asks
for when the required accuracy is unreachable.

---

## The questions that decide it

Answer these about your system. Any single "yes" in the right-hand column
pushes you up a tier.

```text
1. Does the output affect a specific person?
      no  -> minimal
      yes -> continue

2. Can the person easily get a different outcome?
      a different search result tomorrow        -> limited
      a declined loan on their record           -> high

3. Does it gate access to credit, work, housing, education, health,
   benefits, insurance, or law enforcement?
      yes -> HIGH, always

4. Would the person know a system decided, if nobody told them?
      no  -> you owe disclosure regardless of tier

5. Can a person contest it and reach a human with authority to change it?
      no  -> you have an obligation you have not built

6. Does it process biometric, health, or other special-category data?
      yes -> at least HIGH, and probably a legal review first

7. Could it be used to infer a protected characteristic, even indirectly?
      yes -> document why that is acceptable, or remove the capability
```

Question 7 is the one teams get wrong. **A model does not need a protected
attribute to discriminate by it.** Postcode encodes ethnicity; device type
encodes income; name encodes origin. Removing the column removes your ability
to measure the disparity, not the disparity —
[lesson 04](04-fairness-obligations.md) measures exactly that.

Question 5 is the one that is most often answered with an unexamined "yes".
"The customer can call support" is only an appeal route if support can
actually change the outcome and knows how. Test it:
[lesson 05](05-human-oversight.md) shows what happens when oversight is
nominal.

---

## What each tier costs

```text
MINIMAL     ~1 hour     inventory row, owner, an eval number
LIMITED     ~1 day      + subgroup metrics, a disclosure line, a feedback path
HIGH        ~2 weeks    + impact assessment, oversight design, explanation,
                        logging schema, appeal route, review before launch
                        and periodically after
```

Two things follow from those numbers.

**Classify early.** The difference between a day and two weeks is a planning
decision, and finding out at launch that you are in the high tier is a
two-week delay nobody budgeted.

**High-tier cost is mostly design, not documents.** The impact assessment is a
day; the oversight path, the logging and the appeal route are engineering, and
they are much cheaper designed in than retrofitted.

---

## Writing it down

For anything above minimal, the classification itself is a record:

```markdown
# Risk classification: churn-retention-offer

Tier: LIMITED

Q1 affects a specific person          yes — selects who receives an offer
Q2 easily different outcome           yes — offer only; no adverse record
Q3 gates credit/work/housing/...      no
Q4 would the person know              no  -> disclosure added to the email
Q5 can they contest it                yes — agents may issue the offer manually
Q6 special-category data              no
Q7 could infer a protected trait      postcode is in the features; see the
                                      subgroup analysis in EVAL.md, reviewed
                                      2026-09-14, no disparity above 2 points

Decided by: Nour (owner), reviewed by: legal, 2026-09-14
Re-classify if: the offer becomes a price change, or the model starts
deciding eligibility rather than targeting.
```

The **"re-classify if"** line is the one that earns its place. Systems migrate
upward quietly — a targeting model becomes an eligibility model through three
sensible product decisions — and that line is the trigger that catches it.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| One governance standard for every system | Too heavy for trivial systems, so ignored for all |
| Classifying after building | The high tier is two weeks of design, not documents |
| Assuming "no protected attributes" means no risk | Postcode and device encode them |
| Counting an unusable appeal route as oversight | Support cannot change the outcome |
| No "re-classify if" trigger | Systems migrate upward silently |
| Treating the top tier as negotiable | Sometimes the engineering answer is no |
| Classifying the model, not the decision | The same model at a different threshold is a different tier |

---

## Exercises

1. Classify every system in your inventory. How many are high tier?
2. For your highest-tier system, answer all seven questions in writing.
3. Find a system that has migrated upward since it was built.
4. Test one appeal route end to end, as a customer would.
5. Write the "re-classify if" line for your most important system.

---

**Next:** [Lesson 03 — The Record](03-the-record.md)
