# Lesson 02 — The One-Page Answer

**Goal:** write the document that gets a decision made, in one page, with the
answer in the first line.

## What you will learn

- The six-block structure
- Answer first, always
- A full worked example, before and after
- The rules that make it fit on one page

---

## The structure

```text
1. THE ANSWER        one sentence, with the number and its interval
2. THE DECISION      what to do, who does it, by when
3. EVIDENCE          3-5 findings, each with a number
4. METHOD            six sentences. Enough to reproduce, not to admire
5. LIMITATIONS       five, including one that could reverse the conclusion
6. NEXT              the experiment, the follow-up, or the review date
```

Six blocks, in that order, on one page. The order is not negotiable: a reader
who stops after block 1 has the answer, after block 2 knows what to do, and
after block 3 believes you.

---

## Before

> **Churn Analysis — Q3**
>
> ## Introduction
> Customer retention is a key driver of revenue. This document presents the
> results of an investigation into churn prediction carried out over the last
> three weeks.
>
> ## Data
> We used the subscriber table, which contains 12,000 records with 13 columns
> covering demographics, plan information and usage. After removing duplicates
> and handling missing values we were left with 12,000 rows. The data covers the
> period from January 2025 to June 2026.
>
> ## Methodology
> We trained several models including logistic regression, decision trees,
> random forests and gradient boosting. Hyperparameters were tuned using grid
> search with 5-fold cross-validation. Feature engineering included the creation
> of several ratio features.
>
> ## Results
> The best model achieved an AUC of 0.682. Logistic regression outperformed the
> ensemble methods. The most important feature was logins in the last 30 days.
>
> ## Conclusion
> The model shows promise and with further work could be deployed to help the
> retention team prioritise their outreach efforts.

Four hundred words, and a decision-maker reading it learns: nothing to decide,
no number they care about, "shows promise", "further work".

Notice what is missing entirely: **what it is worth, what to do, and what it
costs.**

---

## After

> **Recommendation: call the 1,000 riskiest subscribers every Monday.**
> Worth about **41,700 EGP a month** (range 30,000-50,000 depending on offer
> acceptance), at the retention team's existing call capacity.
>
> **Decision needed:** approval from the Head of Retention to start on 6
> October, using the existing Monday call slot. No new headcount.
>
> **Evidence**
> 1. The model finds **256 churners in the top 1,000** against the current
>    rule's 197 — **59 more customers saved per month**, at identical call cost.
> 2. The riskiest 10% of subscribers contain **21% of next month's
>    cancellations** — 2.1x their share.
> 3. Calling 1,000 costs 50,000 EGP; the 256 saves are worth 91,700 EGP at a
>    30% offer-acceptance rate.
> 4. At the team's current 250-call capacity the model finds only **4 more**
>    churners than the rule. **The value depends entirely on reaching 1,000.**
>
> **Method:** logistic regression on 9,000 subscribers, 8 features, validated on
> 3,000 held out. Compared against the current failed-payment rule on the same
> data. Two columns were excluded because they are recorded after cancellation.
> Threshold set by call capacity, not by the model. Reproducible from
> `churn/run.py`, data hash `6b3f8cdb`.
>
> **Limitations**
> 1. The 30% offer-acceptance rate is an assumption, not a measurement. At 15%
>    the monthly value falls to about 4,000 EGP and the project is marginal.
> 2. The model never contacts **pro** customers (0.9% selected), whose churn
>    costs four times as much. Ranking by expected revenue would change this.
> 3. Validated on historical data only; no live test has been run.
> 4. Retention offers change future churn, so the measured churn rate will drop
>    for reasons other than fewer cancellations.
> 5. The 14-day cancellation-reversal rule was applied; a different definition of
>    churn moves every number here.
>
> **Next:** run it for four weeks with a 10% holdout, and measure the actual
> offer-acceptance rate — that is the assumption the whole case rests on. Review
> 3 November.

Same work. Roughly the same length. The difference is that a decision-maker can
act on it in ninety seconds, and a peer can attack it, because both the number
and its weakest assumption are visible.

---

## Rules that make it fit

**Every sentence in Evidence contains a number.** If it does not, it is method,
limitation or padding. Move it or cut it.

**The answer contains its interval.** "41,700 EGP" alone is a claim to
precision you do not have. "About 41,700, range 30,000-50,000" is honest and
takes six extra words.

**Name the baseline in the evidence, not the method.** "256 against the current
rule's 197" is the finding. "We compared against a baseline" is not.

**One limitation must be able to reverse the conclusion.** Limitation 1 above
does: at 15% acceptance the project is marginal. A limitations section full of
"the dataset could be larger" is decoration, and experienced readers discount
the whole document when they see it.

**Method is six sentences.** Enough to reproduce, not enough to admire. The full
version lives in the linked notebook, for the one reader in ten who wants it.

**Cut every sentence that would survive being deleted.** "Customer retention is
a key driver of revenue" tells the Head of Retention nothing they do not know.

---

## The headline test

Read only the bold first line. Can the reader act?

| Headline | Can they act? |
|---|---|
| "Churn analysis results" | No |
| "The model achieved 0.68 AUC" | No — AUC is not a decision |
| "The model is ready for deployment" | No — to do what, at what cost? |
| "Call the 1,000 riskiest subscribers every Monday; worth ~41,700 EGP/month" | **Yes** |

If your headline fails the test, you have not finished the analysis. That is
not a writing problem — **a finding you cannot state as an action is usually a
finding you have not finished thinking about.**

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Introduction, Data, Method, Results, Conclusion | The answer lands on page four |
| "Shows promise", "further work needed" | No decision is possible |
| A metric as the headline | AUC is not something anyone can do |
| Evidence sentences without numbers | Indistinguishable from opinion |
| Limitations that cannot change anything | The reader discounts all of them |
| No baseline in the evidence | The improvement has no reference |
| A point estimate with no range | False precision, and it will be quoted |

---

## Exercises

1. Rewrite your most recent report into the six blocks. How much did you delete?
2. Write the headline for your current project. Apply the headline test.
3. Take your limitations section and mark the one that could reverse the
   conclusion. If none can, write one that could.
4. Strip every sentence from your Evidence that has no number. What survives?
5. Give the one-pager to someone unfamiliar with the project and ask what they
   would do. Time how long it takes them.

---

**Next:** [Lesson 03 — Writing About Numbers](03-writing-about-numbers.md)
