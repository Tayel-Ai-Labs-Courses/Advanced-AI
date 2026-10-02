# Lesson 07 — SQL for Machine Learning

**Goal:** build a training table in SQL without leaking the future into it.

## What you will learn

- The leaky join everybody writes first
- The one line that fixes it
- Building a panel with window functions
- Where feature logic should live

---

## The setup

```python
import sqlite3

con = sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE events (customer_id INT, event_at TEXT, kind TEXT);
CREATE TABLE labels (customer_id INT, as_of TEXT, churned INT);
INSERT INTO events VALUES
  (1,'2026-01-05','login'), (1,'2026-01-20','login'), (1,'2026-02-10','login'),
  (1,'2026-03-02','login'), (1,'2026-03-25','ticket'),
  (2,'2026-01-08','login'), (2,'2026-02-14','ticket'), (2,'2026-03-18','login'),
  (3,'2026-01-11','login'), (3,'2026-01-12','login'), (3,'2026-01-13','login');
INSERT INTO labels VALUES
  (1,'2026-03-01',0), (2,'2026-03-01',1), (3,'2026-03-01',1);
""")

print("events:", con.execute("SELECT COUNT(*) FROM events").fetchone()[0])
print("labels:", con.execute("SELECT COUNT(*) FROM labels").fetchone()[0])
```

```text
events: 11
labels: 3
```

Eleven events, three labelled customers, each labelled **as of 2026-03-01**.
Small enough to check every number by hand, which is the point.

---

## The leaky join

```python
leaky = """
SELECT l.customer_id, l.as_of, l.churned,
       COUNT(e.event_at) AS logins_all_time
FROM labels l
LEFT JOIN events e ON e.customer_id = l.customer_id AND e.kind = 'login'
GROUP BY l.customer_id
"""
print(f"{'cust':>5}{'as_of':>13}{'churned':>9}{'logins':>8}")
for r in con.execute(leaky):
    print(f"{r[0]:>5}{r[1]:>13}{r[2]:>9}{r[3]:>8}")
print("\ncustomer 1 shows 4 logins — one of them is AFTER 2026-03-01.")
```

```text
 cust        as_of  churned  logins
    1   2026-03-01        0       4
    2   2026-03-01        1       2
    3   2026-03-01        1       3

customer 1 shows 4 logins — one of them is AFTER 2026-03-01.
```

Customer 1 shows **4 logins**. Look back at the data: one of them is
`2026-03-02` — **the day after the label date.**

That query is a correct SQL join and a broken training table. It is
[Data-Science lesson 02](../../Data-Science/lessons/02-framing-the-problem.md)'s
temporal leakage, and it is the single most common way a model scores
beautifully offline and fails in production: in production the future column is
empty.

The model will learn "customers with recent logins do not churn", which is true
and useless, because at prediction time those logins have not happened yet.

---

## The one line that fixes it

```python
correct = """
SELECT l.customer_id, l.as_of, l.churned,
       COUNT(e.event_at) AS logins_before_asof,
       COUNT(CASE WHEN e.event_at >= date(l.as_of, '-30 day')
                  THEN 1 END) AS logins_last_30d,
       MAX(e.event_at) AS last_login
FROM labels l
LEFT JOIN events e
       ON e.customer_id = l.customer_id
      AND e.kind = 'login'
      AND e.event_at < l.as_of              -- the entire lesson is this line
GROUP BY l.customer_id, l.as_of, l.churned
"""
print(f"{'cust':>5}{'as_of':>13}{'churned':>9}{'all':>5}{'30d':>5}{'last_login':>13}")
for r in con.execute(correct):
    print(f"{r[0]:>5}{r[1]:>13}{r[2]:>9}{r[3]:>5}{r[4]:>5}{str(r[5]):>13}")
print("\ncustomer 1 now shows 3 — the 2026-03-02 login is correctly excluded.")
```

```text
 cust        as_of  churned  all  30d   last_login
    1   2026-03-01        0    3    1   2026-02-10
    2   2026-03-01        1    1    0   2026-01-08
    3   2026-03-01        1    3    0   2026-01-13

customer 1 now shows 3 — the 2026-03-02 login is correctly excluded.
```

```sql
AND e.event_at < l.as_of          -- the entire lesson is this line
```

Customer 1 is now **3**, and the `2026-03-02` login is correctly gone.

Three things to notice about where that condition sits:

**It is in the `ON` clause, not the `WHERE` clause.** On a `LEFT JOIN` that
distinction matters: in `ON` it filters which events match; in `WHERE` it would
discard customers with no qualifying events entirely, turning a count of 0 into
a missing row.

**The 30-day window is a `CASE` inside `COUNT`**, so you get several windows in
one pass over the data instead of one query per window.

**`MAX(event_at)` gives recency for free.** "Days since last login" is one of
the strongest features in most churn problems, and it is a `MAX` and a date
difference.

### The rule

> **Every feature query takes an `as_of` and compares against it.**

A feature function with no cutoff parameter is a leak waiting to be written —
Data-Science lesson 02 says the same thing about Python, and in SQL the cutoff
is a join condition rather than an argument.

---

## A panel, with window functions

Lesson 04's window functions build the customer-month table from
[Data-Science lesson 02](../../Data-Science/lessons/02-framing-the-problem.md)
directly:

```python
con.executescript("""
CREATE TABLE snapshots (customer_id INT, as_of TEXT);
INSERT INTO snapshots VALUES (1,'2026-02-01'),(1,'2026-03-01'),
                             (2,'2026-02-01'),(2,'2026-03-01'),
                             (3,'2026-02-01'),(3,'2026-03-01');
""")
panel = """
SELECT s.customer_id, s.as_of,
       COUNT(e.event_at) AS events_before,
       ROW_NUMBER() OVER (PARTITION BY s.customer_id ORDER BY s.as_of) AS period,
       LAG(COUNT(e.event_at)) OVER (PARTITION BY s.customer_id ORDER BY s.as_of)
         AS events_prev_period
FROM snapshots s
LEFT JOIN events e ON e.customer_id = s.customer_id AND e.event_at < s.as_of
GROUP BY s.customer_id, s.as_of
ORDER BY s.customer_id, s.as_of
"""
print(f"{'cust':>5}{'as_of':>13}{'before':>8}{'period':>8}{'prev':>7}")
for r in con.execute(panel):
    print(f"{r[0]:>5}{r[1]:>13}{r[2]:>8}{r[3]:>8}{str(r[4]) if r[4] is not None else '-':>7}")
print("\nthat is Data-Science lesson 02's customer-month panel, built in SQL")
```

```text
 cust        as_of  before  period   prev
    1   2026-02-01       2       1      -
    1   2026-03-01       3       2      2
    2   2026-02-01       1       1      -
    2   2026-03-01       2       2      1
    3   2026-02-01       3       1      -
    3   2026-03-01       3       2      3

that is Data-Science lesson 02's customer-month panel, built in SQL
```

A `snapshots` table of `(customer_id, as_of)` pairs, joined to events with the
cutoff condition, gives you **one row per customer per period with only the data
available at that moment**. `LAG` then gives the previous period's value — a
trend feature, computed without a loop.

And a reminder that comes with it: this panel has **two rows per customer**, so a
random train/test split puts the same customer on both sides. Data-Science
lesson 02 measured 3,291 of 4,000 customers leaking that way. **Split on
`customer_id`, not on rows.**

---

## Where feature logic should live

| Logic | Where | Why |
|---|---|---|
| Filtering, joining, aggregating, windows | **SQL** | It is what the engine does, at any scale |
| The `as_of` cutoff | **SQL**, in the `ON` clause | One place, auditable |
| Scaling, encoding, imputation | **Python, inside the pipeline** | [Data-Science 04](../../Data-Science/lessons/04-features-and-pipelines.md): it must be fitted on the training fold only |
| Model-specific transforms | Python | Same reason |
| Anything with a library | Python | SQL cannot |

The split is sharper than it looks. **A fitted transformation must never be in
SQL**, because SQL has no concept of "fitted on train only" — put a mean in a
SQL view and you have computed it over every row, which is
[Data-Science lesson 03](../../Data-Science/lessons/03-the-data-you-have.md)'s
0.720-AUC-from-noise leak.

```text
SQL     ->  a point-in-time feature table, no fitting
Python  ->  impute, scale, encode, inside the pipeline
```

That boundary gives you scale from the database and correctness from the
pipeline.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A join with no `as_of` condition | The future is in your features |
| The cutoff in `WHERE` instead of `ON` | Customers with zero events disappear |
| Computing a mean in a SQL view for a feature | Fitted on all rows: leakage |
| One query per time window | A `CASE` inside `COUNT` does all of them |
| A random split on a panel | The same customer in train and test |
| Feature logic duplicated in SQL and Python | They drift, and the model sees a different world in production |

---

## Exercises

1. Take a feature query you already have and check whether it has an `as_of`
   condition. If not, add it and compare the numbers.
2. Move the cutoff from `ON` to `WHERE` and count the rows you lose.
3. Build three time windows (7, 30, 90 days) in one query with `CASE`.
4. Build a snapshot panel for your own data and confirm every row uses only
   prior information.
5. Find a fitted statistic (a mean, a scale, a target encoding) living in a SQL
   view. Move it into the pipeline.

---

**Next:** [Lesson 08 — Beyond Relational](08-beyond-relational.md)
