# Lesson 04 — Window Functions

**Goal:** rank, compare and accumulate **without collapsing your rows** — the
single most useful SQL feature most people never learn.

## What you will learn

- `GROUP BY` against `OVER`, measured
- Ranking within a group
- `LAG`, `LEAD`, and running totals
- Where this replaces a pandas loop

---

## Setup

```python
import sqlite3, time, random

import sqlite3, time, random

def build(n_customers=5_000, n_orders=200_000, seed=0):
    rng = random.Random(seed)
    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            governorate TEXT NOT NULL,
            signup_date TEXT NOT NULL
        );
        CREATE TABLE orders (
            order_id    INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            ordered_at  TEXT NOT NULL,
            amount      REAL NOT NULL,
            status      TEXT NOT NULL
        );
    """)
    govs = ["Cairo", "Giza", "Alexandria", "Dakahlia", "Sharqia"]
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)",
        [(i, f"customer_{i}", rng.choice(govs),
          f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}")
         for i in range(1, n_customers + 1)])
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)",
        [(j, rng.randint(1, n_customers),
          f"2026-{rng.randint(1,9):02d}-{rng.randint(1,28):02d}",
          round(rng.uniform(40, 900), 2),
          rng.choice(["delivered"] * 8 + ["cancelled", "shipped"]))
         for j in range(1, n_orders + 1)])
    con.commit()
    return con
con = build()

def show(sql, header, widths, params=()):
    rows = con.execute(sql, params).fetchall()
    print("".join(f"{h:>{w}}" for h, w in zip(header, widths)))
    for r in rows:
        print("".join(
            f"{str(v) if not isinstance(v,(int,float)) else (f'{v:,.2f}' if isinstance(v,float) else f'{v:,}'):>{w}}"
            for v, w in zip(r, widths)))
    return rows

print("ready")
```

```text
ready
```


---

## The difference, in one number

```python
g = con.execute("SELECT COUNT(*) FROM (SELECT governorate FROM customers GROUP BY governorate)").fetchone()[0]
w = con.execute("SELECT COUNT(*) FROM (SELECT customer_id, COUNT(*) OVER (PARTITION BY governorate) FROM customers)").fetchone()[0]
print(f"GROUP BY governorate  -> {g} rows   (one per group: it COLLAPSES)")
print(f"COUNT(*) OVER (...)   -> {w:,} rows (one per input row: it ADDS a column)")
```

```text
GROUP BY governorate  -> 5 rows   (one per group: it COLLAPSES)
COUNT(*) OVER (...)   -> 5,000 rows (one per input row: it ADDS a column)
```

That is the whole idea.

- **`GROUP BY`** reduces many rows to one per group. You lose the detail.
- **`OVER (PARTITION BY ...)`** computes the same aggregate and **attaches it to
  every row**. You keep the detail.

So the question "what fraction of its governorate's revenue is each customer?"
needs a window function: it requires both the individual value *and* the group
total on the same row. Without windows you need a self-join or a subquery, and
with them it is one clause.

---

## Ranking within a group

```python
show("""
WITH per_customer AS (
  SELECT customer_id, ROUND(SUM(amount),2) AS spend
  FROM orders WHERE status='delivered' GROUP BY customer_id
)
SELECT p.customer_id, c.governorate, p.spend,
       RANK()       OVER (PARTITION BY c.governorate ORDER BY p.spend DESC) AS rank_in_gov,
       ROUND(100.0 * p.spend / SUM(p.spend) OVER (PARTITION BY c.governorate), 2) AS pct_of_gov
FROM per_customer p JOIN customers c USING (customer_id)
WHERE c.governorate = 'Cairo'
ORDER BY rank_in_gov LIMIT 5
""", ["cust", "governorate", "spend", "rank", "pct"], [7, 14, 12, 7, 8])
```

```text
   cust   governorate       spend   rank     pct
  2,993         Cairo   26,597.16      1    0.17
     27         Cairo   24,791.81      2    0.16
  3,931         Cairo   24,658.67      3    0.16
  2,846         Cairo   24,427.75      4    0.16
  3,564         Cairo   23,928.43      5    0.16
```

Two windows in one query: a `RANK` within governorate, and each customer's share
of their governorate's total. The alternative is a self-join plus a subquery,
and it is slower and much harder to read.

**The three ranking functions differ only on ties:**

| Function | 100, 100, 90 becomes |
|---|---|
| `ROW_NUMBER()` | 1, 2, 3 — arbitrary tie-break |
| `RANK()` | 1, 1, 3 — ties share, then skip |
| `DENSE_RANK()` | 1, 1, 2 — ties share, no gap |

Use `ROW_NUMBER()` for "pick exactly one per group" (deduplication), `RANK()`
for a leaderboard, `DENSE_RANK()` when gaps would confuse a reader.

---

## Deduplication: the most common real use

```python
show("""
WITH ranked AS (
  SELECT customer_id, order_id, amount, ordered_at,
         ROW_NUMBER() OVER (PARTITION BY customer_id
                            ORDER BY ordered_at DESC, order_id DESC) AS rn
  FROM orders WHERE status = 'delivered'
)
SELECT customer_id, order_id, amount, ordered_at
FROM ranked WHERE rn = 1
ORDER BY customer_id LIMIT 5
""", ["cust", "order", "amount", "ordered_at"], [7, 9, 10, 14])
```

```text
   cust    order    amount    ordered_at
      1  173,522    505.88    2026-09-27
      2   51,138    648.02    2026-09-26
      3   88,968    251.93    2026-09-17
      4  151,708    460.92    2026-09-28
      5  125,532    409.44    2026-09-18
```

**"The most recent row per customer"** — one `ROW_NUMBER()` and a `WHERE rn = 1`.

This pattern is everywhere: the latest price, the current status, the newest
address, deduplicating a messy import. It is the single most useful thing in
this lesson, and the alternative (a correlated subquery with `MAX`) is both
slower and wrong when there are ties.

---

## Comparing to the previous row

```python
show("""
WITH monthly AS (
  SELECT substr(ordered_at, 1, 7) AS month,
         ROUND(SUM(amount), 2) AS revenue
  FROM orders WHERE status = 'delivered'
  GROUP BY month
)
SELECT month, revenue,
       LAG(revenue) OVER (ORDER BY month) AS prev_month,
       ROUND(revenue - LAG(revenue) OVER (ORDER BY month), 2) AS change,
       ROUND(SUM(revenue) OVER (ORDER BY month), 2) AS running_total
FROM monthly ORDER BY month LIMIT 6
""", ["month", "revenue", "prev", "change", "running"], [10, 14, 14, 12, 16])
```

```text
     month       revenue          prev      change         running
   2026-01  8,314,062.65          None        None    8,314,062.65
   2026-02  8,405,167.76  8,314,062.65   91,105.11   16,719,230.41
   2026-03  8,451,109.36  8,405,167.76   45,941.60   25,170,339.77
   2026-04  8,314,459.03  8,451,109.36 -136,650.33   33,484,798.80
   2026-05  8,385,881.94  8,314,459.03   71,422.91   41,870,680.74
   2026-06  8,447,538.60  8,385,881.94   61,656.66   50,318,219.34
```

Three features in one query:

- **`LAG`** reaches back one row in the ordering — month-on-month change without
  a self-join.
- **`SUM(...) OVER (ORDER BY ...)`** is a running total, because adding `ORDER BY`
  to a window makes its frame cumulative by default.
- The first row's `prev` is `None`, correctly — there is no previous month.

That last point matters for features: a `LAG` column always has NULLs at the
start of each partition, and
[Data-Science lesson 04](../../Data-Science/lessons/04-features-and-pipelines.md)'s
imputer has to handle them. Do not silently `COALESCE` them to 0 — "no previous
month" is different from "zero last month".

---

## Where this replaces pandas

| Task | pandas | SQL |
|---|---|---|
| Latest row per group | `sort_values().groupby().head(1)` | `ROW_NUMBER() ... WHERE rn = 1` |
| Rank within group | `groupby().rank()` | `RANK() OVER (PARTITION BY ...)` |
| Share of group total | `transform('sum')` then divide | `SUM(...) OVER (PARTITION BY ...)` |
| Previous period | `shift(1)` | `LAG(...)` |
| Running total | `cumsum()` | `SUM(...) OVER (ORDER BY ...)` |
| Rolling 7-day | `rolling(7)` | `... OVER (ORDER BY d ROWS 6 PRECEDING)` |

Every one of these is a window function, and doing it in SQL means it happens
**before** the data crosses the connection — lesson 01's "SQL reduces, Python
computes".

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| `GROUP BY` when you needed the detail too | You lost the rows; now you need a self-join |
| A correlated subquery for "latest per group" | Slower, and wrong on ties |
| `RANK()` where you needed `ROW_NUMBER()` | Ties give you two rows where you wanted one |
| Forgetting `PARTITION BY` | The window spans the whole table |
| `COALESCE`-ing a `LAG` NULL to zero | "No previous period" is not "zero" |
| A window in `WHERE` | Windows run after `WHERE`; wrap it in a CTE |

That last one is worth remembering: you cannot write `WHERE ROW_NUMBER() ... = 1`.
Windows are computed after `WHERE`, which is why the deduplication pattern needs
a CTE.

---

## Exercises

1. Write "the latest order per customer" with a window and with a correlated
   subquery. Compare the plans and the times.
2. Add `DENSE_RANK()` beside `RANK()` on a column with ties. Explain the
   difference to a colleague.
3. Compute a 7-day rolling average of daily revenue with `ROWS 6 PRECEDING`.
4. Find a pandas `groupby().transform()` in your code and rewrite it as a window.
5. Try putting a window function in `WHERE`, read the error, and fix it with a CTE.

---

**Next:** [Lesson 05 — Indexes and Query Plans](05-indexes.md)
