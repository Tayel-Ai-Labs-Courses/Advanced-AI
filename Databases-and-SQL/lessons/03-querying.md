# Lesson 03 — Querying

**Goal:** write a query that says what you want, and understand the join that
silently deletes your rows.

## What you will learn

- The six clauses, and the order they really run in
- `INNER` against `LEFT`, measured
- CTEs, which keep SQL reviewable
- `NULL`, which is not a value

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

## The six clauses

```python
show("""
SELECT c.governorate, COUNT(*) AS orders, ROUND(SUM(o.amount),2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status = 'delivered' AND o.ordered_at >= '2026-06-01'
GROUP BY c.governorate
HAVING COUNT(*) > 5000
ORDER BY revenue DESC
LIMIT 3
""", ["governorate", "orders", "revenue"], [14, 9, 14])
```

```text
   governorate   orders       revenue
         Cairo   14,439  6,816,007.44
      Dakahlia   14,444  6,802,233.85
          Giza   14,375  6,755,944.25
```

You write them in one order and the engine runs them in another, which explains
almost every "why can't I use that alias" error:

```text
written:   SELECT ... FROM ... WHERE ... GROUP BY ... HAVING ... ORDER BY ... LIMIT
runs as:   FROM -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY -> LIMIT
```

Two consequences you will meet within a week:

- **`WHERE` cannot see an alias from `SELECT`**, because `SELECT` has not run
  yet. `ORDER BY` can, because it runs after.
- **`WHERE` filters rows; `HAVING` filters groups.** `WHERE` runs before
  grouping, so it cannot refer to `COUNT(*)`; `HAVING` runs after, so it can.

Put every condition you can in `WHERE` rather than `HAVING` — filtering before
grouping means grouping less.

---

## The join that loses rows

```python
for label, join in (("INNER JOIN", "JOIN"), ("LEFT JOIN", "LEFT JOIN")):
    n = con.execute(f"""
        SELECT COUNT(*) FROM customers c
        {join} orders o ON o.customer_id = c.customer_id AND o.amount > 880
    """).fetchone()[0]
    distinct = con.execute(f"""
        SELECT COUNT(DISTINCT c.customer_id) FROM customers c
        {join} orders o ON o.customer_id = c.customer_id AND o.amount > 880
    """).fetchone()[0]
    print(f"{label:<12} rows {n:>7,}   distinct customers {distinct:>6,}")
print("customers total:", con.execute("SELECT COUNT(*) FROM customers").fetchone()[0])
```

```text
INNER JOIN   rows   4,646   distinct customers  3,052
LEFT JOIN    rows   6,594   distinct customers  5,000
customers total: 5000
```

**The inner join silently dropped 1,948 of 5,000 customers** — everyone with no
order above 880 EGP.

If the question was "revenue from large orders by customer", that is correct. If
it was "how many large orders has each customer placed", the inner join answers
it **only for customers who have at least one**, and your average is computed
over the wrong denominator.

```text
INNER JOIN  keeps rows that matched on BOTH sides
LEFT JOIN   keeps every row from the left, NULL where nothing matched
```

The habit that catches it: **after any join, check the row count against what
you expected.** A join that changes the number of rows unexpectedly is either
losing data or duplicating it, and both are silent.

| Symptom | Cause |
|---|---|
| Fewer rows than the left table | An inner join where you wanted left |
| **More** rows than the left table | The join key is not unique on the right — a fan-out |
| Sums larger than reality | The same fan-out, now multiplying your money |

The second and third are the dangerous pair: joining orders to a table with two
rows per order doubles every amount, and nothing warns you.

---

## CTEs keep it readable

```python
show("""
WITH delivered AS (
    SELECT customer_id, amount FROM orders WHERE status = 'delivered'
),
per_customer AS (
    SELECT customer_id, COUNT(*) AS n, ROUND(SUM(amount),2) AS spend
    FROM delivered GROUP BY customer_id
)
SELECT c.governorate,
       COUNT(*)                        AS customers,
       ROUND(AVG(p.spend), 2)          AS avg_spend,
       ROUND(AVG(p.n), 2)              AS avg_orders
FROM per_customer p
JOIN customers c USING (customer_id)
GROUP BY c.governorate
ORDER BY avg_spend DESC
""", ["governorate", "customers", "avg_spend", "avg_orders"], [14, 11, 12, 12])
```

```text
   governorate  customers   avg_spend  avg_orders
      Dakahlia      1,004   15,182.82       32.22
    Alexandria        987   15,102.39       32.06
       Sharqia        970   15,092.28       32.14
          Giza      1,005   15,036.52       31.95
         Cairo      1,034   14,910.60       31.73
```

Each `WITH` block is a named step. Compare with the nested-subquery version of
the same thing, and the difference in reviewability is the whole argument.

**Use a CTE per logical step, name it after what it contains**, and keep each
one short enough to check on its own. This is how a 200-line query stays
verifiable — lesson 01's point about SQL being unreviewable past 50 lines.

---

## NULL is not a value

```python
rows = con.execute("""
    SELECT
      (SELECT COUNT(*) FROM orders WHERE amount = NULL)        AS eq_null,
      (SELECT COUNT(*) FROM orders WHERE amount IS NULL)       AS is_null,
      (SELECT COUNT(*) FROM orders WHERE status != 'delivered') AS not_delivered,
      (SELECT COUNT(*) FROM orders)                            AS total
""").fetchone()
print(f"WHERE amount = NULL     -> {rows[0]:,}   (always 0: NULL = NULL is unknown)")
print(f"WHERE amount IS NULL    -> {rows[1]:,}")
print(f"WHERE status != 'x'     -> {rows[2]:,} of {rows[3]:,}")
print(f"\nNULL is 'unknown'. Comparing with it gives unknown, not true or false.")
```

```text
WHERE amount = NULL     -> 0   (always 0: NULL = NULL is unknown)
WHERE amount IS NULL    -> 0
WHERE status != 'x'     -> 39,917 of 200,000

NULL is 'unknown'. Comparing with it gives unknown, not true or false.
```

`WHERE x = NULL` matches nothing, ever — including rows where `x` is NULL. Use
`IS NULL`.

And the trap that silently loses rows: **`WHERE status != 'delivered'` excludes
rows where `status` is NULL**, because `NULL != 'delivered'` is unknown, not
true. If NULLs are possible, write `WHERE status IS NULL OR status != 'delivered'`
— or constrain the column `NOT NULL` in the first place (lesson 02).

| Expression | Result |
|---|---|
| `NULL = NULL` | unknown (not true) |
| `NULL != 'x'` | unknown |
| `COUNT(column)` | **skips** NULLs |
| `COUNT(*)` | counts every row |
| `SUM`, `AVG` | skip NULLs — so `AVG` has a different denominator than you think |
| `COALESCE(x, 0)` | the fix |

`COUNT(column)` against `COUNT(*)` is the cheapest NULL check there is: if they
differ, you have NULLs.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| `INNER JOIN` where you wanted `LEFT` | 1,948 customers vanished silently |
| Not checking row counts after a join | Fan-out doubles your money with no warning |
| Conditions in `HAVING` that belong in `WHERE` | You grouped rows you were going to discard |
| Using a `SELECT` alias in `WHERE` | `SELECT` has not run yet |
| `= NULL` | Matches nothing, ever |
| `!= 'x'` on a nullable column | Silently excludes the NULLs |
| `AVG` on a column with NULLs | A denominator you did not choose |
| Nested subqueries four deep | Unreviewable; use CTEs |

---

## Exercises

1. Take a query of yours with an inner join. Switch it to `LEFT` and compare the
   row counts. Which was right?
2. Find a join in your codebase that fans out. What does it do to the sums?
3. Rewrite a nested-subquery query of yours with CTEs. Is it clearer?
4. Run `COUNT(*)` against `COUNT(column)` on every column of a real table. Where
   are the NULLs?
5. Find a `!=` filter on a nullable column in your code and work out how many
   rows it is silently dropping.

---

**Next:** [Lesson 04 — Window Functions](04-window-functions.md)
