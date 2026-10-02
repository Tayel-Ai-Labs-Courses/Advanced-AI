# Lesson 01 — Why SQL

**Goal:** know when the database should do the work, and when pandas should.

## What you will learn

- The N+1 problem, measured
- Where the computation should happen
- SQL against pandas, honestly
- What SQL is genuinely bad at

---

## A database to work with

Lessons 01 through 07 use this one, built in memory with **nothing installed** —
`sqlite3` is in the Python standard library.

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
for t in ("customers", "orders"):
    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t:<12}{n:>9,} rows")
```

```text
customers       5,000 rows
orders        200,000 rows
```

---

## One query instead of a loop

```python
# skip-verify: wall-clock timings vary; the RATIO is the lesson
def timed(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    return best, out

def in_sql():
    return con.execute("""
        SELECT c.governorate,
               COUNT(*)        AS orders,
               ROUND(SUM(o.amount), 2) AS revenue,
               ROUND(AVG(o.amount), 2) AS avg_order
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        WHERE o.status = 'delivered'
        GROUP BY c.governorate
        ORDER BY revenue DESC
    """).fetchall()

def in_python():
    rows = con.execute("SELECT customer_id, amount, status FROM orders").fetchall()
    cust = dict(con.execute("SELECT customer_id, governorate FROM customers").fetchall())
    agg = {}
    for cid, amt, status in rows:
        if status != "delivered":
            continue
        g = cust[cid]
        a = agg.setdefault(g, [0, 0.0])
        a[0] += 1; a[1] += amt
    return sorted(((g, c, round(s, 2), round(s / c, 2)) for g, (c, s) in agg.items()),
                  key=lambda r: -r[2])

t_sql, r_sql = timed(in_sql)
t_py, r_py = timed(in_python)
print(f"{'governorate':<14}{'orders':>9}{'revenue':>14}{'avg':>9}")
for g, c, s, a in r_sql:
    print(f"{g:<14}{c:>9,}{s:>14,.2f}{a:>9.2f}")
print(f"\nSQL      : {t_sql*1000:>7.1f} ms")
print(f"Python   : {t_py*1000:>7.1f} ms   ({t_py/t_sql:.1f}x slower)")
print(f"same answer: {r_sql == r_py}")

```

```text
governorate      orders       revenue      avg
Cairo            32,809 15,417,563.49   469.92
Dakahlia         32,350 15,243,553.48   471.21
Giza             32,105 15,111,700.11   470.70
Alexandria       31,643 14,906,055.07   471.07
Sharqia          31,176 14,639,509.36   469.58

SQL      :    48.1 ms
Python   :    78.4 ms   (1.6x slower)
same answer: True
```

**1.6x** — honest, and smaller than people claim. On an in-memory SQLite with
200,000 rows, Python is not catastrophically slower, and anyone promising "SQL
is 100x faster" for an aggregate like this is quoting a different benchmark.

The real arguments for doing it in SQL are not raw speed:

- **Thirteen lines against twenty-six**, and the SQL version says what it wants
  rather than how to get it.
- **The data never leaves the database.** The Python version pulled 200,000 rows
  across the connection to produce five.
- It keeps working when the table is 200 **million** rows, where pulling
  everything into memory stops being an option at all.

That third point is the one that matters, and the next section is where it
becomes dramatic.

---

## The N+1 problem

```python
# skip-verify: wall-clock timings vary; the RATIO is the lesson
ids = [i for i in range(1, 501)]
def n_plus_one():
    out = []
    for cid in ids:
        row = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(amount),0) FROM orders WHERE customer_id = ?",
            (cid,)).fetchone()
        out.append((cid, row[0], row[1]))
    return out
def one_query():
    q = f"""SELECT customer_id, COUNT(*), COALESCE(SUM(amount),0)
            FROM orders WHERE customer_id IN ({','.join('?'*len(ids))})
            GROUP BY customer_id"""
    return con.execute(q, ids).fetchall()
t_n, _ = timed(n_plus_one)
t_1, _ = timed(one_query)
print(f"{len(ids)} separate queries : {t_n*1000:>8.1f} ms")
print(f"one grouped query     : {t_1*1000:>8.1f} ms   ({t_n/t_1:.0f}x faster)")

```

```text
500 separate queries :   1315.9 ms
one grouped query     :      8.2 ms   (161x faster)
```

**161 times faster**, and this is the single most common performance bug in
applications that use a database.

It happens whenever a loop queries inside it:

```python
# no-run
for customer in customers:          # 1 query for the list
    orders = fetch_orders(customer) # + 1 query EACH -> N+1 queries
```

Each query has fixed overhead — parse, plan, round trip — and 500 of those
dominate everything. On a database over a network rather than in memory, the
round trip alone is 1-10 ms, so the same loop takes **5-10 seconds**.

**The fix is always the same shape: one query that returns all of it,** with
`IN (...)`, a join, or a `GROUP BY`. If you are writing a loop that queries,
stop and write the query that would answer all iterations at once.

---

## Where the work should happen

```mermaid
flowchart TD
    Qwhat are you doing? --> A["filtering, joining, aggregating<br/>grouping, ranking, deduplicating"]
    Q --> B["training a model, plotting,<br/>text processing, anything iterative"]
    A --> A1["<b>SQL</b> — in the database"]
    B --> B1["<b>pandas / Python</b> — after SQL<br/>has reduced the rows"]
```

| Task | Where | Why |
|---|---|---|
| Filter 200M rows to 10k | **SQL** | Never move what you will discard |
| Join two tables | **SQL** | It is what the engine is for |
| Group and aggregate | **SQL** | Same |
| Rank within groups | **SQL** (lesson 04) | Window functions |
| Deduplicate | **SQL** | `GROUP BY` or `DISTINCT` |
| Point-in-time features | **SQL** (lesson 07) | A join condition, not a loop |
| Train a model | pandas / sklearn | SQL cannot |
| Plot | pandas / matplotlib | SQL cannot |
| Tokenise, embed, call a model | Python | SQL cannot |
| Anything row-by-row and iterative | Python | SQL is declarative |

**The rule: SQL reduces, Python computes.** Push filtering, joining and
aggregation down; pull the small result up.

---

## What SQL is genuinely bad at

Being fair about this matters, because over-committing to SQL produces
unmaintainable monsters:

| SQL is bad at | Do it in |
|---|---|
| Iterative algorithms | Python |
| Anything needing a library (ML, NLP, images) | Python |
| Complex string processing | Python |
| Logic that needs tests and version control | Python — a 400-line query is unreviewable |
| Being read by a newcomer, past ~50 lines | Break it into CTEs, or move it |

A 300-line query with six nested subqueries is a correctness risk even when it
is fast. **Common table expressions (`WITH`) are how you keep SQL reviewable**,
and lesson 03 covers them.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| A loop that queries inside it | 161x slower here, worse over a network |
| `SELECT *` then filter in pandas | You moved rows you threw away |
| Aggregating in Python what SQL could group | More code, and it stops scaling |
| Putting model logic in SQL | Untestable, unversioned, unreviewable |
| A 300-line query | Nobody can verify it, including you |
| Assuming SQL is always faster | 1.6x here, not 100x |

---

## Exercises

1. Find an N+1 loop in code you have written. Rewrite it as one query and
   measure both.
2. Take a pandas pipeline that loads a whole table and push the filter and the
   aggregation into SQL. How many rows now cross the connection?
3. Write the governorate revenue query without looking at the one above.
4. Find a query in your codebase longer than 50 lines. Can you follow it?
5. List three things your team currently does in pandas that belong in SQL, and
   one thing in SQL that belongs in Python.

---

**Next:** [Lesson 02 — Designing a Schema](02-schema-design.md)
