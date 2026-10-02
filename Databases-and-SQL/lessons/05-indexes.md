# Lesson 05 — Indexes and Query Plans

**Goal:** make a slow query fast, and know what the fix costs.

## What you will learn

- An index, measured: 605x
- What it costs on writes
- The leftmost-prefix rule
- Reading a query plan

---

## An index, measured

```python
# skip-verify: wall-clock timings vary; the RATIOS are the lesson
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

def timed(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out

print("ready:", con.execute("SELECT COUNT(*) FROM orders").fetchone()[0], "orders")
```

```text
ready: 200000 orders
```

```python
# skip-verify: wall-clock timings vary
Q = "SELECT COUNT(*), SUM(amount) FROM orders WHERE customer_id = 2317"
print("=== S1 an index, measured ===")
t_no, _ = timed(lambda: con.execute(Q).fetchone())
plan_no = con.execute("EXPLAIN QUERY PLAN " + Q).fetchall()[0][-1]
con.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
t_yes, _ = timed(lambda: con.execute(Q).fetchone())
plan_yes = con.execute("EXPLAIN QUERY PLAN " + Q).fetchall()[0][-1]
print(f"without index: {t_no*1000:>7.2f} ms   plan: {plan_no}")
print(f"with index   : {t_yes*1000:>7.2f} ms   plan: {plan_yes}")
print(f"speedup      : {t_no/t_yes:>7.0f}x")
```

```text
without index:    3.58 ms   plan: SCAN orders
with index   :    0.01 ms   plan: SEARCH orders USING INDEX idx_orders_customer (customer_id=?)
speedup      :     605x
```

**605 times faster**, from one line of DDL.

The plan tells you why. `SCAN orders` means *read all 200,000 rows and check
each one*. `SEARCH orders USING INDEX` means *jump straight to the matching
rows*. An index is a sorted structure that turns a scan into a seek — the same
difference as looking up a word in an index at the back of a book rather than
reading the book.

**`EXPLAIN QUERY PLAN` is the first thing to run on a slow query**, before
changing anything. Every database has an equivalent (`EXPLAIN ANALYZE` in
Postgres), and the word to look for is always the same: a **scan** of a large
table is the problem.

---

## What it costs

An index is not free. It is a second copy of that column, kept sorted, updated
on every write.

```python
# skip-verify: wall-clock timings vary
def insert_batch(con, n=20_000, start=10_000_000):
    rows = [(start + i, random.randint(1, 5000), "2026-09-01", 100.0, "delivered")
            for i in range(n)]
    t0 = time.perf_counter()
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", rows)
    con.commit()
    return time.perf_counter() - t0

t_with = insert_batch(con, start=10_000_000)
con.execute("DROP INDEX idx_orders_customer")
t_without = insert_batch(con, start=20_000_000)
print(f"insert 20,000 rows WITH an index   : {t_with*1000:>7.0f} ms")
print(f"insert 20,000 rows WITHOUT         : {t_without*1000:>7.0f} ms")
print(f"the index cost {t_with/t_without:.2f}x on writes")
```

```text
insert 20,000 rows WITH an index   :      14 ms
insert 20,000 rows WITHOUT         :       7 ms
the index cost 1.97x on writes
```

**Reads 605x faster, writes 1.97x slower.** That is almost always a trade worth
making — but it is a trade, and it compounds: five indexes on a table means
every insert updates five structures.

| Index when | Do not index when |
|---|---|
| The column appears in `WHERE`, `JOIN` or `ORDER BY` | The table is small (under ~1,000 rows) |
| The table is large and read often | The column is almost all one value |
| A foreign key (most databases do **not** do this for you) | The table is write-heavy and rarely queried that way |
| A query shows up as a `SCAN` in the plan | You are guessing. **Measure first** |

The third row catches people: declaring `REFERENCES customers(customer_id)`
creates a *constraint*, not an index. Joins on an unindexed foreign key are the
second most common cause of a slow query, after N+1.

---

## The leftmost-prefix rule

A composite index on `(a, b)` is sorted by `a`, then by `b`. That shape decides
what it can serve.

```python
# skip-verify: wall-clock timings vary
con.execute("CREATE INDEX idx_cust_date ON orders(customer_id, ordered_at)")
tests = [
    ("customer_id = 100", "SELECT COUNT(*) FROM orders WHERE customer_id = 100"),
    ("customer_id AND date", "SELECT COUNT(*) FROM orders WHERE customer_id = 100 AND ordered_at = '2026-03-14'"),
    ("date only", "SELECT COUNT(*) FROM orders WHERE ordered_at = '2026-03-14'"),
]
print(f"{'query filters on':<24}{'how the index is used':<26}{'ms':>8}")
for label, q in tests:
    plan = con.execute("EXPLAIN QUERY PLAN " + q).fetchall()[0][-1]
    if plan.startswith("SEARCH"):
        used = "SEARCH - seeks directly"
    elif "USING" in plan and "INDEX" in plan:
        used = "SCAN - reads it all"
    else:
        used = "full table scan"
    t, _ = timed(lambda q=q: con.execute(q).fetchone())
    print(f"{label:<24}{used:<26}{t*1000:>8.2f}")
print("\nan index on (a, b) serves a, and (a,b) — never b alone")
```

```text
query filters on        how the index is used           ms
customer_id = 100       SEARCH - seeks directly       0.00
customer_id AND date    SEARCH - seeks directly       0.00
date only               SCAN - reads it all           4.19

an index on (a, b) serves a, and (a,b) — never b alone
```

Note the third row carefully. The plan *mentions* the index — but it says
**SCAN**, not SEARCH: the engine read the whole index from start to finish
because `ordered_at` is the second column and the index is not sorted by it.
4.19 ms against 0.00.

**"The plan mentions my index" is not the same as "my index is being used."**
Look for the word SEARCH.

The rule in practice:

```text
index on (customer_id, ordered_at) serves:
  WHERE customer_id = ?                          yes, SEARCH
  WHERE customer_id = ? AND ordered_at > ?       yes, SEARCH
  WHERE customer_id = ? ORDER BY ordered_at      yes, and the sort is free
  WHERE ordered_at > ?                           NO - needs its own index
```

So **order the columns by selectivity and by how you query**: the column you
filter on exactly goes first, the range or sort column second.

---

## Reading a plan

The four things to look for, in order of how much they cost you:

| In the plan | Means | Fix |
|---|---|---|
| `SCAN <big table>` | Reading everything | An index on the filter column |
| `SCAN` inside a loop over another table | A nested loop with no index | Index the join column |
| `USE TEMP B-TREE FOR ORDER BY` | Sorting at query time | An index matching the sort |
| `SEARCH ... USING INDEX` | What you want | Nothing |

And the habit that prevents most problems: **when you write a query, run its
plan before you ship it.** It costs ten seconds, and it is the difference
between finding a missing index now and finding it when the table reaches ten
million rows.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Not running `EXPLAIN QUERY PLAN` | You are guessing at the cause |
| Assuming a foreign key is indexed | It is a constraint, not an index |
| Indexing every column | Every write pays for all of them |
| Reading "index" in the plan as success | SCAN vs SEARCH is the whole difference |
| A composite index in the wrong column order | It serves the wrong queries |
| Indexing a low-cardinality column alone | A scan of half the table either way |
| Adding an index without measuring | You cannot tell whether it helped |

---

## Exercises

1. Take your slowest real query and run its plan. Which line is the problem?
2. Add the index it needs, re-run the plan, and measure before and after.
3. Measure your insert throughput with and without that index.
4. Build a composite index and find a query it does **not** serve.
5. Find a foreign key in your schema with no index on it.

---

**Next:** [Lesson 06 — Correctness](06-correctness.md)
