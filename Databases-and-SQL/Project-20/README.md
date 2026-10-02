# Project 20 — A Schema You Can Defend

**Do this after the eight lessons.**

You design a schema for a real domain, load real data into it, and prove the
queries are both **correct** and **fast** — with the plans and the timings to
show it.

One to two days. No modelling.

---

## Requirements

### 1. The domain and the schema
- A real domain: your work, a public dataset, or a system you use
- **At least four tables** with foreign keys between them
- Every table: a primary key, `NOT NULL` where it belongs, and **at least three
  `CHECK` constraints** that make wrong data impossible
- At least one **unique constraint on a business key** (lesson 02)
- An ER diagram, as Mermaid, in the repository
- A written note on anything you **denormalised on purpose**, and why

### 2. Prove the constraints work
- A script that attempts **at least six invalid inserts** — a missing foreign
  key, a negative amount, a bad enum, a null, a duplicate business key, one of
  your own
- Each one rejected, with the error named
- `PRAGMA foreign_keys = ON`, or the equivalent, verified

### 3. Load real data
- At least **100,000 rows** in the largest table
- Loaded by a script, idempotently — running it twice must not duplicate
- The row counts, and how many rows the constraints rejected

### 4. Ten queries
Each with its purpose in one line. Between them they must include:

| Must include | Lesson |
|---|---|
| A `GROUP BY` with `HAVING` | 03 |
| A `LEFT JOIN` where `INNER` would be wrong — **with both row counts** | 03 |
| A CTE chain of at least three steps | 03 |
| `ROW_NUMBER()` for "latest per group" | 04 |
| `RANK()` or `DENSE_RANK()` within a partition | 04 |
| `LAG` or `LEAD` for a period-on-period change | 04 |
| A running total with `SUM(...) OVER (ORDER BY ...)` | 04 |
| A point-in-time feature join with an `as_of` cutoff | 07 |
| A query that exposes a NULL trap, and the fix | 03 |

### 5. Make them fast
- `EXPLAIN QUERY PLAN` for **every** one of the ten, before and after indexing
- At least **three indexes**, each justified by a plan that showed a SCAN
- Before/after timings for each
- **One composite index**, with a query it serves and a query it does not
  (lesson 05)
- The **write cost** of your indexes, measured

### 6. Correctness
- One operation that writes to two tables, inside a transaction, with a test
  that proves rollback works
- One **read-modify-write converted to a single statement**
- An **idempotency key** with a unique constraint, and a retry-storm test showing
  one row after five attempts

### 7. The ML table
- A point-in-time feature table built in SQL, with **at least four features** and
  an `as_of` column
- Proof that no feature uses data after `as_of` — show the leaky version's
  numbers beside the correct one
- A snapshot panel with at least two periods per entity
- A note on why the split must be on the entity, not on rows

### 8. The write-up

```text
DOMAIN          what this models, and for whom
SCHEMA          the tables, and the three decisions you had to make
CONSTRAINTS     what is now impossible to insert
REJECTED        how many real rows the constraints caught
QUERIES         ten, each with its purpose
PERFORMANCE     the three worst plans, before and after, with timings
INDEX COST      what the indexes cost on writes
CORRECTNESS     the transaction test and the idempotency test
ML TABLE        the leaky number beside the correct one
WHAT I WOULD DO DIFFERENTLY
```

---

## Deliverables

```text
project-20/
├── REPORT.md             the write-up
├── schema/
│   ├── 01-tables.sql     DDL with constraints
│   ├── 02-indexes.sql    each with a comment naming the query it serves
│   └── er-diagram.md     Mermaid
├── load/
│   └── load.py           idempotent loader, with counts
├── queries/
│   ├── 01-*.sql ... 10-*.sql    each with its purpose in a header comment
│   └── plans.md          EXPLAIN output before and after, with timings
├── tests/
│   ├── test_constraints.py   six invalid inserts, each rejected
│   ├── test_transaction.py   rollback proven
│   └── test_idempotency.py   retry storm, one row
└── ml/
    ├── features.sql      the point-in-time table
    └── leakage.md        the leaky numbers beside the correct ones
```

---

## Marking

| Weight | Criterion |
|---|---|
| 15% | Schema: 4+ tables, foreign keys, 3+ CHECKs, a business-key unique |
| 10% | Six invalid inserts, each rejected and named |
| 10% | 100,000+ rows loaded idempotently, with counts |
| 20% | Ten queries covering all nine required features |
| 20% | Plans before and after, three justified indexes, write cost measured |
| 15% | Transaction rollback test, read-modify-write fix, idempotency test |
| 10% | The point-in-time table, with the leaky comparison |

Automatic deductions: a schema with no CHECK constraints; an index added without
a plan justifying it; a timing claim with no before/after; a feature query with
no `as_of`; a loader that duplicates on a second run; a `LEFT JOIN` requirement
met without showing both row counts.

---

## Before you submit

- [ ] Every table has a primary key and `NOT NULL` where it belongs
- [ ] At least three `CHECK` constraints exist and are tested
- [ ] Foreign key enforcement is **on**, and verified
- [ ] Six invalid inserts are rejected, each with its error named
- [ ] The loader is idempotent — proven by running it twice
- [ ] All nine required query features appear
- [ ] Every query has an `EXPLAIN` before and after
- [ ] Every index has a comment naming the query it serves
- [ ] The composite index has a query it does **not** serve, documented
- [ ] The transaction rollback test passes
- [ ] The retry storm leaves exactly one row
- [ ] The leaky feature numbers are shown beside the correct ones
