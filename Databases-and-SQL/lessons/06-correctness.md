# Lesson 06 — Correctness

**Goal:** make concurrent writes safe, and stop paying a refund twice.

## What you will learn

- Transactions, and the four guarantees
- The lost-update race, demonstrated
- Isolation levels, and which one you have
- Idempotency at the database level

---

## A transaction is all-or-nothing

```python
import sqlite3
con = sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL NOT NULL CHECK (balance >= 0));
INSERT INTO accounts VALUES (1, 500.0), (2, 100.0);
""")

def transfer(con, src, dst, amount, atomic=True):
    try:
        if atomic:
            con.execute("BEGIN")
        con.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, src))
        con.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, dst))
        con.execute("COMMIT" if atomic else "SELECT 1")
        return "ok"
    except sqlite3.Error as e:
        if atomic:
            con.execute("ROLLBACK")
        return f"failed: {type(e).__name__}"

print("start      ", con.execute("SELECT id, balance FROM accounts").fetchall())
print("transfer 50", transfer(con, 1, 2, 50))
print("after      ", con.execute("SELECT id, balance FROM accounts").fetchall())
print("transfer 10000 (would overdraw):", transfer(con, 1, 2, 10_000))
print("after      ", con.execute("SELECT id, balance FROM accounts").fetchall())
```

```text
start       [(1, 500.0), (2, 100.0)]
transfer 50 ok
after       [(1, 450.0), (2, 150.0)]
transfer 10000 (would overdraw): failed: IntegrityError
after       [(1, 450.0), (2, 150.0)]
```

The failed transfer left the balances **exactly as they were.** The first
`UPDATE` had already run; the `CHECK` constraint rejected the second state, and
`ROLLBACK` undid the first.

Without the transaction, that first `UPDATE` would have stuck: 10,000 EGP would
have left account 1 and arrived nowhere.

**ACID, in one line each:**

| Guarantee | Means |
|---|---|
| **Atomicity** | All the statements, or none of them |
| **Consistency** | Constraints hold at every commit |
| **Isolation** | Concurrent transactions do not see each other's half-finished work |
| **Durability** | Once committed, it survives a crash |

Atomicity is the one you will reach for daily. **Any operation that touches two
rows and must agree belongs in a transaction** — a transfer, an order plus its
items, a status change plus its audit row.

---

## The lost update

Two processes read the same row, both modify it, both write. One update
disappears.

```python
con2 = sqlite3.connect(":memory:")
con2.executescript("CREATE TABLE counters (name TEXT PRIMARY KEY, n INTEGER NOT NULL);"
                   "INSERT INTO counters VALUES ('calls', 0);")

def read_modify_write(con, by):
    current = con.execute("SELECT n FROM counters WHERE name='calls'").fetchone()[0]
    con.execute("UPDATE counters SET n = ? WHERE name='calls'", (current + by,))

# two "processes", interleaved the way they would be in production
read_modify_write(con2, 1)       # A reads 0, writes 1
read_modify_write(con2, 1)       # B reads 1, writes 2
print("sequential  ->", con2.execute("SELECT n FROM counters").fetchone()[0])

con2.execute("UPDATE counters SET n = 0")
a = con2.execute("SELECT n FROM counters WHERE name='calls'").fetchone()[0]   # A reads 0
b = con2.execute("SELECT n FROM counters WHERE name='calls'").fetchone()[0]   # B reads 0
con2.execute("UPDATE counters SET n = ?", (a + 1,))                          # A writes 1
con2.execute("UPDATE counters SET n = ?", (b + 1,))                          # B writes 1
print("interleaved ->", con2.execute("SELECT n FROM counters").fetchone()[0],
      "(should be 2 — one update was lost)")

con2.execute("UPDATE counters SET n = 0")
con2.execute("UPDATE counters SET n = n + 1")        # the fix: compute IN the database
con2.execute("UPDATE counters SET n = n + 1")
print("n = n + 1   ->", con2.execute("SELECT n FROM counters").fetchone()[0])
```

```text
sequential  -> 2
interleaved -> 1 (should be 2 — one update was lost)
n = n + 1   -> 2
```

**`n = n + 1` is safe. Read-modify-write is not.**

The fix is to let the database do the arithmetic. Any pattern of the shape *read
a value into your application, change it, write it back* is a race, and the
window is as long as your round trip.

| Unsafe | Safe |
|---|---|
| `SELECT balance`, compute, `UPDATE balance = ?` | `UPDATE balance = balance - ?` |
| `SELECT MAX(id)+1` then insert | `AUTOINCREMENT` / a sequence |
| Check-then-insert | `INSERT ... ON CONFLICT DO NOTHING` |
| Read a counter, increment, write | `UPDATE n = n + 1` |

When the logic genuinely cannot be expressed in one statement, you need a lock
(`SELECT ... FOR UPDATE` in Postgres) or an optimistic version column — but try
the one-statement version first, because it is always cheaper.

---

## Idempotency, at the database level

[AI-Agents lesson 06](../../AI-Agents/lessons/06-reliability.md) measured three
retries paying out **1,350 EGP for one 450 EGP refund.** Here is the fix one
level down, where it cannot be bypassed:

```python
con3 = sqlite3.connect(":memory:")
con3.executescript("""
CREATE TABLE refunds (
    idempotency_key TEXT PRIMARY KEY,     -- the whole mechanism
    order_id        INTEGER NOT NULL,
    amount          REAL NOT NULL
);
""")

def refund(con, order_id, amount, key):
    try:
        con.execute("INSERT INTO refunds VALUES (?,?,?)", (key, order_id, amount))
        return "applied"
    except sqlite3.IntegrityError:
        return "already applied"

for attempt in range(3):
    print(f"attempt {attempt + 1}: {refund(con3, 1001, 450.0, 'refund:1001')}")
total = con3.execute("SELECT COUNT(*), COALESCE(SUM(amount),0) FROM refunds").fetchone()
print(f"\nrows: {total[0]}   paid out: {total[1]:,.0f} EGP")
```

```text
attempt 1: applied
attempt 2: already applied
attempt 3: already applied

rows: 1   paid out: 450 EGP
```

**Three attempts, one refund, 450 EGP.** The primary key does the work, and it
holds regardless of which service, retry loop or agent is calling.

Note the key: `refund:1001`, derived from **what the operation is**, not a random
UUID per attempt. A random key makes every retry a new operation — which is
exactly the bug it was meant to prevent.

---

## Isolation levels, briefly

| Level | Prevents | Still allows |
|---|---|---|
| Read uncommitted | nothing | dirty reads |
| **Read committed** (Postgres default) | dirty reads | the same query returning different rows twice |
| Repeatable read (MySQL default) | + that | phantom rows in a range |
| **Serializable** | everything | nothing — but transactions may be aborted and need a retry |

**Know which one your database uses by default**, because the guarantees you
assume are probably stronger than the ones you have. Postgres's read-committed
means two `SELECT`s in the same transaction can disagree, which surprises
people writing reports.

For analytics, this rarely matters. For money, use an explicit transaction,
compute in the database, and add a unique constraint on the business key.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Two related writes with no transaction | A half-finished state, permanently |
| Read-modify-write | A lost update, silently |
| `SELECT MAX(id)+1` | Two rows with the same id |
| A random idempotency key per attempt | Every retry is a new operation |
| Assuming serializable isolation | You almost certainly have read-committed |
| Idempotency in application code only | The next caller bypasses it |
| A long-running transaction | It holds locks and blocks everyone |

---

## Exercises

1. Find a read-modify-write in your codebase and convert it to one statement.
2. Add a unique constraint that makes your most expensive duplicate impossible.
3. Check your database's default isolation level. Was it what you assumed?
4. Write the two-write operation in your system that currently has no
   transaction. Add one.
5. Run a retry storm against your refund or charge path and count the rows.

---

**Next:** [Lesson 07 — SQL for Machine Learning](07-sql-for-ml.md)
