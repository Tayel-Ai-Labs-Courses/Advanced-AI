# Lesson 02 — Designing a Schema

**Goal:** design tables that make wrong data impossible to insert.

## What you will learn

- Normalisation, in one example
- Keys, and what they guarantee
- Constraints as the cheapest tests you will ever write
- When to denormalise on purpose

---

## The spreadsheet shape, and why it hurts

Here is the table everybody starts with:

```text
order_id  customer_name  customer_gov  product  qty  price  total
1001      Mona Saleh     Cairo         latte    2    45.0   90.0
1002      Mona Saleh     Cairo         cake     1    60.0   60.0
1003      Mona Salah     Cairo         tea      3    30.0   90.0    <- typo
1004      Omar Nabil     Giza          latte    1    45.0   50.0    <- wrong total
```

Four rows, three bugs waiting:

| Problem | Name | Consequence |
|---|---|---|
| `Mona Saleh` stored twice, once misspelled | **Update anomaly** | She is now two customers. Any count by customer is wrong |
| `total` can disagree with `qty * price` | **Derived data stored** | Row 1004 says 50.0 and should say 45.0 |
| Nothing stops `qty = -3` or `price = 0` | **No constraints** | The bad row exists forever, and you find it in a report |
| Changing her governorate means updating every order | **Redundancy** | Miss one, and she lives in two places |

None of these are performance problems. They are **correctness** problems, and
a schema is the cheapest place to prevent them.

---

## The normalised version

```sql
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    governorate  TEXT    NOT NULL,
    signup_date  TEXT    NOT NULL
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    ordered_at   TEXT    NOT NULL,
    amount       REAL    NOT NULL CHECK (amount > 0),
    status       TEXT    NOT NULL CHECK (status IN ('delivered','shipped','cancelled','refunded'))
);
```

Each fact lives in exactly one place:

- **A customer's name is stored once.** Correcting a typo is one `UPDATE`.
- **`customer_id` is a foreign key.** An order for a customer who does not exist
  cannot be inserted.
- **`CHECK (amount > 0)`** makes a negative amount impossible, not merely
  unlikely.
- **`status` is constrained to four values**, so `"Delivered"`, `"deliverd"` and
  `"DELIVERED"` never coexist.
- **`total` is gone.** It is `qty * price`, computed when needed. Derived data
  stored is derived data that will disagree.

That last rule is the most frequently broken, and the one worth fighting for:
**store facts, compute derivations.**

---

## Constraints are tests that run on every write

```python
import sqlite3
con = sqlite3.connect(":memory:")
con.execute("PRAGMA foreign_keys = ON")          # SQLite needs this, every connection
con.executescript("""
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    amount      REAL NOT NULL CHECK (amount > 0),
    status      TEXT NOT NULL CHECK (status IN ('delivered','cancelled'))
);
INSERT INTO customers VALUES (1, 'Mona');
""")

attempts = [
    ("valid row",            (1, 1, 90.0, 'delivered')),
    ("customer 99 missing",  (2, 99, 90.0, 'delivered')),
    ("negative amount",      (3, 1, -5.0, 'delivered')),
    ("status typo",          (4, 1, 90.0, 'Delivered')),
    ("null amount",          (5, 1, None, 'delivered')),
]
for label, row in attempts:
    try:
        con.execute("INSERT INTO orders VALUES (?,?,?,?)", row)
        print(f"{label:<22} accepted")
    except sqlite3.Error as e:
        print(f"{label:<22} REJECTED - {type(e).__name__}")
```

```text
valid row              accepted
customer 99 missing    REJECTED - IntegrityError
negative amount        REJECTED - IntegrityError
status typo            REJECTED - IntegrityError
null amount            REJECTED - IntegrityError
```

**Four bad rows, four rejections, zero application code.** Every one of those
would otherwise be a validation function somebody forgets to call, or a bug
found three months later in a report.

This is the same argument as
[Data-Science lesson 08](../../Data-Science/lessons/08-shipping-the-model.md)'s
input validation and
[AI-Agents lesson 02](../../AI-Agents/lessons/02-tools.md)'s business rules
inside the tool: **put the guarantee where it cannot be bypassed.** A `CHECK`
constraint holds for every writer — your app, a script, a colleague's notebook,
a migration — forever.

Note the `PRAGMA foreign_keys = ON`. SQLite does not enforce foreign keys by
default, per connection. Postgres and MySQL do. It is the single most common
cause of "but I declared the reference".

---

## Keys

| Key | Guarantees | Example |
|---|---|---|
| **Primary key** | Unique, not null, one per table | `order_id` |
| **Foreign key** | The referenced row exists | `orders.customer_id -> customers` |
| **Unique** | No duplicates, but may be null | `customers.email` |
| **Natural key** | Meaningful in the business | a national ID, an ISBN |
| **Surrogate key** | Meaningless, stable | an auto-increment integer |

**Prefer surrogate primary keys.** Natural keys change — people change phone
numbers, companies re-issue SKUs, a country splits — and a changing primary key
means updating every row that references it.

And the one that prevents duplicate data at the source:

```sql
CREATE UNIQUE INDEX uq_one_order_per_customer_per_day
    ON orders(customer_id, ordered_at, amount);
```

A unique constraint on the natural business key is how you make
**idempotency** a database property rather than application logic — which is
[AI-Agents lesson 06](../../AI-Agents/lessons/06-reliability.md)'s refund
problem, solved one level down.

---

## When to denormalise

Normalisation is the default, not a religion. Break it deliberately when:

| Reason | Example | Keep safe by |
|---|---|---|
| **Analytics** | A wide table for a dashboard | Rebuild it from the normalised source |
| **A point-in-time snapshot** | The price *at the time of order* | It is a different fact, not redundancy |
| Expensive joins at scale | A warehouse fact table | It is derived; document the source |
| A cache | Current balance | Recomputable, with a refresh path |

The second row deserves care, because it looks like redundancy and is not. The
price when the order was placed is **historical fact**; the product's price
today is a different thing. Storing `price_at_order` on the order is correct
normalisation, not a violation.

**The test: if the source is deleted, can you rebuild the copy?** If yes, it is
a cache. If no, it was a fact and belongs there.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Storing a derived column | It will disagree with its inputs |
| The same text in two tables | One typo becomes two entities |
| No foreign keys | Orphan rows, discovered in a report |
| No `CHECK` constraints | Negative amounts and status typos live forever |
| Forgetting `PRAGMA foreign_keys = ON` | You declared a reference that is not enforced |
| A natural primary key | It changes, and now every reference must too |
| Denormalising before measuring | Complexity, for a problem you did not have |

---

## Exercises

1. Take a spreadsheet you use and normalise it. How many tables, and which
   constraints?
2. Add `CHECK` constraints to a table you own. How many existing rows violate
   them?
3. Find a derived column in your schema. Can you prove it agrees with its
   inputs today?
4. Add a unique constraint that makes a duplicate insert impossible, and test it.
5. Find a foreign key with no index (lesson 05) and no enforcement.

---

**Next:** [Lesson 03 — Querying](03-querying.md)
