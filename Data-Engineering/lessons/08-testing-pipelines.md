# Lesson 08 — Testing Pipelines

**Goal:** test data code the way you would test any other code.

## What you will learn

- Why data code is testable
- Unit tests for transformations
- Fixtures, and testing against a real database
- What to test, and what not to

---

## Two kinds of test

| | Data tests | Code tests |
|---|---|---|
| Ask | Is **this data** valid? | Is **this logic** correct? |
| Run | Every pipeline run | Every commit |
| Tool | dbt test, great_expectations | pytest |
| Lesson | 07 | **this one** |

You need both. Lesson 07's checks catch a bad batch; these catch the bug you
are about to deploy.

---

## Make the transformation a function

```python
import pandas as pd

# NOT testable: reads, transforms and writes in one blob
def daily_job_bad():
    frame = pd.read_csv("/tmp/orders.csv")
    frame["amount_with_vat"] = frame["amount"] * 1.14
    frame.to_sql("orders", "postgresql://...")

# Testable: the logic is a pure function
def add_vat(frame, rate=0.14):
    """Add a VAT-inclusive column. Pure: no IO, no globals."""
    result = frame.copy()
    result["amount_with_vat"] = (result["amount"] * (1 + rate)).round(2)
    return result

def daily_job(read, write, rate=0.14):
    """IO at the edges, logic in the middle."""
    write(add_vat(read(), rate))

sample = pd.DataFrame({"order_id": [1, 2], "amount": [100.0, 85.5]})
print(add_vat(sample).to_string(index=False))
```

```text
 order_id  amount  amount_with_vat
        1   100.0           114.00
        2    85.5            97.47
```

The shape that makes data code testable: **IO at the edges, logic in the
middle.** `add_vat` takes a DataFrame and returns one — no files, no database,
no clock.

---

## Unit tests

```python
import pandas as pd
import numpy as np

def test_add_vat_computes_correctly():
    frame = pd.DataFrame({"amount": [100.0]})
    assert add_vat(frame)["amount_with_vat"].iloc[0] == 114.0

def test_add_vat_does_not_mutate_input():
    frame = pd.DataFrame({"amount": [100.0]})
    add_vat(frame)
    assert "amount_with_vat" not in frame.columns

def test_add_vat_handles_empty():
    frame = pd.DataFrame({"amount": []})
    result = add_vat(frame)
    assert len(result) == 0 and "amount_with_vat" in result.columns

def test_add_vat_propagates_nulls():
    frame = pd.DataFrame({"amount": [100.0, np.nan]})
    assert bool(add_vat(frame)["amount_with_vat"].isna().iloc[1])

def test_add_vat_handles_zero():
    frame = pd.DataFrame({"amount": [0.0]})
    assert add_vat(frame)["amount_with_vat"].iloc[0] == 0.0

for test in [test_add_vat_computes_correctly, test_add_vat_does_not_mutate_input,
             test_add_vat_handles_empty, test_add_vat_propagates_nulls,
             test_add_vat_handles_zero]:
    test()
    print(f"PASS {test.__name__}")
```

```text
PASS test_add_vat_computes_correctly
PASS test_add_vat_does_not_mutate_input
PASS test_add_vat_handles_empty
PASS test_add_vat_propagates_nulls
PASS test_add_vat_handles_zero
```

The five cases that matter for **any** transformation:

1. The happy path
2. **Empty input** — a source with no rows on a quiet day
3. **Nulls** — what should they become?
4. **Edge values** — zero, negative, the maximum
5. **No mutation** of the input

That second one catches more production failures than the first. Empty
DataFrames lose their dtypes, break aggregations, and turn a quiet Sunday into
an incident.

---

## Testing joins

```python
import pandas as pd

def enrich_orders(orders, customers):
    """Left join, with a guard against fan-out."""
    if customers["customer_id"].duplicated().any():
        raise ValueError("customers has duplicate keys; the join would fan out")
    before = len(orders)
    result = orders.merge(customers, on="customer_id", how="left")
    if len(result) != before:
        raise ValueError(f"join changed the row count: {before} -> {len(result)}")
    return result

orders = pd.DataFrame({"order_id": [1, 2], "customer_id": [10, 11]})
good_customers = pd.DataFrame({"customer_id": [10, 11], "city": ["Cairo", "Giza"]})
duplicated_customers = pd.DataFrame({"customer_id": [10, 10, 11],
                                     "city": ["Cairo", "Cairo", "Giza"]})

print("clean join:", len(enrich_orders(orders, good_customers)), "rows")
try:
    enrich_orders(orders, duplicated_customers)
except ValueError as error:
    print("caught:", error)
```

```text
clean join: 2 rows
caught: customers has duplicate keys; the join would fan out
```

**Assert the row count around every join.** A duplicate key on the right side
silently multiplies rows, and every sum downstream is then wrong — the most
expensive bug in the Machine Learning course's pandas lesson, and it belongs
in a test.

---

## Fixtures and a real database

```python
import pandas as pd
from sqlalchemy import create_engine, text

def make_test_database():
    """A real SQLite database, created fresh for each test."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, "
            "customer_id INTEGER, amount REAL)"))
        connection.execute(
            text("INSERT INTO orders VALUES (:i, :c, :a)"),
            [{"i": 1, "c": 10, "a": 100.0}, {"i": 2, "c": 11, "a": 200.0}])
    return engine

def daily_revenue(engine):
    return pd.read_sql(text("SELECT SUM(amount) AS revenue FROM orders"), engine)

def test_daily_revenue():
    engine = make_test_database()
    assert daily_revenue(engine)["revenue"].iloc[0] == 300.0

test_daily_revenue()
print("PASS test_daily_revenue")
```

```text
PASS test_daily_revenue
```

In pytest this is a fixture:

```python
import pytest

@pytest.fixture
def engine():
    return make_test_database()

def test_daily_revenue(engine):
    assert daily_revenue(engine)["revenue"].iloc[0] == 300.0
```

**SQLite in memory is enough for most SQL logic**, and it runs in
milliseconds. Use `testcontainers` with a real Postgres when you depend on
Postgres-specific behaviour — window functions with `FILTER`, `ON CONFLICT`,
array types.

---

## Idempotency is a test

```python
import pandas as pd
from sqlalchemy import create_engine, text

def test_pipeline_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE mart (day TEXT PRIMARY KEY, revenue REAL)"))

    def load(batch):
        with engine.begin() as connection:
            days = batch["day"].tolist()
            placeholders = ",".join(f":d{i}" for i in range(len(days)))
            connection.execute(text(f"DELETE FROM mart WHERE day IN ({placeholders})"),
                               {f"d{i}": d for i, d in enumerate(days)})
            batch.to_sql("mart", connection, if_exists="append", index=False)

    batch = pd.DataFrame({"day": ["2026-01-01"], "revenue": [100.0]})
    load(batch)
    first = pd.read_sql("SELECT * FROM mart", engine)
    load(batch)
    second = pd.read_sql("SELECT * FROM mart", engine)

    assert first.equals(second), "running twice changed the result"
    return len(second)

print("PASS test_pipeline_is_idempotent, rows:", test_pipeline_is_idempotent())
```

```text
PASS test_pipeline_is_idempotent, rows: 1
```

**Every pipeline should have this test.** It is five lines, and it is the
difference between a re-run you can do without thinking and one you do at 2am
while holding your breath.

---

## What to test, and what not to

| Test | Do not test |
|---|---|
| Your transformation logic | That pandas can join |
| Row counts around joins | That SQLite stores integers |
| Empty and null handling | Third-party library behaviour |
| Idempotency of writes | The network |
| Business rules (VAT, discounts, statuses) | Data content — that is lesson 07 |
| Schema contracts between stages | Exact float equality |

Float comparison deserves a note:

```python
import pandas as pd

result = pd.DataFrame({"amount": [0.1 + 0.2]})
print("exact equality:", result["amount"].iloc[0] == 0.3)
print("with tolerance:", abs(result["amount"].iloc[0] - 0.3) < 1e-9)
```

```text
exact equality: False
with tolerance: True
```

Use `pytest.approx` or `pandas.testing.assert_frame_equal(..., atol=...)`. An
exact float assertion will fail on a different machine, next month, for no
reason you can reproduce.

---

## The test suite for a pipeline

```text
tests/
  test_transformations.py    pure functions: happy path, empty, nulls, edges
  test_joins.py              row counts, fan-out guards
  test_idempotency.py        run twice, same result
  test_schema.py             the contract each stage produces
  test_sql.py                queries against SQLite fixtures
  test_quality_rules.py      lesson 07's rules, against known-bad samples
```

Six files, and they run in seconds on every commit.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Logic tangled with IO | Untestable without a database |
| No empty-input test | A quiet Sunday breaks production |
| No row-count assertion on joins | Silent fan-out |
| Exact float equality | Flaky tests |
| Testing against production data | Slow, non-deterministic, and a data-leak risk |
| Only data tests, no code tests | You catch bad batches, not bad code |

---

## Exercises

1. Refactor one of your jobs into `read`, `transform`, `write`.
2. Write the five standard tests for that transformation.
3. Add a row-count assertion to every join in one pipeline.
4. Write the idempotency test, and make it pass.
