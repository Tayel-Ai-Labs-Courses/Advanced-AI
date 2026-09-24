# Lesson 06 — Incremental Loading

**Goal:** load only what changed, and survive being run twice.

## What you will learn

- Full refresh versus incremental
- Watermarks, and the late-data trap
- Idempotent writes: delete-insert and merge
- Backfills

---

## Full or incremental

```python
rows_total = 500_000_000
rows_per_day = 2_000_000
seconds_per_million = 8

full = rows_total / 1_000_000 * seconds_per_million
incremental = rows_per_day / 1_000_000 * seconds_per_million

print(f"full refresh: {full / 60:>7.1f} minutes")
print(f"incremental:  {incremental:>7.1f} seconds")
print(f"ratio:        {full / incremental:>7.0f}x")
```

```text
full refresh:    66.7 minutes
incremental:     16.0 seconds
ratio:            250x
```

| | Full refresh | Incremental |
|---|---|---|
| Correctness | **Always correct** | Correct if the logic is right |
| Cost | Grows forever | Constant |
| Complexity | Trivial | Watermarks, dedup, backfills |
| Late data | Handled automatically | **Must be handled explicitly** |
| Deletes | Handled automatically | Need soft deletes or CDC |

**Full-refresh until it hurts.** It is one line, it is always right, and at
250× the cost it is still only 67 minutes. Move to incremental when the run
time or the bill forces you to — and accept that you have taken on four new
failure modes.

---

## The watermark

```python
import pandas as pd

source = pd.DataFrame({
    "order_id": range(1, 11),
    "amount": [i * 10.0 for i in range(1, 11)],
    "updated_at": pd.to_datetime([
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05",
        "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10"]),
})

def extract_since(source, watermark):
    return source[source["updated_at"] > watermark]

watermark = pd.Timestamp("2026-01-07")
batch = extract_since(source, watermark)
print(f"watermark {watermark.date()} -> {len(batch)} rows "
      f"({batch['order_id'].min()}..{batch['order_id'].max()})")
print("new watermark:", batch["updated_at"].max().date())
```

```text
watermark 2026-01-07 -> 3 rows (8..10)
new watermark: 2026-01-10
```

Simple, and it has a hole.

---

## Late data breaks strict watermarks

```python
import pandas as pd

processed_until = pd.Timestamp("2026-01-10")

late_arrival = pd.DataFrame({
    "order_id": [11],
    "amount": [500.0],
    "updated_at": pd.to_datetime(["2026-01-08"]),      # event happened earlier
})

source_now = pd.concat([source, late_arrival], ignore_index=True)

strict = source_now[source_now["updated_at"] > processed_until]
overlap = source_now[source_now["updated_at"] > processed_until - pd.Timedelta(days=3)]

print(f"strict watermark:   {len(strict)} rows  -> order 11 is MISSED")
print(f"3-day overlap:      {len(overlap)} rows -> order 11 is caught")
print("but the overlap re-reads orders:",
      sorted(overlap[overlap['updated_at'] <= processed_until]['order_id'].tolist()))
```

```text
strict watermark:   0 rows  -> order 11 is MISSED
3-day overlap:      4 rows -> order 11 is caught
but the overlap re-reads orders: [8, 9, 10, 11]
```

An order recorded with Thursday's timestamp but delivered to you on Sunday is
**invisible** to a strict watermark. The fix is a lookback window — and the
lookback re-reads rows you already have, which is exactly the situation that
duplicated revenue in lesson 01.

**Overlap plus idempotent writes is the only combination that is safe.** One
without the other is a bug waiting for a busy Monday.

---

## Idempotent writes

```python
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///:memory:")
with engine.begin() as connection:
    connection.execute(text(
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, amount REAL, updated_at TEXT)"))

def naive_append(batch, engine):
    batch.to_sql("orders", engine, if_exists="append", index=False)

def merge_upsert(batch, engine, key="order_id"):
    """Delete the incoming keys, then insert. Safe to re-run."""
    with engine.begin() as connection:
        placeholders = ",".join(f":k{i}" for i in range(len(batch)))
        connection.execute(
            text(f"DELETE FROM orders WHERE {key} IN ({placeholders})"),
            {f"k{i}": int(value) for i, value in enumerate(batch[key])})
        batch.to_sql("orders", connection, if_exists="append", index=False)

batch = pd.DataFrame({"order_id": [1, 2], "amount": [10.0, 20.0],
                      "updated_at": ["2026-01-01", "2026-01-02"]})

merge_upsert(batch, engine)
merge_upsert(batch, engine)
merge_upsert(batch.assign(amount=[99.0, 20.0]), engine)     # an update arrives

print(pd.read_sql("SELECT * FROM orders ORDER BY order_id", engine).to_string(index=False))
```

```text
 order_id  amount updated_at
        1    99.0 2026-01-01
        2    20.0 2026-01-02
```

Three runs, two rows, and the updated amount won. That is the behaviour you
need: **re-running is free, and the latest version wins.**

| Pattern | How | Use when |
|---|---|---|
| **Delete + insert** | Delete the keys, insert the batch | The default; simple and portable |
| **MERGE / upsert** | `MERGE INTO ... WHEN MATCHED` | The warehouse supports it |
| **Partition swap** | Write a new partition, swap atomically | Partitioned lakes |
| **Append-only + dedup on read** | Keep every version; pick latest in a view | Full audit history needed |

A primary key or a deduplication key is what makes all of these possible.
Without one, you cannot tell an update from a new row — and **that is a data
modelling problem, not a pipeline problem.**

---

## Deduplicating on read

```python
import pandas as pd

events = pd.DataFrame({
    "order_id": [1, 1, 2, 2, 2],
    "status": ["placed", "paid", "placed", "paid", "refunded"],
    "updated_at": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 11:00",
                                  "2026-01-02 09:00", "2026-01-02 10:00",
                                  "2026-01-03 15:00"]),
})

latest = (events.sort_values("updated_at")
                .groupby("order_id", as_index=False)
                .last())
print(latest.to_string(index=False))
```

```text
 order_id   status          updated_at
        1     paid 2026-01-01 11:00:00
        2 refunded 2026-01-03 15:00:00
```

Append everything, resolve on read. It costs storage and query time, and it
gives you the full history for free — which is often worth more than the
saving.

In SQL this is a window function:

```sql
select * from (
    select *, row_number() over (
        partition by order_id order by updated_at desc) as rn
    from raw_events
) where rn = 1
```

---

## Backfills

```python
import pandas as pd

def date_partitions(start, end):
    return [d.date().isoformat() for d in pd.date_range(start, end, freq="D")]

def backfill(start, end, run_partition):
    """Reprocess a date range, partition by partition, most recent first."""
    partitions = date_partitions(start, end)
    results = []
    for partition in reversed(partitions):          # recent days matter most
        results.append((partition, run_partition(partition)))
    return results

def fake_run(partition):
    return {"rows": 1_000, "status": "ok"}

results = backfill("2026-01-01", "2026-01-05", fake_run)
for partition, result in results:
    print(f"{partition}  {result['status']}  {result['rows']} rows")
```

```text
2026-01-05  ok  1000 rows
2026-01-04  ok  1000 rows
2026-01-03  ok  1000 rows
2026-01-02  ok  1000 rows
2026-01-01  ok  1000 rows
```

Four rules for backfills:

- **One partition at a time**, so a failure costs one day, not the whole range.
- **Most recent first** — the recent days are usually the ones someone is
  waiting for.
- **Idempotent**, or the backfill doubles the data it was meant to fix.
- **Rate-limited**, or you take the warehouse down for everyone else.

A pipeline that cannot be backfilled cannot be fixed. Design for it from the
start: parameterise the job by partition, and never let "today" be implicit.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Strict watermark, no overlap | Late data is silently lost |
| Overlap without idempotent writes | Duplicated rows, inflated numbers |
| Watermark from the local clock | Timezone-dependent gaps |
| No primary key | Updates cannot be distinguished from inserts |
| Backfill as one giant job | Fails at 80%, restarts from zero |
| Incremental from day one | Complexity before it is needed |
| Deletes ignored | Rows live forever in the warehouse |

---

## Exercises

1. Show the late-data hole on your own data with a strict watermark.
2. Implement delete-insert and prove idempotency by running it three times.
3. Deduplicate an append-only table with a window function.
4. Write a backfill that processes one partition at a time and can resume.
