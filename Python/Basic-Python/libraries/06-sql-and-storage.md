# SQL and Storage — SQLAlchemy, SQLite, Parquet

**What this covers:** getting data out of databases, and choosing the file
format you store it in.

**Role:** data engineering. Also the part of data science everyone skips and
later regrets.

---

## SQL from Python

The standard library ships with SQLite — a full SQL database in a single file,
no server to install.

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect(":memory:")       # ":memory:" or "shop.db"

conn.execute("CREATE TABLE orders (id INTEGER, item TEXT, amount REAL)")
conn.executemany(
    "INSERT INTO orders VALUES (?, ?, ?)",
    [(1, "Latte", 60.0), (2, "V60", 85.0), (3, "Latte", 60.0)],
)
conn.commit()

df = pd.read_sql("SELECT item, SUM(amount) AS total FROM orders GROUP BY item", conn)
print(df)
conn.close()
```

```text
    item  total
0  Latte  120.0
1    V60   85.0
```

Two things to carry forward:

- **Parameters go in as `?`, never as string formatting.** This is correct:
  `conn.execute("SELECT * FROM orders WHERE item = ?", (name,))`. Building that
  query with an f-string is a SQL injection hole, and it also breaks the moment
  a value contains a quote.
- `pd.read_sql` hands you a DataFrame directly. The database does the grouping,
  which is usually much faster than pulling everything into Python.

---

## SQLAlchemy — one interface for every database

```python
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine("sqlite:///shop.db")
# postgresql+psycopg://user:pass@host:5432/dbname
# mysql+pymysql://user:pass@host:3306/dbname

with engine.connect() as conn:
    df = pd.read_sql(text("SELECT * FROM orders WHERE amount > :limit"),
                     conn, params={"limit": 50})

df.to_sql("orders_big", engine, if_exists="replace", index=False)
```

The connection string is the only line that changes between SQLite on your
laptop and PostgreSQL in production. Write the rest once.

Keep credentials out of the string in real code:

```python
import os
from sqlalchemy import create_engine

engine = create_engine(os.environ["DATABASE_URL"])
```

---

## Reading a large table without exploding

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///shop.db")

total = 0
for chunk in pd.read_sql("SELECT * FROM orders", engine, chunksize=50_000):
    total += chunk["amount"].sum()

print(total)
```

```text
245.0
```

`chunksize` streams the result instead of loading fifty million rows into
memory at once. The same argument exists on `read_csv`.

---

## Push work into SQL

```python
# Slow: pull 10 million rows, then group in pandas
df = pd.read_sql("SELECT * FROM orders", engine)
result = df.groupby("item")["amount"].sum()

# Fast: let the database do it, pull 20 rows
result = pd.read_sql(
    "SELECT item, SUM(amount) AS total FROM orders GROUP BY item", engine
)
```

The database has indexes, a query planner, and it is next to the data. Filter
and aggregate there; bring back only what you need.

---

## File formats

```mermaid
flowchart LR
    CSV["CSV<br/>text, row-based<br/>no types"] --> P["Parquet<br/>binary, columnar<br/>typed, compressed"]
    P --> USE["read one column<br/>without reading the file"]
```

| | CSV | Parquet |
|---|---|---|
| Types | Lost — everything is text | Kept |
| Size | Large | 5–10× smaller |
| Read one column | Reads the whole file | Reads that column only |
| Human-readable | Yes | No |
| Use it for | Handing a file to a person | Everything a program reads |

```python
import pandas as pd

df = pd.DataFrame({"item": ["Latte", "V60"], "amount": [60.0, 85.0]})

df.to_parquet("orders.parquet", index=False)

part = pd.read_parquet("orders.parquet", columns=["amount"])
print(part)
```

```text
   amount
0    60.0
1    85.0
```

Only the `amount` column was read off disk. On a 200-column table that is the
difference between two seconds and two minutes.

---

## Partitioning

```python
df.to_parquet("events/", partition_cols=["year", "month"])
```

```text
events/
  year=2026/
    month=08/part-0.parquet
    month=09/part-0.parquet
```

Reading with a filter on `year` then skips whole directories without opening
them. This is how every data lake is laid out, and it is the cheapest
performance win in data engineering.

---

## A minimal ETL

```python
import pandas as pd
from sqlalchemy import create_engine

def etl(source_csv, engine):
    """Extract from CSV, clean, load into the warehouse."""
    df = pd.read_csv(source_csv)                        # extract

    df = df.dropna(subset=["item", "amount"])           # transform
    df["item"] = df["item"].str.strip().str.title()
    df["amount"] = df["amount"].astype(float)
    df["loaded_at"] = pd.Timestamp.utcnow()

    df.to_sql("orders", engine, if_exists="append", index=False)   # load
    return len(df)
```

Extract, transform, load. Every pipeline you will build is this shape, with
more error handling and a scheduler around it.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| f-strings in SQL | Injection, and broken quoting |
| `SELECT *` on a wide table | You move gigabytes you do not need |
| Grouping in pandas what SQL could group | 100× slower |
| CSV as the internal format | Lost dtypes, slow reads, huge files |
| `if_exists="replace"` in production | The table is dropped — check twice |
| Credentials in the connection string | Leaked on the first commit |

---

## Exercises

1. Create a SQLite table, insert ten rows, and read a `GROUP BY` into pandas.
2. Rewrite an injection-prone query with bound parameters.
3. Save a DataFrame as CSV and Parquet; compare file size and load time.
4. Write an `etl()` that is safe to run twice without duplicating rows.
