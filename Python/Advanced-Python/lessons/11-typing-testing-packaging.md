# Lesson 11 — Typing, Testing, Packaging

**Goal:** write code other people — including future you — can rely on.

## What you will learn

- Type hints, and what they do and do not do
- pytest, and what is worth testing
- Project layout and environments
- Logging instead of printing

---

## Type hints

```python
def with_tax(price: float, rate: float = 0.14) -> float:
    """Return the price including tax."""
    return round(price * (1 + rate), 2)

print(with_tax(60))
```

```text
68.4
```

Python does **not** enforce these at runtime. `with_tax("60")` still runs and
still breaks — the hint is documentation that tools can check.

What they buy you: autocomplete in your editor, errors caught by `mypy` before
you run anything, and a signature that answers "what do I pass this?" without
reading the body.

```python
from typing import Optional
from collections.abc import Iterable

def clean(names: list[str]) -> list[str]:
    return [n.strip().title() for n in names]

def find(names: list[str], target: str) -> Optional[int]:
    """Return the index, or None if absent."""
    return names.index(target) if target in names else None

def total(values: Iterable[float]) -> float:
    return sum(values)

Row = dict[str, str | float]

def summarise(rows: list[Row]) -> dict[str, float]:
    return {"count": float(len(rows))}

print(clean([" adam ", "sara"]), find(["a", "b"], "b"), total([1.0, 2.0]))
```

```text
['Adam', 'Sara'] 1 3.0
```

Modern syntax: `list[str]` not `List[str]`, `str | None` not
`Optional[str]`. Check with:

```bash
pip install mypy
mypy src/
```

Type the public functions — the ones other modules call. Typing every local
variable is noise.

---

## Testing

```python
# src/clean.py
def normalise_name(name: str) -> str:
    """Strip whitespace and title-case a name."""
    if not isinstance(name, str):
        raise TypeError(f"expected str, got {type(name).__name__}")
    return " ".join(name.split()).title()
```

```python
# tests/test_clean.py
import pytest
from src.clean import normalise_name

def test_strips_and_titles():
    assert normalise_name("  adam  yasser ") == "Adam Yasser"

def test_collapses_inner_spaces():
    assert normalise_name("adam     yasser") == "Adam Yasser"

def test_empty_string():
    assert normalise_name("") == ""

def test_rejects_non_string():
    with pytest.raises(TypeError):
        normalise_name(123)

@pytest.mark.parametrize("raw,expected", [
    ("adam", "Adam"),
    ("ADAM", "Adam"),
    ("  a  ", "A"),
])
def test_cases(raw, expected):
    assert normalise_name(raw) == expected
```

```bash
pytest -v
```

```text
tests/test_clean.py::test_strips_and_titles PASSED
tests/test_clean.py::test_collapses_inner_spaces PASSED
tests/test_clean.py::test_empty_string PASSED
tests/test_clean.py::test_rejects_non_string PASSED
tests/test_clean.py::test_cases[adam-Adam] PASSED
tests/test_clean.py::test_cases[ADAM-Adam] PASSED
tests/test_clean.py::test_cases[  a  -A] PASSED

7 passed in 0.03s
```

### What to test

Not everything. Test:

- The **edge cases**: empty, one item, missing value, wrong type
- The **bugs you have already fixed** — a test stops them coming back
- The **business rules**: the tax calculation, the grading boundary
- **Data assumptions**: no negative prices, no duplicate ids, the row count
  after a merge

Do not test that pandas can group, or that Python can add. Test your logic.

### Fixtures

```python
import pytest
import pandas as pd

@pytest.fixture
def sample_orders():
    return pd.DataFrame({
        "item": ["Latte", "V60", "Latte"],
        "amount": [60.0, 85.0, 60.0],
    })

def test_revenue_by_item(sample_orders):
    totals = sample_orders.groupby("item")["amount"].sum()
    assert totals["Latte"] == 120.0
```

A fixture is shared setup. Ask for it by name in the test's parameters.

---

## Project layout

```text
my-project/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── clean.py
│       └── model.py
├── tests/
│   └── test_clean.py
└── notebooks/
    └── exploration.ipynb
```

- **`src/`** holds the code that must work. Functions, tested, importable.
- **`notebooks/`** is where you think. Anything that proves useful gets moved
  into `src/` and imported back into the notebook.

A notebook is a bad home for logic: it has hidden state, it runs out of order,
and it cannot be imported or tested. Notebooks explore; modules ship.

---

## Environments

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.txt
```

One environment per project. "It works on my machine" is almost always a
dependency that is installed globally on yours and nowhere else.

A minimal `pyproject.toml`:

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pandas>=2.2", "scikit-learn>=1.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`, at minimum:

```text
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/
.env
```

Never commit data, secrets, or the virtual environment.

---

## Logging, not printing

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

def load(path: str) -> list[str]:
    log.info("loading %s", path)
    try:
        with open(path, encoding="utf-8") as f:
            rows = f.readlines()
    except FileNotFoundError:
        log.error("file not found: %s", path)
        raise
    log.info("loaded %d rows", len(rows))
    return rows
```

```text
2026-09-22 14:03:11,204 INFO __main__: loading menu.csv
2026-09-22 14:03:11,205 INFO __main__: loaded 151 rows
```

`print` cannot be switched off, carries no timestamp or severity, and cannot be
routed to a file. Logging can do all three, and you configure it once.

Levels: `DEBUG` for development detail, `INFO` for milestones, `WARNING` for
"this is suspicious", `ERROR` for failures.

---

## Formatting

```bash
pip install ruff
ruff format .      # format
ruff check .       # lint
```

One tool, fast, and it ends every argument about style. Run it before you
commit.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Believing type hints are enforced | They are not — they are checked, not applied |
| Logic living in notebooks | Cannot be tested, imported, or reviewed |
| No virtual environment | Dependency conflicts across projects |
| `print` for diagnostics in production | No levels, no timestamps, no off switch |
| Committing `.env` or `data/` | Leaked secrets, bloated repository |
| Tests only for the happy path | The edge cases are where the bugs are |

---

## Exercises

1. Add type hints to three functions from Project 1 and run `mypy`.
2. Write five pytest tests for a cleaning function, including two edge cases.
3. Restructure a project into `src/`, `tests/`, `notebooks/`.
4. Replace every `print` in a script with logging at the right levels.
