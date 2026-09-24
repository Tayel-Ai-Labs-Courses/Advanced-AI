#!/usr/bin/env python3
"""Build .ipynb notebooks from the markdown lessons.

The markdown is the single source of truth. Every lesson `.md` in every course
becomes a notebook next to it:

    prose            -> markdown cells
    ```python blocks -> runnable code cells
    ```text blocks   -> dropped (they are the expected output; run the cell)

Usage:
    python3 tools/build_notebooks.py            # build every course
    python3 tools/build_notebooks.py --check    # fail if anything is stale

Standard library only, so it runs anywhere the lessons do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories whose markdown files become notebooks, per course.
LESSON_DIRS = [
    REPO_ROOT / "Python" / "Basic-Python" / "lessons",
    REPO_ROOT / "Python" / "Basic-Python" / "libraries",
    REPO_ROOT / "Python" / "Advanced-Python" / "lessons",
    REPO_ROOT / "Machine-Learning" / "lessons",
    REPO_ROOT / "Deep-Learning" / "lessons",
    REPO_ROOT / "Optimization" / "lessons",
    REPO_ROOT / "NLP" / "lessons",
    REPO_ROOT / "Computer-Vision" / "lessons",
    REPO_ROOT / "Data-Engineering" / "lessons",
]

HEADER = (
    "> Generated from the matching `.md` file — **edit the markdown, not this "
    "notebook**, then run `python3 tools/build_notebooks.py`.\n"
    ">\n"
    "> Some examples are illustrative (Spark, external APIs, large models) and "
    "will not run without that service or package installed."
)


def split_cells(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (kind, source) pairs, kind being 'markdown' or 'code'."""
    cells: list[tuple[str, str]] = []
    prose: list[str] = []
    lines = markdown.splitlines()
    i = 0

    def flush_prose() -> None:
        text = "\n".join(prose).strip("\n")
        prose.clear()
        if text.strip():
            cells.append(("markdown", text))

    while i < len(lines):
        line = lines[i]

        if line.startswith("```python"):
            flush_prose()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1                                   # closing fence
            cells.append(("code", "\n".join(code).strip("\n")))

            # Drop a blank line plus an expected-output block, if present.
            look = i
            while look < len(lines) and not lines[look].strip():
                look += 1
            if look < len(lines) and lines[look].startswith("```text"):
                look += 1
                while look < len(lines) and not lines[look].startswith("```"):
                    look += 1
                i = look + 1
            continue

        prose.append(line)
        i += 1

    flush_prose()
    return cells


def build_notebook(markdown: str) -> dict:
    """Turn lesson markdown into a notebook document."""
    cells = [{"cell_type": "markdown", "id": "cell-0", "metadata": {},
              "source": _lines(HEADER)}]

    for number, (kind, source) in enumerate(split_cells(markdown), start=1):
        # Ids are stable across builds so a rebuild produces no spurious diff.
        cell = {"cell_type": kind, "id": f"cell-{number}", "metadata": {},
                "source": _lines(source)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _lines(text: str) -> list[str]:
    """Notebook sources are lists of lines, each keeping its newline but the last."""
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + [parts[-1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report stale notebooks instead of writing them")
    args = parser.parse_args()

    stale: list[Path] = []
    written = 0

    for directory in LESSON_DIRS:
        if not directory.is_dir():
            print(f"skipping missing directory: {directory}", file=sys.stderr)
            continue

        for md_path in sorted(directory.glob("*.md")):
            if md_path.name == "README.md":
                continue

            notebook = build_notebook(md_path.read_text(encoding="utf-8"))
            text = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
            nb_path = md_path.with_suffix(".ipynb")

            if args.check:
                current = nb_path.read_text(encoding="utf-8") if nb_path.exists() else ""
                if current != text:
                    stale.append(nb_path.relative_to(REPO_ROOT))
                continue

            nb_path.write_text(text, encoding="utf-8")
            code_cells = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
            print(f"{nb_path.relative_to(REPO_ROOT)}  ({code_cells} code cells)")
            written += 1

    if args.check:
        if stale:
            print("stale notebooks — run tools/build_notebooks.py:", file=sys.stderr)
            for path in stale:
                print(f"  {path}", file=sys.stderr)
            return 1
        print("all notebooks up to date")
        return 0

    print(f"\n{written} notebooks built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
