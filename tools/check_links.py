"""Check that every relative markdown link in the repository resolves.

Links inside fenced code blocks are ignored: `operations["square"](5)` is Python,
not a link. Exits non-zero when something is broken, so CI can run it.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```.*?```", re.S)
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(#[^)]*)?\)")
EXTERNAL = ("http://", "https://", "mailto:")


def broken_links(path):
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    for match in LINK.finditer(text):
        target = match.group(1)
        if target.startswith(EXTERNAL):
            continue
        if not (path.parent / target).exists():
            yield target


def main():
    failures = 0
    for markdown in sorted(REPO_ROOT.rglob("*.md")):
        if ".git" in markdown.parts:
            continue
        for target in broken_links(markdown):
            print(f"{markdown.relative_to(REPO_ROOT)}: {target}")
            failures += 1
    print(f"broken links: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
