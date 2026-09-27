# Lesson 05 — Code Documentation

**Goal:** write the documentation that saves someone a day, and stop writing the
kind that wastes one.

## What you will learn

- The test a comment must pass
- Docstrings that answer the real questions
- The README a stranger can follow
- Decision records, for the reasoning code cannot hold

---

## Two functions

```python
import ast

SAMPLE = '''
def process(df, t=0.5, f=True):
    # loop over rows
    out = []
    for i, r in df.iterrows():
        if r["score"] > t:
            out.append(r)
    return out


def calculate_refund_amount(order_total, days_since_delivery, is_member):
    """Return the refund due on an order, in EGP.

    Refunds are full within 14 days and half between 15 and 30 days.
    Members get the full amount for the whole 30 days. Outside 30 days the
    refund is zero. See policy PR-2026-03; the 30-day cliff is deliberate.
    """
    if days_since_delivery > 30:
        return 0.0
    if days_since_delivery <= 14 or is_member:
        return order_total
    return order_total * 0.5
'''
```

```python
def audit_docs(source):
    tree = ast.parse(source)
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            args = [a.arg for a in node.args.args]
            named_args = sum(1 for a in args if len(a) > 2)
            rows.append({
                "name": node.name,
                "has_docstring": doc is not None,
                "doc_words": len(doc.split()) if doc else 0,
                "args": len(args),
                "descriptive_arg_names": f"{named_args}/{len(args)}",
                "name_length": len(node.name),
            })
    return rows

print(f"{'function':<26}{'doc':>5}{'words':>7}{'args':>6}{'clear names':>13}")
for r in audit_docs(SAMPLE):
    print(f"{r['name']:<26}{'yes' if r['has_docstring'] else 'NO':>5}"
          f"{r['doc_words']:>7}{r['args']:>6}{r['descriptive_arg_names']:>13}")
```

```text
function                    doc  words  args  clear names
process                      NO      0     3          0/3
calculate_refund_amount     yes     47     3          3/3
```

`process(df, t=0.5, f=True)` gives a reader nothing. What is `t`? What does `f`
switch? What comes back — rows, indices, a frame?

`calculate_refund_amount(order_total, days_since_delivery, is_member)` answers
all three in its signature. **The best documentation is a name**, and it costs
nothing to maintain because the compiler keeps it honest.

---

## The comment test

```text
  # loop over rows                  restates the code. Zero information.
  the 30-day cliff is deliberate    cannot be recovered from the code.
```

**Could a competent reader derive this from the code itself?** If yes, delete
it. If no, it is the comment worth writing.

| Comment | Verdict |
|---|---|
| `# increment counter` | Delete |
| `# loop over rows` | Delete |
| `# TODO: clean this up` | Delete, or make it a ticket with a date |
| `# the 30-day cliff is policy PR-2026-03, not an approximation` | **Keep** |
| `# sorted before grouping because groupby assumes order here` | **Keep** |
| `# 0.192 comes from call capacity, not from the model` | **Keep** |
| `# do not vectorise: the API rate-limits at 10/s` | **Keep** |

Every keeper has the same shape: **it explains a decision, a constraint or a
surprise** — something that was in the author's head and nowhere in the syntax.

The most valuable comment in any codebase is the one that says why the obvious
improvement is wrong. It has saved a day for every person who read it.

---

## Docstrings

Answer four questions, in this order:

```text
1. What does it return?      the caller's actual question
2. What does it assume?      preconditions that will bite
3. When does it fail?        exceptions, edge cases, empty input
4. Why does it work this way? only when surprising
```

A good one:

```python
# no-run
def score_batch(frame, bundle, top_k=None):
    """Score a frame of customers and return the ranked call list.

    Returns a DataFrame indexed like `frame`, with `probability`, `action`
    ("call" or "skip") and `model`, sorted by probability descending.

    Assumes `frame` has every column in `bundle["feature_order"]`; raises
    ValueError listing the specific problems if not. An unseen category is
    encoded as all-zeros and scored silently, so validate first.

    `top_k` is the call-centre's capacity, not a model parameter: changing it
    does not require retraining.
    """
```

Note what is **not** there: how it works. The caller does not care, and the
implementation will change.

Note what is: the return shape, the precondition, the failure mode, and the
surprise (`handle_unknown="ignore"` scoring silently). Each of those is a
support question someone will not now have to ask you.

---

## READMEs

```python
REQUIRED = {
    "what it does": ["## what", "# what", "purpose"],
    "how to install": ["install", "pip install", "requirements"],
    "how to run it": ["usage", "how to run", "quickstart", "```bash"],
    "how to test": ["test", "pytest"],
    "where data comes from": ["data", "dataset", "source"],
    "who owns it": ["owner", "maintainer", "contact"],
    "known limitations": ["limitation", "caveat", "not supported", "known issue"],
}
EXAMPLE_README = "\n".join([
    "# Churn model",
    "Predicts which subscribers cancel next month.",
    "## Install",
    "pip install -r requirements.txt",
    "## Usage",
    "    python score_batch.py --date 2026-10-05",
])
def check_readme(text):
    low = text.lower()
    return {k: any(t in low for t in v) for k, v in REQUIRED.items()}
res = check_readme(EXAMPLE_README)
for k, ok in res.items():
    print(f"  {'[x]' if ok else '[ ]'} {k}")
print(f"\n  {sum(res.values())}/{len(res)} sections present")
```

```text
  [ ] what it does
  [x] how to install
  [x] how to run it
  [ ] how to test
  [ ] where data comes from
  [ ] who owns it
  [ ] known limitations

  2/7 sections present
```

Two of seven — and this is a better README than most, because it has a runnable
command in it.

The seven sections, and what each one prevents:

| Section | Prevents |
|---|---|
| **What it does**, in one sentence | Someone reading the code to find out |
| **How to install** | An hour of dependency archaeology |
| **How to run it**, with a real command | A message to you |
| **How to test** | A change nobody dares make |
| **Where the data comes from** | Rediscovering the source |
| **Who owns it** | The project quietly becoming nobody's |
| **Known limitations** | Someone using it outside its scope |

Two rules that make READMEs survive:

**Every command must be copy-pasteable and correct.** A README with a broken
first command teaches the reader to distrust the rest of it. Run the commands
from a clean checkout before you commit.

**Write it for someone on their first day.** If it says "just run the usual
pipeline", it is for you, and you are not the reader.

---

## Decision records

Code says what. Git says when. **Nothing says why**, and why is the expensive
thing to reconstruct.

```text
# ADR-014: Use logistic regression for churn scoring

Date: 2026-09-27
Status: accepted
Owner: adam

## Context
Gradient boosting and random forests were candidates. Cross-validated AUC:
logistic 0.682 +/- 0.011, boosting 0.662 +/- 0.019, forest 0.613 +/- 0.015.

## Decision
Logistic regression.

## Why
It won on the metric, it is 20x faster to fit, and its probabilities are
calibrated (predicted 0.360 vs observed 0.364) which the discount engine
requires. The forest's probabilities were not (0.492 predicted, 0.232 observed).

## Consequences
We cannot capture interactions without engineering them by hand. If the feature
set grows past ~30 columns, revisit.

## What would change this
A downstream use that does not need calibration, plus a feature set where
interactions matter.
```

Half a page, written once, and it answers the question a new team member will
otherwise spend a day on — or, worse, will not ask, and will "improve" the model
to a random forest whose probabilities break the discount engine.

**Write one for every decision you had to think about for more than an hour.**

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| `process(df, t, f)` | Three unanswered questions in the signature |
| Comments that restate the code | Noise that must be maintained |
| Docstrings describing the implementation | The caller wanted the contract |
| No failure modes documented | Every edge case becomes a message to you |
| A README missing "how to test" | Nobody dares change it |
| A README command that does not work | The reader distrusts everything after it |
| No decision records | The reasoning is gone, and someone undoes it |
| `# TODO` with no date or owner | A comment that will outlive the project |

---

## Exercises

1. Run the docstring audit over one of your modules. Report the fraction of
   functions with docstrings and the fraction with clear argument names.
2. Find five comments in your codebase that restate the code. Delete them.
3. Find one place where the obvious improvement is wrong, and write the comment
   that says so.
4. Run the README check on your repository. Fill in the missing sections.
5. Write the decision record for the biggest choice in your current project,
   including "what would change this".

---

**Next:** [Lesson 06 — Artifacts](06-artifacts.md)
