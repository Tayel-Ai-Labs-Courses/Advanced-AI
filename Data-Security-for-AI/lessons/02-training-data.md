# Lesson 02 — Training Data and Re-identification

**Goal:** see why removing the name is not anonymisation, on data you can hold
in your head.

## What you will learn

- Quasi-identifiers, and how few you need
- k-anonymity, measured
- Why aggregate queries leak too
- What "anonymised" has to mean before you can say it

---

## A dataset with the names removed

```python
import numpy as np
import pandas as pd

import numpy as np, pandas as pd

rng = np.random.default_rng(0)
n = 5_000
df = pd.DataFrame({
    "name": [f"person_{i}" for i in range(n)],
    "birth_year": rng.integers(1950, 2006, n),
    "gender": rng.choice(["M", "F"], n),
    "governorate": rng.choice(
        ["Cairo", "Giza", "Alexandria", "Dakahlia", "Sharqia", "Qalyubia",
         "Beheira", "Gharbia", "Minya", "Asyut"], n,
        p=[.22, .16, .12, .10, .09, .08, .07, .06, .05, .05]),
    "job": rng.choice(["engineer", "teacher", "doctor", "driver", "sales",
                       "student", "retired"], n),
    "salary": rng.normal(12_000, 4_000, n).round(0),
})

print(df.head(3).to_string(index=False))
```

```text
    name  birth_year gender governorate     job  salary
person_0        1997      F     Gharbia teacher  8486.0
person_1        1985      F    Qalyubia  driver 12222.0
person_2        1978      F  Alexandria  driver 16016.0
```

This is the shape of almost every "anonymised export" in existence: the direct
identifier is dropped, everything else is kept because "it's just demographics".

---

## Four ordinary facts

```python
for quasi in (["governorate"], ["governorate", "gender"],
              ["governorate", "gender", "birth_year"],
              ["governorate", "gender", "birth_year", "job"]):
    sizes = df.groupby(quasi).size()
    unique = int((sizes == 1).sum())
    k = int(sizes.min())
    covered = int(sizes[sizes == 1].sum())
    print(f"{' + '.join(quasi):<44} k={k:<4} {unique:>4} groups of one, "
          f"{covered / n:>6.1%} of people uniquely identified")
```

```text
governorate                                  k=244     0 groups of one,   0.0% of people uniquely identified
governorate + gender                         k=114     0 groups of one,   0.0% of people uniquely identified
governorate + gender + birth_year            k=1     120 groups of one,   2.4% of people uniquely identified
governorate + gender + birth_year + job      k=1    2314 groups of one,  46.3% of people uniquely identified
```

**Four ordinary attributes uniquely identify 46.3% of 5,000 people.**

Governorate alone is safe — the smallest group has 244 people. Add gender: still
114. Add birth year and it collapses, and adding a job takes nearly half the
dataset to a group of one.

This is how re-identification works everywhere. Nobody attacks the name column;
they **join on the quasi-identifiers** — the attributes that are individually
unremarkable and jointly unique. Anyone who knows four facts about a colleague
can find their row.

`k` is the size of the smallest group. **k=1 means at least one person in your
"anonymous" export is identifiable by anyone who knows their basic details.**

---

## Generalisation

```python
df2 = df.copy()
df2["birth_decade"] = (df2["birth_year"] // 10 * 10).astype(str) + "s"
df2["region"] = df2["governorate"].map(
    lambda g: "Greater Cairo" if g in ("Cairo", "Giza", "Qalyubia") else "Other")
for quasi, label in [(["governorate", "gender", "birth_year", "job"], "raw"),
                     (["region", "gender", "birth_decade", "job"], "generalised")]:
    sizes = df2.groupby(quasi).size()
    print(f"{label:<14} k={sizes.min():<5} uniquely identified "
          f"{int((sizes == 1).sum()):>4} people  "
          f"({(sizes == 1).sum() / n:>6.1%})")
```

```text
raw            k=1     uniquely identified 2314 people  ( 46.3%)
generalised    k=11    uniquely identified    0 people  (  0.0%)
```

Birth year to decade, governorate to region: **k goes from 1 to 11**, and nobody
is uniquely identified.

The cost is analytical precision, and it is real — you can no longer study
year-on-year cohorts or governorate-level differences. That is the trade, and it
belongs to the data owner, not to the engineer doing the export.

| Technique | What it does | Cost |
|---|---|---|
| **Generalisation** | Year -> decade, city -> region | Loses granularity |
| **Suppression** | Drop the rows in small groups | Loses the rare cases, which are often the interesting ones |
| **Top/bottom coding** | "65+", "over 100,000 EGP" | Loses the tails |
| **Noise** | Add a random amount | Lesson 07; needs a budget to be meaningful |
| **Sampling** | Release 10% of rows | Weakens linkage; does not prevent it |
| **Pseudonymisation** | Replace the id with a token | **Not anonymisation.** Reversible by whoever holds the mapping |

The last row matters legally as well as technically. Pseudonymised data is still
personal data; a token that maps back to a person is a person.

---

## Aggregates leak too

"We only expose aggregates" is the most common reassurance, and it is wrong for
the same reason.

```python
quasi = ["governorate", "gender", "birth_year", "job"]
sizes = df.groupby(quasi).size()
singleton = sizes[sizes == 1].index[0]
mask = np.ones(len(df), bool)
for col, val in zip(quasi, singleton):
    mask &= (df[col] == val)
person = df[mask].iloc[0]
print("an attacker who knows four ordinary facts about someone:")
print(f"  {dict(zip(quasi, singleton))}")
print(f"  matching rows in the 'anonymised' table: {mask.sum()}")
print(f"  their salary, which was meant to be private: {person.salary:,.0f}")

print("\naggregate queries leak the same way:")
q = df[(df.governorate == singleton[0]) & (df.gender == singleton[1]) &
       (df.job == singleton[3])]
print(f"  'mean salary of {singleton[1]} {singleton[3]}s in {singleton[0]}': "
      f"{q.salary.mean():,.0f} over {len(q)} people  (safe)")
q2 = q[q.birth_year == singleton[2]]
print(f"  add one more filter (born {singleton[2]}): "
      f"{q2.salary.mean():,.0f} over {len(q2)} person  (that is a salary)")
```

```text
an attacker who knows four ordinary facts about someone:
  {'governorate': 'Alexandria', 'gender': 'F', 'birth_year': 1950, 'job': 'driver'}
  matching rows in the 'anonymised' table: 1
  their salary, which was meant to be private: 570

aggregate queries leak the same way:
  'mean salary of F drivers in Alexandria': 12,455 over 52 people  (safe)
  add one more filter (born 1950): 570 over 1 person  (that is a salary)
```

The first query is a genuine aggregate over 52 people. Add **one more filter**
and the "average" is one person's salary, reported to the attacker by your
dashboard, in a field labelled *mean*.

Defences, in the order they are usually needed:

1. **A minimum group size**, enforced in the query layer: refuse any result
   computed from fewer than *n* rows. Ten is a common floor.
2. **Suppress complementary results** too. If a filter returns fewer than ten,
   returning "suppressed" for that one and a real number for its complement lets
   the attacker subtract.
3. **A query budget.** Repeated differencing queries reconstruct individuals
   even when each answer is suppressed or noisy; lesson 07 is about bounding
   that formally.

---

## Before you call data anonymised

- [ ] The direct identifiers are gone (name, national id, phone, email, exact
      address, account number)
- [ ] **k has been measured** on the quasi-identifiers that remain, and it is
      above your threshold
- [ ] Free-text fields have been checked — a "notes" column is an identifier
      farm
- [ ] Dates are generalised. A precise timestamp is close to a fingerprint
- [ ] Location is generalised. GPS to four decimals is a household
- [ ] You have listed the **external datasets** someone could join against
- [ ] The minimum group size is enforced in any query interface
- [ ] Someone other than the person who built the export has reviewed it

The sixth item is the one that is easy to skip and impossible to undo. A dataset
is not anonymous in isolation; it is anonymous **relative to what else exists**,
and the other dataset may be published next year.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| "We removed the name" | Four other columns identified 46.3% of people |
| Pseudonymisation called anonymisation | A reversible token is still personal data |
| Keeping exact dates of birth | The single strongest quasi-identifier here |
| Free-text fields in an export | Names, phone numbers and addresses live there |
| "We only expose aggregates" | One extra filter turned a mean into a salary |
| No minimum group size in the query layer | The dashboard does the attacker's work |
| Ignoring joinable external data | Anonymity is relative to what else exists |

---

## Exercises

1. Measure k on a real export from your own work, with the quasi-identifiers you
   actually ship. Report the percentage of rows in groups of one.
2. Generalise until k >= 10 and quantify what analysis you lost.
3. Implement a minimum-group-size guard in a query function and find a pair of
   queries that defeats it by subtraction.
4. List the external datasets that could be joined against your export. For each,
   name the columns that would join.
5. Take a free-text column and count how many rows contain something that looks
   like a phone number, an email or a full name.

---

**Next:** [Lesson 03 — Memorisation and Membership Inference](03-memorisation.md)
