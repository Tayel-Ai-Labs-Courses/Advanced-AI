# Lesson 11 — Experiment Tracking and the Model Registry

**Read after [lesson 07](07-reproducibility.md).**

**Goal:** move from hand-written JSON run records to a tool, and know exactly
what the tool adds and what it does not fix.

## What you will learn

- MLflow in ten lines, locally, with no account
- What a tracking table shows you that a folder of JSON does not
- The registry: which model is in production, as data rather than a README line
- The trap a tracking UI makes easier, not harder

---

## Where this sits

Lesson 07 ended with a four-level ladder:

```text
0  notebook, no seeds, results in your head
1  seeds, one script, printed metrics
2  JSON run records, data fingerprints, committed code   <- lesson 07 built this
3  MLflow / W&B, artefacts stored, model registry        <- this lesson
```

Level 2 is reachable in an afternoon and answers most questions. **Do not skip
it waiting for permission to install a platform.** This lesson is what you add
when there are more runs than you can read, or more people than you, or an
auditor.

---

## MLflow, locally, in ten lines

```bash
pip install mlflow
```

No account, no server, no cloud. Everything below writes to a folder.

```python
# no-run: needs mlflow installed
import os
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns_demo"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow

mlflow.set_experiment("churn")
with mlflow.start_run(run_name="logreg-C0.1"):
    mlflow.log_params({"model": "LogisticRegression", "C": 0.1, "seed": 0})
    mlflow.log_metrics({"cv_auc_mean": 0.6818, "holdout_auc": 0.6828})
    mlflow.set_tags({"dataset": "subscribers", "rows": 12000})
```

Three calls — `log_params`, `log_metrics`, `set_tags` — and the same four groups
lesson 07 wrote by hand: code, data, parameters, environment. MLflow captures
the git commit automatically when you run from a repository.

**A note on the second environment variable.** MLflow 3 deprecated the plain
file store and wants a database (`sqlite:///mlflow.db`). The flag above opts
back in, which is fine for one person on a laptop. For a team, use the SQLite
backend from day one — a shared folder of files is a merge conflict waiting to
happen.

---

## Four candidates, tracked

```python
# no-run: needs mlflow installed
import os, warnings, shutil
warnings.filterwarnings("ignore")
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns_demo"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
shutil.rmtree("/tmp/mlruns_demo", ignore_errors=True)

import mlflow, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("/tmp/subscribers.parquet")
numeric = ["tenure_days", "logins_last_30d", "support_tickets_last_30d",
           "payment_failures_last_90d", "monthly_fee"]
categorical = ["plan", "country", "age_band"]
X, y = df[numeric + categorical], df["churned_next_30d"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

def pre():
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])

mlflow.set_experiment("churn")
CANDIDATES = {
    "logreg-C0.1": (LogisticRegression(C=0.1, max_iter=2000), {"C": 0.1}),
    "logreg-C1.0": (LogisticRegression(C=1.0, max_iter=2000), {"C": 1.0}),
    "tree-d4": (DecisionTreeClassifier(max_depth=4, random_state=0), {"max_depth": 4}),
    "hgb-default": (HistGradientBoostingClassifier(random_state=0), {}),
}
for name, (est, params) in CANDIDATES.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("pre", pre()), ("model", est)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        holdout = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        mlflow.log_params({"model": type(est).__name__, "seed": 0, **params})
        mlflow.log_metrics({"cv_auc_mean": float(scores.mean()),
                            "cv_auc_std": float(scores.std()),
                            "holdout_auc": float(holdout)})
        mlflow.set_tags({"dataset": "subscribers", "rows": len(df)})

runs = mlflow.search_runs(experiment_names=["churn"])
cols = ["tags.mlflow.runName", "params.model", "metrics.cv_auc_mean",
        "metrics.cv_auc_std", "metrics.holdout_auc"]
table = runs[cols].rename(columns=lambda c: c.split(".")[-1])
table = table.sort_values("cv_auc_mean", ascending=False)
print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n{len(runs)} runs recorded in {mlflow.get_tracking_uri()}")
```

```text
    runName                          model  cv_auc_mean  cv_auc_std  holdout_auc
logreg-C0.1             LogisticRegression       0.6818      0.0113       0.6828
logreg-C1.0             LogisticRegression       0.6817      0.0113       0.6826
hgb-default HistGradientBoostingClassifier       0.6640      0.0164       0.6649
    tree-d4         DecisionTreeClassifier       0.6604      0.0123       0.6835
4 runs recorded in file:///tmp/mlruns_demo
```

Those are lesson 05's numbers, reproduced through a tool instead of a print
statement. And the table contains the trap.

**`tree-d4` has the worst cross-validated score (0.6604) and the best holdout
score (0.6835).**

Sort this table by `holdout_auc` — which is one click in the MLflow UI, and the
column with the most convincing name — and you ship the fourth-best model.
Lesson 07 measured why: a single split moves by 0.0149 on this data, and the
0.0007 gap between `tree-d4` and `logreg-C0.1` on holdout is a fifth of that.

**A tracking UI makes this mistake easier, not harder**, because it makes
sorting by any column effortless and shows no intervals. Two habits fix it:

1. **Always log `cv_auc_std` beside `cv_auc_mean`.** A mean without its spread
   invites a ranking it cannot support.
2. **Write down the selection metric before the runs**, in the experiment
   description. "We select on `cv_auc_mean`; `holdout_auc` is reported for
   traceability only."

---

## What the tool adds

| Capability | Level 2 (JSON) | Level 3 (MLflow) |
|---|---|---|
| Record params and metrics | yes | yes |
| Compare 5 runs | read 5 files | a table |
| Compare 500 runs | **no** | **yes**, with filters |
| Find "all runs on data sha X" | grep | a query |
| Store the model artefact | you write it | `log_model`, with its environment |
| Store the plot / confusion matrix | you write it | `log_artifact` |
| Share with a colleague | send files | a URL |
| Which model is in production | **a README line** | **the registry** |
| Approval trail | git history | stages + who moved it |

The two rows that justify the install are **500 runs** and **the registry**.
Everything above them you can do with a folder.

---

## The registry

Lesson 07 ended with the most valuable line in the repository:

```text
PRODUCTION: logreg-C0.1  (run 2026-09-27, data sha 6b3f8cdb, git 4a91c02)
```

A registry makes that line **data instead of prose**:

```python
# no-run
mlflow.register_model("runs:/<run_id>/model", "churn")
client.set_registered_model_alias("churn", "production", version=3)
```

Then the scoring job asks the registry rather than trusting a filename:

```python
# no-run
model = mlflow.sklearn.load_model("models:/churn@production")
```

What this buys, concretely:

- **The deployed version is queryable**, so "what is live?" has one answer
- **Promotion is an event** with a user and a timestamp — an approval trail
- **Rollback is an alias change**, not a redeploy
- **The scoring job cannot drift** from the intended model, because it does not
  hold a path

What it does **not** buy: any of the discipline. A registry with one model
called `final_v2_new` and no metrics is a folder with extra steps.

---

## W&B, and choosing

| Tool | Good at | Cost |
|---|---|---|
| **JSON files** (lesson 07) | Starting today, auditable, zero deps | Manual comparison |
| **MLflow** | Local or self-hosted, registry, sklearn/PyTorch integration | You run the server |
| **Weights & Biases** | Live training curves, sharing, sweeps | SaaS; check where your data goes |
| DVC | Data and model versioning in git | Steeper, more opinionated |

For an Egyptian company handling customer data, the **"where does the data go"**
column is not a footnote — see Data-Security lesson 08. MLflow self-hosted keeps
everything inside your network; W&B is excellent and is somebody else's server.

**Log metadata, never raw personal data.** A tracking system is not a data
store, and a metric named `customer_12345_score` is a privacy incident with a
dashboard.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Installing a platform before doing level 2 | The discipline is the point; the tool is not |
| Sorting the UI by the most impressive column | `tree-d4` wins on holdout and is the fourth-best model |
| Logging a mean without its spread | The table invites a ranking it cannot support |
| No selection metric agreed in advance | It gets chosen after the results are in |
| A registry with no metrics attached | A folder with extra steps |
| Personal data in metric names or tags | A privacy incident with a dashboard |
| A shared file-store tracking folder | Concurrent writes, corrupted runs |
| Never deleting failed runs | 500 runs, 400 of them noise |

---

## Exercises

1. Re-run lesson 05's ladder under MLflow and confirm the numbers match.
2. Add `cv_auc_std` to every run, then write the query that returns only runs
   whose mean exceeds the best mean minus one standard deviation.
3. Register two versions and move the `production` alias between them. Where is
   that event recorded?
4. Log the calibration table from lesson 06 as an artifact and open it from the
   UI.
5. Write your experiment's description field, stating the selection metric,
   before your next set of runs.

---

**Next:** [Lesson 12 — Prototypes That Get a Decision](12-prototypes.md)
