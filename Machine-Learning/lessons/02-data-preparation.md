# Lesson 02 — Data Preparation

**Goal:** turn a messy table into a numeric matrix a model can learn from.

## What you will learn

- Why models need numbers
- Missing values, done deliberately
- Encoding categories
- Scaling, and which models care
- Leakage — the mistake that invalidates everything

---

## Models eat numbers

```python
import pandas as pd

orders = pd.DataFrame({
    "city":      ["Cairo", "Giza", "Cairo", "Alex"],
    "plan":      ["basic", "pro", "pro", "basic"],
    "age":       [25, 34, None, 45],
    "spend":     [120.0, 340.0, 210.0, 90.0],
    "churned":   [0, 1, 0, 1],
})

print(orders.dtypes)
print(orders.isna().sum().sum(), "missing values")
```

```text
city        object
plan        object
age        float64
spend      float64
churned      int64
dtype: object
1 missing values
```

Two problems before any model can run: `city` and `plan` are text, and `age`
has a hole. Everything in this lesson is fixing those two things without
cheating.

---

## Missing values

First, look at how much is missing and where:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age":    [25, np.nan, 40, np.nan, 31],
    "income": [5000, 7000, np.nan, 4000, 6200],
})

print(df.isna().sum())
print((df.isna().mean() * 100).round(1).to_dict(), "% missing")
```

```text
age       2
income    1
dtype: int64
{'age': 40.0, 'income': 20.0} % missing
```

Then choose, and write down why:

| Situation | Do |
|---|---|
| A few rows, and the label is missing | Drop the rows |
| A column is 80% empty | Drop the column |
| Numeric, missing at random | Fill with the median |
| Categorical | Fill with the mode, or a `"missing"` category |
| Time series | `ffill` — carry the last value forward |
| Missing *means* something | Add an `is_missing` flag, then fill |

That last row matters more than it looks. A blank `last_login` may mean the
user never logged in — which is the most predictive fact in the table. Deleting
it throws away the signal.

```python
from sklearn.impute import SimpleImputer
import numpy as np

X = np.array([[25.0, 5000.0], [np.nan, 7000.0], [40.0, np.nan]])

imputer = SimpleImputer(strategy="median")
print(imputer.fit_transform(X))
```

```text
[[  25.  5000. ]
 [  32.5 7000. ]
 [  40.  6000. ]]
```

Use `SimpleImputer` rather than `df.fillna()`, because the imputer *remembers*
the median from training and applies that same number at prediction time. Using
the test set's own median is a leak.

---

## Encoding categories

### One-hot — for categories with no order

```python
import pandas as pd

df = pd.DataFrame({"city": ["Cairo", "Giza", "Cairo", "Alex"]})
print(pd.get_dummies(df, columns=["city"], drop_first=False).astype(int))
```

```text
   city_Alex  city_Cairo  city_Giza
0          0           1          0
1          0           0          1
2          0           1          0
3          1           0          0
```

One column per category, a single 1 per row. Use it for city, product,
payment method — anything where no category is "more" than another.

The cost is width: 1,000 cities become 1,000 columns. For high-cardinality
columns, group the rare values into `"other"`, or use target encoding.

### Ordinal — only when the order is real

```python
import pandas as pd

sizes = pd.DataFrame({"size": ["S", "L", "M", "S"]})
order = {"S": 0, "M": 1, "L": 2}
sizes["size_encoded"] = sizes["size"].map(order)
print(sizes)
```

```text
  size  size_encoded
0    S             0
1    L             2
2    M             1
3    S             0
```

Small < medium < large is a genuine order, so numbers carry meaning.

**The classic error** is doing this to unordered categories: `Cairo=0,
Giza=1, Alex=2` tells a linear model that Alex is twice Giza and that the
average of Cairo and Alex is Giza. It is nonsense, and the model will believe
it.

(Tree models survive it. Linear models, KNN and SVM do not.)

---

## Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

X = np.array([[25.0, 50_000.0], [40.0, 80_000.0], [31.0, 62_000.0]])

print(StandardScaler().fit_transform(X).round(2))
print(MinMaxScaler().fit_transform(X).round(2))
```

```text
[[-1.14 -1.14]
 [ 1.3   1.3 ]
 [-0.16 -0.16]]
[[0.  0. ]
 [1.  1. ]
 [0.4 0.4]]
```

Age runs 20–60; income runs 20,000–200,000. To any model that measures
distance, income is the only column that exists — not because it matters more,
but because its numbers are bigger.

| Model | Needs scaling? |
|---|---|
| Linear / logistic regression | Yes (and required for regularisation to be fair) |
| KNN, SVM, K-Means | **Yes** — they are distance, and nothing else |
| Neural networks | Yes |
| Decision tree, random forest, boosting | **No** — they split on thresholds |

`StandardScaler` (mean 0, std 1) is the default. `MinMaxScaler` (0 to 1) when
you need a bounded range. `RobustScaler` when outliers are ruining the other
two.

---

## Leakage

**Leakage is using information at training time that will not exist at
prediction time.** It is the most expensive mistake in this course, because it
does not look like a bug — it looks like success.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = np.arange(20, dtype=float).reshape(-1, 1)
y = np.array([0] * 10 + [1] * 10)

# WRONG — the scaler saw the test set
scaler = StandardScaler().fit(X)
X_all_scaled = scaler.transform(X)

# RIGHT — fit on train only, apply to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("train mean:", X_train_scaled.mean().round(3))
print("test mean: ", X_test_scaled.mean().round(3))
```

```text
train mean: -0.0
test mean:  0.758
```

The test mean is **not** 0, and that is correct. Test data is supposed to be
unseen; if scaling made it perfectly centred, information had crossed the line.

Four leaks to watch for:

1. **Fitting any transformer on all the data** — scalers, imputers, encoders,
   PCA, feature selection.
2. **A feature that contains the answer** — `total_paid` when predicting
   whether they paid.
3. **Future information** — a column recorded after the moment you predict.
4. **Duplicate rows split across train and test** — the model memorises and
   you score its memory.

The tell is an accuracy that surprises you. Investigate good news as hard as
bad news.

---

## Putting it together

`ColumnTransformer` applies different treatment to different columns, and
because it lives inside a pipeline, it is fitted on the training fold only —
leakage becomes structurally impossible.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    "city":  ["Cairo", "Giza", "Cairo", "Alex", "Giza", "Cairo"],
    "age":   [25, 34, None, 45, 29, 52],
    "spend": [120.0, 340.0, 210.0, 90.0, 260.0, 410.0],
})
y = [0, 1, 0, 1, 1, 0]

numeric = ["age", "spend"]
categorical = ["city"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

model = Pipeline([("prep", preprocess), ("clf", LogisticRegression())])
model.fit(df, y)

print("features after preprocessing:", model.named_steps["prep"].transform(df).shape)
print("prediction:", model.predict(df.head(1)))
```

```text
features after preprocessing: (6, 5)
prediction: [0]
```

Two numeric columns plus three cities gives five features.
`handle_unknown="ignore"` is what stops a city that appears only in production
from crashing the service at 3am.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| `fit` on all data, then split | Leakage — scores you cannot reproduce in production |
| Label-encoding unordered categories | A false order the model believes |
| Dropping every row with any NaN | Half the dataset gone, and a biased remainder |
| Forgetting to scale for KNN/SVM | The largest-unit column decides everything |
| `handle_unknown` left at default | `ValueError` on the first unseen category |
| Encoding after splitting, separately | Train and test end up with different columns |

---

## Exercises

1. Take any CSV. Report missing values per column as a percentage.
2. One-hot encode a categorical column; explain what `drop_first=True` changes
   and when it matters.
3. Demonstrate the leak: score a model with the scaler fitted on all data, and
   again fitted on train only. Report both numbers.
4. Build a `ColumnTransformer` for a table with numeric, ordinal and nominal
   columns.
