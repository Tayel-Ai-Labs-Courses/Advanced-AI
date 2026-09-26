# Lesson 07 — Visualising

**Goal:** make a chart that answers the question, and does not mislead.

## What you will learn

- Choosing the chart from the question
- The four things every chart needs
- Axes, and how they lie
- What to delete

---

## The chart follows the question

| Question | Chart |
|---|---|
| How did it change over time? | **Line** |
| Which category is bigger? | **Horizontal bar**, sorted |
| How is this distributed? | **Histogram** or box plot |
| Are these two related? | **Scatter** |
| What is the composition? | Stacked bar — or a table |
| How do groups compare over time? | Multi-line, or small multiples |
| What is the exact number? | **A table.** Not a chart |

Two rules that remove most bad charts:

- **Pie charts**: use a sorted horizontal bar instead. Humans compare lengths
  accurately and angles badly, and a pie with more than three slices is
  unreadable.
- **Fewer than six numbers**: write the numbers. A chart of four values is
  decoration.

---

## The four requirements

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet").set_index("ordered_at")
weekly = orders["amount"].resample("W").sum().iloc[1:-1]      # drop partial weeks

figure, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(weekly.index, weekly.values, linewidth=2, color="#2563eb")
ax.set_title("Weekly revenue is flat at ~20,000 EGP, with no trend")   # 1. the finding
ax.set_xlabel("Week starting")                                         # 2. labelled axes
ax.set_ylabel("Revenue (EGP)")                                         # 3. units
ax.set_ylim(0, weekly.max() * 1.15)                                    # 4. honest axis
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
figure.tight_layout()
figure.savefig("/tmp/weekly_revenue.png", dpi=150)

print("weeks plotted:", len(weekly))
print("range:", f"{weekly.min():,.0f} to {weekly.max():,.0f} EGP")
print("saved: /tmp/weekly_revenue.png")
```

```text
weeks plotted: 25
range: 17,740 to 22,985 EGP
saved: /tmp/weekly_revenue.png
```

1. **The title states the finding**, not the columns. "Weekly revenue" is a
   label; "Weekly revenue is flat at ~20,000 EGP" is information.
2. **Both axes labelled.**
3. **Units in the label** — EGP, not "amount".
4. **The y-axis starts at zero** for a quantity.

And one thing not in the list: `iloc[1:-1]` drops the partial first and last
weeks. Lesson 06's trap, applied before the chart rather than explained
underneath it.

---

## The truncated axis

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

branch_revenue = pd.Series({"Zamalek": 192150, "Maadi": 152355,
                            "Heliopolis": 104950, "Giza": 79890})

figure, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(branch_revenue.index, branch_revenue.values, color="#2563eb")
axes[0].set_ylim(70_000, 200_000)
axes[0].set_title("Truncated axis: Zamalek looks 5x Giza")

axes[1].bar(branch_revenue.index, branch_revenue.values, color="#2563eb")
axes[1].set_ylim(0, 200_000)
axes[1].set_title("Honest axis: Zamalek is 2.4x Giza")

figure.tight_layout()
figure.savefig("/tmp/axis_comparison.png", dpi=150)

truncated_ratio = ((branch_revenue["Zamalek"] - 70_000) /
                   (branch_revenue["Giza"] - 70_000))
print(f"true ratio:            {branch_revenue['Zamalek'] / branch_revenue['Giza']:.2f}x")
print(f"apparent ratio at y=70k: {truncated_ratio:.2f}x")
```

```text
true ratio:            2.41x
apparent ratio at y=70k: 12.35x
```

Starting the axis at 70,000 makes Zamalek's bar look **twelve times** Giza's
when the true ratio is 2.4. The numbers are correct; the chart lies.

| Chart type | Zero baseline |
|---|---|
| Bar chart of a quantity | **Required.** The length *is* the value |
| Line chart of a trend | Optional — the shape is the message |
| Line chart of an index or rate | Optional, but label it clearly |
| Anything shown to a decision-maker | Say what the axis starts at |

---

## Composition

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet")
pivot = orders.pivot_table(index="branch", columns="product",
                           values="amount", aggfunc="sum")
share = pivot.div(pivot.sum(axis=1), axis=0) * 100

figure, ax = plt.subplots(figsize=(9, 4))
bottom = pd.Series(0.0, index=share.index)
for product in share.columns:
    ax.barh(share.index, share[product], left=bottom, label=product)
    bottom += share[product]
ax.set_xlabel("Share of branch revenue (%)")
ax.set_title("Product mix is nearly identical across branches")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
figure.tight_layout()
figure.savefig("/tmp/product_mix.png", dpi=150)

print(share.round(1).to_string())
```

```text
product     cake  espresso  juice  latte   tea
branch                                        
Giza        15.7      22.9   11.3   28.8  21.3
Heliopolis  15.9      22.2   12.6   28.0  21.3
Maadi       14.8      23.7   11.0   29.5  20.9
Zamalek     14.1      24.4   10.9   30.0  20.7
```

The title says what the chart shows. A reader who sees only the title has the
finding; the chart is the evidence.

**Stacked bars are hard to read** beyond the bottom segment — only the first
category shares a baseline. When the comparison matters, use grouped bars or
small multiples instead.

---

## Distributions

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

orders = pd.read_parquet("/tmp/orders_clean.parquet")
amount = orders["amount"]

figure, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(amount, bins=30, color="#2563eb", edgecolor="white")
axes[0].axvline(amount.mean(), color="#dc2626", linestyle="--",
                label=f"mean {amount.mean():.0f}")
axes[0].axvline(amount.median(), color="#16a34a", linestyle="--",
                label=f"median {amount.median():.0f}")
axes[0].set_xlabel("Order amount (EGP)")
axes[0].set_ylabel("Orders")
axes[0].legend(frameon=False)
axes[0].set_title("Order amounts cluster at 60-180 EGP")

orders.boxplot(column="amount", by="branch", ax=axes[1], grid=False)
axes[1].set_title("Similar distribution in every branch")
axes[1].set_xlabel("")
axes[1].set_ylabel("Order amount (EGP)")
figure.suptitle("")
figure.tight_layout()
figure.savefig("/tmp/distributions.png", dpi=150)

print(f"mean {amount.mean():.1f}, median {amount.median():.1f}")
print(amount.describe().round(1).to_string())
```

```text
mean 106.2, median 105.0
count    4985.0
mean      106.2
std        55.5
min        30.0
25%        60.0
50%       105.0
75%       140.0
max       240.0
```

**Plot the distribution before reporting any average.** The histogram shows
whether the mean is representative; here mean and median are 106.2 and 105.0,
so it is. In lesson 02, before cleaning, they were 258.5 and 105.0 — and a
histogram would have shown that immediately.

---

## Delete things

| Delete | Why |
|---|---|
| Gridlines you are not reading against | Ink without information |
| The top and right spines | The data does not need a box |
| 3-D effects | They distort the values |
| A legend for one series | Put it in the title |
| Decimal places nobody needs | 106.2, not 106.234891 |
| A chart of four numbers | Write the numbers |

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

figure, ax = plt.subplots(figsize=(7, 4))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ax.tick_params(length=0)
figure.savefig("/tmp/clean_style.png", dpi=150)
print("four lines, and the chart stops competing with the data")
```

```text
four lines, and the chart stops competing with the data
```

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Title naming the columns | The reader has to find the finding |
| Truncated bar axis | Differences exaggerated several-fold |
| No units | "Amount" — of what? |
| Pie chart with eight slices | Unreadable |
| Chart of four numbers | A table would be faster |
| Partial periods at the edges | A cliff that is a calendar |
| Default matplotlib styling | Boxed, gridded, hard to read |

---

## Exercises

1. Take a chart you have made and rewrite the title as the finding.
2. Draw the same bar chart with a truncated and a zero axis; compare.
3. Plot the distribution of your main measure; is the mean representative?
4. Remove every element from one chart that is not carrying information.
