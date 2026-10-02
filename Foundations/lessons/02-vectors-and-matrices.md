# Lesson 02 — Vectors and Matrices

**Goal:** stop seeing a matrix as a grid of numbers and start seeing it as
something that *does* one thing.

## What you will learn

- A matrix as a transformation, and what the determinant measures
- The dot product as similarity — the operation behind embedding search
- Shapes, and the error you will hit most often
- Where each one shows up in the rest of the track

---

## A matrix transforms space

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)

A = np.array([[2.0, 1.0], [0.0, 1.5]])
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
print("the unit square's corners:\n", square)
print("\nafter A:\n", A @ square)
print(f"\narea before: 1.0   after: {abs(np.linalg.det(A)):.2f}  (that is |det A|)")
```

```text
the unit square's corners:
 [[0 1 1 0]
 [0 0 1 1]]

after A:
 [[0.  2.  3.  1. ]
 [0.  0.  1.5 1.5]]

area before: 1.0   after: 3.00  (that is |det A|)
```

`A` stretched the square by 2 horizontally, by 1.5 vertically, and sheared it.
The **determinant, 3.0, is the factor by which area changed.**

Two consequences worth carrying:

- **`det(A) = 0` means the transformation flattened space** — it squashed a
  dimension away, and the operation cannot be undone. That is exactly what
  "singular matrix" means when NumPy raises it at you.
- Every layer of a neural network is a matrix multiply plus a non-linearity.
  Training is searching for matrices whose combined transformation separates
  your classes.

---

## The dot product is a similarity

```python
def unit(v):
    return v / np.linalg.norm(v)
pairs = [("same direction", [1, 0], [2, 0]),
         ("45 degrees", [1, 0], [1, 1]),
         ("orthogonal", [1, 0], [0, 1]),
         ("opposite", [1, 0], [-1, 0])]
print(f"{'relationship':<18}{'cosine':>9}{'angle':>9}")
for name, a, b in pairs:
    c = float(unit(np.array(a, float)) @ unit(np.array(b, float)))
    print(f"{name:<18}{c:>9.3f}{np.degrees(np.arccos(np.clip(c,-1,1))):>8.0f}d")
print("\nthis is exactly the similarity in LLM lesson 05's embedding search")
```

```text
relationship         cosine    angle
same direction        1.000       0d
45 degrees            0.707      45d
orthogonal            0.000      90d
opposite             -1.000     180d

this is exactly the similarity in LLM lesson 05's embedding search
```

The dot product of two unit vectors **is** the cosine of the angle between them.
That single fact is why:

- [LLM lesson 05](../../LLM-and-GenAI/lessons/05-embeddings-and-search.md)
  searches a corpus with one matrix multiply — normalise everything, then `V @ q`
  is every similarity at once.
- Attention in a transformer scores query against key with a dot product.
- A logistic regression's `w @ x` is "how aligned is this example with the
  direction the model learned?".

**Normalise before comparing.** Without it the dot product mixes direction and
magnitude, and a long document beats a relevant one.

---

## Shapes, and the error you will actually hit

```python
A = np.random.default_rng(0).normal(size=(3, 4))
x = np.arange(4)
print("A", A.shape, "@ x", x.shape, "->", (A @ x).shape)
try:
    x @ A
except ValueError as e:
    print("x @ A raises:", str(e)[:60])
print("x @ A.T ->", (x @ A.T).shape)
```

```text
A (3, 4) @ x (4,) -> (3,)
x @ A raises: matmul: Input operand 1 has a mismatch in its core dimension
x @ A.T -> (3,)
```

The rule is the only one you need: **the inner dimensions must match, and the
outer ones survive.** `(3,4) @ (4,) -> (3,)`.

Three habits that remove most shape bugs:

1. **Write the shape in a comment** above any non-obvious line.
2. **Name the axes**: `(batch, features)`, `(batch, seq, dim)`.
3. When a broadcast surprises you, `print(a.shape, b.shape)` before debugging
   anything else.

Broadcasting is the other half:

```python
X = np.arange(6).reshape(2, 3).astype(float)
mean = X.mean(axis=0)            # (3,) - one per feature
print("X:\n", X)
print("column means:", mean)
print("centred:\n", X - mean)   # (2,3) - (3,) broadcasts along rows
```

```text
X:
 [[0. 1. 2.]
 [3. 4. 5.]]
column means: [1.5 2.5 3.5]
centred:
 [[-1.5 -1.5 -1.5]
 [ 1.5  1.5  1.5]]
```

That is `StandardScaler` with the mean part written out. Knowing it is a
broadcast, not a loop, is why the whole of scikit-learn is fast.

---

## Where this appears

| Operation | Appears in |
|---|---|
| `w @ x` | Every linear model; every network layer |
| Normalise, then dot | [LLM 05](../../LLM-and-GenAI/lessons/05-embeddings-and-search.md) embedding search |
| `X - mean` broadcast | [Data-Science 04](../../Data-Science/lessons/04-features-and-pipelines.md) scaling |
| `det = 0` | "Singular matrix", perfectly correlated features |
| Matrix shapes | Every error message you will see in Deep-Learning |

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Treating a matrix as a table of numbers | You cannot predict what it does |
| Comparing un-normalised vectors | Length beats relevance |
| Guessing at shapes | Ten minutes per bug, forever |
| Looping where you could broadcast | 100x slower, and harder to read |
| Ignoring `det = 0` | Your features are linearly dependent |

---

## Exercises

1. Build a matrix that rotates by 30 degrees. Confirm its determinant is 1 and
   explain why.
2. Take two sentences, embed them with any model, and compute the cosine
   similarity by hand with `@`. Compare with the library's answer.
3. Find a pair of features in a dataset you use whose correlation makes the
   matrix nearly singular. What does a linear model do with them?
4. Rewrite a loop from your own code as a broadcast. Time both.
5. Write down the shapes for one forward pass of a network you have trained,
   layer by layer.

---

**Next:** [Lesson 03 — Decomposition](03-decomposition.md)
