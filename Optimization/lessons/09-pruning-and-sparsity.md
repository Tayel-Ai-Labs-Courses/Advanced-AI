# Lesson 09 — Pruning and Sparsity

**Goal:** remove weights that do not earn their place.

## What you will learn

- Unstructured versus structured pruning
- `torch.nn.utils.prune`
- Iterative pruning with fine-tuning
- Why sparsity often does not make things faster

---

## The idea

Trained networks are over-parameterised. Many weights are near zero and
contribute almost nothing; setting them to exactly zero changes the output
very little.

```mermaid
flowchart LR
    D["dense weights<br/>most near zero"] --> P["prune<br/>zero the smallest"]
    P --> F["fine-tune<br/>the survivors adapt"]
    F --> R["repeat"]
```

The catch, stated up front: **zeros save memory only if you store them
sparsely, and save time only if your hardware skips them.** Dense hardware
multiplies by zero at full speed. Most of this lesson is about when that
matters.

---

## Magnitude pruning

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

torch.manual_seed(0)
layer = nn.Linear(64, 64)

prune.l1_unstructured(layer, name="weight", amount=0.5)      # zero the smallest 50%

print("weight_orig exists:", hasattr(layer, "weight_orig"))
print("mask zeros:", int((layer.weight_mask == 0).sum()), "of", layer.weight_mask.numel())
print("actual zeros in weight:", int((layer.weight == 0).sum()))

prune.remove(layer, "weight")                                 # make it permanent
print("after remove — zeros:", int((layer.weight == 0).sum()))
print("still a plain Linear:", isinstance(layer, nn.Linear))
```

```text
weight_orig exists: True
mask zeros: 2048 of 4096
actual zeros in weight: 2048
after remove — zeros: 2048
still a plain Linear: True
```

Before `prune.remove`, PyTorch keeps the original weights in `weight_orig` and
a `weight_mask`, recomputing `weight = weight_orig * weight_mask` on every
forward pass. That costs memory and time — **`prune.remove` is not optional**
once you are done.

---

## How much can you remove?

```python
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_model(model, amount):
    """Globally prune the smallest `amount` of all weights, permanently."""
    pruned = copy.deepcopy(model)
    parameters = [(m, "weight") for m in pruned.modules() if isinstance(m, nn.Linear)]
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured,
                              amount=amount)
    for module, name in parameters:
        prune.remove(module, name)
    return pruned

def sparsity(model):
    zeros = sum(int((p == 0).sum()) for p in model.parameters() if p.dim() > 1)
    total = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    return zeros / total

# `model` is the small breast-cancer classifier from lesson 08
with torch.inference_mode():
    baseline = (model(test_X).argmax(1) == test_y).float().mean().item()
print(f"dense   sparsity {sparsity(model):.2f}  accuracy {baseline:.4f}")

for amount in [0.5, 0.8, 0.9, 0.95, 0.99]:
    pruned = prune_model(model, amount)
    with torch.inference_mode():
        accuracy = (pruned(test_X).argmax(1) == test_y).float().mean().item()
    print(f"pruned  sparsity {sparsity(pruned):.2f}  accuracy {accuracy:.4f}")
```

```text
dense   sparsity 0.00  accuracy 0.9561
pruned  sparsity 0.50  accuracy 0.9561
pruned  sparsity 0.80  accuracy 0.9649
pruned  sparsity 0.90  accuracy 0.9386
pruned  sparsity 0.95  accuracy 0.9211
pruned  sparsity 0.99  accuracy 0.3684
```

**Eighty per cent of the weights can be deleted with no loss at all** — 0.9649
against the dense model's 0.9561, which is the same number plus noise, with no
retraining whatsoever.

Then it falls off a cliff: 90% costs almost two points, 95% costs three and a
half, and at 99% the model collapses to 0.3684 — *worse than predicting the
majority class every time*.

That shape — flat, then a knee, then collapse — is typical. Find your knee by
sweeping, and stop one step before it.

---

## Iterative pruning with fine-tuning

Pruning in one shot at high ratios is wasteful. Prune a little, let the
survivors adapt, repeat.

```python
import torch
import torch.nn as nn

def iterative_prune(model, train_X, train_y, steps=5, final_amount=0.9, epochs=30):
    """Prune gradually, fine-tuning between steps."""
    loss_fn = nn.CrossEntropyLoss()
    current = copy.deepcopy(model)

    for step in range(1, steps + 1):
        amount = final_amount * step / steps            # 18%, 36%, ... 90%
        current = prune_model(model, amount)            # prune from the original

        optimiser = torch.optim.AdamW(current.parameters(), lr=1e-3)
        for _ in range(epochs):                          # let it recover
            optimiser.zero_grad()
            loss_fn(current(train_X), train_y).backward()
            optimiser.step()

    return current

recovered = iterative_prune(model, train_X, train_y)
with torch.inference_mode():
    accuracy = (recovered(test_X).argmax(1) == test_y).float().mean().item()
print(f"iterative to 90%: accuracy {accuracy:.4f}  sparsity {sparsity(recovered):.2f}")
```

```text
iterative to 90%: accuracy 0.9561  sparsity 0.06
```

Read the second number. The accuracy recovered — 0.9386 back up to 0.9561,
the dense model's score — but **the sparsity is 0.06, not 0.90.**

The model is not pruned at all. Ninety per cent of its weights were set to
zero and then thirty epochs of fine-tuning moved almost all of them straight
back off zero, because `prune.remove` had already discarded the mask and
nothing was holding them there.

This is the most common pruning bug there is, and it is silent: you get a
model with a good accuracy that you believe is 90% sparse, and it is dense.

The fix is to keep the mask **during** fine-tuning — prune without calling
`prune.remove`, fine-tune while `weight = weight_orig * weight_mask` is
recomputed each forward pass, and only call `prune.remove` at the very end.
Always print the achieved sparsity afterwards, exactly as this code does; the
number you asked for and the number you got are different things.

---

## Unstructured versus structured

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

torch.manual_seed(0)
unstructured = nn.Linear(64, 64)
prune.l1_unstructured(unstructured, "weight", amount=0.5)
prune.remove(unstructured, "weight")

torch.manual_seed(0)
structured = nn.Linear(64, 64)
prune.ln_structured(structured, "weight", amount=0.5, n=2, dim=0)   # whole rows
prune.remove(structured, "weight")

print("unstructured — zeros:", int((unstructured.weight == 0).sum()),
      " fully-zero rows:", int((unstructured.weight.abs().sum(1) == 0).sum()))
print("structured   — zeros:", int((structured.weight == 0).sum()),
      " fully-zero rows:", int((structured.weight.abs().sum(1) == 0).sum()))
```

```text
unstructured — zeros: 2048  fully-zero rows: 0
structured   — zeros: 2048  fully-zero rows: 32
```

Same number of zeros; completely different consequences.

| | Unstructured | Structured |
|---|---|---|
| Removes | Individual weights | Whole neurons, channels, heads |
| Accuracy at a given sparsity | Better | Worse |
| **Actual speedup on normal hardware** | **None** | **Real** |
| Result | A sparse matrix | A genuinely smaller dense matrix |

The structured version zeroed 32 entire output rows — those neurons can be
**deleted**, giving a 64→32 layer that is smaller and faster on any hardware.
The unstructured version produced scattered zeros that a dense matrix multiply
runs through at exactly the same speed.

This is the point most tutorials skip: **unstructured pruning does not make
your model faster** unless you have sparse kernels (Ampere's 2:4 structured
sparsity, or a sparse runtime). It makes the *file* smaller if you store it
compressed. Structured pruning makes the model smaller and faster everywhere,
and costs more accuracy per weight removed.

---

## Where pruning actually pays

| Situation | Verdict |
|---|---|
| Shrinking a download or a checkpoint | Yes — compress the zeros |
| Real speedup on a CPU or a normal GPU | Only structured pruning |
| Ampere or newer GPU | 2:4 sparsity gives a real ~1.5× |
| Getting a smaller model quickly | **Distillation is usually better** — lesson 10 |
| An already-small model | No — you will just lose accuracy |

In practice the order is: quantise first (int8 is nearly free), distil second,
prune third. Pruning is the most fiddly of the three and usually returns the
least.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Forgetting `prune.remove` | Masks recomputed every forward pass; slower |
| Fine-tuning without keeping the mask | The zeros fill back in, as above |
| Expecting unstructured pruning to be faster | No speedup on dense hardware |
| Pruning without fine-tuning at high ratios | Needless accuracy loss |
| One-shot pruning to 95% | Iterative pruning would have kept more |
| Pruning a model that was already small | Pure loss |

---

## Exercises

1. Sweep sparsity from 0 to 0.99 and plot the accuracy. Where is your knee?
2. Compare unstructured and structured pruning at the same sparsity.
3. Fine-tune with the mask kept in place; confirm the sparsity survives.
4. Compress a dense and a 90%-sparse checkpoint with gzip. Compare the sizes.
