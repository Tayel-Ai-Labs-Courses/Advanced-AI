# Lesson 05 — Training an Image Classifier

**Goal:** train end to end, with the checks that catch the real bugs.

## What you will learn

- The dataset and loader for images
- The three checks before training
- A training loop with the vision-specific parts
- Reading per-class results

---

## Data first

```python
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

train_full = datasets.FashionMNIST("./data", train=True, download=True,
                                   transform=train_transform)
test_full = datasets.FashionMNIST("./data", train=False, download=True,
                                  transform=eval_transform)

train_ds = Subset(train_full, range(12_000))          # a subset, for speed
test_ds = Subset(test_full, range(3_000))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

images, labels = next(iter(train_loader))
print("batch:", tuple(images.shape), images.dtype)
print("labels:", tuple(labels.shape), labels[:8].tolist())
print("classes:", train_full.classes[:4], "...")
print(f"pixel range after normalisation: {images.min():.2f} to {images.max():.2f}")
```

```text
batch: (128, 1, 28, 28) torch.float32
labels: (128,) [3, 1, 8, 6, 4, 8, 0, 3]
classes: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress'] ...
pixel range after normalisation: -0.81 to 2.02
```

Two transform pipelines, as always: flips for training, nothing random for
evaluation.

---

## Three checks before you train

### 1. Look at the images

```python
import matplotlib.pyplot as plt

images, labels = next(iter(train_loader))
figure, axes = plt.subplots(2, 6, figsize=(12, 4))
for ax, image, label in zip(axes.flat, images, labels):
    ax.imshow(image.squeeze(), cmap="grey")
    ax.set_title(train_full.classes[label], fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

```text
(a grid of 12 clothing images, each titled with its class name)
```

Thirty seconds of looking catches: wrong labels, upside-down images, a channel
bug, an empty class, and augmentation that has destroyed the content. No
metric catches any of them.

### 2. Class balance

```python
import torch
from collections import Counter

counts = Counter(int(train_full.targets[i]) for i in train_ds.indices)
for index in sorted(counts):
    print(f"{train_full.classes[index]:<14}{counts[index]:>6}")
print("imbalance ratio:", round(max(counts.values()) / min(counts.values()), 2))
```

```text
T-shirt/top     1122
Trouser         1220
Pullover        1201
Dress           1212
Coat            1181
Sandal          1204
Shirt           1244
Sneaker         1192
Bag             1195
Ankle boot      1229
imbalance ratio: 1.11
```

Balanced here. On a real dataset it rarely is — and if the ratio is above 5,
accuracy stops being a usable metric (ML lesson 06).

### 3. Overfit one batch

```python
import torch
import torch.nn as nn

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

def build_model(n_classes=10):
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(0.25), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, n_classes),
    )

torch.manual_seed(0)
model = build_model().to(device)
images, labels = next(iter(train_loader))
images, labels = images[:16].to(device), labels[:16].to(device)

optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for step in range(1, 101):
    optimiser.zero_grad()
    loss = loss_fn(model(images), labels)
    loss.backward()
    optimiser.step()
    if step % 25 == 0:
        print(f"step {step:>3}  loss {loss.item():.6f}")
```

```text
step  25  loss 0.000458
step  50  loss 0.000056
step  75  loss 0.000047
step 100  loss 0.000046
```

Sixteen images driven to a loss of 0.00005 within 25 steps. The pipeline — loading, labels,
model, loss, optimiser — is wired correctly.

**If this does not reach near zero, stop and debug.** Training on 12,000
images will not fix a bug that 16 images expose in ten seconds.

---

## The loop

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = build_model().to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=6)
loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total

for epoch in range(1, 7):
    model.train()
    running = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimiser.step()
        running += loss.item() * images.size(0)
    scheduler.step()
    print(f"epoch {epoch}  train loss {running / len(train_ds):.4f}  "
          f"test accuracy {evaluate(test_loader):.4f}  "
          f"lr {optimiser.param_groups[0]['lr']:.5f}")
```

```text
epoch 1  train loss 0.6349  test accuracy 0.8223  lr 0.00093
epoch 2  train loss 0.3981  test accuracy 0.8783  lr 0.00075
epoch 3  train loss 0.3437  test accuracy 0.8637  lr 0.00050
epoch 4  train loss 0.2955  test accuracy 0.8727  lr 0.00025
epoch 5  train loss 0.2671  test accuracy 0.8870  lr 0.00007
epoch 6  train loss 0.2475  test accuracy 0.8897  lr 0.00000
```

Six epochs, 12,000 images, a laptop, and 89.0%.

Notice epoch 3: the training loss fell (0.398 → 0.344) and the **test accuracy
went down** (0.878 → 0.864). That is normal mid-training noise, not a bug —
and it is exactly why you never judge a run by a single epoch, and why the
cosine schedule matters: as the learning rate anneals towards zero the
accuracy stops bouncing and settles.

---

## Per-class results

The headline number hides everything that matters.

```python
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

model.eval()
predictions, truths = [], []
with torch.no_grad():
    for images, labels in test_loader:
        predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
        truths.extend(labels.tolist())

print(classification_report(truths, predictions,
                            target_names=train_full.classes, digits=3, zero_division=0))
```

```text
              precision    recall  f1-score   support

 T-shirt/top      0.839     0.831     0.835       302
     Trouser      0.981     0.981     0.981       308
    Pullover      0.843     0.832     0.838       310
       Dress      0.883     0.886     0.884       298
        Coat      0.842     0.824     0.833       324
      Sandal      0.965     0.961     0.963       285
       Shirt      0.702     0.735     0.718       298
     Sneaker      0.916     0.973     0.944       293
         Bag      0.979     0.949     0.964       297
  Ankle boot      0.967     0.937     0.952       285

    accuracy                          0.890      3000
   macro avg      0.892     0.891     0.891      3000
weighted avg      0.891     0.890     0.890      3000
```

Trousers (0.981), bags (0.964) and sandals (0.963) are nearly solved.
**Shirt is at 0.718**, twenty-six points below the mean — and the confusion
matrix says why:

```python
matrix = confusion_matrix(truths, predictions)
shirt = train_full.classes.index("Shirt")
confusions = sorted(
    ((count, train_full.classes[i]) for i, count in enumerate(matrix[shirt]) if i != shirt),
    reverse=True)[:3]
print("Shirt is predicted as:", confusions)
```

```text
Shirt is predicted as: [(35, 'T-shirt/top'), (19, 'Pullover'), (17, 'Coat')]
```

Seventy-one of the 298 shirts went to T-shirt, pullover and coat — all
upper-body garments that look nearly identical at 28×28 in greyscale.
**That is a data ceiling, not a model failure.** Higher resolution or colour
would help; a bigger network would not.

Finding that out took two cells, and it is the difference between "we need a
better model" and "we need better images".

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Never plotting the images | Label and orientation bugs survive |
| Skipping the one-batch check | Hours lost to a wiring bug |
| Augmenting the test set | Noisy, pessimistic validation |
| Reporting only the headline accuracy | The failing class stays invisible |
| Normalisation differing between train and test | Silent degradation |
| `shuffle=True` on the test loader | Predictions no longer align with labels |

---

## Exercises

1. Plot 24 training images with their labels. Do any look mislabelled?
2. Run the one-batch check; then break the labels deliberately and watch it
   fail.
3. Train for 15 epochs and plot train and test curves on one chart.
4. Print the per-class report and name the two classes most confused.
