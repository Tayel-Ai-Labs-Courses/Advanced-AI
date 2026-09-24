# Lesson 06 — Transfer Learning

**Goal:** start from ImageNet weights, and get the preprocessing exactly right.

## What you will learn

- Replacing the head on `torchvision` models
- Freezing, and two learning rates
- Matching the preprocessing to the weights
- Measuring it against a from-scratch baseline

---

## Replace the head

Every `torchvision` classifier has a head sized for ImageNet's 1,000 classes.
It is **not** always called `fc`.

```python
import torch.nn as nn
from torchvision import models

resnet = models.resnet18(weights=None)
efficientnet = models.efficientnet_b0(weights=None)
mobilenet = models.mobilenet_v3_small(weights=None)

print("resnet18       ", resnet.fc)
print("efficientnet_b0", efficientnet.classifier[-1])
print("mobilenet_v3   ", mobilenet.classifier[-1])

resnet.fc = nn.Linear(resnet.fc.in_features, 10)
efficientnet.classifier[-1] = nn.Linear(efficientnet.classifier[-1].in_features, 10)
mobilenet.classifier[-1] = nn.Linear(mobilenet.classifier[-1].in_features, 10)

print("after replacement:", resnet.fc.out_features,
      efficientnet.classifier[-1].out_features, mobilenet.classifier[-1].out_features)
```

```text
resnet18        Linear(in_features=512, out_features=1000, bias=True)
efficientnet_b0 Linear(in_features=1280, out_features=1000, bias=True)
mobilenet_v3    Linear(in_features=1024, out_features=1000, bias=True)
after replacement: 10 10 10
```

Read `in_features` from the existing layer rather than hard-coding 512 — it
differs per model, and hard-coding it is how a model swap turns into a shape
error.

---

## The preprocessing must match the weights

```python
from torchvision import models

weights = models.ResNet18_Weights.DEFAULT
print(weights.transforms())
print("\nmeta:", {"num_params": weights.meta["num_params"]})
print("classes:", len(weights.meta["categories"]))
```

```text
ImageClassification(
    crop_size=[224]
    resize_size=[256]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    interpolation=InterpolationMode.BILINEAR
)

meta: {'num_params': 11689512}
classes: 1000
```

`weights.transforms()` gives you the **exact** pipeline the model was trained
with: resize to 256, centre-crop to 224, ImageNet normalisation, bilinear
interpolation.

Use it. Substituting your own dataset's statistics shifts every feature the
backbone learned, and the model gets quietly worse for a reason no error
message will mention.

Greyscale input needs one more step:

```python
from torchvision import transforms

adapted = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),      # 1 channel -> 3
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

---

## Freeze, or fine-tune

```python
import torch.nn as nn
from torchvision import models

def frozen_model(n_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model

def finetuned_model(n_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():       # last block only
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model

for name, builder in [("frozen", frozen_model), ("layer4 unfrozen", finetuned_model)]:
    model = builder()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{name:<18}{trainable:>12,} of {total:,} trainable "
          f"({100 * trainable / total:.2f}%)")
```

```text
frozen                   5,130 of 11,181,642 trainable (0.05%)
layer4 unfrozen      8,398,858 of 11,181,642 trainable (75.11%)
```

Two strategies, differing by a factor of 1,600 in what they train.

---

## Measure all three

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = Subset(datasets.FashionMNIST("./data", train=True, download=True,
                                        transform=transform), range(2_000))
test_ds = Subset(datasets.FashionMNIST("./data", train=False, download=True,
                                       transform=transform), range(2_000))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

def train_and_score(model, lr, epochs=3):
    model = model.to(device)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            loss_fn(model(images), labels).backward()
            optimiser.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(test_ds)

torch.manual_seed(42)
scratch = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 16 * 16, 10))

print(f"from scratch     {train_and_score(scratch, 1e-3):.4f}")
torch.manual_seed(42)
print(f"frozen backbone  {train_and_score(frozen_model(), 1e-3):.4f}")
torch.manual_seed(42)
print(f"layer4 at 1e-4   {train_and_score(finetuned_model(), 1e-4):.4f}")
```

```text
from scratch     0.8165
frozen backbone  0.7195
layer4 at 1e-4   0.8470
```

Three results, and the middle one is the important one.

**The frozen ImageNet backbone lost to a six-line CNN** — 0.720 against 0.817.
Its features were learned on colour photographs of animals, vehicles and
objects at photographic resolution; upscaled 28×28 greyscale clothing has
almost none of that structure, and a 5,130-parameter head cannot compensate.

Unfreeze one block at a gentle `1e-4` and the same backbone reaches **0.847**,
the best of the three. Given permission to adjust its last block, it adapts
the general features to this domain while keeping everything useful.

| Your data | Strategy | Head lr | Body lr |
|---|---|---|---|
| Similar to ImageNet, < 1,000 images | Freeze everything | 1e-3 | — |
| Similar, a few thousand | Unfreeze `layer4` | 1e-3 | 1e-4 |
| Different domain (medical, satellite, greyscale) | Unfreeze more | 1e-3 | 1e-5 – 1e-4 |
| Very different, and lots of data | Full fine-tune | 1e-3 | 1e-5 |

**And always run the from-scratch baseline.** It is six lines, and here it beat
one of the two transfer strategies.

---

## Two learning rates, properly

```python
import torch
import torch.nn as nn
from torchvision import models

model = finetuned_model()
optimiser = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},   # pretrained: gentle
    {"params": model.fc.parameters(), "lr": 1e-3},       # random head: faster
])
print([group["lr"] for group in optimiser.param_groups])
```

```text
[0.0001, 0.001]
```

The head is random and must move; the pretrained block already knows
something and a large step destroys it. One optimiser, two groups.

---

## Common mistakes

| Mistake | What happens |
|---|---|
| Your own normalisation with pretrained weights | Quietly worse, hard to trace |
| Hard-coding 512 as `in_features` | Shape error on any other backbone |
| Fine-tuning the body at 1e-3 | Catastrophic forgetting |
| Greyscale into a 3-channel model | Shape error — use `Grayscale(3)` |
| No from-scratch baseline | You cannot tell whether transfer helped |
| Freezing on a very different domain | Worse than training from scratch, as above |

---

## Exercises

1. Replace the head on three different backbones without hard-coding sizes.
2. Print `weights.transforms()` and build your pipeline from it.
3. Reproduce the three-way comparison on your own images.
4. Fine-tune the body at 1e-3 and at 1e-5; show catastrophic forgetting.
