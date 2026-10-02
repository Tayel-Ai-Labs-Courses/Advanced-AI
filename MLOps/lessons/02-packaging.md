# Lesson 02 — Packaging and Environments

**Goal:** make "it works on my machine" into a claim anyone can check.

## What you will learn

- Why unpinned dependencies make reproduction impossible, measured
- Lockfiles, containers, and what each guarantees
- What goes in the image and what does not
- The four things to pin that are not Python packages

---

## Unpinned dependencies do not survive

```python
import numpy as np

def prob_all_same(n_deps, p_change_per_month, months):
    return (1 - p_change_per_month) ** (n_deps * months)
print(f"{'dependencies':>14}{'1 month':>11}{'3 months':>11}{'12 months':>12}")
for n in (5, 20, 60, 150):
    row = "".join(f"{prob_all_same(n, 0.04, m):>11.3f}" for m in (1, 3, 12))
    print(f"{n:>14}{row}")
print("\nP(every unpinned dependency is unchanged), at 4% churn per month each")
print("a 60-dependency project is byte-identical after a year with probability", 
      f"{prob_all_same(60, 0.04, 12):.4f}")
```

```text
  dependencies    1 month   3 months   12 months
             5      0.815      0.542      0.086
            20      0.442      0.086      0.000
            60      0.086      0.001      0.000
           150      0.002      0.000      0.000

P(every unpinned dependency is unchanged), at 4% churn per month each
a 60-dependency project is byte-identical after a year with probability 0.0000
```

**A 60-dependency project has a 0.086 chance of resolving identically after one
month**, and effectively zero after a year.

A typical ML project has 60-150 transitive dependencies. So `pip install -r
requirements.txt` with unpinned versions does not install the same software
twice, which means:

- Your colleague cannot reproduce your number
- Your CI cannot reproduce your local run
- **Your production rebuild cannot reproduce your tested artefact**

That last one is the expensive version. A rebuild of an "unchanged" service can
ship a different scikit-learn, and
[Data-Science lesson 07](../../Data-Science/lessons/07-reproducibility.md)
measured what that does: defaults change between versions, and the same code and
data give a different number.

---

## What each level guarantees

| Level | Pins | Reproducible? | Effort |
|---|---|---|---|
| `requirements.txt`, unpinned | nothing | **No** — the table above | none |
| `requirements.txt`, `==` on direct deps | your imports | Partly — transitives still float | low |
| **A lockfile** (`pip-compile`, `poetry.lock`, `uv.lock`) | **every transitive dependency, by hash** | Yes, for Python | **low** |
| A container image | Python + system libraries + OS | Yes, for everything userspace | medium |
| Image + pinned base **by digest** | the base too | **Yes** | medium |

**A lockfile is the single highest-value change here**, and it is one command:

```bash
# no-run
pip install pip-tools
pip-compile requirements.in -o requirements.txt --generate-hashes
pip-sync requirements.txt
```

`--generate-hashes` is what makes it a guarantee rather than a hope: install
fails if a package's contents changed, even under the same version number.

---

## The four non-Python pins

A lockfile covers Python. These four are outside it and cause most "works
locally, fails in CI" incidents:

```text
1. The base image, BY DIGEST    python:3.12-slim@sha256:...  — not ":latest", not ":3.12"
2. The CUDA / driver version    a torch built for CUDA 12 will not run on a CUDA 11 driver
3. System libraries             libgomp, libglib — pulled by apt, versioned separately
4. The model artefact itself    a file hash, recorded with the run (Data-Science 07)
```

The first is the one people get wrong most: `FROM python:3.12-slim` is **not
pinned**. That tag moves. Pin by digest, and update it deliberately.

---

## What goes in the image

| In the image | Not in the image |
|---|---|
| Code, lockfile, system deps | **Data** — mount or download it |
| The entrypoint | **Secrets** — inject at runtime |
| Small static assets | **Model weights**, usually — pull from the registry by version |
| | Anything you would have to rebuild the image to change |

**Model weights are the interesting case.** Baking them in gives you one
immutable artefact — easy to roll back, slow to update. Pulling them at startup
from a registry (Data-Science 11) lets you change the model without a rebuild —
faster, and now the image alone does not determine behaviour.

**Bake them in if deploys are rare and rollback matters most.** Pull them if you
retrain often — and then log the model version on every response
([AI-System-Design 03](../../AI-System-Design/lessons/03-interfaces.md)) so the
pair is always identifiable.

---

## A minimal Dockerfile

```dockerfile
# no-run
FROM python:3.12-slim@sha256:0a1b2c3d...        # pinned by digest

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

COPY src/ ./src/
ENV MODEL_VERSION=churn-v1

HEALTHCHECK --interval=30s CMD python -c "import urllib.request; \
  urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

Five details that matter:

- **`--require-hashes`** makes the lockfile's guarantee real.
- **`COPY requirements.txt` before `COPY src/`** so a code change does not
  reinstall every dependency. Layer order is the difference between a 10-second
  and a 4-minute build.
- **`--no-install-recommends`** and cleaning the apt lists keeps the image small.
- **A healthcheck**, which the orchestrator uses to decide the container is
  alive — and which should report the model version
  ([Data-Science 08](../../Data-Science/lessons/08-shipping-the-model.md)).
- **No secrets**, no data, no weights baked by accident.

---

## Verifying it

A pinned environment you never test is a hope. Two checks, both cheap:

```text
1. Build twice from a clean cache. Are the image digests equal?
2. Run the eval set inside the container and compare with your local number.
   A difference means the environment changed something.
```

Check 2 is the one that catches a silent library upgrade, and it is the natural
thing for CI to run — lesson 04.

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Unpinned `requirements.txt` | 0.086 chance of an identical install after a month |
| Pinning direct dependencies only | Transitives still float |
| `FROM python:3.12-slim` | The tag moves; pin by digest |
| No CUDA/driver pin | Works locally, crashes on the GPU |
| `COPY . .` before installing | Every code change reinstalls everything |
| Secrets or data in the image | They leak, and the image is cached everywhere |
| Never testing the built image | The pin was a hope |

---

## Exercises

1. Count your project's transitive dependencies. Look up its row in the table.
2. Generate a lockfile with hashes and install from it in a clean environment.
3. Find every unpinned tag in your Dockerfiles. Pin one by digest.
4. Build your image twice from a clean cache. Are the digests equal?
5. Run your eval set inside the container and compare with your local number.

---

**Next:** [Lesson 03 — The Pipeline as Code](03-pipeline-as-code.md)
