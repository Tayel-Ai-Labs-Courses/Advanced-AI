# Advanced AI — Tayel AI Labs

Course material for the Advanced AI track.

## Courses

| Folder | Course |
|---|---|
| [`Python/`](Python/) | Python for AI — basic track, advanced track, and two projects |
| [`Machine-Learning/`](Machine-Learning/) | Machine learning from first principles, and a third project |
| [`Deep-Learning/`](Deep-Learning/) | Neural networks in PyTorch, vision and text, and a fourth project |
| [`Optimization/`](Optimization/) | Making models faster, smaller and cheaper, and a fifth project |
| [`NLP/`](NLP/) | Text from TF-IDF to transformers, including Arabic, and a sixth project |
| [`Computer-Vision/`](Computer-Vision/) | Images: classical CV, CNNs, detection, segmentation, and a seventh project |
| [`Data-Engineering/`](Data-Engineering/) | Pipelines, storage, quality, orchestration, and an eighth project |
| [`Data-Analysis/`](Data-Analysis/) | Basic and advanced analysis tracks, and a ninth project |
| [`Data-Science/`](Data-Science/) | Problem to decision: leakage, thresholds, shipping, drift, and a tenth project |

## Working on this repository

Lessons are written in markdown. The notebooks are generated from them:

```bash
python3 tools/build_notebooks.py
```

Edit the `.md`, never the `.ipynb` — a rebuild overwrites the notebook.
One builder covers every course.

Two checks run on every push, via [`.github/workflows/check.yml`](.github/workflows/check.yml):

```bash
python3 tools/build_notebooks.py --check   # notebooks match their markdown
python3 tools/check_links.py               # no broken relative links
```

Both are standard library only. If the first one fails, you edited a `.md`
without rebuilding — run the builder and push again.
