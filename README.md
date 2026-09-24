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

## Working on this repository

Lessons are written in markdown. The notebooks are generated from them:

```bash
python3 tools/build_notebooks.py
```

Edit the `.md`, never the `.ipynb` — a rebuild overwrites the notebook.
One builder covers every course; `--check` reports stale notebooks without
writing, which is what CI should run.
