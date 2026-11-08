# Contributing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest --cov
python tools/evaluate.py          # JSON scoring against scikit-learn, robustness, speed
```

- **Scoring uses the standard library only.** scikit-learn and numpy are for training (`[train]`). Keep it that way: a model file is JSON, never a pickle.
- **A new feature** goes in `features.NUMERIC` (and `numeric_features`, in the same order) with a test in `tests/test_features.py`. Changing the feature list makes old model files unloadable on purpose: retrain, and update `docs/report.json` and the numbers in the README (`tests/test_docs.py` checks they agree).
- **Do not add a feature because it raises the score** until you have checked what it measures. Two features that did (the path, the scheme) were removed because the public dataset separates its classes on them for reasons that do not hold on the web. `docs/evaluation.md` tells the story.
- **Datasets are never committed.** `docs/evaluation.md` says which one to download and the command that reproduces the report.
