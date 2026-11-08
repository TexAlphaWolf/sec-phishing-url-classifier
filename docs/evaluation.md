# Evaluation

Every figure in the README comes from here, and `docs/report.json` is the unedited output of the command in section 7. `tests/test_docs.py` and `tests/test_bundled.py` check that the README, this page, the report and the bundled model agree.

## 1. The data and the split

**Dataset:** PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository, #967; CC BY 4.0), 235,795 URLs: 134,850 labelled legitimate and 100,945 phishing. Only the `URL` and `label` columns are used; the dataset's own 54 derived features are not (some are computed from page content, some from other features). It is not in this repository: download it yourself.

**Cleaning.** The model sees only the registrable domain (plus user info and port when present), so URLs with the same origin are one example, however many pages they name. 14,364 rows were dropped as repeats of an origin with the same label, and 54 rows because their origin carried both labels. 221,377 examples remain.

**Split by site.** All URLs of one registrable domain (Public Suffix List, snapshot 2026-11-06) go to the same part, chosen by a fixed hash of the domain: 70% train, 15% validation, 15% test. Splitting URL by URL would test the model on sites it trained on.

| Part | URLs | Phishing | Legitimate | Sites |
|---|---|---|---|---|
| train | 155,855 | 61,587 | 94,268 | 138,046 |
| validation | 32,669 | 12,272 | 20,397 | 29,695 |
| test | 32,853 | 12,680 | 20,173 | 29,550 |

**What is chosen where.** Training tries seven feature sets and four regularisation strengths (C from 0.03 to 1). The **validation** part picks the feature set and C (by AUC) and the threshold (the lowest one that flags at most 1% of the legitimate validation URLs). The **test** part takes part in none of these choices; the figures below are its single reading.

## 2. Result

On the test part, with 95% Wilson intervals. Selected: numeric features + character 3-grams of the domain name, C = 0.1, threshold 0.737.

| Measure | Value |
|---|---|
| ROC AUC | **0.883** |
| Recall (phishing found) | **52.2%** (51.3 - 53.1%) |
| Precision | 96.9% (96.4 - 97.2%) |
| False-positive rate | 1.07% (0.93 - 1.22%) |
| F1 | 0.678 |
| Recall at a 0.1% false-positive budget | 37.3% |
| Confusion | 6,617 true positives, 215 false positives, 19,958 true negatives, 6,063 false negatives |

The seven feature sets (the validation AUC is what chose between them):

| Feature set | Validation AUC | Test AUC | Test recall | Test false-positive rate |
|---|---|---|---|---|
| numeric features only | 0.729 | 0.750 | 31.2% | 0.99% |
| 3-grams | 0.873 | 0.880 | 51.5% | 1.23% |
| 3- and 4-grams | 0.869 | 0.877 | 46.6% | 1.09% |
| 3-, 4- and 5-grams | 0.869 | 0.873 | 49.3% | 1.12% |
| **numeric + 3-grams** (selected) | **0.877** | **0.883** | **52.2%** | 1.07% |
| numeric + 3- and 4-grams | 0.875 | 0.882 | 47.8% | 1.03% |
| numeric + 3-, 4- and 5-grams | 0.874 | 0.881 | 51.5% | 1.08% |

Longer n-grams do not help; the numeric features add little on top of them (test AUC 0.880 to 0.883). Other models on the same split and threshold rule, none of them saved or used:

| Model | AUC | Recall | False-positive rate |
|---|---|---|---|
| gradient boosting, numeric features only | 0.796 | 36.8% | 0.87% |
| linear, name length alone | 0.503 | 7.1% | 0.68% |
| linear, `https` alone | 0.777 | 55.4% | 0.00% |
| linear, selected features **plus the scheme** | 0.975 | 86.0% | 1.10% |

What this says:

- The domain name's character pieces carry nearly all the signal; the hand-made numeric features alone reach only 0.750. Gradient boosting on them is no better, so the linear model, which can be stored as JSON and scored in pure Python, gives nothing up.
- Name length alone is a coin toss (AUC 0.503): the model is not saying "long names are phishing".
- The last two rows are why the scheme is not a feature. See section 3.
- 52% recall at a 1% false-positive rate is modest. It is what the name of a registrable domain alone supports on this data. Section 6 and `limitations.md` say what that means in use.

## 3. How the numbers became honest

The first complete run was far better, and wrong. Each step below was caused by looking at the data, and each lowered the score.

| Step | What the model could use | AUC | Recall | False positives |
|---|---|---|---|---|
| 1 | the whole URL, path included | 0.999 | 99.7% | 0.92% |
| 2 | origin only (scheme, host, port) | 0.984 | 93.6% | 0.93% |
| 3 | + the `https` column removed | 0.978 | 88.4% | 1.06% |
| 4 | + the scheme also removed from lengths and ratios | 0.910 | 65.4% | 1.07% |
| 5 | **registrable domain only (final)** | **0.883** | **52.2%** | 1.07% |

1. **The path.** Of the 20,178 legitimate URLs in that run's test part, **none** had a path, against 5,319 of the phishing ones. The model had learnt "has a path = phishing". The fix is a decision: the path and query are not read.
2. **The scheme.** Every legitimate URL in the data is `https` and starts with `www.`; 44.65% and 47.86% of the phishing ones do. The one-line rule "not https" finds 55% of the phishing with no false positive. In step 3 the `https` column was removed and the result barely moved, because the scheme was still inside the lengths and character ratios (they were computed on a string starting with `https://`). Computing them without the scheme (step 4) took **23 points of recall** away.
3. **The subdomains.** After removing `www.`, 3.14% of the legitimate test URLs have a subdomain, against 30.24% of the phishing ones. That is how the list of legitimate sites was built (home pages of popular sites), not how legitimate sites look: real ones use `mail.`, `accounts.`, `cdn.`. A model that learns "has a subdomain = phishing" would flag them. Step 5 scores the registrable domain and reports subdomain tricks as plain-rule warnings instead (`paypal.com.example.net`, a bare `@`, an IP, an odd port). That cost 13 points of recall (AUC 0.910 to 0.883), which is consistent with that signal having been a dataset effect; this evaluation cannot prove it, because it has no legitimate URLs with subdomains to test on.

The profile the report prints for the test part: legitimate URLs are 100% `https`, 100% `www.`, 0% with a path, 3.14% with another subdomain; phishing ones 44.65%, 47.86%, 16.81% and 30.24%; the median domain name is 15 characters in both classes.

## 4. Scoring is exactly the trained model

`urlclf train` fits scikit-learn's logistic regression and writes its weights as JSON; `urlclf score` recomputes the same sum in plain Python. `tools/evaluate.py` compares the two on every URL of a made-up test part, for **all seven feature sets**: the largest difference in the log-odds is of the order of 1e-14 (CI requires 1e-8). The same check is a unit test (`tests/test_model.py`).

## 5. Robustness and speed

20,004 damaged URLs (random insertions of `%`, `@`, `[`, NUL, right-to-left marks, lone surrogates; deletions; 100,000-character strings; 3,000-label hosts) go through scoring: **0 raised**, the slowest took about 20 ms. Scoring one URL takes about 0.3 ms (median) and 0.5 ms (99th percentile), on a cloud VM with Python 3.13. The real training run (seven feature sets, comparison models, 155,855 training URLs) took 646 seconds in a `python:3.13-slim` container.

## 6. What this does not show

- Nothing here measures performance on traffic. See [limitations](limitations.md): legitimate means popular sites, phishing means this dataset's collection, there are no dates, and the share of phishing in real traffic is far lower than the 39% of the test part. At 1 phishing URL in 100, only about one flagged URL in three would be phishing (0.522 x 0.01 / (0.522 x 0.01 + 0.0107 x 0.99) = 33%).
- `tools/evaluate.py` and CI train on **made-up URLs** from `urlclf synth`. They prove the pipeline runs and that scoring matches scikit-learn. The generator writes the very patterns the features look for, so its accuracy means nothing and is not reported.

## 7. Reproduce

```bash
# 1. download the dataset (14.7 MB zip) from https://archive.ics.uci.edu/dataset/967 and unzip it into datasets/
# 2. train and report (about 11 minutes, with progress on standard error)
pip install ".[train]"
urlclf train --data datasets/PhiUSIIL_Phishing_URL_Dataset.csv --preset phiusiil \
  --out model.json --report report.json
```

The bundled model, `src/urlclf/data/model.json`, is the `model.json` this command produced for the figures above, and `docs/report.json` its `report.json`.
