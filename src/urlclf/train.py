"""Training and evaluation (needs scikit-learn: ``pip install "urlclf[train]"``).

The model that gets saved is always linear: it can be written as JSON and scored without any
library. ``train`` tries several feature sets, chooses among them **by AUC on the validation part
only**, and reports the test part once. The test part never takes part in any choice.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from urlclf import __version__, features, metrics
from urlclf.data import Sample, Split
from urlclf.model import Model

C_GRID = (0.03, 0.1, 0.3, 1.0)
MAX_FPR = 0.01  # the false-positive budget used to choose the threshold
ALL = features.GROUPS


@dataclass(frozen=True)
class Variant:
    name: str
    numeric: bool
    groups: tuple[str, ...]


VARIANTS = (
    Variant("numeric features only", True, ()),
    Variant("3-grams", False, ("h3", "t")),
    Variant("3- and 4-grams", False, ("h3", "h4", "t")),
    Variant("3-, 4- and 5-grams", False, ("h3", "h4", "h5", "t")),
    Variant("numeric + 3-grams", True, ("h3", "t")),
    Variant("numeric + 3- and 4-grams", True, ("h3", "h4", "t")),
    Variant("numeric + 3-, 4- and 5-grams", True, ("h3", "h4", "h5", "t")),
)
FULL = VARIANTS[-1]


def _log(progress: bool, message: str) -> None:
    if progress:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _require() -> tuple[Any, Any]:
    try:
        import numpy as np
        import scipy.sparse as sp
        import sklearn
    except ImportError as exc:  # pragma: no cover - exercised only without scikit-learn
        raise SystemExit(
            'training needs scikit-learn: pip install "urlclf[train]"  (' + str(exc) + ")"
        ) from None
    del sklearn
    return np, sp


class Vectors:
    """The features of a list of samples, ready for scikit-learn."""

    def __init__(self, samples: list[Sample]) -> None:
        scoped = [features.scoped_parts(s.url) for s in samples]
        self.numeric = [features.numeric_features(p) for p in scoped]
        self.by_group: dict[str, list[list[int]]] = {g: [] for g in ALL}
        for p in scoped:
            for g, toks in features.tokens_by_group(p).items():
                self.by_group[g].append(sorted({features.bucket(t) for t in toks}))
        self.https = [float(features.origin_parts(s.url).scheme == "https") for s in samples]
        self.labels = [s.label for s in samples]

    def __len__(self) -> int:
        return len(self.labels)


def _standardiser(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(rows)
    cols = len(features.NUMERIC)
    mean = [sum(r[j] for r in rows) / n for j in range(cols)]
    var = [sum((r[j] - mean[j]) ** 2 for r in rows) / n for j in range(cols)]
    # A column that never varies gets scale 1: it contributes nothing and cannot divide by zero.
    scale = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
    return mean, scale


def _matrix(
    v: Vectors,
    mean: list[float],
    scale: list[float],
    *,
    numeric: bool,
    groups: tuple[str, ...],
    scheme: bool = False,
) -> Any:
    np, sp = _require()
    blocks = []
    if numeric:
        arr = (np.asarray(v.numeric, dtype=float) - np.asarray(mean)) / np.asarray(scale)
        blocks.append(sp.csr_matrix(arr))
    if groups:
        rows, cols = [], []
        for i in range(len(v)):
            buckets = sorted(set().union(*(v.by_group[g][i] for g in groups)))
            rows.extend([i] * len(buckets))
            cols.extend(buckets)
        data = np.ones(len(rows))
        blocks.append(sp.csr_matrix((data, (rows, cols)), shape=(len(v), features.HASH_SIZE)))
    if scheme:  # never part of the saved model: shown only to measure what it would do
        blocks.append(sp.csr_matrix(np.asarray(v.https, dtype=float).reshape(-1, 1)))
    return sp.hstack(blocks, format="csr")


def _fit_linear(x: Any, y: list[int], c: float, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=seed)
    clf.fit(x, y)
    return clf


def _pick(
    train: Vectors,
    val: Vectors,
    mean: list[float],
    scale: list[float],
    seed: int,
    variant: Variant,
    scheme: bool = False,
    progress: bool = False,
) -> tuple[Any, float, float]:
    """Train for each C in the grid; keep the one with the best validation AUC."""
    kw: dict[str, Any] = {"numeric": variant.numeric, "groups": variant.groups, "scheme": scheme}
    xt = _matrix(train, mean, scale, **kw)
    xv = _matrix(val, mean, scale, **kw)
    best: tuple[float, float, Any] | None = None
    for c in C_GRID:
        t0 = time.time()
        clf = _fit_linear(xt, train.labels, c, seed)
        auc = metrics.roc_auc(list(clf.predict_proba(xv)[:, 1]), val.labels)
        _log(progress, f"    C={c}: validation AUC {auc:.4f} ({time.time() - t0:.0f} s)")
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    assert best is not None
    return best[2], best[1], best[0]


def build_model(clf: Any, mean: list[float], scale: list[float], variant: Variant) -> Model:
    """The saveable model with the weights of a fitted scikit-learn ``clf`` trained on the columns
    ``_matrix`` builds for ``variant``."""
    n = len(features.NUMERIC)
    coef = [float(x) for x in clf.coef_[0]]
    if variant.numeric:
        w_numeric, rest = coef[:n], coef[n:]
    else:
        w_numeric, rest = [0.0] * n, coef
    w_hash = {i: w for i, w in enumerate(rest[: features.HASH_SIZE]) if abs(w) > 1e-12}
    return Model(
        mean, scale, w_numeric, float(clf.intercept_[0]), w_hash, groups=tuple(variant.groups)
    )


def _summary(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    cm = metrics.confusion(scores, labels, threshold)
    pr = metrics.wilson(cm.tp, cm.tp + cm.fp)
    rc = metrics.wilson(cm.tp, cm.tp + cm.fn)
    fp = metrics.wilson(cm.fp, cm.fp + cm.tn)
    return {
        "threshold": round(threshold, 6),
        "auc": round(metrics.roc_auc(scores, labels), 5),
        "precision": round(cm.precision, 5),
        "precision_ci95": [round(x, 5) for x in pr],
        "recall": round(cm.recall, 5),
        "recall_ci95": [round(x, 5) for x in rc],
        "false_positive_rate": round(cm.fpr, 5),
        "false_positive_rate_ci95": [round(x, 5) for x in fp],
        "f1": round(cm.f1, 5),
        "recall_at_fpr_0.1pct": round(metrics.recall_at_fpr(scores, labels, 0.001), 5),
        "confusion": {"tp": cm.tp, "fp": cm.fp, "tn": cm.tn, "fn": cm.fn},
    }


def _evaluate(
    val_scores: list[float], val_labels: list[int], te_scores: list[float], te_labels: list[int]
) -> dict[str, Any]:
    t = metrics.threshold_for_fpr(val_scores, val_labels, MAX_FPR)
    return _summary(te_scores, te_labels, t)


def profile(samples: list[Sample]) -> dict[str, Any]:
    """How the two classes differ in ways a model could use as a shortcut. If one class is almost
    always ``https`` or never has a path, a model can look good by learning that."""
    out: dict[str, Any] = {}
    for name, label in (("phishing", 1), ("legitimate", 0)):
        rows = [s for s in samples if s.label == label]
        parsed = [features.parse_url(s.url) for s in rows]
        origin = [features.origin_parts(s.url) for s in rows]
        n = len(rows) or 1
        lengths = sorted(len(p.registered) for p in parsed)
        out[name] = {
            "urls": len(rows),
            "share_https": round(sum(p.scheme == "https" for p in parsed) / n, 4),
            "share_www_prefix": round(sum(p.host.startswith("www.") for p in parsed) / n, 4),
            "share_with_path_or_query": round(
                sum(bool(p.path.strip("/") or p.query) for p in parsed) / n, 4
            ),
            "share_ip_host": round(sum(p.host_is_ip for p in parsed) / n, 4),
            "share_with_subdomain_besides_www": round(
                sum(bool(o.subdomain) for o in origin) / n, 4
            ),
            "median_domain_length": lengths[len(lengths) // 2] if lengths else 0,
        }
    return out


def train(
    split: Split, seed: int = 1, compare: bool = True, progress: bool = False
) -> tuple[Model, dict[str, Any]]:
    """Fit on ``split.train``, choose the feature set and C on ``split.val``, report on
    ``split.test``. With ``compare=False`` only the full feature set is trained (quick runs)."""
    started = time.time()
    _log(
        progress,
        f"extracting features of {len(split.train) + len(split.val) + len(split.test)} URLs",
    )
    tr, va, te = Vectors(split.train), Vectors(split.val), Vectors(split.test)
    mean, scale = _standardiser(tr.numeric)

    fitted = []
    todo = VARIANTS if compare else (FULL,)
    for k, variant in enumerate(todo, 1):
        _log(progress, f"[{k}/{len(todo)}] {variant.name}")
        clf, c, val_auc = _pick(tr, va, mean, scale, seed, variant, progress=progress)
        kw: dict[str, Any] = {"numeric": variant.numeric, "groups": variant.groups}
        va_p = list(clf.predict_proba(_matrix(va, mean, scale, **kw))[:, 1])
        te_p = list(clf.predict_proba(_matrix(te, mean, scale, **kw))[:, 1])
        fitted.append((variant, clf, c, val_auc, va_p, te_p))

    best = max(fitted, key=lambda f: f[3])  # by validation AUC; the test part plays no part
    variant, clf, c, val_auc, va_p, te_p = best
    model = build_model(clf, mean, scale, variant)
    model.threshold = metrics.threshold_for_fpr(va_p, va.labels, MAX_FPR)
    main = _evaluate(va_p, va.labels, te_p, te.labels)

    report: dict[str, Any] = {
        "urlclf_version": __version__,
        "data": split.stats,
        "selected": variant.name,
        "chosen_C": c,
        "main_model": main,
        "test_set_profile": profile(split.test),
    }
    if compare:
        report["variants"] = [
            {
                "name": v.name,
                "C": cc,
                "validation_auc": round(vauc, 5),
                "test": _evaluate(vp, va.labels, tp, te.labels),
            }
            for v, _, cc, vauc, vp, tp in fitted
        ]
        _log(progress, "comparison models")
        report["comparison"] = _compare(tr, va, te, mean, scale, seed)
    _log(progress, f"selected: {variant.name} (validation AUC {val_auc:.4f})")
    report["seconds"] = round(time.time() - started, 1)
    model.meta = {
        "urlclf_version": __version__,
        "feature_set": variant.name,
        "trained_on_urls": len(split.train),
        "validation_urls": len(split.val),
        "chosen_C": c,
        "scope": "registrable domain; path, query, scheme, www. and other subdomains are not scored",
        "threshold_rule": f"lowest threshold with at most {MAX_FPR:.0%} of validation legitimate URLs flagged",
        "test": main,
    }
    return model, report


def _compare(
    tr: Vectors, va: Vectors, te: Vectors, mean: list[float], scale: list[float], seed: int
) -> dict[str, Any]:
    """The same test split, other models. None of them is saved or used."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    out: dict[str, Any] = {}
    # What the scheme would have done, had it been a feature.
    clf, _, _ = _pick(tr, va, mean, scale, seed, FULL, scheme=True)
    kw: dict[str, Any] = {"numeric": True, "groups": FULL.groups, "scheme": True}
    out["linear, selected features plus the scheme (https): not used"] = _evaluate(
        list(clf.predict_proba(_matrix(va, mean, scale, **kw))[:, 1]),
        va.labels,
        list(clf.predict_proba(_matrix(te, mean, scale, **kw))[:, 1]),
        te.labels,
    )
    j = features.NUMERIC.index("len_name")
    columns: dict[str, Callable[[Vectors], list[list[float]]]] = {
        "name length alone": lambda v: [[r[j]] for r in v.numeric],
        "https alone": lambda v: [[x] for x in v.https],
    }
    for label, column in columns.items():
        lr = LogisticRegression(max_iter=1000).fit(column(tr), tr.labels)
        out[f"linear, {label}"] = _evaluate(
            list(lr.predict_proba(column(va))[:, 1]),
            va.labels,
            list(lr.predict_proba(column(te))[:, 1]),
            te.labels,
        )
    gb = HistGradientBoostingClassifier(random_state=seed, max_iter=200).fit(tr.numeric, tr.labels)
    out["gradient boosting, numeric features only"] = _evaluate(
        list(gb.predict_proba(va.numeric)[:, 1]),
        va.labels,
        list(gb.predict_proba(te.numeric)[:, 1]),
        te.labels,
    )
    return out
