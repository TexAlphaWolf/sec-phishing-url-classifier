from __future__ import annotations

import random

import pytest

from urlclf import metrics


def test_auc_known_cases():
    assert metrics.roc_auc([0.1, 0.9], [0, 1]) == 1.0
    assert metrics.roc_auc([0.9, 0.1], [0, 1]) == 0.0
    assert metrics.roc_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    with pytest.raises(ValueError):
        metrics.roc_auc([0.1, 0.2], [1, 1])


def test_auc_matches_scikit_learn_with_ties():
    from sklearn.metrics import roc_auc_score

    rng = random.Random(1)
    for _ in range(20):
        n = rng.randint(20, 300)
        labels = [rng.randint(0, 1) for _ in range(n)]
        if len(set(labels)) < 2:
            continue
        scores = [round(rng.random() + 0.3 * y, 1) for y in labels]  # rounding makes ties
        assert metrics.roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


def test_wilson():
    lo, hi = metrics.wilson(0, 100)
    assert lo == 0.0
    assert hi == pytest.approx(0.0370, abs=1e-3)
    lo, hi = metrics.wilson(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)
    assert metrics.wilson(0, 0) == (0.0, 1.0)


def test_confusion():
    cm = metrics.confusion([0.9, 0.8, 0.2, 0.1, 0.6], [1, 0, 1, 0, 1], 0.5)
    assert (cm.tp, cm.fp, cm.tn, cm.fn) == (2, 1, 1, 1)
    assert cm.precision == pytest.approx(2 / 3)
    assert cm.recall == pytest.approx(2 / 3)
    assert cm.fpr == pytest.approx(0.5)
    assert cm.f1 == pytest.approx(2 / 3)
    assert cm.accuracy == pytest.approx(3 / 5)
    empty = metrics.Confusion(0, 0, 0, 0)
    assert empty.precision == empty.recall == empty.fpr == empty.f1 == empty.accuracy == 0.0


def test_threshold_for_fpr_respects_the_budget():
    rng = random.Random(2)
    for max_fpr in (0.0, 0.001, 0.01, 0.1):
        labels = [rng.randint(0, 1) for _ in range(2000)]
        scores = [min(0.999, max(0.001, rng.gauss(0.65 if y else 0.35, 0.2))) for y in labels]
        t = metrics.threshold_for_fpr(scores, labels, max_fpr)
        assert 0 < t < 1
        assert metrics.confusion(scores, labels, t).fpr <= max_fpr + 1e-12
    with pytest.raises(ValueError):
        metrics.threshold_for_fpr([0.5], [1], 0.01)


def test_recall_at_fpr_is_monotone():
    rng = random.Random(3)
    labels = [rng.randint(0, 1) for _ in range(3000)]
    scores = [rng.gauss(0.7 if y else 0.3, 0.2) for y in labels]
    a = metrics.recall_at_fpr(scores, labels, 0.001)
    b = metrics.recall_at_fpr(scores, labels, 0.01)
    c = metrics.recall_at_fpr(scores, labels, 0.1)
    assert a <= b <= c
