"""Evaluation numbers, in plain Python (checked against scikit-learn in the tests).

Positive class = phishing. ``scores`` are probabilities of phishing, ``labels`` are 1 for
phishing and 0 for legitimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve = chance a random phishing URL outscores a random legitimate one
    (ties count half), computed from ranks."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        raise ValueError("both classes are needed")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(labels[order[k]] for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0


def confusion(scores: list[float], labels: list[int], threshold: float) -> Confusion:
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels, strict=True):
        flagged = s >= threshold
        if y and flagged:
            tp += 1
        elif y:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    return Confusion(tp, fp, tn, fn)


def threshold_for_fpr(scores: list[float], labels: list[int], max_fpr: float) -> float:
    """The lowest threshold whose false-positive rate on legitimate URLs stays within
    ``max_fpr``: it keeps as many phishing URLs as that allows."""
    legit = sorted((s for s, y in zip(scores, labels, strict=True) if not y), reverse=True)
    if not legit:
        raise ValueError("no legitimate URLs to calibrate on")
    allowed = int(max_fpr * len(legit))
    # Everything strictly above legit[allowed] is flagged: at most `allowed` legitimate URLs.
    cut = legit[allowed] if allowed < len(legit) else 0.0
    return min(max(math.nextafter(cut, 1.0), 1e-9), 1 - 1e-9)


def recall_at_fpr(scores: list[float], labels: list[int], max_fpr: float) -> float:
    return confusion(scores, labels, threshold_for_fpr(scores, labels, max_fpr)).recall
