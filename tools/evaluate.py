"""Measurements that need no dataset: equivalence with scikit-learn, robustness, speed, and the
pipeline run on made-up URLs. Prints JSON, then OK or FAILED.

The made-up URLs prove the pipeline runs end to end. They say NOTHING about real phishing: the
generator invents the patterns the features look for. Real figures: docs/evaluation.md.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urlclf import data, features, synthetic
from urlclf.model import parse_model
from urlclf.train import VARIANTS, Vectors, _matrix, _pick, _standardiser, build_model, train


def equivalence(split: data.Split) -> dict:
    """Pure-Python JSON scoring against scikit-learn's own decision function, for every feature
    set that can be selected."""
    tr, va = Vectors(split.train), Vectors(split.val)
    mean, scale = _standardiser(tr.numeric)
    te = Vectors(split.test)
    worst = 0.0
    for variant in VARIANTS:
        clf, _, _ = _pick(tr, va, mean, scale, 1, variant)
        model = parse_model(build_model(clf, mean, scale, variant).to_json())
        expected = clf.decision_function(
            _matrix(te, mean, scale, numeric=variant.numeric, groups=variant.groups)
        )
        for s, want in zip(split.test, expected, strict=True):
            f = features.extract(s.url, variant.groups)
            worst = max(worst, abs(model.logit(f) - want))
    return {
        "feature_sets": len(VARIANTS),
        "urls_compared": len(te.labels),
        "max_abs_logit_difference": worst,
    }


def damaged(rng: random.Random, base: list[str], n: int) -> list[str]:
    pieces = [
        "%",
        "%zz",
        "@",
        "[",
        "]",
        "::",
        "\x00",
        "‮",
        "xn--",
        "//",
        "..",
        "\ud800",
        "é",
        " ",
        "\t",
    ]
    out = []
    for _ in range(n):
        s = rng.choice(base)
        for _ in range(rng.randint(1, 4)):
            i = rng.randint(0, len(s))
            kind = rng.randint(0, 2)
            if kind == 0:
                s = s[:i] + rng.choice(pieces) + s[i:]
            elif kind == 1:
                s = s[:i] + s[i + rng.randint(1, 8) :]
            else:
                s = s + "".join(rng.choice("abc./:-_") for _ in range(rng.randint(1, 300)))
        out.append(s)
    return out


def robustness(model, samples) -> dict:
    rng = random.Random(11)
    urls = damaged(rng, [s.url for s in samples[:500]], 20000)
    urls += ["http://" + "a." * 3000, "x" * 100000, "%" * 50000, "http://[" + "1:" * 2000]
    bad = 0
    worst = 0.0
    for u in urls:
        t = time.perf_counter()
        try:
            p = model.score(u).probability
            assert 0.0 <= p <= 1.0
        except Exception:
            bad += 1
        worst = max(worst, time.perf_counter() - t)
    return {"urls": len(urls), "raised": bad, "slowest_ms": round(worst * 1000, 2)}


def speed(model, samples) -> dict:
    urls = [s.url for s in samples[:2000]]
    per = []
    for u in urls:
        t = time.perf_counter()
        model.score(u)
        per.append((time.perf_counter() - t) * 1000)
    return {
        "median_ms": round(statistics.median(per), 4),
        "p99_ms": round(sorted(per)[int(len(per) * 0.99)], 4),
    }


def main() -> int:
    samples = synthetic.generate(8000, seed=1)
    split = data.split(samples, seed=1)
    model, report = train(split, seed=1, compare=False)
    first = data.split(samples, seed=1)
    result = {
        "equivalence_with_scikit_learn": equivalence(split),
        "split_is_deterministic": first.test == split.test,
        "robustness": robustness(model, samples),
        "speed": speed(model, samples),
        "made_up_data_pipeline": {
            "urls": len(samples),
            "test_auc": report["main_model"]["auc"],
            "note": "synthetic: shows the pipeline runs, proves nothing about real phishing",
        },
    }
    print(json.dumps(result, indent=2))
    ok = (
        result["equivalence_with_scikit_learn"]["max_abs_logit_difference"] < 1e-8
        and result["split_is_deterministic"]
        and result["robustness"]["raised"] == 0
        and result["speed"]["p99_ms"] < 50
    )
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
