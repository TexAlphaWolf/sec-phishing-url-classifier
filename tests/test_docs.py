"""The README, the evaluation page, the report and the bundled model must not drift apart."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
EVAL = (ROOT / "docs/evaluation.md").read_text(encoding="utf-8")
LIMITS = (ROOT / "docs/limitations.md").read_text(encoding="utf-8")
REPORT = json.loads((ROOT / "docs/report.json").read_text(encoding="utf-8"))


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def test_headline_numbers_come_from_the_report():
    m = REPORT["main_model"]
    for text in (README, EVAL):
        assert f"{m['auc']:.3f}" in text
        assert pct(m["recall"]) in text
        assert pct(m["false_positive_rate"], 2) in text
        assert pct(m["precision"]) in text
    lo, hi = m["recall_ci95"]
    assert f"{lo * 100:.1f} - {hi * 100:.1f}%" in README
    assert f"{lo * 100:.1f} - {hi * 100:.1f}%" in EVAL
    cm = m["confusion"]
    assert f"{cm['tp']:,} true positives" in EVAL
    assert f"{cm['fn']:,} false negatives" in EVAL
    assert f"{m['threshold']:.3f}" in EVAL


def test_split_table_matches_the_report():
    for part in ("train", "val", "test"):
        assert f"{REPORT['data'][part]['urls']:,}" in EVAL
    assert f"{REPORT['data']['rows_read']:,}" in EVAL
    assert f"{REPORT['data']['duplicates_removed']:,}" in EVAL
    assert f"{REPORT['data']['test']['urls']:,}" in README


def test_variants_table_matches_the_report():
    for v in REPORT["variants"]:
        assert f"{v['validation_auc']:.3f}" in EVAL, v["name"]
        assert f"{v['test']['auc']:.3f}" in EVAL, v["name"]
        assert pct(v["test"]["recall"]) in EVAL, v["name"]


def test_comparison_table_matches_the_report():
    for c in REPORT["comparison"].values():
        assert f"{c['auc']:.3f}" in EVAL
        assert pct(c["recall"]) in EVAL


def test_profile_numbers_match():
    p = REPORT["test_set_profile"]
    assert pct(p["phishing"]["share_https"], 2) in EVAL
    assert pct(p["phishing"]["share_www_prefix"], 2) in EVAL
    assert pct(p["phishing"]["share_with_path_or_query"], 2) in LIMITS
    assert pct(p["phishing"]["share_with_subdomain_besides_www"], 2) in EVAL
    assert pct(p["legitimate"]["share_with_subdomain_besides_www"], 2) in EVAL
    assert pct(p["legitimate"]["share_with_subdomain_besides_www"], 2) in LIMITS


def test_false_alarm_arithmetic_is_right():
    m = REPORT["main_model"]
    r, f = m["recall"], m["false_positive_rate"]
    precision = r * 0.01 / (r * 0.01 + f * 0.99)
    assert f"{precision * 100:.0f}%" in EVAL  # "33%"
    assert f"{round(precision * 100)} of every 100" in LIMITS


def test_every_local_link_exists():
    for doc in (ROOT / "README.md", ROOT / "docs/evaluation.md", ROOT / "docs/limitations.md"):
        for target in re.findall(r"\]\(([^)#]+)\)", doc.read_text(encoding="utf-8")):
            if target.startswith("http"):
                continue
            assert (doc.parent / target).exists(), f"{doc.name} links to missing {target}"


def test_readme_counts():
    from urlclf import features

    assert f"{len(features.NUMERIC)} " in README or "eighteen" in README
    assert len(features.NUMERIC) == 18
