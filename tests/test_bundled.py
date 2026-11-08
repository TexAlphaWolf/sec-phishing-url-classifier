"""The model shipped inside the package, and the report it came from."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urlclf import features
from urlclf.model import load_model, verdict

ROOT = Path(__file__).resolve().parent.parent
REPORT = json.loads((ROOT / "docs/report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bundled():
    return load_model()


def test_it_loads_and_is_the_one_in_the_report(bundled):
    assert bundled.meta["feature_set"] == REPORT["selected"]
    assert bundled.meta["test"] == REPORT["main_model"]
    assert bundled.threshold == pytest.approx(REPORT["main_model"]["threshold"], abs=1e-6)
    assert bundled.groups == ("h3", "t")
    assert bundled.meta["trained_on_urls"] == REPORT["data"]["train"]["urls"]


def test_obvious_cases(bundled):
    def v(url):
        s = bundled.score(url)
        return verdict(s.probability, bundled.threshold, s.warnings)

    assert v("https://www.wikipedia.org/wiki/Phishing") == "legitimate"
    assert v("https://example.com/docs/index.html") == "legitimate"
    assert v("http://203.0.113.7/login.php") == "phishing"
    assert v("http://user:pass@bank-update-24.com:8081/confirm") == "phishing"
    assert v("http://paypal.com.secure-login-x9.example.net/verify") == "suspicious"


def test_scheme_path_www_and_subdomains_do_not_change_the_score(bundled):
    base = bundled.score("example.org").probability
    for url in [
        "https://example.org",
        "http://www.example.org/a/b?c=d",
        "https://mail.example.org/inbox",
        "https://a.b.c.example.org:443/",
    ]:
        assert bundled.score(url).probability == base, url


def test_docs_report_is_the_whole_story():
    names = [v["name"] for v in REPORT["variants"]]
    assert len(names) == 7
    best = max(REPORT["variants"], key=lambda v: v["validation_auc"])
    assert best["name"] == REPORT["selected"]
    assert best["test"] == REPORT["main_model"]
    cm = REPORT["main_model"]["confusion"]
    data = REPORT["data"]["test"]
    assert cm["tp"] + cm["fn"] == data["phishing"]
    assert cm["fp"] + cm["tn"] == data["legitimate"]
    assert data["phishing"] + data["legitimate"] == data["urls"]


def test_feature_list_matches_the_report_version():
    assert REPORT["urlclf_version"] == "1.0.0"
    assert len(features.NUMERIC) == 18
