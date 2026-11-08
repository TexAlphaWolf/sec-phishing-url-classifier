from __future__ import annotations

import json
import math

import pytest

from urlclf import features
from urlclf.model import (
    FORMAT,
    Model,
    ModelError,
    load_model,
    parse_model,
    sigmoid,
)


def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(1000) == 1.0
    assert 0.0 < sigmoid(-1000) < 1e-300 or sigmoid(-1000) == 0.0
    assert sigmoid(2) == pytest.approx(1 - sigmoid(-2))


def test_pure_python_scoring_equals_scikit_learn(split):
    """The saved JSON model must give the same logit as the scikit-learn object it came from."""
    from urlclf.train import VARIANTS, Vectors, _matrix, _pick, _standardiser, build_model

    tr, va = Vectors(split.train), Vectors(split.val)
    mean, scale = _standardiser(tr.numeric)
    for variant in (VARIANTS[0], VARIANTS[1], VARIANTS[-1]):  # no tokens, no numbers, both
        clf, _, _ = _pick(tr, va, mean, scale, 3, variant)
        model = build_model(clf, mean, scale, variant)
        again = parse_model(model.to_json())  # ... and still after a round trip through the text
        sample = split.test[:300]
        expected = clf.decision_function(
            _matrix(Vectors(sample), mean, scale, numeric=variant.numeric, groups=variant.groups)
        )
        for s, want in zip(sample, expected, strict=True):
            f = features.extract(s.url, variant.groups)
            assert model.logit(f) == pytest.approx(want, abs=1e-9)
            assert again.logit(f) == pytest.approx(want, abs=1e-9)


def test_round_trip(model, tmp_path):
    f = tmp_path / "m.json"
    model.save(f)
    again = load_model(f)
    url = "http://paypal.com.secure-login.example.tk/verify/account"
    assert again.probability(url) == pytest.approx(model.probability(url), abs=1e-12)
    assert again.threshold == model.threshold


def test_trained_model_separates_the_made_up_classes(model):
    bad = model.score("http://paypal.com.secure-login-x9.tk/verify/12345.php")
    good = model.score("https://www.kalomi.com/about")
    assert bad.phishing and not good.phishing
    assert bad.probability > good.probability
    assert bad.reasons and all(isinstance(k, str) for k, _ in bad.reasons)
    assert len(model.score("http://example.com", top=2).reasons) <= 2


def test_report_numbers_come_from_the_test_part(trained):
    _, report = trained
    assert report["data"]["test"]["urls"] > 0
    m = report["main_model"]
    cm = m["confusion"]
    assert cm["tp"] + cm["fn"] == report["data"]["test"]["phishing"]
    assert cm["fp"] + cm["tn"] == report["data"]["test"]["legitimate"]
    assert 0.5 < m["auc"] <= 1.0
    assert set(report["test_set_profile"]) == {"phishing", "legitimate"}


def test_compare_runs_all_baselines(split):
    from urlclf.train import train

    _, report = train(split, seed=3, compare=True)
    names = set(report["comparison"])
    assert "linear, name length alone" in names
    assert "linear, https alone" in names
    assert any(n.startswith("gradient boosting") for n in names)
    assert any("plus the scheme" in n for n in names)
    assert len(names) == 4
    assert report["variants"][0]["name"] == "numeric features only"
    assert len(report["variants"]) == 7
    chosen = max(report["variants"], key=lambda v: v["validation_auc"])
    assert report["selected"] == chosen["name"]  # chosen on validation, never on test
    assert report["main_model"] == chosen["test"]


def _good_doc(model: Model) -> dict:
    return json.loads(model.to_json())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(format="x"), "not a"),
        (lambda d: d.update(numeric=["a"]), "feature list"),
        (lambda d: d.update(hash_bits=3), "hash size"),
        (lambda d: d.update(token_groups=["zz"]), "token_groups"),
        (lambda d: d.update(token_groups=["h3", "h3"]), "repeats"),
        (lambda d: d.update(mean=[0.0]), "'mean'"),
        (lambda d: d["scale"].__setitem__(0, 0.0), "positive"),
        (lambda d: d["w_numeric"].__setitem__(0, "x"), "finite"),
        (lambda d: d["w_numeric"].__setitem__(0, True), "finite"),
        (lambda d: d.update(bias=float("nan")), "finite"),
        (lambda d: d.update(w_hash=[]), "w_hash"),
        (lambda d: d["w_hash"].update({"99999999": 1.0}), "bad bucket"),
        (lambda d: d["w_hash"].update({"-1": 1.0}), "bad bucket"),
        (lambda d: d.update(threshold=1.5), "threshold"),
        (lambda d: d.update(meta=[]), "meta"),
    ],
)
def test_bad_models_are_refused(model, mutate, message):
    doc = _good_doc(model)
    mutate(doc)
    with pytest.raises(ModelError, match=message):
        parse_model(json.dumps(doc, allow_nan=True))


def test_not_json_and_not_an_object():
    with pytest.raises(ModelError, match="not valid JSON"):
        parse_model("{")
    with pytest.raises(ModelError, match=FORMAT):
        parse_model("[1, 2]")


def test_load_errors(tmp_path):
    with pytest.raises(ModelError, match="cannot read"):
        load_model(tmp_path / "missing.json")
    f = tmp_path / "bin.json"
    f.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ModelError, match="UTF-8"):
        load_model(f)


def test_a_model_file_cannot_run_code(tmp_path, model):
    """A pickle would execute on load. This format is read with json and nothing else."""
    f = tmp_path / "evil.json"
    f.write_text('{"format": "urlclf-linear-1", "__reduce__": "os.system"}', encoding="utf-8")
    with pytest.raises(ModelError):
        load_model(f)


def test_oversized_model_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("urlclf.model.MAX_MODEL_BYTES", 10)
    f = tmp_path / "big.json"
    f.write_text("x" * 100, encoding="utf-8")
    with pytest.raises(ModelError, match="larger"):
        load_model(f)


def test_scores_are_finite_for_hostile_input(model):
    for url in ["", "\x00" * 50, "http://" + "a" * 5000, "%" * 5000, "http://[::1", "ÿ" * 100]:
        s = model.score(url)
        assert math.isfinite(s.probability) and 0.0 <= s.probability <= 1.0


def test_bundled_model_missing_gives_a_clear_error(monkeypatch):
    class Fake:
        def joinpath(self, _):
            return self

        def is_file(self):
            return False

    monkeypatch.setattr("urlclf.model.resources.files", lambda _: Fake())
    with pytest.raises(ModelError, match="no bundled model"):
        load_model()


def test_path_and_query_do_not_change_the_score(model):
    base = model.score("http://paypal.com.secure-login-x9.tk").probability
    for url in [
        "http://paypal.com.secure-login-x9.tk/",
        "http://paypal.com.secure-login-x9.tk/a/b/c.php?x=1",
        "http://www.paypal.com.secure-login-x9.tk/z",
    ]:
        assert model.score(url).probability == base


def test_verdict_rules():
    from urlclf.model import verdict

    assert verdict(0.9, 0.5, []) == "phishing"
    assert verdict(0.9, 0.5, ["x"]) == "phishing"
    assert verdict(0.1, 0.5, ["x"]) == "suspicious"
    assert verdict(0.1, 0.5, []) == "legitimate"
