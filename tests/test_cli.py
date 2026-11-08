from __future__ import annotations

import io
import json

import pytest

from urlclf import cli, synthetic

BAD = "http://paypal.com.secure-login-x9.tk/verify/12345.php"
GOOD = "https://www.kalomi.com/about"


@pytest.fixture(scope="module")
def model_file(tmp_path_factory, model):
    f = tmp_path_factory.mktemp("m") / "model.json"
    model.save(f)
    return f


def test_score_text(model_file, capsys):
    assert cli.main(["score", "--model", str(model_file), BAD, GOOD]) == 0
    out = [ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith(" ")]
    assert out[0].startswith("phishing") and BAD in out[0]
    assert out[1].startswith("legitimate")


def test_score_explain_and_json(model_file, capsys):
    cli.main(["score", "--model", str(model_file), "--explain", BAD])
    assert "+" in capsys.readouterr().out
    cli.main(["score", "--model", str(model_file), "--format", "json", "--top", "3", BAD])
    doc = json.loads(capsys.readouterr().out)
    assert doc["results"][0]["verdict"] == "phishing"
    assert len(doc["results"][0]["reasons"]) <= 3


def test_threshold_override_and_fail_flag(model_file, capsys):
    assert cli.main(["score", "--model", str(model_file), "--fail-on-phishing", BAD]) == 1
    assert cli.main(["score", "--model", str(model_file), "--fail-on-phishing", GOOD]) == 0
    assert (
        cli.main(
            [
                "score",
                "--model",
                str(model_file),
                "--threshold",
                "0.00001",
                "--fail-on-phishing",
                GOOD,
            ]
        )
        == 1
    )
    assert cli.main(["score", "--model", str(model_file), "--threshold", "1.5", GOOD]) == 2
    capsys.readouterr()


def test_warnings_are_printed(model_file, capsys):
    cli.main(["score", "--model", str(model_file), "http://paypal.com.login.example.tk/x"])
    out = capsys.readouterr().out
    assert "! a well-known brand name in a subdomain" in out
    assert "! a domain suffix such as 'com' used as a subdomain" in out


def test_score_reads_stdin(model_file, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{BAD}\n\n{GOOD}\n"))
    cli.main(["score", "--model", str(model_file), "-"])
    assert len([ln for ln in capsys.readouterr().out.splitlines() if not ln.startswith(" ")]) == 2


def test_control_characters_do_not_reach_the_terminal(model_file, capsys):
    cli.main(["score", "--model", str(model_file), "http://a.example/\x1b[31mred"])
    assert "\x1b" not in capsys.readouterr().out


def test_missing_model_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["score", "--model", str(tmp_path / "none.json"), GOOD]) == 2
    assert "urlclf:" in capsys.readouterr().err


def test_synth_then_train_then_score(tmp_path, capsys):
    csv_file = tmp_path / "s.csv"
    assert cli.main(["synth", "--out", str(csv_file), "-n", "1500", "--seed", "2"]) == 0
    first = csv_file.read_text(encoding="utf-8").splitlines()
    assert first[0] == "URL,label" and len(first) == 1501
    out = tmp_path / "m.json"
    rep = tmp_path / "r.json"
    code = cli.main(
        ["train", "--data", str(csv_file), "--out", str(out), "--report", str(rep), "--no-compare"]
    )
    assert code == 0
    assert json.loads(rep.read_text(encoding="utf-8"))["main_model"]["auc"] > 0.9
    capsys.readouterr()
    assert cli.main(["score", "--model", str(out), BAD]) == 0
    assert capsys.readouterr().out.startswith("phishing")


def test_train_with_bad_data_is_a_clean_error(tmp_path, capsys):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    assert cli.main(["train", "--data", str(f)]) == 2
    assert "no column" in capsys.readouterr().err


def test_train_with_preset(tmp_path, capsys):
    rows = synthetic.generate(1200, seed=4)
    f = tmp_path / "p.csv"
    with f.open("w", encoding="utf-8") as h:
        h.write("FILENAME,URL,label\n")
        for i, s in enumerate(rows):
            h.write(f"{i}.txt,{s.url},{0 if s.label else 1}\n")  # PhiUSIIL: 0 = phishing
    out = tmp_path / "m.json"
    assert (
        cli.main(
            ["train", "--data", str(f), "--preset", "phiusiil", "--out", str(out), "--no-compare"]
        )
        == 0
    )
    assert out.is_file()
    capsys.readouterr()


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "urlclf 1." in capsys.readouterr().out
