from __future__ import annotations

from pathlib import Path

import pytest

from urlclf import data
from urlclf.data import DataError, Sample


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_phiusiil_layout_label_zero_is_phishing(tmp_path):
    f = write(
        tmp_path / "d.csv",
        "FILENAME,URL,Domain,label\n1.txt,https://good.example/,good.example,1\n"
        "2.txt,http://bad.example/login,bad.example,0\n",
    )
    rows = data.load(f, "phiusiil")
    assert rows == [Sample("https://good.example/", 0), Sample("http://bad.example/login", 1)]


def test_custom_columns(tmp_path):
    f = write(tmp_path / "d.csv", "u,kind\nhttp://a.example/,phish\nhttp://b.example/,ok\n")
    rows = data.load(f, None, url_column="u", label_column="kind", phishing_value="phish")
    assert [r.label for r in rows] == [1, 0]


def test_missing_column_and_empty_and_unknown_preset(tmp_path):
    f = write(tmp_path / "d.csv", "a,b\n1,2\n")
    with pytest.raises(DataError, match="no column"):
        data.load(f, None)
    e = write(tmp_path / "e.csv", "URL,label\n,\n")
    with pytest.raises(DataError, match="no usable rows"):
        data.load(e, None)
    with pytest.raises(DataError, match="unknown preset"):
        data.load(f, "nope")


def test_bom_and_limit(tmp_path):
    f = tmp_path / "d.csv"
    f.write_bytes(b"\xef\xbb\xbfURL,label\nhttp://a.example/,1\nhttp://b.example/,0\n")
    assert len(list(data.read_csv(f))) == 2
    assert len(list(data.read_csv(f, limit=1))) == 1


def test_clean_removes_duplicates_and_conflicts():
    rows = [
        Sample("http://a.example/", 0),
        Sample("http://a.example/", 0),
        Sample("http://b.example/", 1),
        Sample("http://b.example/", 0),
        Sample(" http://c.example/ ", 1),
        Sample("http://www.c.example/other/page?x=1", 1),  # same origin as the line above
    ]
    out, stats = data.clean(rows)
    assert [s.url for s in out] == ["http://a.example/", "http://c.example/"]
    assert stats == {"rows_read": 6, "duplicates_removed": 2, "conflicting_removed": 2}


def test_split_keeps_every_site_on_one_side(samples):
    s = data.split(samples, seed=3)
    sites = [{data.group_of(x.url) for x in part} for part in (s.train, s.val, s.test)]
    assert not (sites[0] & sites[1]) and not (sites[0] & sites[2]) and not (sites[1] & sites[2])
    total = len(s.train) + len(s.val) + len(s.test)
    assert total == (
        s.stats["rows_read"] - s.stats["duplicates_removed"] - s.stats["conflicting_removed"]
    )
    assert 0.6 < len(s.train) / total < 0.8
    assert len(s.val) > 0 and len(s.test) > 0


def test_split_is_deterministic_and_seed_changes_it(samples):
    a = data.split(samples, seed=3)
    b = data.split(samples, seed=3)
    c = data.split(samples, seed=4)
    assert a.test == b.test
    assert a.test != c.test


def test_same_site_different_pages_stay_together():
    rows = [Sample(f"http://shop.example.com/p{i}", i % 2) for i in range(50)]
    rows += [Sample(f"http://www.example.com/q{i}", 0) for i in range(50)]  # same registered domain
    s = data.split(rows, seed=1)
    non_empty = [p for p in (s.train, s.val, s.test) if p]
    assert len(non_empty) == 1
