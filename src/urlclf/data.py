"""Reading labelled URL lists, and splitting them so that no site appears on both sides.

Why group the split: a phishing kit is deployed on many URLs of one domain, and a legitimate site
has many pages. Split URL by URL, and the model is tested on pages of sites it has trained on, which
flatters it. Here every URL of one registrable domain goes to the same part.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from urlclf import features

# label 1 = phishing in everything below. The dataset files may use the opposite.
PRESETS = {
    # PhiUSIIL Phishing URL Dataset (UCI #967, CC BY 4.0): column "URL", label 1 = legitimate.
    "phiusiil": {"url": "URL", "label": "label", "phishing": "0"},
}


@dataclass(frozen=True)
class Sample:
    url: str
    label: int  # 1 phishing, 0 legitimate


class DataError(ValueError):
    pass


def read_csv(
    path: Path,
    *,
    url_column: str = "URL",
    label_column: str = "label",
    phishing_value: str = "1",
    limit: int | None = None,
) -> Iterator[Sample]:
    """Rows of a CSV with a URL column and a label column. ``phishing_value`` is the text in the
    label column that means phishing; any other value means legitimate."""
    csv.field_size_limit(min(sys.maxsize, 2**24))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        for needed in (url_column, label_column):
            if needed not in names:
                raise DataError(f"{path.name} has no column {needed!r}; it has {names[:8]}")
        count = 0
        for row in reader:
            url = (row.get(url_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not url or not label:
                continue
            yield Sample(url, int(label == phishing_value))
            count += 1
            if limit is not None and count >= limit:
                return


def load(
    path: Path,
    preset: str | None,
    url_column: str = "URL",
    label_column: str = "label",
    phishing_value: str = "1",
) -> list[Sample]:
    if preset is not None:
        if preset not in PRESETS:
            raise DataError(f"unknown preset {preset!r}; known: {', '.join(PRESETS)}")
        p = PRESETS[preset]
        samples = list(
            read_csv(
                path, url_column=p["url"], label_column=p["label"], phishing_value=p["phishing"]
            )
        )
    else:
        samples = list(
            read_csv(
                path,
                url_column=url_column,
                label_column=label_column,
                phishing_value=phishing_value,
            )
        )
    if not samples:
        raise DataError(f"no usable rows in {path.name}")
    return samples


@dataclass
class Split:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    stats: dict[str, Any] = field(default_factory=dict)


def _key(url: str) -> str:
    """What the model can tell apart: the origin. Ten pages of one phishing host are one example."""
    return features.origin_parts(url).raw.lower()


def clean(samples: list[Sample]) -> tuple[list[Sample], dict[str, int]]:
    """Keep one row per origin and label, and drop every row of an origin that carries both labels
    (it would teach nothing). The counts are in rows."""
    seen: dict[str, int] = {}
    conflicting: set[str] = set()
    keys = [_key(s.url) for s in samples]
    for key, s in zip(keys, samples, strict=True):
        if key in seen and seen[key] != s.label:
            conflicting.add(key)
        seen.setdefault(key, s.label)
    out = []
    done: set[str] = set()
    conflicting_rows = 0
    for key, s in zip(keys, samples, strict=True):
        if key in conflicting:
            conflicting_rows += 1
            continue
        if key in done:
            continue
        done.add(key)
        out.append(Sample(s.url.strip(), s.label))
    return out, {
        "rows_read": len(samples),
        "duplicates_removed": len(samples) - len(out) - conflicting_rows,
        "conflicting_removed": conflicting_rows,
    }


def group_of(url: str) -> str:
    return features.parse_url(url).registered or url


def part_of(group: str, seed: int) -> str:
    """Which part a group falls in, from a fixed hash: 70% train, 15% validation, 15% test."""
    bucket = features.fnv1a(f"{seed}:{group}") % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def split(samples: list[Sample], seed: int = 1) -> Split:
    cleaned, counts = clean(samples)
    stats: dict[str, Any] = dict(counts)
    parts: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for s in cleaned:
        g = group_of(s.url)
        name = part_of(g, seed)
        parts[name].append(s)
        groups[name].add(g)
    stats["seed"] = seed
    for name, rows in parts.items():
        c = Counter(s.label for s in rows)
        stats[name] = {
            "urls": len(rows),
            "phishing": c[1],
            "legitimate": c[0],
            "sites": len(groups[name]),
        }
    return Split(parts["train"], parts["val"], parts["test"], stats)
