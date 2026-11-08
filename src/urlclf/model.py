"""The model: a logistic regression kept as plain JSON, scored in pure Python.

A model file is data, never code: it is read with ``json`` and checked field by field, so opening
one cannot run anything (unlike a pickled scikit-learn object).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from urlclf import features

FORMAT = "urlclf-linear-2"
MAX_MODEL_BYTES = 8 * 1024 * 1024


class ModelError(ValueError):
    """The file is not a model this version can use."""


@dataclass
class Model:
    mean: list[float]
    scale: list[float]
    w_numeric: list[float]
    bias: float
    w_hash: dict[int, float]
    threshold: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)
    groups: tuple[str, ...] = features.GROUPS

    def logit(self, feats: features.Features) -> float:
        z = self.bias
        for x, m, s, w in zip(feats.numeric, self.mean, self.scale, self.w_numeric, strict=True):
            z += w * (x - m) / s
        for b in feats.buckets:
            z += self.w_hash.get(b, 0.0)
        return z

    def probability(self, url: str) -> float:
        return sigmoid(self.logit(features.extract(url, self.groups)))

    def score(self, url: str, top: int = 5) -> Score:
        parts = features.scoped_parts(url)
        feats = features.Features(
            features.numeric_features(parts), features.hashed(parts, self.groups)
        )
        z = self.logit(feats)
        prob = sigmoid(z)
        return Score(
            url=url,
            probability=prob,
            phishing=prob >= self.threshold,
            reasons=self._reasons(parts, feats, top),
            truncated=parts.truncated,
            warnings=features.warnings(url),
        )

    def _reasons(
        self, parts: features.Parts, feats: features.Features, top: int
    ) -> list[tuple[str, float]]:
        """The pieces that moved the score most, in logit units (positive = towards phishing)."""
        items: list[tuple[str, float]] = []
        for name, x, m, s, w in zip(
            features.NUMERIC, feats.numeric, self.mean, self.scale, self.w_numeric, strict=True
        ):
            items.append((f"{name}={_fmt(x)}", w * (x - m) / s))
        seen: set[int] = set()
        for token in features.tokens(parts, self.groups):
            b = features.bucket(token)
            if b in seen or b not in self.w_hash:
                continue
            seen.add(b)
            items.append((token, self.w_hash[b]))
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return [(k, v) for k, v in items[:top] if v != 0.0]

    def to_json(self) -> str:
        doc = {
            "format": FORMAT,
            "numeric": list(features.NUMERIC),
            "hash_bits": features.HASH_BITS,
            "token_groups": list(self.groups),
            "mean": self.mean,
            "scale": self.scale,
            "w_numeric": self.w_numeric,
            "bias": self.bias,
            "w_hash": {str(k): v for k, v in sorted(self.w_hash.items())},
            "threshold": self.threshold,
            "meta": self.meta,
        }
        return json.dumps(doc, indent=1, sort_keys=False) + "\n"

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")


def verdict(probability: float, threshold: float, warnings: list[str]) -> str:
    """``phishing`` when the score reaches the threshold; ``suspicious`` when it does not but a
    plain rule found something in the part of the URL the model does not score; else
    ``legitimate`` (which means "no sign found", not "safe")."""
    if probability >= threshold:
        return "phishing"
    return "suspicious" if warnings else "legitimate"


@dataclass(frozen=True)
class Score:
    url: str
    probability: float
    phishing: bool
    reasons: list[tuple[str, float]]
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) else f"{x:.2f}"


def _floats(doc: dict[str, Any], key: str, n: int) -> list[float]:
    value = doc.get(key)
    if not isinstance(value, list) or len(value) != n:
        raise ModelError(f"{key!r} must be a list of {n} numbers")
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ModelError(f"{key!r} holds a value that is not a finite number")
        out.append(float(v))
    return out


def parse_model(text: str) -> Model:
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise ModelError(f"not valid JSON: {exc}") from None
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ModelError(f"not a {FORMAT} model")
    if doc.get("numeric") != list(features.NUMERIC):
        raise ModelError("the feature list differs from this version of urlclf: retrain the model")
    if doc.get("hash_bits") != features.HASH_BITS:
        raise ModelError("the hash size differs from this version of urlclf: retrain the model")
    groups = doc.get("token_groups")
    if not isinstance(groups, list) or not all(g in features.GROUPS for g in groups):
        raise ModelError(f"'token_groups' must be a list of: {', '.join(features.GROUPS)}")
    if len(set(groups)) != len(groups):
        raise ModelError("'token_groups' repeats a group")
    n = len(features.NUMERIC)
    mean = _floats(doc, "mean", n)
    scale = _floats(doc, "scale", n)
    if any(s <= 0 for s in scale):
        raise ModelError("'scale' must be positive")
    w_numeric = _floats(doc, "w_numeric", n)
    bias = _floats({"b": [doc.get("bias")]}, "b", 1)[0]
    raw_hash = doc.get("w_hash")
    if not isinstance(raw_hash, dict):
        raise ModelError("'w_hash' must be an object")
    w_hash: dict[int, float] = {}
    for key, value in raw_hash.items():
        if not (key.isdigit() and int(key) < features.HASH_SIZE):
            raise ModelError(f"bad bucket {key!r}")
        w_hash[int(key)] = _floats({"v": [value]}, "v", 1)[0]
    threshold = _floats({"t": [doc.get("threshold")]}, "t", 1)[0]
    if not 0.0 < threshold < 1.0:
        raise ModelError("'threshold' must lie between 0 and 1")
    meta = doc.get("meta", {})
    if not isinstance(meta, dict):
        raise ModelError("'meta' must be an object")
    return Model(mean, scale, w_numeric, bias, w_hash, threshold, meta, tuple(groups))


def load_model(path: Path | str | None = None) -> Model:
    """Load ``path``, or the model shipped inside the package when ``path`` is None."""
    if path is None:
        resource = resources.files("urlclf").joinpath("data/model.json")
        if not resource.is_file():
            raise ModelError(
                "this copy has no bundled model; train one with 'urlclf train' and pass --model"
            )
        return parse_model(resource.read_text("utf-8"))
    file = Path(path)
    try:
        if file.stat().st_size > MAX_MODEL_BYTES:
            raise ModelError(f"{file} is larger than {MAX_MODEL_BYTES} bytes")
        return parse_model(file.read_text("utf-8"))
    except OSError as exc:
        raise ModelError(f"cannot read {file}: {exc.strerror}") from None
    except UnicodeDecodeError:
        raise ModelError(f"{file} is not UTF-8 text") from None
