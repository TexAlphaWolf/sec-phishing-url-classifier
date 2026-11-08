"""Made-up URLs for tests, CI and the demo. They are NOT evidence of anything.

The generator invents the very patterns the features look for, so a model trained on its output
scores well by construction. It exists to run the whole pipeline (read, split, train, save, load,
score) without downloading a dataset. Real figures come from a real dataset (docs/evaluation.md).
"""

from __future__ import annotations

import random

from urlclf.data import Sample

_SYLL = ("ka", "lo", "mi", "ra", "te", "no", "vi", "su", "pe", "da", "or", "an", "el", "ti", "bo")
_WORDS = (
    "news", "blog", "shop", "about", "contact", "products", "docs", "help", "team", "careers",
    "events", "gallery", "press", "learn", "guide", "pricing", "download", "forum", "store",
)  # fmt: skip
_BRANDS = ("paypal", "apple", "microsoft", "google", "amazon", "netflix", "intesa", "unicredit")
_KEYS = ("login", "verify", "secure", "account", "update", "confirm", "signin", "billing")
_TLDS = ("com", "net", "org", "it", "de", "io", "co.uk", "info")
_BAD_TLDS = ("tk", "xyz", "top", "click", "icu", "info", "ru", "cc")


def _word(rng: random.Random, n: int | None = None) -> str:
    return "".join(rng.choice(_SYLL) for _ in range(n or rng.randint(2, 4)))


def _legit(rng: random.Random) -> str:
    domain = f"{_word(rng)}.{rng.choice(_TLDS)}"
    host = rng.choice(["", "www.", "blog.", "shop."]) + domain
    path = "/".join(rng.choice(_WORDS) for _ in range(rng.choice([0, 0, 1, 2, 3])))
    if rng.random() < 0.15:
        path += f"?id={rng.randint(1, 9999)}"
    scheme = "https" if rng.random() < 0.9 else "http"
    return f"{scheme}://{host}/{path}".rstrip("/") if path else f"{scheme}://{host}"


def _phish(rng: random.Random) -> str:
    kind = rng.choice(["brand-sub", "ip", "hyphens", "userinfo", "random", "plain"])
    tld = rng.choice(_BAD_TLDS)
    key = rng.choice(_KEYS)
    brand = rng.choice(_BRANDS)
    scheme = "https" if rng.random() < 0.6 else "http"
    if kind == "brand-sub":
        host = f"{brand}.com.{key}-{_word(rng)}.{tld}"
    elif kind == "ip":
        host = ".".join(str(rng.randint(1, 254)) for _ in range(4))
    elif kind == "hyphens":
        host = f"{brand}-{key}-{_word(rng)}-{rng.randint(10, 999)}.{tld}"
    elif kind == "userinfo":
        host = f"{brand}.com@{_word(rng)}{rng.randint(10, 99)}.{tld}"
    elif kind == "random":
        host = (
            "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(14))
            + f".{tld}"
        )
    else:  # looks ordinary on purpose: the hard cases
        host = f"{_word(rng)}.{rng.choice(_TLDS)}"
    path = "/".join([key, _word(rng), f"{rng.randint(1000, 99999)}.php"])
    return f"{scheme}://{host}/{path}"


def generate(n: int, seed: int = 1, phishing_share: float = 0.43) -> list[Sample]:
    """``n`` labelled URLs, deterministic for a given ``seed``."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < phishing_share:
            out.append(Sample(_phish(rng), 1))
        else:
            out.append(Sample(_legit(rng), 0))
    return out
