"""Features from the text of a URL. Nothing here ever contacts the address it reads.

What the model scores is the **registrable domain** (``example.co.uk`` for ``a.b.example.co.uk``),
plus a user-info part, a non-default port or an IP address when there is one. Not scored: the path,
the query, the scheme, a leading ``www.`` and the subdomains. This is deliberate. In the public
dataset used to measure the model, the legitimate URLs are bare ``https://www.`` domains of
popular sites: they have no path, are all ``https`` and almost never have another subdomain. A
model fed any of those would learn them as "phishing signs", which is true of the dataset and false
of the web (see docs/limitations.md). Tricks in the subdomain (``paypal.com.example.tk``) are caught
by plain rules instead, reported as ``warnings`` next to the score.

Two kinds of feature feed the model:

* ``NUMERIC``: counts, ratios and flags about the domain's name (length, digits, hyphens, how
  pronounceable it looks, brand names stuck onto it, ...).
* hashed tokens: character n-grams of the name and its suffix, each folded into one of
  ``2**HASH_BITS`` buckets with FNV-1a (a fixed function, so the same URL gives the same buckets on
  every machine and Python version). ``GROUPS`` names the kinds of token.
"""

from __future__ import annotations

import ipaddress
import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources
from urllib.parse import urlsplit

from urlclf import psl

MAX_HOST = 253  # longest valid DNS name; a longer "host" is junk and skips the list lookup
MAX_URL = 4096  # longer input is cut: nobody needs more to decide, and it bounds the work
HASH_BITS = 15
HASH_SIZE = 1 << HASH_BITS

KEYWORDS = (
    "login",
    "signin",
    "verify",
    "secure",
    "account",
    "update",
    "confirm",
    "password",
    "banking",
    "wallet",
    "invoice",
    "suspend",
    "unlock",
    "billing",
    "support",
    "recover",
    "validate",
    "authenticate",
)
SHORTENERS = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "rebrand.ly",
        "cutt.ly",
        "shorturl.at",
        "tiny.cc",
        "rb.gy",
    }
)
_DEFAULT_PORTS = {"http": 80, "https": 443, "": 80}
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


@lru_cache(maxsize=1)
def brands() -> tuple[str, ...]:
    text = resources.files("urlclf").joinpath("data/brands.txt").read_text("utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip() and line[0] != "#")


@dataclass(frozen=True)
class Parts:
    """A URL taken apart tolerantly: odd input gives odd parts, never an exception."""

    raw: str
    scheme: str
    userinfo: str
    host: str
    port: int | None
    path: str
    query: str
    truncated: bool = False
    explicit_scheme: bool = True

    @property
    def host_is_ip(self) -> bool:
        return _ip(self.host) is not None

    @property
    def registered(self) -> str:
        """The registrable domain (``example.co.uk`` for ``a.b.example.co.uk``); the host if none."""
        if self.host_is_ip or len(self.host) > MAX_HOST:
            return self.host
        return psl.organizational_domain(self.host) or self.host

    @property
    def subdomain(self) -> str:
        if self.host_is_ip or self.registered == self.host:
            return ""
        return self.host[: -len(self.registered)].rstrip(".")

    @property
    def tld(self) -> str:
        if self.host_is_ip:
            return ""
        if len(self.host) > MAX_HOST:
            return self.host.rsplit(".", 1)[-1]
        return (psl.public_suffix(self.host) or self.host.rsplit(".", 1)[-1]).lower()


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    text = host.strip("[]")
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        pass
    # 0x7f000001, 2130706433 and 0177.0.0.1 all mean an address to a browser.
    try:
        if re.fullmatch(r"0[xX][0-9a-fA-F]{1,8}", text):
            return ipaddress.IPv4Address(int(text, 16))
        if re.fullmatch(r"[0-9]{8,10}", text) and int(text) < 2**32:
            return ipaddress.IPv4Address(int(text))
    except ValueError:
        return None
    return None


def parse_url(raw: str) -> Parts:
    """Split ``raw`` into parts. A missing scheme is read as http, like a browser's address bar."""
    text = raw.strip()
    truncated = len(text) > MAX_URL
    text = text[:MAX_URL]
    explicit = True
    if text.startswith("//"):
        text = "http:" + text
        explicit = False
    elif not _SCHEME.match(text) or re.match(r"^[A-Za-z0-9.-]+:[0-9]+(/|$)", text):
        text = "http://" + text
        explicit = False
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower().rstrip(".")
        try:
            port = parts.port
        except ValueError:
            port = None
        userinfo = parts.netloc.rpartition("@")[0] if "@" in parts.netloc else ""
        return Parts(
            raw=raw,
            scheme=parts.scheme.lower(),
            userinfo=userinfo,
            host=host,
            port=port,
            path=parts.path,
            query=parts.query,
            truncated=truncated,
            explicit_scheme=explicit,
        )
    except ValueError:
        # urlsplit refuses, for instance, an unbalanced "[" in the host. Keep what is readable.
        rest = text.split("://", 1)[-1]
        host = re.split(r"[/?#]", rest, maxsplit=1)[0].rpartition("@")[2].lower()
        return Parts(raw, "", "", host, None, "", "", truncated, False)


def origin_parts(raw: str) -> Parts:
    """The part of a URL the model reads: scheme, user info, host (without a leading ``www.``) and
    port. ``Parts.raw`` becomes the rebuilt origin, so lengths and ratios describe the origin."""
    p = parse_url(raw)
    host = p.host[4:] if p.host.startswith("www.") and len(p.host) > 4 else p.host
    netloc = f"[{host}]" if ":" in host else host
    if p.userinfo:
        netloc = f"{p.userinfo}@{netloc}"
    if p.port is not None:
        netloc += f":{p.port}"
    origin = f"{p.scheme}://{netloc}" if p.explicit_scheme and p.scheme else netloc
    return replace(p, raw=origin, host=host, path="", query="")


def entropy(text: str) -> float:
    """Shannon entropy in bits per character: random-looking names score high."""
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


GROUPS = ("h3", "h4", "h5", "t")  # character 3-, 4- and 5-grams of the name; the suffix

NUMERIC = (
    "len_name",  # log1p: the model sees "twice as long", not "5 more characters"
    "name_digit_ratio",
    "name_hyphens",
    "name_entropy",
    "name_vowel_ratio",
    "name_max_consonant_run",
    "name_max_digit_run",
    "name_digit_letter_switches",
    "name_max_repeat",
    "name_keywords",
    "name_brand_plus_extra",
    "suffix_labels",
    "tld_length",
    "host_is_ip",
    "host_punycode",
    "has_userinfo",
    "odd_port",
    "shortener",
)

_VOWELS = frozenset("aeiou")


def _run(text: str, test: Callable[[str], bool]) -> int:
    best = cur = 0
    for c in text:
        cur = cur + 1 if test(c) else 0
        best = max(best, cur)
    return best


def scoped_parts(raw: str) -> Parts:
    """What the model reads: the origin with the host cut to its registrable domain."""
    p = origin_parts(raw)
    if p.host_is_ip or not p.host:
        return p
    host = p.registered
    netloc = f"{p.userinfo}@{host}" if p.userinfo else host
    if p.port is not None:
        netloc += f":{p.port}"
    return replace(p, raw=netloc, host=host)


def name_of(parts: Parts) -> str:
    """The registrable domain without its public suffix (``example`` for ``example.co.uk``)."""
    if parts.host_is_ip or not parts.host:
        return parts.host
    suffix = parts.tld
    host = parts.host
    return host[: -len(suffix) - 1] if suffix and host.endswith("." + suffix) else host


def numeric_features(parts: Parts) -> list[float]:
    """The values for ``NUMERIC``, in that order. ``parts`` should come from ``scoped_parts``."""
    name = name_of(parts)
    letters = [c for c in name if c.isalpha()]
    all_brands = brands()
    extra = any(b in name and b != name for b in all_brands) if not parts.host_is_ip else False
    switches = sum(
        1
        for x, y in itertools.pairwise(name)
        if x.isdigit() != y.isdigit() and (x.isalnum() and y.isalnum())
    )
    return [
        math.log1p(len(name)),
        _ratio(sum(c.isdigit() for c in name), len(name)),
        float(name.count("-")),
        entropy(name),
        _ratio(sum(c in _VOWELS for c in letters), len(letters)),
        float(_run(name, lambda c: c.isalpha() and c not in _VOWELS)),
        float(_run(name, str.isdigit)),
        float(switches),
        float(_run_repeat(name)),
        float(sum(name.count(k) for k in KEYWORDS)),
        float(extra),
        float(len(parts.tld.split(".")) if parts.tld else 0),
        float(len(parts.tld)),
        float(parts.host_is_ip),
        float("xn--" in parts.host),  # the registrable domain is stored as punycode
        float(bool(parts.userinfo)),
        float(parts.port is not None and parts.port != _DEFAULT_PORTS.get(parts.scheme, -1)),
        float(parts.registered in SHORTENERS),
    ]


def _run_repeat(text: str) -> int:
    best = cur = 1 if text else 0
    for x, y in itertools.pairwise(text):
        cur = cur + 1 if x == y else 1
        best = max(best, cur)
    return best


def warnings(raw: str) -> list[str]:
    """Plain-rule signs that are visible in the part of the URL the model does not score. They do
    not change the probability; they are reported next to it."""
    p = origin_parts(raw)
    out: list[str] = []
    sub = p.subdomain
    labels = [x for x in sub.split(".") if x]
    low = sub.lower()
    if p.userinfo:
        out.append("user info before the host (text before '@' is not the site)")
    if p.host_is_ip:
        out.append("the host is an IP address")
    if p.port is not None and p.port != _DEFAULT_PORTS.get(p.scheme, -1):
        out.append(f"unusual port {p.port}")
    if "xn--" in p.host:
        out.append("punycode (internationalised) host name")
    elif any(ord(c) > 127 for c in p.host):
        out.append("non-ASCII characters in the host name")
    if any(b in low for b in brands()):
        out.append("a well-known brand name in a subdomain of an unrelated domain")
    if any(w in labels for w in ("com", "net", "org", "gov", "edu")):
        out.append("a domain suffix such as 'com' used as a subdomain")
    if len(labels) >= 4:
        out.append(f"{len(labels)} subdomain levels")
    if p.registered in SHORTENERS:
        out.append("a URL shortener: the target is hidden")
    return out


def fnv1a(text: str) -> int:
    """32-bit FNV-1a over the UTF-8 bytes: a fixed, documented hash (Python's ``hash`` is not)."""
    h = 0x811C9DC5
    for byte in text.encode("utf-8", "surrogatepass"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h


def tokens_by_group(parts: Parts) -> dict[str, list[str]]:
    """Strings that stand for pieces of the name and its suffix, grouped by kind."""
    out: dict[str, list[str]] = {g: [] for g in GROUPS}
    if parts.host_is_ip:
        out["t"].append("t:<ip>")
        return out
    name = name_of(parts)
    if not parts.host:
        return out
    padded = f"^{name}$"
    for n, group in ((3, "h3"), (4, "h4"), (5, "h5")):
        out[group].extend(f"{group}:{padded[i : i + n]}" for i in range(len(padded) - n + 1))
    out["t"].append("t:" + parts.tld)
    return out


def tokens(parts: Parts, groups: Sequence[str] = GROUPS) -> list[str]:
    by = tokens_by_group(parts)
    return [t for g in groups for t in by[g]]


def bucket(token: str) -> int:
    return fnv1a(token) & (HASH_SIZE - 1)


def hashed(parts: Parts, groups: Sequence[str] = GROUPS) -> list[int]:
    """The distinct buckets the tokens fall into, sorted."""
    return sorted({bucket(t) for t in tokens(parts, groups)})


@dataclass(frozen=True)
class Features:
    numeric: list[float]
    buckets: list[int]


def extract(raw: str, groups: Sequence[str] = GROUPS) -> Features:
    parts = scoped_parts(raw)
    return Features(numeric_features(parts), hashed(parts, groups))
