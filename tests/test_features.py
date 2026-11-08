from __future__ import annotations

import random

import pytest

from urlclf import features
from urlclf.features import NUMERIC, extract, fnv1a, hashed, numeric_features, parse_url, tokens


def feat(url: str) -> dict[str, float]:
    return dict(zip(NUMERIC, numeric_features(features.scoped_parts(url)), strict=True))


def test_fnv1a_known_values():
    # Published test vectors for 32-bit FNV-1a.
    assert fnv1a("") == 0x811C9DC5
    assert fnv1a("a") == 0xE40C292C
    assert fnv1a("foobar") == 0xBF9CF968


@pytest.mark.parametrize(
    ("url", "scheme", "host", "port", "path"),
    [
        ("https://www.Example.com/a/b?x=1", "https", "www.example.com", None, "/a/b"),
        ("example.com/login", "http", "example.com", None, "/login"),
        ("//example.com/x", "http", "example.com", None, "/x"),
        ("http://example.com:8080/", "http", "example.com", 8080, "/"),
        ("example.com:8080/x", "http", "example.com", 8080, "/x"),
        ("  https://example.com./  ", "https", "example.com", None, "/"),
        ("ftp://files.example.org/pub", "ftp", "files.example.org", None, "/pub"),
    ],
)
def test_parse(url, scheme, host, port, path):
    p = parse_url(url)
    assert (p.scheme, p.host, p.port, p.path) == (scheme, host, port, path)


def test_parse_userinfo_and_registered():
    p = parse_url("http://paypal.com@evil.example.tk/x")
    assert p.userinfo == "paypal.com"
    assert p.host == "evil.example.tk"
    assert p.registered == "example.tk"
    assert p.subdomain == "evil"
    assert p.tld == "tk"


def test_registered_uses_the_public_suffix_list():
    p = parse_url("https://a.b.shop.example.co.uk/")
    assert p.registered == "example.co.uk"
    assert p.subdomain == "a.b.shop"
    assert p.tld == "co.uk"


@pytest.mark.parametrize(
    "host", ["192.168.0.1", "[2001:db8::1]", "0x7f000001", "2130706433", "8.8.8.8"]
)
def test_ip_hosts(host):
    p = parse_url(f"http://{host}/x")
    assert p.host_is_ip
    assert p.registered == p.host
    assert feat(f"http://{host}/x")["host_is_ip"] == 1.0


def test_not_ip():
    assert not parse_url("http://example.com").host_is_ip
    assert not parse_url("http://123.example.com").host_is_ip
    assert not parse_url("http://99999999999/").host_is_ip  # too large for an address


@pytest.mark.parametrize(
    "url", ["http://[::1", "http://[", "http://a:99999/x", "http://", "", "   "]
)
def test_parse_never_raises(url):
    parse_url(url)
    extract(url)


def test_numeric_vector_has_one_value_per_name():
    assert len(numeric_features(features.scoped_parts("https://example.com"))) == len(NUMERIC)
    assert len(set(NUMERIC)) == len(NUMERIC)


def test_name_features():
    f = feat("http://secure-login-x9.tk:8081/verify")
    assert f["odd_port"] == 1
    assert f["name_keywords"] == 2  # "secure" and "login"; the path is not read
    assert f["name_hyphens"] == 2
    assert f["name_max_digit_run"] == 1
    assert f["name_digit_letter_switches"] == 1  # x9
    g = feat("https://www.example.com/")
    assert g["odd_port"] == 0
    assert g["name_keywords"] == 0
    assert g["name_max_repeat"] == 1
    assert feat("http://aaaabbb.com")["name_max_repeat"] == 4
    assert feat("http://bcdfgh.com")["name_max_consonant_run"] == 6
    assert feat("http://bcdfgh.com")["name_vowel_ratio"] == 0
    assert feat("http://aeiou.com")["name_vowel_ratio"] == 1
    assert feat("http://example.co.uk")["suffix_labels"] == 2
    assert feat("http://example.com")["suffix_labels"] == 1


def test_subdomains_are_not_scored():
    base = extract("http://example.com")
    assert extract("http://a.b.c.mail.example.com") == base
    assert extract("http://paypal.com.login.example.com") == base


def test_brand_stuck_onto_a_name():
    assert feat("http://paypal-secure.example/")["name_brand_plus_extra"] == 1
    assert feat("http://paypal.com/")["name_brand_plus_extra"] == 0
    assert feat("http://www.example.com/")["name_brand_plus_extra"] == 0
    assert feat("http://203.0.113.5/")["name_brand_plus_extra"] == 0


def test_warnings_for_the_part_the_model_does_not_score():
    w = features.warnings("http://paypal.com.login.example.tk:8081/x")
    assert any("brand name in a subdomain" in x for x in w)
    assert any("'com' used as a subdomain" in x for x in w)
    assert any("unusual port 8081" in x for x in w)
    assert features.warnings("https://www.example.com/a?b=c") == []
    assert any("user info" in x for x in features.warnings("http://paypal.com@example.tk/"))
    assert any("IP address" in x for x in features.warnings("http://203.0.113.9/"))
    assert any("punycode" in x for x in features.warnings("http://xn--pypal-4ve.com/"))
    assert any("non-ASCII" in x for x in features.warnings("http://pаypal.com/"))
    assert any("4 subdomain levels" in x for x in features.warnings("http://a.b.c.d.example.com/"))
    assert any("shortener" in x for x in features.warnings("https://bit.ly/abc"))


def test_default_ports_are_not_odd():
    assert feat("https://example.com:443/")["odd_port"] == 0
    assert feat("http://example.com:80/")["odd_port"] == 0
    assert feat("http://example.com:443/")["odd_port"] == 1


def test_userinfo_punycode_nonascii_shortener_noscheme():
    assert feat("http://a@b.example.com/")["has_userinfo"] == 1
    assert feat("http://xn--pypal-4ve.com/")["host_punycode"] == 1
    assert feat("http://pаypal.com/")["host_punycode"] == 1  # Cyrillic а: stored as punycode
    assert feat("https://bit.ly/abc")["shortener"] == 1
    assert not parse_url("example.com/x").explicit_scheme
    assert not parse_url("//example.com/x").explicit_scheme
    assert parse_url("https://example.com/x").explicit_scheme


def test_entropy():
    assert features.entropy("") == 0.0
    assert features.entropy("aaaa") == 0.0
    assert features.entropy("abcd") == pytest.approx(2.0)


def test_tokens_and_buckets_are_stable():
    p = features.scoped_parts("https://login.example.com/Verify/account-12345/Update")
    by = features.tokens_by_group(p)
    assert by["t"] == ["t:com"]
    assert by["h3"][0] == "h3:^ex" and by["h3"][-1] == "h3:le$"
    assert "h4:exam" in by["h4"] and "h5:examp" in by["h5"]
    assert not any("login" in t or "verify" in t for t in tokens(p))  # only the name is read
    assert tokens(p, ("t",)) == ["t:com"]
    assert hashed(p) == sorted(set(hashed(p)))
    assert all(0 <= b < features.HASH_SIZE for b in hashed(p))
    assert hashed(p, ()) == []
    ip = features.scoped_parts("http://1.2.3.4/x")
    assert tokens(ip) == ["t:<ip>"]


def test_long_input_is_cut():
    p = features.origin_parts("http://example.com/" + "a" * 100_000)
    assert p.truncated
    assert len(numeric_features(p)) == len(NUMERIC)


def test_random_text_never_raises():
    rng = random.Random(5)
    alphabet = "abc./:@[]%?#&=-_ \t\x00é‮\ud800" + "0123456789"
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        f = extract(text)
        assert len(f.numeric) == len(NUMERIC)
        assert all(x == x and abs(x) < 1e9 for x in f.numeric)  # no NaN, nothing absurd


def test_brands_file_is_clean():
    b = features.brands()
    assert "paypal" in b
    assert all(x == x.lower() and x.strip() == x for x in b)
    assert len(b) == len(set(b))


def test_absurdly_long_hosts_are_cheap_and_safe():
    p = parse_url("http://" + "a." * 3000 + "com/")
    assert p.registered == p.host
    assert p.tld == p.host.rsplit(".", 1)[-1]  # the URL was cut at 4096 characters
    assert p.subdomain == ""


def test_origin_ignores_path_query_and_www():
    base = extract("https://example.com")
    for url in [
        "https://example.com/",
        "https://example.com/login?next=/a&b=1",
        "https://www.example.com/a/b/c.php",
        "HTTPS://WWW.EXAMPLE.COM:443/x",
    ]:
        got = extract(url)
        if ":443" in url:  # an explicit default port is still a port the user typed
            continue
        assert got == base, url
    assert extract("https://example.com:8443/") != base


def test_origin_parts_rebuilds_the_origin():
    assert (
        features.origin_parts("http://u:p@www.Example.com:81/x?y").raw
        == "http://u:p@example.com:81"
    )
    assert features.origin_parts("example.com/x").raw == "example.com"
    assert features.origin_parts("https://[2001:db8::1]:8080/x").raw == "https://[2001:db8::1]:8080"
    assert features.origin_parts("http://www.").host == "www"  # nothing left to strip safely


def test_the_scheme_is_not_a_feature():
    assert extract("http://example.com") == extract("https://example.com")
    assert extract("example.com") == extract("https://example.com")
    assert not {"is_https", "no_scheme"} & set(NUMERIC)
