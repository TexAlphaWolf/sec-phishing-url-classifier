from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from urlclf.serve import MAX_URLS, make_server


@pytest.fixture(scope="module")
def base(model):
    server = make_server(model, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(base, body, path="/score", raw=False):
    data = body if raw else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        with e:
            return e.code, json.loads(e.read())


def test_health(base):
    with urllib.request.urlopen(base + "/health", timeout=5) as r:
        assert json.loads(r.read())["status"] == "ok"
        assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_score(base):
    status, doc = post(
        base,
        {"urls": ["http://paypal.com.secure-login-x9.tk/verify/1.php", "https://www.kalomi.com/"]},
    )
    assert status == 200
    assert [r["verdict"] for r in doc["results"]] == ["phishing", "legitimate"]


@pytest.mark.parametrize(
    "body",
    [b"not json", b'{"urls": "x"}', b'{"urls": [1]}', b"{}", b"[]"],
)
def test_bad_bodies(base, body):
    status, doc = post(base, body, raw=True)
    assert status == 400 and "error" in doc


def test_limits_and_unknown_paths(base):
    status, _ = post(base, {"urls": ["http://a.example"] * (MAX_URLS + 1)})
    assert status == 413
    status, _ = post(base, {"urls": []}, path="/other")
    assert status == 404
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(base + "/nothing", timeout=5)
    with info.value:
        assert info.value.code == 404


def test_oversized_body_refused(base):
    status, _ = post(base, b"x" * (300 * 1024), raw=True)
    assert status == 413


def test_body_beyond_the_drain_limit_never_hangs_or_succeeds(base):
    try:
        status, _ = post(base, b"x" * (3 * 1024 * 1024), raw=True)
    except OSError:  # the server cut the connection: also a refusal
        return
    assert status == 413
