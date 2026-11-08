"""A small local HTTP API: POST /score with {"urls": [...]} -> scores. Standard library only.

It listens on 127.0.0.1 unless told otherwise, caps the body and the number of URLs, and never
fetches any address it is given.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from urlclf.model import Model, verdict

MAX_BODY = 256 * 1024
MAX_URLS = 200


def score_dict(model: Model, url: str, top: int = 5) -> dict[str, Any]:
    s = model.score(url, top)
    return {
        "url": s.url,
        "probability": round(s.probability, 6),
        "verdict": verdict(s.probability, model.threshold, s.warnings),
        "truncated": s.truncated,
        "warnings": s.warnings,
        "reasons": [{"feature": k, "logit": round(v, 4)} for k, v in s.reasons],
    }


def make_server(model: Model, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "urlclf"
        sys_version = ""

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok", "threshold": model.threshold})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/score":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self._send(413 if length > MAX_BODY else 411, {"error": "bad or too large body"})
                return
            try:
                doc = json.loads(self.rfile.read(length))
                urls = doc["urls"]
                if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                self._send(400, {"error": 'send {"urls": ["...", ...]}'})
                return
            if len(urls) > MAX_URLS:
                self._send(413, {"error": f"at most {MAX_URLS} URLs per request"})
                return
            self._send(200, {"results": [score_dict(model, u) for u in urls]})

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ThreadingHTTPServer((host, port), Handler)
