# Security policy

urlclf reads URLs that may be hostile, and model files that anyone could hand you. Two kinds of problem matter:

- **Input that makes it crash, hang, or print control characters**: a URL is untrusted text. It is cut at 4,096 characters, never fetched or resolved, and shown with unprintable characters replaced.
- **A model file that does more than hold numbers**: model files are read with `json` and validated field by field (format tag, list lengths, finite numbers, bucket range, size limit). Nothing is unpickled.

The `serve` command listens on `127.0.0.1` by default, has no authentication, caps the request body (256 KiB) and the number of URLs per request (200), and only scores. Do not expose it to a network you do not trust.

Please report a problem through GitHub's **private vulnerability reporting** (Security tab, "Report a vulnerability") rather than a public issue. You will get an answer within a few days.

A score is not a safety guarantee. What it can and cannot tell is in [docs/limitations.md](docs/limitations.md).

Supported: the latest release.
