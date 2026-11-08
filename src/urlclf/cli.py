"""Command line: urlclf score | train | synth | serve."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from urlclf import __version__, data, synthetic
from urlclf.model import ModelError, load_model, verdict
from urlclf.serve import make_server, score_dict


def _score(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    urls = args.urls
    if urls == ["-"] or not urls:
        urls = [line.strip() for line in sys.stdin if line.strip()]
    threshold = args.threshold if args.threshold is not None else model.threshold
    if not 0.0 <= threshold <= 1.0:
        print("urlclf: --threshold must lie between 0 and 1", file=sys.stderr)
        return 2
    results = []
    for u in urls:
        d = score_dict(model, u, args.top)
        d["verdict"] = verdict(d["probability"], threshold, d["warnings"])
        results.append(d)
    if args.format == "json":
        print(json.dumps({"threshold": threshold, "results": results}, indent=2))
    else:
        for d in results:
            safe = "".join(c if c.isprintable() else "?" for c in d["url"])
            print(f"{d['verdict']:<10} {d['probability']:.3f}  {safe}")
            for w in d["warnings"]:
                print(f"             ! {w}")
            if args.explain:
                for r in d["reasons"]:
                    print(f"             {r['logit']:+.2f}  {r['feature']}")
    flagged = any(d["verdict"] == "phishing" for d in results)
    return 1 if (args.fail_on_phishing and flagged) else 0


def _train(args: argparse.Namespace) -> int:
    from urlclf.train import train

    if args.preset:
        samples = data.load(args.data, args.preset)
    else:
        samples = data.load(
            args.data,
            None,
            url_column=args.url_column,
            label_column=args.label_column,
            phishing_value=args.phishing_value,
        )
    parts = data.split(samples, args.seed)
    model, report = train(parts, args.seed, compare=not args.no_compare, progress=True)
    model.save(args.out)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"model written to {args.out}", file=sys.stderr)
    return 0


def _synth(args: argparse.Namespace) -> int:
    rows = synthetic.generate(args.n, args.seed)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        handle.write("URL,label\n")
        for s in rows:
            handle.write(f"{s.url},{s.label}\n")
    print(f"{len(rows)} made-up URLs written to {args.out}", file=sys.stderr)
    return 0


def _serve(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    server = make_server(model, args.host, args.port)
    print(
        f"listening on http://{args.host}:{args.port}  (POST /score, GET /health)", file=sys.stderr
    )
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="urlclf", description="Phishing scores for URLs, from their text alone."
    )
    p.add_argument("--version", action="version", version=f"urlclf {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="score URLs (arguments, or lines on standard input)")
    s.add_argument("urls", nargs="*", help="URLs; '-' or none reads standard input")
    s.add_argument("--model", type=Path, help="model file (default: the bundled one)")
    s.add_argument("--format", choices=["text", "json"], default="text")
    s.add_argument("--explain", action="store_true", help="show what moved each score most")
    s.add_argument("--top", type=int, default=5, help="how many reasons to keep")
    s.add_argument("--threshold", type=float, help="override the model's threshold (0-1)")
    s.add_argument("--fail-on-phishing", action="store_true", help="exit 1 if any URL is flagged")
    s.set_defaults(run=_score)

    t = sub.add_parser("train", help="train on a labelled CSV and report on a held-out test part")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument(
        "--preset", choices=sorted(data.PRESETS), help="column layout of a known dataset"
    )
    t.add_argument("--url-column", default="URL")
    t.add_argument("--label-column", default="label")
    t.add_argument("--phishing-value", default="1", help="label text that means phishing")
    t.add_argument("--out", type=Path, default=Path("model.json"))
    t.add_argument("--report", type=Path, help="also write the report here")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--no-compare", action="store_true", help="skip the comparison models")
    t.set_defaults(run=_train)

    y = sub.add_parser("synth", help="write made-up labelled URLs (for trying the pipeline only)")
    y.add_argument("--out", type=Path, required=True)
    y.add_argument("-n", type=int, default=5000)
    y.add_argument("--seed", type=int, default=1)
    y.set_defaults(run=_synth)

    v = sub.add_parser("serve", help="local JSON API")
    v.add_argument("--model", type=Path)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8080)
    v.set_defaults(run=_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (ModelError, data.DataError) as exc:
        print(f"urlclf: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"urlclf: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
