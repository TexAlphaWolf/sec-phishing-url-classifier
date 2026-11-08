"""Render docs/demo.png from a real run of the bundled model on examples/urls.txt.

pip install -e ".[dev]" && playwright install chromium
python tools/screenshot.py
"""

from __future__ import annotations

import contextlib
import html
import io
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from urlclf import cli


def run(urls: list[str]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cli.main(["score", "--top", "2", *urls])
    return buffer.getvalue()


def colour(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"^(phishing )", r'<span class="high">\1</span>', text, flags=re.M)
    text = re.sub(r"^(suspicious)", r'<span class="medium">\1</span>', text, flags=re.M)
    text = re.sub(r"^(legitimate)", r'<span class="ok">\1</span>', text, flags=re.M)
    text = re.sub(r"^(\s+! .*)$", r'<span class="medium">\1</span>', text, flags=re.M)
    return re.sub(r"^(\$ .*)$", r'<span class="cmd">\1</span>', text, flags=re.M)


def main() -> None:
    urls = [u for u in (ROOT / "examples/urls.txt").read_text(encoding="utf-8").split() if u]
    out = "\n".join(
        ln
        for ln in run(urls).splitlines()
        if not ln.startswith("             +") and not ln.startswith("             -")
    )
    body = colour("$ urlclf score < examples/urls.txt\n" + out)
    page = f"""<!doctype html><meta charset="utf-8"><style>
body{{margin:0;background:#0d1117}}
pre{{margin:0;padding:22px 26px;font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;color:#c9d1d9;width:1000px;box-sizing:border-box}}
.high{{color:#ff7b72;font-weight:700}}.medium{{color:#e3b341}}.ok{{color:#7ee787;font-weight:700}}.cmd{{color:#79c0ff}}
</style><pre>{body}</pre>"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        shot = browser.new_page(viewport={"width": 1000, "height": 400}, device_scale_factor=2)
        shot.set_content(page)
        shot.locator("pre").screenshot(path=str(ROOT / "docs" / "demo.png"))
        browser.close()
    print("wrote docs/demo.png")


if __name__ == "__main__":
    main()
