#!/usr/bin/env python3
"""Build the self-contained 2026 fantasy luck explorer HTML.

The page layout, styles and scripts live in templates/explorer.html. This
script embeds output/explorer_data.json into that template and writes the
finished page to output/fantasy_luck_2026.html and index.html.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
TEMPLATE = ROOT / "templates" / "explorer.html"
PLACEHOLDER = "/*__EXPLORER_DATA__*/null"

data = json.loads((OUT / "explorer_data.json").read_text())
# Escape "<" so player names can never close the <script> tag.
payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")

template = TEMPLATE.read_text()
if template.count(PLACEHOLDER) != 1:
    raise SystemExit(f"Expected exactly one {PLACEHOLDER!r} in {TEMPLATE}")
html = template.replace(PLACEHOLDER, payload)

path = OUT / "fantasy_luck_2026.html"
path.write_text(html)
index = ROOT / "index.html"
index.write_text(html)
print("Wrote", path, "and", index, "bytes", path.stat().st_size)
