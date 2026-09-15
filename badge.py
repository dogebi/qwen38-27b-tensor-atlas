#!/usr/bin/env python3
"""Add the NTT internal badge + copyright footer to every atlas page (idempotent).

matches the reference implementation already live on glm53-flash-atlas:
  CSS appended at the end of the head's <style>, badge right after <body>,
  footer <p> right before </body>.
"""
from __future__ import annotations

import pathlib
import sys

CSS = """
    .internal-badge {
        position: fixed;
        top: 20px;
        right: -60px;
        width: 200px;
        text-align: center;
        background: #ffeb3b;
        color: #d32f2f;
        font-weight: bold;
        font-size: 16px;
        padding: 8px 0;
        transform: rotate(45deg);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        pointer-events: none;
    }
    .footer{ 
      text-align: center; 
      margin-top: 1.5rem; 
      color: var(--text-faint); 
      font-size: 0.75rem; 
    }
"""
BADGE = '<div class="internal-badge">NTT社内用</div>\n'
FOOTER = '<p class="footer">Copyright NTT Communications China 2026</p>\n'


def inject(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "internal-badge" in text:
        return "already present"
    if "</body>" not in text:
        return "SKIPPED: no </body>"
    # 1. CSS at the end of the last <style> block in the head
    head_end = text.find("</head>")
    style_end = text.rfind("</style>", 0, head_end if head_end > 0 else len(text))
    if style_end > 0:
        text = text[:style_end] + CSS + text[style_end:]
    else:
        text = text.replace("</head>", f"<style>{CSS}</style>\n</head>", 1)
    # 2. badge right after <body>
    i = text.find("<body>")
    if i < 0:
        return "SKIPPED: no <body>"
    text = text[:i + len("<body>")] + "\n  " + BADGE + text[i + len("<body>"):]
    # 3. footer before the last </body>
    j = text.rfind("</body>")
    text = text[:j] + FOOTER + text[j:]
    path.write_text(text, encoding="utf-8")
    return "injected"


BUILDER_HOOK = '''

    # the NTT internal badge + copyright footer are part of the published page
    import badge
    print("badge ·", badge.inject(OUT))
    return 0
'''


def hook_builder(d: pathlib.Path) -> str:
    b = d / "build.py"
    if not b.exists():
        return "no builder"
    s = b.read_text(encoding="utf-8")
    if "import badge" in s:
        return "hook present"
    marker = "\n    return 0\n\n\nif __name__"
    if marker not in s:
        return "hook point not found"
    s = s.replace(marker, BUILDER_HOOK + "\n\nif __name__", 1)
    b.write_text(s, encoding="utf-8")
    (d / "badge.py").write_text(pathlib.Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    return "hook added"


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = pathlib.Path(arg)
        target = p / "index.html"
        print(f"{p.name:32s} page={inject(target) if target.exists() else 'no index.html'} · builder={hook_builder(p)}")
