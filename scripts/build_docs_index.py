#!/usr/bin/env python3
"""Build site/docs-index.json: the passages the web "Explain" panel searches (BM25, in the browser).

    python scripts/build_docs_index.py            # writes site/docs-index.json
    python scripts/build_docs_index.py --check    # exit 1 if the committed file is stale (CI)

Each docs/*.md file (plus README.md when present) is split at headings into passages of at
most ~1200 characters. Code fences are kept as plain text, Markdown link syntax is reduced
to its text, and nothing is interpreted: the browser shows passages with textContent only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "docs-index.json"
REPO_URL = "https://github.com/Normansrule/jepa-studio/blob/main/"
MAX_CHARS = 1200


def slug(text: str) -> str:
    s = re.sub(r"[^\w\- ]", "", text.lower()).strip().replace(" ", "-")
    return re.sub(r"-+", "-", s)


def clean(md: str) -> str:
    md = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", md)       # images -> alt text
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)        # links -> text
    md = re.sub(r"</?(?:br|p|div|span|details|summary|sub|sup|img|a|b|i|em|strong|kbd|code)\b[^>]*>", "", md,
                flags=re.I)                                  # common inline HTML (not <id>-style placeholders)
    md = re.sub(r"^\s*```.*$", "", md, flags=re.M)           # fence markers
    md = re.sub(r"(?<![\w*])(\*{1,3}|_{1,3})([^*_\n]+?)\1(?![\w*])", r"\2", md)  # emphasis, not snake_case
    md = re.sub(r"`([^`\n]+)`", r"\1", md)                   # inline code marks
    md = re.sub(r"^\s*\|?\s*-{3,}.*$", "", md, flags=re.M)   # table rules
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def split_passages(path: Path) -> list[dict]:
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    out: list[dict] = []
    heading, anchor, buf = path.stem, "", []
    in_fence = False

    def flush():
        body = clean("\n".join(buf))
        if not body:
            return
        paras = re.split(r"\n\s*\n", body)
        chunk = ""
        for p in paras:
            if chunk and len(chunk) + len(p) > MAX_CHARS:
                out.append({"file": rel, "heading": heading, "text": chunk.strip(), "url": REPO_URL + rel + anchor})
                chunk = ""
            chunk += p + "\n\n"
        if chunk.strip():
            out.append({"file": rel, "heading": heading, "text": chunk.strip()[: MAX_CHARS * 2], "url": REPO_URL + rel + anchor})

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush()
            buf = []
            heading = m.group(2).strip().strip("#").strip()
            anchor = "#" + slug(heading)
        else:
            buf.append(line)
    flush()
    return out


def build() -> dict:
    files = sorted((ROOT / "docs").glob("*.md"))
    readme = ROOT / "README.md"
    if readme.exists():
        files.insert(0, readme)
    passages = []
    for f in files:
        passages += split_passages(f)
    return {"version": 1, "source": "docs/*.md", "files": [f.relative_to(ROOT).as_posix() for f in files],
            "passages": passages}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if site/docs-index.json is out of date")
    args = ap.parse_args()
    data = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != data:
            print("site/docs-index.json is stale; run scripts/build_docs_index.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(data, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(json.loads(data)['passages'])} passages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
