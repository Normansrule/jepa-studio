#!/usr/bin/env python3
"""Turn failed tests in a JUnit XML report into GitHub Actions error annotations.

    python scripts/annotate_junit.py pytest-report.xml

Each failure becomes one `::error` line (test id as title, the first ~40 lines of the failure
as message), so failures show on the run summary page and through the check-runs API without
downloading logs. At most 10 annotations (GitHub's per-step limit for errors).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def esc(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def esc_prop(s: str) -> str:
    """Property values (title=...) must also escape ':' and ','."""
    return esc(s).replace(":", "%3A").replace(",", "%2C")


def main(path: str) -> int:
    root = ET.parse(path).getroot()
    shown = 0
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            el = case.find(kind)
            if el is None:
                continue
            test = f"{case.get('classname', '')}::{case.get('name', '')}"
            body = (el.get("message") or "") + "\n" + (el.text or "")
            lines = [ln for ln in body.splitlines() if ln.strip()]
            msg = "\n".join(lines[-40:])[-3500:]
            print(f"::error title={esc_prop(test)}::{esc(msg)}")
            shown += 1
            if shown >= 10:
                return 0
    print(f"{shown} failing test(s) annotated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "pytest-report.xml"))
