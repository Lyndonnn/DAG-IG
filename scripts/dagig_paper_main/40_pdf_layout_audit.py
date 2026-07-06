#!/usr/bin/env python3
"""Audit compiled PDF layout metadata and LaTeX log risks.

The post-compile content audit verifies that important claims render in the PDF.
This companion audit checks mechanical layout signals: page count, page size,
encryption status, text extraction by page, and LaTeX warning patterns.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
DEFAULT_PDF = ASSETS / "submission_bundle/main.pdf"
DEFAULT_LOG = ASSETS / "submission_bundle/main.log"
DEFAULT_JSON = ASSETS / "pdf_layout_audit.json"
DEFAULT_MD = ASSETS / "PDF_LAYOUT_AUDIT.md"


WARNING_PATTERNS = [
    r"Warning",
    r"Overfull",
    r"Underfull",
    r"Undefined",
    r"undefined",
]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def parse_pdfinfo(pdf_path: Path) -> dict[str, Any]:
    if shutil.which("pdfinfo") is None:
        return {"available": False, "error": "pdfinfo unavailable"}
    result = run(["pdfinfo", str(pdf_path)])
    info: dict[str, Any] = {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        return info
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    info["parsed"] = parsed
    pages = parsed.get("Pages")
    info["pages"] = int(pages) if pages and pages.isdigit() else None
    info["encrypted"] = parsed.get("Encrypted")
    info["page_size"] = parsed.get("Page size")
    info["pdf_version"] = parsed.get("PDF version")
    info["file_size"] = parsed.get("File size")
    return info


def extract_page_text_lengths(pdf_path: Path) -> dict[str, Any]:
    if shutil.which("pdftotext") is None:
        return {"available": False, "error": "pdftotext unavailable"}
    result = run(["pdftotext", "-layout", str(pdf_path), "-"])
    info: dict[str, Any] = {
        "available": True,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        return info
    pages = result.stdout.split("\f")
    if pages and pages[-1].strip() == "":
        pages = pages[:-1]
    lengths = [len(page.strip()) for page in pages]
    info["page_count_by_text"] = len(lengths)
    info["page_text_lengths"] = lengths
    info["empty_pages"] = [idx + 1 for idx, length in enumerate(lengths) if length == 0]
    return info


def scan_latex_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"exists": False, "warning_lines": [], "output_pages": None, "output_bytes": None}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    warning_lines: list[dict[str, Any]] = []
    patterns = [re.compile(pattern) for pattern in WARNING_PATTERNS]
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in patterns):
            warning_lines.append({"line": lineno, "text": line.strip()})
    output_pages = None
    output_bytes = None
    match = re.search(r"Output written on .+?\((\d+) pages?, ([0-9]+) bytes\)", text)
    if match:
        output_pages = int(match.group(1))
        output_bytes = int(match.group(2))
    return {
        "exists": True,
        "warning_patterns": WARNING_PATTERNS,
        "warning_lines": warning_lines,
        "output_pages": output_pages,
        "output_bytes": output_bytes,
    }


def build_audit(pdf_path: Path, log_path: Path, max_pages: int | None) -> dict[str, Any]:
    pdf_exists = pdf_path.exists() and pdf_path.stat().st_size > 0
    pdfinfo = parse_pdfinfo(pdf_path) if pdf_exists else {"available": shutil.which("pdfinfo") is not None}
    text_pages = extract_page_text_lengths(pdf_path) if pdf_exists else {"available": shutil.which("pdftotext") is not None}
    log = scan_latex_log(log_path)

    pages = pdfinfo.get("pages")
    checks: dict[str, dict[str, Any]] = {}
    checks["pdf_exists"] = {
        "passed": pdf_exists,
        "details": f"{pdf_path.stat().st_size} bytes" if pdf_exists else "missing or empty",
    }
    checks["pdfinfo_pages"] = {
        "passed": isinstance(pages, int) and pages > 0,
        "details": pages,
    }
    checks["pdf_not_encrypted"] = {
        "passed": str(pdfinfo.get("encrypted", "")).lower().startswith("no"),
        "details": pdfinfo.get("encrypted"),
    }
    checks["page_size_present"] = {
        "passed": bool(pdfinfo.get("page_size")),
        "details": pdfinfo.get("page_size"),
    }
    if max_pages is not None:
        checks["max_pages"] = {
            "passed": isinstance(pages, int) and pages <= max_pages,
            "details": f"pages={pages}, max_pages={max_pages}",
        }
    checks["text_pages_match_pdfinfo"] = {
        "passed": (
            isinstance(pages, int)
            and text_pages.get("returncode") == 0
            and text_pages.get("page_count_by_text") == pages
        ),
        "details": f"text_pages={text_pages.get('page_count_by_text')}, pdfinfo_pages={pages}",
    }
    checks["no_empty_text_pages"] = {
        "passed": text_pages.get("returncode") == 0 and not text_pages.get("empty_pages"),
        "details": text_pages.get("empty_pages", []),
    }
    checks["latex_log_exists"] = {
        "passed": bool(log.get("exists")),
        "details": str(log_path),
    }
    checks["latex_log_no_warning_patterns"] = {
        "passed": bool(log.get("exists")) and not log.get("warning_lines"),
        "details": f"{len(log.get('warning_lines', []))} warning-pattern lines",
    }
    checks["latex_log_pages_match_pdfinfo"] = {
        "passed": isinstance(pages, int) and log.get("output_pages") == pages,
        "details": f"log_pages={log.get('output_pages')}, pdfinfo_pages={pages}",
    }

    passed = all(item["passed"] for item in checks.values())
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf_path": str(pdf_path),
        "log_path": str(log_path),
        "max_pages": max_pages,
        "pdfinfo": pdfinfo,
        "text_pages": text_pages,
        "latex_log": log,
        "checks": checks,
        "passed": passed,
    }


def build_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# PDF Layout Audit\n\n")
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- pdf_path: `{audit['pdf_path']}`\n")
    lines.append(f"- log_path: `{audit['log_path']}`\n")
    lines.append(f"- overall pass: `{audit['passed']}`\n")
    lines.append("\n")
    pdfinfo = audit.get("pdfinfo", {})
    lines.append("## PDF Metadata\n\n")
    lines.append(f"- pages: `{pdfinfo.get('pages')}`\n")
    lines.append(f"- page size: `{pdfinfo.get('page_size')}`\n")
    lines.append(f"- encrypted: `{pdfinfo.get('encrypted')}`\n")
    lines.append(f"- pdf version: `{pdfinfo.get('pdf_version')}`\n")
    lines.append(f"- file size: `{pdfinfo.get('file_size')}`\n")
    lines.append("\n")
    lines.append("## Checks\n\n")
    lines.append("| check | passed | details |\n")
    lines.append("|---|---:|---|\n")
    for name, result in audit["checks"].items():
        lines.append(f"| {name} | `{result['passed']}` | `{result.get('details')}` |\n")
    warning_lines = audit.get("latex_log", {}).get("warning_lines", [])
    if warning_lines:
        lines.append("\n## LaTeX Warning Pattern Lines\n\n")
        for item in warning_lines[:80]:
            lines.append(f"- line {item['line']}: `{item['text']}`\n")
        if len(warning_lines) > 80:
            lines.append(f"- ... {len(warning_lines) - 80} more lines omitted\n")
    lines.append("\n## Interpretation\n\n")
    if audit["passed"]:
        lines.append(
            "The compiled generic PDF has valid page metadata, extractable text on every page, "
            "matching PDF/log page counts, and no LaTeX warning-pattern lines.\n"
        )
    else:
        lines.append(
            "The compiled PDF has at least one layout or log-risk issue. Inspect the failed checks "
            "before treating the PDF as submission-ready.\n"
        )
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args.pdf, args.log, args.max_pages)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.write_text(build_markdown(audit), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if args.require_pass and not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
