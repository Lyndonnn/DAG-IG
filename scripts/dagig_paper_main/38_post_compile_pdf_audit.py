#!/usr/bin/env python3
"""Audit the rendered DAG-IG paper PDF after LaTeX compilation.

This is a post-compile guard. Source-level checks can pass while a venue
template still drops a table, hides limitations, or fails to render references.
The script extracts text from the compiled PDF and checks for the main paper
invariants that should be visible to reviewers.
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
DEFAULT_JSON = ASSETS / "post_compile_pdf_audit.json"
DEFAULT_MD = ASSETS / "POST_COMPILE_PDF_AUDIT.md"


def run_pdftotext(pdf_path: Path) -> str | None:
    if shutil.which("pdftotext") is None:
        return None
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "pdftotext failed")
    return result.stdout


def run_python_pdf_reader(pdf_path: Path) -> str | None:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            return None
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_text(pdf_path: Path) -> tuple[str, str]:
    text = run_pdftotext(pdf_path)
    if text is not None:
        return text, "pdftotext"
    text = run_python_pdf_reader(pdf_path)
    if text is not None:
        return text, "python_pdf_reader"
    raise RuntimeError("No PDF text extractor found. Install poppler-utils, pypdf, or PyPDF2.")


def normalize(text: str) -> str:
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text.lower())
    return text.strip()


def has_all(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    missing = [needle for needle in needles if normalize(needle) not in text]
    return not missing, missing


def has_any(text: str, groups: list[list[str]]) -> tuple[bool, list[list[str]]]:
    missing = [group for group in groups if not any(normalize(needle) in text for needle in group)]
    return not missing, missing


def check_regex(text: str, patterns: list[str]) -> tuple[bool, list[str]]:
    missing = [pattern for pattern in patterns if re.search(pattern, text) is None]
    return not missing, missing


def build_checks(norm_text: str, raw_text: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    checks["title_and_abstract"] = dict(
        zip(
            ["passed", "missing"],
            has_all(
                norm_text,
                [
                    "DAG-IG",
                    "Node-Level Counterfactual Credit Assignment",
                    "multimodal search agents",
                    "Format-SFT",
                ],
            ),
        )
    )

    schema_passed, schema_missing = has_all(
        norm_text,
        [
            "visual observation",
            "search query",
            "final answer",
            "strict success",
        ],
    )
    evidence_passed, evidence_missing = has_any(norm_text, [["top-k evidence", "top-5 evidence", "top-5 documents"]])
    checks["rollout_schema"] = {
        "passed": schema_passed and evidence_passed,
        "missing": schema_missing + ["/".join(group) for group in evidence_missing],
    }

    checks["main_result_table"] = dict(
        zip(
            ["passed", "missing"],
            has_all(
                norm_text,
                [
                    "DAG-IG seed42 main",
                    "DAG-IG seed43 confirm",
                    "Goldfixed control",
                    "42.9",
                    "49.0",
                    "34.4",
                    "40.6",
                    "57.1",
                    "51.6",
                ],
            ),
        )
    )

    checks["reward_table"] = dict(
        zip(
            ["passed", "missing"],
            has_all(
                norm_text,
                [
                    "reward audit",
                    "seed42 main",
                    "seed43 confirm",
                    "goldfixed control",
                    "0.974",
                    "0.984",
                    "0.960",
                ],
            ),
        )
    )

    checks["limitations"] = dict(
        zip(
            ["passed", "missing"],
            has_all(
                norm_text,
                [
                    "offline",
                    "frozen BM25 corpus",
                    "live web-search generalization",
                    "modest",
                    "remaining bottleneck",
                ],
            ),
        )
    )

    checks["diagnostic_boundaries"] = dict(
        zip(
            ["passed", "missing"],
            has_all(
                norm_text,
                [
                    "DAG-SFT",
                    "diagnostic",
                    "query reranking",
                    "answer repair",
                    "not the main method",
                ],
            ),
        )
    )

    checks["references_rendered"] = dict(
        zip(
            ["passed", "missing"],
            has_all(
                norm_text,
                [
                    "references",
                    "retrieval-augmented",
                    "react",
                    "qwen",
                ],
            ),
        )
    )

    bad_citation_patterns = [r"\[\?\]", r"citation undefined", r"undefined citation"]
    bad_citations = [pattern for pattern in bad_citation_patterns if re.search(pattern, norm_text)]
    checks["no_obvious_citation_failures"] = {
        "passed": not bad_citations,
        "bad_patterns": bad_citations,
    }

    checks["pdf_text_nontrivial"] = {
        "passed": len(raw_text.strip()) > 5000,
        "text_chars": len(raw_text.strip()),
    }

    return checks


def build_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Post-Compile PDF Audit\n\n")
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- pdf_path: `{audit['pdf_path']}`\n")
    lines.append(f"- extractor: `{audit.get('extractor')}`\n")
    lines.append(f"- overall pass: `{audit['passed']}`\n")
    lines.append("\n")
    if audit.get("error"):
        lines.append("## Error\n\n")
        lines.append(f"```text\n{audit['error']}\n```\n\n")
    lines.append("## Checks\n\n")
    lines.append("| check | passed | details |\n")
    lines.append("|---|---:|---|\n")
    for name, result in audit.get("checks", {}).items():
        details = []
        if result.get("missing"):
            details.append("missing: " + ", ".join(result["missing"]))
        if result.get("bad_patterns"):
            details.append("bad patterns: " + ", ".join(result["bad_patterns"]))
        if "text_chars" in result:
            details.append(f"text chars: {result['text_chars']}")
        lines.append(f"| {name} | `{result.get('passed')}` | {'; '.join(details) or 'ok'} |\n")
    lines.append("\n")
    lines.append("## Interpretation\n\n")
    if audit["passed"]:
        lines.append(
            "The rendered PDF text contains the main result numbers, reward diagnostics, "
            "rollout schema, limitation language, diagnostic-branch boundaries, and references.\n"
        )
    else:
        lines.append(
            "The rendered PDF did not pass all checks. Inspect the missing items above before "
            "submitting or converting to the final venue template.\n"
        )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    audit: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf_path": str(args.pdf),
        "passed": False,
        "checks": {},
    }
    try:
        if not args.pdf.exists():
            raise FileNotFoundError(f"PDF not found: {args.pdf}")
        if args.pdf.stat().st_size == 0:
            raise RuntimeError(f"PDF is empty: {args.pdf}")
        raw_text, extractor = extract_text(args.pdf)
        norm_text = normalize(raw_text)
        checks = build_checks(norm_text, raw_text)
        audit["extractor"] = extractor
        audit["checks"] = checks
        audit["passed"] = all(result.get("passed") for result in checks.values())
    except Exception as exc:
        audit["error"] = str(exc)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(audit), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if args.require_pass and not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
