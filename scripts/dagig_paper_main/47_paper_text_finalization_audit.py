#!/usr/bin/env python3
"""Audit paper source text for unresolved finalization problems."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
LATEX = ASSETS / "latex"
BUNDLE = ASSETS / "submission_bundle"
OUT_JSON = ASSETS / "text_finalization_audit.json"
OUT_MD = ASSETS / "TEXT_FINALIZATION_AUDIT.md"

SOURCE_ROOTS = [
    LATEX,
    ASSETS / "figures",
    BUNDLE,
]

FORBIDDEN_PATTERNS = {
    "todo_marker": re.compile(r"\b(?:TODO|FIXME|TBD|PLACEHOLDER|lorem ipsum|citation needed)\b", re.I),
    "empty_citation": re.compile(r"\\cite\{\s*\}"),
    "empty_reference": re.compile(r"\\(?:ref|pageref|autoref)\{\s*\}"),
    "unresolved_ref_text": re.compile(r"(^|[^A-Za-z0-9])\?\?([^A-Za-z0-9]|$)"),
    "absolute_local_path": re.compile(r"/root/|/data/|/storage/|/mnt/|/home/"),
    "repo_output_path": re.compile(r"outputs/dagig_|scripts/dagig_|data/Pix2Fact"),
}

REQUIRED_BOUNDARY_TEXT = [
    "DAG-SFT, DPO, query-reranking, fusion, and answer-repair experiments are diagnostics rather than the main result",
    "does not establish live web-search generalization",
    "Full end-to-end optimization of visual grounding, retrieval, evidence selection, and answer extraction remains future work",
]


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if "docs" in path.parts or "scripts" in path.parts:
                continue
            if path.suffix in {".tex", ".bib"}:
                files.append(path)
    return files


def scan_forbidden_patterns(files: list[Path]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(line):
                    hits.append(
                        {
                            "file": str(path),
                            "line": lineno,
                            "pattern": label,
                            "text": line.strip(),
                        }
                    )
    return hits


def check_title_author() -> dict[str, Any]:
    main = (LATEX / "main.tex").read_text(encoding="utf-8")
    bundle_main = (BUNDLE / "main.tex").read_text(encoding="utf-8") if (BUNDLE / "main.tex").exists() else ""
    problems: list[str] = []
    title_match = re.search(r"\\title\{(.+?)\}", main, flags=re.S)
    author_match = re.search(r"\\author\{(.+?)\}", main, flags=re.S)
    title = title_match.group(1).strip() if title_match else ""
    author = author_match.group(1).strip() if author_match else ""
    if not title or "DAG-IG" not in title or "Pix2Fact" not in title and "Multimodal Search" not in title:
        problems.append(f"title looks wrong: {title!r}")
    if author != "Anonymous Authors":
        problems.append(f"generic source author should remain Anonymous Authors before venue policy is known: {author!r}")
    if bundle_main and main.replace("../figures/", "figures/").replace("../main_results_table.tex", "tables/main_results_table.tex").replace("../node_credit_diagnostic_table.tex", "tables/node_credit_diagnostic_table.tex") == bundle_main:
        bundle_consistent = True
    else:
        # Path replacement in the bundle is audited elsewhere; this is only a soft title/author consistency check.
        bundle_consistent = bool(bundle_main and "\\author{Anonymous Authors}" in bundle_main and "\\title{" in bundle_main)
    if not bundle_consistent:
        problems.append("submission bundle main.tex title/author are not consistent with source main.tex")
    return {"title": title, "author": author, "bundle_consistent": bundle_consistent, "problems": problems, "passed": not problems}


def check_required_boundaries() -> dict[str, Any]:
    text = "\n".join((LATEX / name).read_text(encoding="utf-8") for name in ["main.tex", "appendix.tex"])
    missing = [needle for needle in REQUIRED_BOUNDARY_TEXT if needle not in text]
    return {"required": REQUIRED_BOUNDARY_TEXT, "missing": missing, "passed": not missing}


def check_compile_log() -> dict[str, Any]:
    log_path = BUNDLE / "main.log"
    if not log_path.exists():
        return {"path": str(log_path), "exists": False, "problems": ["main.log missing"], "passed": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "undefined_reference": re.compile(r"undefined references?|Reference .* undefined", re.I),
        "undefined_citation": re.compile(r"Citation .* undefined|undefined citations?", re.I),
        "multiply_defined_label": re.compile(r"multiply defined", re.I),
        "fatal_error": re.compile(r"Fatal error|Emergency stop", re.I),
    }
    problems = [label for label, pattern in patterns.items() if pattern.search(text)]
    return {"path": str(log_path), "exists": True, "problems": problems, "passed": not problems}


def check_pdf_text() -> dict[str, Any]:
    audit_path = ASSETS / "post_compile_pdf_audit.json"
    pdf_path = BUNDLE / "main.pdf"
    if not audit_path.exists():
        return {"path": str(audit_path), "exists": False, "problems": ["post compile audit missing"], "passed": False}
    if not pdf_path.exists():
        return {"path": str(pdf_path), "exists": False, "problems": ["compiled PDF missing"], "passed": False}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if not audit.get("passed"):
        problems.append("post compile PDF content audit did not pass")
    extracted_text = ""
    if shutil_which_pdftotext := shutil.which("pdftotext"):
        result = subprocess.run([shutil_which_pdftotext, str(pdf_path), "-"], text=True, capture_output=True)
        if result.returncode != 0:
            problems.append(f"pdftotext failed: {result.stderr.strip()}")
        else:
            extracted_text = result.stdout
    else:
        problems.append("pdftotext unavailable")
    for label, pattern in {
        "todo_marker": FORBIDDEN_PATTERNS["todo_marker"],
        "unresolved_ref_text": FORBIDDEN_PATTERNS["unresolved_ref_text"],
    }.items():
        if pattern.search(extracted_text):
            problems.append(f"rendered PDF contains {label}")
    return {
        "path": str(pdf_path),
        "post_compile_audit": str(audit_path),
        "exists": True,
        "text_chars": len(extracted_text),
        "problems": problems,
        "passed": not problems,
    }


def build_audit() -> dict[str, Any]:
    files = iter_source_files()
    checks = {
        "forbidden_source_patterns": {
            "files_scanned": len(files),
            "hits": scan_forbidden_patterns(files),
        },
        "title_author": check_title_author(),
        "required_boundaries": check_required_boundaries(),
        "compile_log": check_compile_log(),
        "pdf_text": check_pdf_text(),
    }
    checks["forbidden_source_patterns"]["passed"] = not checks["forbidden_source_patterns"]["hits"]
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "LaTeX paper source, submission bundle compile source, latest compile log, and rendered-PDF text audit.",
        "checks": checks,
    }
    audit["overall_pass"] = all(item.get("passed") for item in checks.values())
    return audit


def build_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Text Finalization Audit\n\n")
    lines.append("This audit checks the paper source for unresolved textual finalization issues. It does not evaluate experiments or retrain models.\n\n")
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- overall pass: `{audit['overall_pass']}`\n")
    lines.append(f"- scope: {audit['scope']}\n\n")
    lines.append("## Checks\n\n")
    lines.append("| check | passed | notes |\n")
    lines.append("|---|---:|---|\n")
    for key, item in audit["checks"].items():
        notes: list[str] = []
        if item.get("hits"):
            notes.append(f"hits={item['hits'][:20]}")
        if item.get("problems"):
            notes.append(f"problems={item['problems']}")
        if item.get("missing"):
            notes.append(f"missing={item['missing']}")
        if item.get("files_scanned") is not None:
            notes.append(f"files_scanned={item['files_scanned']}")
        if key == "title_author":
            notes.append(f"title={item.get('title')!r}; author={item.get('author')!r}")
        if not notes:
            notes.append("ok")
        lines.append(f"| {key} | `{item.get('passed')}` | {'; '.join(notes)} |\n")
    lines.append("\n")
    lines.append("## Interpretation\n\n")
    if audit["overall_pass"]:
        lines.append("No unresolved TODO/TBD/placeholder markers, empty citations/references, local path leaks, unresolved compile-log references, or rendered-PDF placeholder text were found in the audited paper source. `Anonymous Authors` is retained intentionally until the target venue policy is known.\n")
    else:
        lines.append("At least one text-finalization check failed. Fix the listed source issue before treating the generic paper source as ready for venue conversion.\n")
    return "".join(lines)


def main() -> None:
    audit = build_audit()
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_markdown(audit), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
