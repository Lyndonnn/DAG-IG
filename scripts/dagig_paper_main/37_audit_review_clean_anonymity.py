#!/usr/bin/env python3
"""Audit the review-clean source bundle for anonymity and local-path leaks."""

from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
CLEAN_BUNDLE = ASSETS / "review_clean_bundle"
CLEAN_ZIP = ASSETS / "DAGIG_Pix2Fact_review_clean_source_bundle.zip"
OUT_JSON = ASSETS / "REVIEW_CLEAN_ANONYMITY_AUDIT.json"
OUT_MD = ASSETS / "REVIEW_CLEAN_ANONYMITY_AUDIT.md"


FORBIDDEN_PATTERNS = {
    "absolute_root_path": r"/root/",
    "autodl_path": r"autodl",
    "workspace_name": r"search-test",
    "user_name": r"zhengxiang",
    "github_repo": r"github\.com/Lyndonnn",
    "google_drive": r"drive\.google\.com",
    "internal_outputs_path": r"outputs/dagig",
    "internal_data_path": r"data/Pix2Fact",
    "codex_attachment": r"\.codex/attachments",
}

HUMAN_SUFFIXES = {".tex", ".bib", ".md"}


def scan_text(name: str, text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if re.search(pattern, line, flags=re.I):
                hits.append(
                    {
                        "file": name,
                        "line": lineno,
                        "pattern": label,
                        "text": line.strip(),
                    }
                )
    return hits


def audit_bundle_files() -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    bad_names: list[str] = []
    scanned_files = 0
    for path in sorted(p for p in CLEAN_BUNDLE.rglob("*") if p.is_file()):
        rel = str(path.relative_to(CLEAN_BUNDLE))
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if re.search(pattern, rel, flags=re.I):
                bad_names.append(rel)
        if path.suffix in HUMAN_SUFFIXES:
            scanned_files += 1
            hits.extend(scan_text(rel, path.read_text(encoding="utf-8")))
    return {
        "scanned_files": scanned_files,
        "forbidden_text_hits": hits,
        "forbidden_filename_hits": sorted(set(bad_names)),
        "passed": not hits and not bad_names,
    }


def audit_zip_entries() -> dict[str, Any]:
    if not CLEAN_ZIP.exists():
        return {"passed": False, "error": f"missing {CLEAN_ZIP}"}
    hits: list[dict[str, Any]] = []
    bad_names: list[str] = []
    disallowed_entries: list[str] = []
    scanned_files = 0
    with zipfile.ZipFile(CLEAN_ZIP) as zf:
        names = zf.namelist()
        for name in names:
            if name.startswith("docs/") or name.startswith("scripts/") or name.endswith(".json"):
                disallowed_entries.append(name)
            for label, pattern in FORBIDDEN_PATTERNS.items():
                if re.search(pattern, name, flags=re.I):
                    bad_names.append(name)
            suffix = Path(name).suffix
            if suffix in HUMAN_SUFFIXES:
                scanned_files += 1
                hits.extend(scan_text(name, zf.read(name).decode("utf-8")))
    return {
        "scanned_files": scanned_files,
        "forbidden_text_hits": hits,
        "forbidden_filename_hits": sorted(set(bad_names)),
        "disallowed_entries": disallowed_entries,
        "passed": not hits and not bad_names and not disallowed_entries,
    }


def build_report(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Review-Clean Anonymity Audit\n\n")
    lines.append("- This audit checks the review-clean source bundle, not the full handoff bundle.\n")
    lines.append(f"- overall pass: `{audit['overall_pass']}`\n")
    lines.append(f"- clean bundle: `{audit['clean_bundle']}`\n")
    lines.append(f"- clean zip: `{audit['clean_zip']}`\n")
    lines.append("\n")
    lines.append("## Bundle Directory\n\n")
    bundle = audit["bundle_directory"]
    lines.append(f"- scanned human-facing files: `{bundle['scanned_files']}`\n")
    lines.append(f"- forbidden filename hits: `{bundle['forbidden_filename_hits']}`\n")
    lines.append(f"- forbidden text hit count: `{len(bundle['forbidden_text_hits'])}`\n")
    lines.append("\n")
    lines.append("## Zip Package\n\n")
    zip_audit = audit["zip_package"]
    lines.append(f"- scanned human-facing files: `{zip_audit.get('scanned_files')}`\n")
    lines.append(f"- forbidden filename hits: `{zip_audit.get('forbidden_filename_hits')}`\n")
    lines.append(f"- forbidden text hit count: `{len(zip_audit.get('forbidden_text_hits', []))}`\n")
    lines.append(f"- disallowed entries: `{zip_audit.get('disallowed_entries')}`\n")
    if not audit["overall_pass"]:
        lines.append("\n## Hits\n\n")
        for section_name in ["bundle_directory", "zip_package"]:
            section = audit[section_name]
            for hit in section.get("forbidden_text_hits", []):
                lines.append(f"- {section_name}: `{hit['file']}:{hit['line']}` {hit['pattern']} -> `{hit['text']}`\n")
    return "".join(lines)


def main() -> None:
    bundle = audit_bundle_files()
    zip_audit = audit_zip_entries()
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_bundle": str(CLEAN_BUNDLE),
        "clean_zip": str(CLEAN_ZIP),
        "forbidden_patterns": FORBIDDEN_PATTERNS,
        "bundle_directory": bundle,
        "zip_package": zip_audit,
        "overall_pass": bundle["passed"] and zip_audit["passed"],
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_report(audit), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
