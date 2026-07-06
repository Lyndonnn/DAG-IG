#!/usr/bin/env python3
"""Compile the review-clean source zip from a fresh extraction.

The review-clean package intentionally excludes internal docs/scripts, so this
verification uses repository-side audit scripts after compiling the extracted
anonymous source bundle.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
CLEAN_ZIP = ASSETS / "DAGIG_Pix2Fact_review_clean_source_bundle.zip"
VERIFY_ROOT = ASSETS / "review_clean_compile_verification"
OUT_JSON = ASSETS / "review_clean_compile_verification.json"
OUT_MD = ASSETS / "REVIEW_CLEAN_COMPILE_VERIFICATION.md"

FORBIDDEN_BUILD_ARTIFACTS = {"main.pdf", "main.aux", "main.bbl", "main.blg", "main.log", "main.out"}


def run(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    return {
        "cmd": " ".join(cmd),
        "cwd": str(cwd) if cwd else None,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def safe_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def verify() -> dict[str, Any]:
    safe_clean(VERIFY_ROOT)
    extract_error = ""
    names: list[str] = []
    try:
        with zipfile.ZipFile(CLEAN_ZIP) as zf:
            bad = zf.testzip()
            if bad:
                extract_error = f"zip test failed at {bad}"
            else:
                names = zf.namelist()
                zf.extractall(VERIFY_ROOT)
    except Exception as exc:  # pragma: no cover - defensive report path
        extract_error = str(exc)

    forbidden_entries = [
        name
        for name in names
        if (Path(name).name in FORBIDDEN_BUILD_ARTIFACTS and len(Path(name).parts) == 1)
        or name.startswith("docs/")
        or name.startswith("scripts/")
        or name.endswith(".json")
    ]

    make_check = make_clean = make_all = post_audit = layout_audit = anonymity_audit = None
    pdf_path = VERIFY_ROOT / "main.pdf"
    if not extract_error:
        make_check = run(["make", "check"], cwd=VERIFY_ROOT)
        make_clean = run(["make", "clean"], cwd=VERIFY_ROOT)
        make_all = run(["make", "all"], cwd=VERIFY_ROOT)
        if make_all["returncode"] == 0 and pdf_path.exists():
            post_audit = run(
                [
                    "python",
                    "scripts/dagig_paper_main/38_post_compile_pdf_audit.py",
                    "--pdf",
                    str(pdf_path),
                    "--output_json",
                    str(VERIFY_ROOT / "post_compile_pdf_audit.review_clean.json"),
                    "--output_md",
                    str(VERIFY_ROOT / "POST_COMPILE_PDF_AUDIT.review_clean.md"),
                    "--require-pass",
                ]
            )
            layout_audit = run(
                [
                    "python",
                    "scripts/dagig_paper_main/40_pdf_layout_audit.py",
                    "--pdf",
                    str(pdf_path),
                    "--log",
                    str(VERIFY_ROOT / "main.log"),
                    "--output_json",
                    str(VERIFY_ROOT / "pdf_layout_audit.review_clean.json"),
                    "--output_md",
                    str(VERIFY_ROOT / "PDF_LAYOUT_AUDIT.review_clean.md"),
                    "--require-pass",
                ]
            )
        anonymity_audit = run(["python", "scripts/dagig_paper_main/37_audit_review_clean_anonymity.py"])

    passed = (
        not extract_error
        and not forbidden_entries
        and make_check is not None
        and make_check["returncode"] == 0
        and make_all is not None
        and make_all["returncode"] == 0
        and post_audit is not None
        and post_audit["returncode"] == 0
        and layout_audit is not None
        and layout_audit["returncode"] == 0
        and anonymity_audit is not None
        and anonymity_audit["returncode"] == 0
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_zip": str(CLEAN_ZIP),
        "extract_root": str(VERIFY_ROOT),
        "entry_count": len(names),
        "extract_error": extract_error,
        "forbidden_entries": forbidden_entries,
        "make_check": make_check,
        "make_clean": make_clean,
        "make_all": make_all,
        "pdf_path": str(pdf_path),
        "post_compile_audit": post_audit,
        "layout_audit": layout_audit,
        "anonymity_audit": anonymity_audit,
        "passed": passed,
    }


def build_report(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Review-Clean Compile Verification\n\n")
    lines.append(
        "This verifies that the anonymous/review-clean source zip can be cleanly extracted, "
        "compiled, rendered-content audited, layout-audited, and anonymity-audited.\n\n"
    )
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- overall pass: `{audit['passed']}`\n")
    lines.append(f"- clean zip: `{audit['clean_zip']}`\n")
    lines.append(f"- extract root: `{audit['extract_root']}`\n")
    lines.append(f"- pdf: `{audit['pdf_path']}`\n")
    lines.append("\n")
    lines.append("## Results\n\n")
    lines.append("| check | returncode / status |\n")
    lines.append("|---|---:|\n")
    lines.append(f"| forbidden entries | `{audit['forbidden_entries']}` |\n")
    for key in ["make_check", "make_all", "post_compile_audit", "layout_audit", "anonymity_audit"]:
        item = audit.get(key)
        lines.append(f"| {key} | `{item.get('returncode') if item else None}` |\n")
    if not audit["passed"]:
        lines.append("\n## Failure Details\n\n")
        for key in ["make_all", "post_compile_audit", "layout_audit", "anonymity_audit"]:
            item = audit.get(key)
            if item and item.get("returncode") != 0:
                lines.append(f"### {key}\n\n```text\n")
                lines.append(item.get("stdout_tail", ""))
                if item.get("stderr_tail"):
                    lines.append("\n--- stderr ---\n")
                    lines.append(item["stderr_tail"])
                lines.append("\n```\n")
    return "".join(lines)


def main() -> None:
    audit = verify()
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_report(audit), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
