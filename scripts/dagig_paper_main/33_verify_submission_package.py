#!/usr/bin/env python3
"""Verify the outer submission zip/tarball by checksum, extraction, and make check."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
INDEX = ASSETS / "SUBMISSION_PACKAGE_INDEX.json"
VERIFY_ROOT = ASSETS / "package_extract_verification"
OUT_JSON = ASSETS / "package_extract_verification.json"
OUT_MD = ASSETS / "PACKAGE_EXTRACT_VERIFICATION_REPORT.md"
FORBIDDEN_BUILD_ARTIFACTS = {"main.pdf", "main.aux", "main.bbl", "main.blg", "main.log", "main.out"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    return {
        "cmd": " ".join(cmd),
        "cwd": str(cwd) if cwd else None,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def safe_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def verify_zip(path: Path, expected_sha: str) -> dict[str, Any]:
    extract_root = VERIFY_ROOT / "zip"
    safe_clean(extract_root)
    actual_sha = sha256_file(path)
    namelist: list[str] = []
    extract_error = ""
    make_check: dict[str, Any] | None = None
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                extract_error = f"zip test failed at {bad}"
            else:
                namelist = zf.namelist()
                zf.extractall(extract_root)
    except Exception as exc:  # pragma: no cover - defensive report path
        extract_error = str(exc)
    if not extract_error:
        make_check = run(["make", "check"], cwd=extract_root)
    forbidden_entries = [name for name in namelist if Path(name).name in FORBIDDEN_BUILD_ARTIFACTS and len(Path(name).parts) == 1]
    passed = (
        actual_sha == expected_sha
        and not extract_error
        and make_check is not None
        and make_check["returncode"] == 0
        and not forbidden_entries
    )
    return {
        "path": str(path),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "sha_match": actual_sha == expected_sha,
        "extract_root": str(extract_root),
        "entry_count": len(namelist),
        "required_entries_present": all(
            entry in namelist
            for entry in [
                "main.tex",
                "README.md",
                "docs/FINAL_HANDOFF_PROMPT.md",
                "docs/GOAL_COMPLETION_AUDIT.md",
                "docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.md",
                "docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.json",
                "docs/SUBMISSION_ROUTE_GUIDE.md",
                "docs/VENUE_DECISION_AUDIT.md",
                "docs/FINAL_SUBMISSION_GATE.md",
                "tables/main_results_table.tex",
                "figures/dagig_method_diagram.tex",
                "scripts/prepare_venue_workspace.py",
                "scripts/pdf_layout_audit.py",
                "scripts/audit_venue_decision_form.py",
                "scripts/audit_final_submission_gate.py",
            ]
        ),
        "forbidden_build_artifacts": forbidden_entries,
        "extract_error": extract_error,
        "make_check": make_check,
        "passed": passed,
    }


def verify_tarball(path: Path, expected_sha: str) -> dict[str, Any]:
    extract_root = VERIFY_ROOT / "tarball"
    safe_clean(extract_root)
    actual_sha = sha256_file(path)
    names: list[str] = []
    extract_error = ""
    make_check: dict[str, Any] | None = None
    bundle_root: Path | None = None
    try:
        with tarfile.open(path, "r:gz") as tf:
            names = tf.getnames()
            tf.extractall(extract_root)
    except Exception as exc:  # pragma: no cover - defensive report path
        extract_error = str(exc)
    if not extract_error:
        candidates = [p for p in extract_root.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            bundle_root = candidates[0]
            make_check = run(["make", "check"], cwd=bundle_root)
        else:
            extract_error = f"expected one extracted root directory, found {len(candidates)}"
    forbidden_entries = [
        name
        for name in names
        if Path(name).name in FORBIDDEN_BUILD_ARTIFACTS and Path(name).parent == Path("DAGIG_Pix2Fact_paper_source_bundle")
    ]
    passed = (
        actual_sha == expected_sha
        and not extract_error
        and make_check is not None
        and make_check["returncode"] == 0
        and not forbidden_entries
    )
    return {
        "path": str(path),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "sha_match": actual_sha == expected_sha,
        "extract_root": str(extract_root),
        "bundle_root": str(bundle_root) if bundle_root else None,
        "entry_count": len(names),
        "required_entries_present": all(
            entry in names
            for entry in [
                "DAGIG_Pix2Fact_paper_source_bundle/main.tex",
                "DAGIG_Pix2Fact_paper_source_bundle/README.md",
                "DAGIG_Pix2Fact_paper_source_bundle/docs/FINAL_HANDOFF_PROMPT.md",
                "DAGIG_Pix2Fact_paper_source_bundle/docs/GOAL_COMPLETION_AUDIT.md",
                "DAGIG_Pix2Fact_paper_source_bundle/docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.md",
                "DAGIG_Pix2Fact_paper_source_bundle/docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.json",
                "DAGIG_Pix2Fact_paper_source_bundle/docs/SUBMISSION_ROUTE_GUIDE.md",
                "DAGIG_Pix2Fact_paper_source_bundle/docs/VENUE_DECISION_AUDIT.md",
                "DAGIG_Pix2Fact_paper_source_bundle/docs/FINAL_SUBMISSION_GATE.md",
                "DAGIG_Pix2Fact_paper_source_bundle/tables/main_results_table.tex",
                "DAGIG_Pix2Fact_paper_source_bundle/figures/dagig_method_diagram.tex",
                "DAGIG_Pix2Fact_paper_source_bundle/scripts/prepare_venue_workspace.py",
                "DAGIG_Pix2Fact_paper_source_bundle/scripts/pdf_layout_audit.py",
                "DAGIG_Pix2Fact_paper_source_bundle/scripts/audit_venue_decision_form.py",
                "DAGIG_Pix2Fact_paper_source_bundle/scripts/audit_final_submission_gate.py",
            ]
        ),
        "forbidden_build_artifacts": forbidden_entries,
        "extract_error": extract_error,
        "make_check": make_check,
        "passed": passed,
    }


def build_report(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Package Extract Verification Report\n\n")
    lines.append("This verifies the outer zip/tarball packages by checksum, extraction, required entries, and `make check`.\n\n")
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- overall pass: `{audit['overall_pass']}`\n")
    lines.append(f"- package index: `{audit['package_index']}`\n")
    lines.append("\n")
    lines.append("## Results\n\n")
    lines.append("| package | sha match | required entries | no build artifacts | make check | passed |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for key in ["zipfile", "tarball"]:
        item = audit[key]
        make_rc = item.get("make_check", {}).get("returncode") if item.get("make_check") else None
        lines.append(
            f"| {key} | `{item['sha_match']}` | `{item['required_entries_present']}` | "
            f"`{not item.get('forbidden_build_artifacts')}` | `{make_rc}` | `{item['passed']}` |\n"
        )
    lines.append("\n")
    for key in ["zipfile", "tarball"]:
        item = audit[key]
        lines.append(f"## {key}\n\n")
        lines.append(f"- path: `{item['path']}`\n")
        lines.append(f"- expected sha256: `{item['expected_sha256']}`\n")
        lines.append(f"- actual sha256: `{item['actual_sha256']}`\n")
        lines.append(f"- extract root: `{item['extract_root']}`\n")
        if item.get("extract_error"):
            lines.append(f"- extract error: `{item['extract_error']}`\n")
        if item.get("forbidden_build_artifacts"):
            lines.append(f"- forbidden build artifacts: `{item['forbidden_build_artifacts']}`\n")
        if item.get("make_check"):
            lines.append(f"- make check stdout: `{item['make_check'].get('stdout')}`\n")
            if item["make_check"].get("stderr"):
                lines.append(f"- make check stderr: `{item['make_check'].get('stderr')}`\n")
        lines.append("\n")
    return "".join(lines)


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    zip_info = index["zipfile"]
    tar_info = index["tarball"]
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_index": str(INDEX),
        "zipfile": verify_zip(Path(zip_info["path"]), zip_info["sha256"]),
        "tarball": verify_tarball(Path(tar_info["path"]), tar_info["sha256"]),
    }
    audit["overall_pass"] = audit["zipfile"]["passed"] and audit["tarball"]["passed"]
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_report(audit), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
