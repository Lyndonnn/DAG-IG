#!/usr/bin/env python3
"""Compile cleanly extracted source bundles and run rendered-PDF audits.

This verifies that the deliverable zip/tarball, after extraction into a fresh
directory, is self-contained and can build the paper without relying on the
working `submission_bundle` directory.
"""

from __future__ import annotations

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
VERIFY_ROOT = ASSETS / "source_bundle_compile_verification"
OUT_JSON = ASSETS / "source_bundle_compile_verification.json"
OUT_MD = ASSETS / "SOURCE_BUNDLE_COMPILE_VERIFICATION.md"


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


def extract_zip(path: Path, dest: Path) -> Path:
    safe_clean(dest)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)
    return dest


def extract_tarball(path: Path, dest: Path) -> Path:
    safe_clean(dest)
    with tarfile.open(path, "r:gz") as tf:
        tf.extractall(dest)
    candidates = [p for p in dest.iterdir() if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one tarball root under {dest}, found {len(candidates)}")
    return candidates[0]


def verify_one(label: str, package_path: Path, extract_dest: Path, is_tarball: bool) -> dict[str, Any]:
    extract_error = ""
    root: Path | None = None
    try:
        root = extract_tarball(package_path, extract_dest) if is_tarball else extract_zip(package_path, extract_dest)
    except Exception as exc:  # pragma: no cover - defensive report path
        extract_error = str(exc)

    make_check = make_clean = make_all = post_audit = layout_audit = None
    pdf_path = None
    if root and not extract_error:
        make_check = run(["make", "check"], cwd=root)
        make_clean = run(["make", "clean"], cwd=root)
        make_all = run(["make", "all"], cwd=root)
        pdf_path_obj = root / "main.pdf"
        pdf_path = str(pdf_path_obj)
        if make_all["returncode"] == 0 and pdf_path_obj.exists():
            post_audit = run(
                [
                    "python",
                    "scripts/post_compile_pdf_audit.py",
                    "--pdf",
                    "main.pdf",
                    "--output_json",
                    "post_compile_pdf_audit.clean_extract.json",
                    "--output_md",
                    "POST_COMPILE_PDF_AUDIT.clean_extract.md",
                    "--require-pass",
                ],
                cwd=root,
            )
            layout_audit = run(
                [
                    "python",
                    "scripts/pdf_layout_audit.py",
                    "--pdf",
                    "main.pdf",
                    "--log",
                    "main.log",
                    "--output_json",
                    "pdf_layout_audit.clean_extract.json",
                    "--output_md",
                    "PDF_LAYOUT_AUDIT.clean_extract.md",
                    "--require-pass",
                ],
                cwd=root,
            )

    passed = (
        not extract_error
        and root is not None
        and make_check is not None
        and make_check["returncode"] == 0
        and make_all is not None
        and make_all["returncode"] == 0
        and post_audit is not None
        and post_audit["returncode"] == 0
        and layout_audit is not None
        and layout_audit["returncode"] == 0
    )
    return {
        "label": label,
        "package_path": str(package_path),
        "extract_root": str(root) if root else None,
        "extract_error": extract_error,
        "make_check": make_check,
        "make_clean": make_clean,
        "make_all": make_all,
        "pdf_path": pdf_path,
        "post_compile_audit": post_audit,
        "layout_audit": layout_audit,
        "passed": passed,
    }


def build_report(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Source Bundle Compile Verification\n\n")
    lines.append(
        "This verifies that cleanly extracted zip/tarball deliverables can compile from source "
        "and pass rendered-PDF audits without relying on the working `submission_bundle` directory.\n\n"
    )
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- overall pass: `{audit['overall_pass']}`\n")
    lines.append("\n")
    lines.append("## Results\n\n")
    lines.append("| package | make check | make all | post audit | layout audit | passed |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for key in ["zipfile", "tarball"]:
        item = audit[key]
        lines.append(
            f"| {key} | `{item.get('make_check', {}).get('returncode') if item.get('make_check') else None}` | "
            f"`{item.get('make_all', {}).get('returncode') if item.get('make_all') else None}` | "
            f"`{item.get('post_compile_audit', {}).get('returncode') if item.get('post_compile_audit') else None}` | "
            f"`{item.get('layout_audit', {}).get('returncode') if item.get('layout_audit') else None}` | "
            f"`{item['passed']}` |\n"
        )
    lines.append("\n")
    for key in ["zipfile", "tarball"]:
        item = audit[key]
        lines.append(f"## {key}\n\n")
        lines.append(f"- package: `{item['package_path']}`\n")
        lines.append(f"- extract root: `{item.get('extract_root')}`\n")
        lines.append(f"- pdf: `{item.get('pdf_path')}`\n")
        if item.get("extract_error"):
            lines.append(f"- extract error: `{item['extract_error']}`\n")
        for log_key, title in [
            ("make_all", "make all"),
            ("post_compile_audit", "post audit"),
            ("layout_audit", "layout audit"),
        ]:
            result = item.get(log_key)
            if result and result.get("returncode") != 0:
                lines.append(f"\n### {title} failure tail\n\n```text\n")
                lines.append(result.get("stdout_tail", ""))
                if result.get("stderr_tail"):
                    lines.append("\n--- stderr ---\n")
                    lines.append(result["stderr_tail"])
                lines.append("\n```\n")
        lines.append("\n")
    return "".join(lines)


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_index": str(INDEX),
        "zipfile": verify_one(
            "zipfile",
            Path(index["zipfile"]["path"]),
            VERIFY_ROOT / "zip",
            is_tarball=False,
        ),
        "tarball": verify_one(
            "tarball",
            Path(index["tarball"]["path"]),
            VERIFY_ROOT / "tarball",
            is_tarball=True,
        ),
    }
    audit["overall_pass"] = audit["zipfile"]["passed"] and audit["tarball"]["passed"]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_report(audit), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
