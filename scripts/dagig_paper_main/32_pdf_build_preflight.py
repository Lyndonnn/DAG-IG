#!/usr/bin/env python3
"""Preflight and optional PDF build for the DAG-IG paper source bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
BUNDLE = ASSETS / "submission_bundle"
OUT_JSON = ASSETS / "pdf_build_preflight.json"
OUT_MD = ASSETS / "PDF_BUILD_PREFLIGHT_REPORT.md"


def run(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    return {
        "cmd": " ".join(cmd),
        "cwd": str(cwd) if cwd else None,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def build_report(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# PDF Build Preflight Report\n\n")
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- bundle_root: `{audit['bundle_root']}`\n")
    lines.append(f"- make available: `{audit['tools']['make']}`\n")
    lines.append(f"- pdflatex available: `{audit['tools']['pdflatex']}`\n")
    lines.append(f"- bibtex available: `{audit['tools']['bibtex']}`\n")
    lines.append(f"- source check passed: `{audit['source_check_passed']}`\n")
    lines.append(f"- pdf compilation attempted: `{audit['pdf_compilation_attempted']}`\n")
    lines.append(f"- pdf compilation passed: `{audit['pdf_compilation_passed']}`\n")
    lines.append("\n")

    lines.append("## Source Check\n\n")
    check = audit["make_check"]
    lines.append(f"- command: `{check.get('cmd')}`\n")
    lines.append(f"- returncode: `{check.get('returncode')}`\n")
    if check.get("stdout"):
        lines.append(f"- stdout: `{check['stdout']}`\n")
    if check.get("stderr"):
        lines.append(f"- stderr: `{check['stderr']}`\n")
    lines.append("\n")

    lines.append("## PDF Build\n\n")
    if audit["pdf_compilation_passed"]:
        lines.append("PDF compilation succeeded. The built PDF is:\n\n")
        lines.append(f"- `{audit['pdf_path']}`\n")
    elif not audit["pdf_compilation_attempted"]:
        lines.append(
            "PDF compilation was not attempted because this environment lacks "
            "`pdflatex` and/or `bibtex`, or `--compile` was not requested. "
            "Use a TeX-enabled environment or Overleaf and run:\n\n"
        )
        lines.append("```bash\ncd submission_bundle\nmake check\nmake all\n```\n")
    else:
        lines.append("PDF compilation was attempted but failed. Inspect the command logs below.\n")
    lines.append("\n")

    if audit.get("make_all"):
        lines.append("## make all Log\n\n")
        build = audit["make_all"]
        lines.append(f"- command: `{build.get('cmd')}`\n")
        lines.append(f"- returncode: `{build.get('returncode')}`\n")
        if build.get("stdout"):
            lines.append("\n### stdout\n\n```text\n")
            lines.append(build["stdout"][-4000:])
            lines.append("\n```\n")
        if build.get("stderr"):
            lines.append("\n### stderr\n\n```text\n")
            lines.append(build["stderr"][-4000:])
            lines.append("\n```\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Run make all if TeX tools are available.")
    parser.add_argument("--require-pdf", action="store_true", help="Exit nonzero if PDF compilation is not verified.")
    args = parser.parse_args()

    tools = {
        "make": shutil.which("make") is not None,
        "pdflatex": shutil.which("pdflatex") is not None,
        "bibtex": shutil.which("bibtex") is not None,
    }
    make_check = run(["make", "check"], cwd=BUNDLE) if tools["make"] and BUNDLE.exists() else {
        "cmd": "make check",
        "cwd": str(BUNDLE),
        "returncode": None,
        "stdout": "",
        "stderr": "make unavailable or bundle missing",
    }

    can_compile = tools["make"] and tools["pdflatex"] and tools["bibtex"] and make_check["returncode"] == 0
    make_all = None
    if args.compile and can_compile:
        run(["make", "clean"], cwd=BUNDLE)
        make_all = run(["make", "all"], cwd=BUNDLE)

    pdf_path = BUNDLE / "main.pdf"
    pdf_passed = bool(make_all and make_all["returncode"] == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0)
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(BUNDLE),
        "tools": tools,
        "make_check": make_check,
        "source_check_passed": make_check["returncode"] == 0,
        "pdf_compilation_attempted": make_all is not None,
        "pdf_compilation_passed": pdf_passed,
        "pdf_path": str(pdf_path),
        "make_all": make_all,
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_report(audit), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if args.require_pdf and not pdf_passed:
        raise SystemExit(1)
    if not audit["source_check_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
