#!/usr/bin/env python3
"""Audit the generated venue-conversion workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
DEFAULT_WORKSPACE = ASSETS / "venue_workspace"
DEFAULT_SOURCE = ASSETS / "submission_bundle"
OUT_JSON = ASSETS / "venue_workspace_audit.json"
OUT_MD = ASSETS / "VENUE_WORKSPACE_AUDIT.md"
POST_AUDIT_SCRIPT = Path("scripts/dagig_paper_main/38_post_compile_pdf_audit.py")
LAYOUT_AUDIT_SCRIPT = Path("scripts/dagig_paper_main/40_pdf_layout_audit.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "cmd": " ".join(cmd),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def check_required_files(workspace: Path) -> dict[str, Any]:
    required = [
        "README_VENUE_WORKSPACE.md",
        "VENUE_WORKSPACE_MANIFEST.json",
        "main_generic_check.tex",
        "main_generic_check.pdf",
        "main_generic_check.log",
        "paper_parts/title.tex",
        "paper_parts/abstract.tex",
        "paper_parts/abstract_environment.tex",
        "paper_parts/body_sections.tex",
        "paper_parts/appendix_sections.tex",
        "tables/main_results_table.tex",
        "tables/node_credit_diagnostic_table.tex",
        "figures/dagig_method_diagram.tex",
        "figures/dagig_reward_equations.tex",
        "algorithm_dagig_grpo.tex",
        "diagnostic_branches_table.tex",
        "references.bib",
        "docs/MAINLINE_EVIDENCE_CHAIN.md",
        "docs/TEXT_FINALIZATION_AUDIT.md",
        "docs/FINAL_HANDOFF_PROMPT.md",
        "docs/SUBMISSION_ROUTE_GUIDE.md",
        "docs/VENUE_DECISION_AUDIT.md",
        "docs/VENUE_DECISION_FORM.md",
    ]
    missing = [path for path in required if not (workspace / path).exists()]
    empty = [path for path in required if (workspace / path).exists() and (workspace / path).stat().st_size == 0]
    return {"required": required, "missing": missing, "empty": empty, "passed": not missing and not empty}


def compare_files(workspace: Path, source: Path) -> dict[str, Any]:
    pairs = [
        ("paper_parts/title.tex", "venue_template_parts/title.tex"),
        ("paper_parts/abstract.tex", "venue_template_parts/abstract.tex"),
        ("paper_parts/abstract_environment.tex", "venue_template_parts/abstract_environment.tex"),
        ("paper_parts/body_sections.tex", "venue_template_parts/body_sections.tex"),
        ("paper_parts/appendix_sections.tex", "venue_template_parts/appendix_sections.tex"),
        ("tables/main_results_table.tex", "tables/main_results_table.tex"),
        ("tables/node_credit_diagnostic_table.tex", "tables/node_credit_diagnostic_table.tex"),
        ("figures/dagig_method_diagram.tex", "figures/dagig_method_diagram.tex"),
        ("figures/dagig_reward_equations.tex", "figures/dagig_reward_equations.tex"),
        ("algorithm_dagig_grpo.tex", "algorithm_dagig_grpo.tex"),
        ("diagnostic_branches_table.tex", "diagnostic_branches_table.tex"),
        ("references.bib", "references.bib"),
        ("docs/MAINLINE_EVIDENCE_CHAIN.md", "docs/MAINLINE_EVIDENCE_CHAIN.md"),
        ("docs/TEXT_FINALIZATION_AUDIT.md", "docs/TEXT_FINALIZATION_AUDIT.md"),
        ("docs/SUBMISSION_ROUTE_GUIDE.md", "docs/SUBMISSION_ROUTE_GUIDE.md"),
        ("docs/VENUE_DECISION_AUDIT.md", "docs/VENUE_DECISION_AUDIT.md"),
    ]
    mismatches: list[str] = []
    compared: list[dict[str, Any]] = []
    for workspace_rel, source_rel in pairs:
        workspace_path = workspace / workspace_rel
        source_path = source / source_rel
        if not workspace_path.exists() or not source_path.exists():
            mismatches.append(f"missing pair {workspace_rel} / {source_rel}")
            continue
        workspace_sha = sha256_file(workspace_path)
        source_sha = sha256_file(source_path)
        compared.append(
            {
                "workspace": workspace_rel,
                "source": source_rel,
                "workspace_sha256": workspace_sha,
                "source_sha256": source_sha,
                "match": workspace_sha == source_sha,
            }
        )
        if workspace_sha != source_sha:
            mismatches.append(f"{workspace_rel} differs from source {source_rel}")
    return {"compared": compared, "mismatches": mismatches, "passed": not mismatches}


def check_manifest(workspace: Path) -> dict[str, Any]:
    path = workspace / "VENUE_WORKSPACE_MANIFEST.json"
    if not path.exists():
        return {"path": str(path), "problems": ["manifest missing"], "passed": False}
    manifest = load_json(path)
    problems: list[str] = []
    if not manifest.get("compile_check_requested"):
        problems.append("compile_check_requested is false")
    compile_check = manifest.get("compile_check") or {}
    if compile_check.get("returncode") != 0:
        problems.append(f"compile_check returncode is {compile_check.get('returncode')}")
    if not (workspace / "venue_template_original").exists():
        problems.append("venue_template_original directory missing")
    if not manifest.get("files"):
        problems.append("manifest file list is empty")
    return {
        "path": str(path),
        "compile_check_requested": manifest.get("compile_check_requested"),
        "compile_check_returncode": compile_check.get("returncode"),
        "venue_template": manifest.get("venue_template"),
        "file_count": len(manifest.get("files", [])),
        "problems": problems,
        "passed": not problems,
    }


def run_pdf_audits(workspace: Path) -> dict[str, Any]:
    pdf = workspace / "main_generic_check.pdf"
    log = workspace / "main_generic_check.log"
    post_json = workspace / "venue_workspace_post_compile_pdf_audit.json"
    post_md = workspace / "VENUE_WORKSPACE_POST_COMPILE_PDF_AUDIT.md"
    layout_json = workspace / "venue_workspace_pdf_layout_audit.json"
    layout_md = workspace / "VENUE_WORKSPACE_PDF_LAYOUT_AUDIT.md"
    post = run(
        [
            "python",
            str(POST_AUDIT_SCRIPT),
            "--pdf",
            str(pdf),
            "--output_json",
            str(post_json),
            "--output_md",
            str(post_md),
            "--require-pass",
        ]
    )
    layout = run(
        [
            "python",
            str(LAYOUT_AUDIT_SCRIPT),
            "--pdf",
            str(pdf),
            "--log",
            str(log),
            "--output_json",
            str(layout_json),
            "--output_md",
            str(layout_md),
            "--require-pass",
        ]
    )
    return {
        "post_compile": post,
        "layout": layout,
        "post_compile_report": str(post_md),
        "layout_report": str(layout_md),
        "passed": post["returncode"] == 0 and layout["returncode"] == 0,
    }


def build_audit(workspace: Path, source: Path) -> dict[str, Any]:
    checks = {
        "required_files": check_required_files(workspace),
        "source_consistency": compare_files(workspace, source),
        "manifest": check_manifest(workspace),
        "pdf_audits": run_pdf_audits(workspace),
    }
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "source_bundle": str(source),
        "checks": checks,
    }
    audit["overall_pass"] = all(item.get("passed") for item in checks.values())
    return audit


def build_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Venue Workspace Audit\n\n")
    lines.append("This audit verifies that the generated venue-conversion workspace is mechanically usable before a target venue template is supplied.\n\n")
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- workspace: `{audit['workspace']}`\n")
    lines.append(f"- source bundle: `{audit['source_bundle']}`\n")
    lines.append(f"- overall pass: `{audit['overall_pass']}`\n\n")

    lines.append("## Checks\n\n")
    lines.append("| check | passed | notes |\n")
    lines.append("|---|---:|---|\n")
    for name, item in audit["checks"].items():
        notes: list[str] = []
        for field in ["missing", "empty", "mismatches", "problems"]:
            if item.get(field):
                notes.append(f"{field}={item[field]}")
        if name == "manifest":
            notes.append(f"compile_check_returncode={item.get('compile_check_returncode')}")
            notes.append(f"file_count={item.get('file_count')}")
        if name == "pdf_audits":
            notes.append(f"post_compile_rc={item['post_compile']['returncode']}")
            notes.append(f"layout_rc={item['layout']['returncode']}")
        if not notes:
            notes.append("ok")
        lines.append(f"| {name} | `{item.get('passed')}` | {'; '.join(notes)} |\n")
    lines.append("\n")

    lines.append("## Interpretation\n\n")
    if audit["overall_pass"]:
        lines.append("The venue workspace contains the expected paper parts, matches the verified source bundle, compiles through the generic wrapper, and passes rendered-PDF content/layout audits. It is ready for target venue template insertion once the venue policy is known.\n")
    else:
        lines.append("The venue workspace has at least one mechanical issue. Fix the failed check before using it for target venue conversion.\n")
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source_bundle", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_json", type=Path, default=OUT_JSON)
    parser.add_argument("--output_md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args.workspace, args.source_bundle)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.write_text(build_markdown(audit), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
