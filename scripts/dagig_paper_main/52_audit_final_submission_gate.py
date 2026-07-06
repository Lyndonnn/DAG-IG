#!/usr/bin/env python3
"""Audit final-submission readiness after venue conversion.

This is the last gate before uploading a venue-formatted paper. By default it
reports why final submission is not ready; pass --require-ready only after a
venue-specific PDF/source has been produced and audited.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
DEFAULT_VENUE_ROOT = ASSETS / "venue_workspace"
DEFAULT_VENUE_DECISION = ASSETS / "venue_decision_audit.json"
DEFAULT_OUT_JSON = ASSETS / "final_submission_gate.json"
DEFAULT_OUT_MD = ASSETS / "FINAL_SUBMISSION_GATE.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def default_venue_root() -> Path:
    if DEFAULT_VENUE_ROOT.exists():
        return DEFAULT_VENUE_ROOT
    return Path(".")


def default_venue_decision_audit() -> Path:
    if DEFAULT_VENUE_DECISION.exists():
        return DEFAULT_VENUE_DECISION
    bundle_decision = Path("docs/venue_decision_audit.json")
    if bundle_decision.exists():
        return bundle_decision
    if Path("docs/VENUE_DECISION_FORM.md").exists():
        return bundle_decision
    local_decision = Path("venue_decision_audit.json")
    if local_decision.exists():
        return local_decision
    if Path("VENUE_DECISION_FORM.md").exists():
        return local_decision
    return DEFAULT_VENUE_DECISION


def default_output_paths(venue_decision_audit: Path) -> tuple[Path, Path]:
    if str(venue_decision_audit).startswith(str(ASSETS)):
        return DEFAULT_OUT_JSON, DEFAULT_OUT_MD
    if venue_decision_audit.parent.exists():
        return venue_decision_audit.parent / "final_submission_gate.json", venue_decision_audit.parent / "FINAL_SUBMISSION_GATE.md"
    return Path("final_submission_gate.json"), Path("FINAL_SUBMISSION_GATE.md")


def file_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "bytes": 0}
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def find_pdf(venue_root: Path, explicit: Path | None, allow_generic_check: bool) -> tuple[Path | None, bool]:
    if explicit:
        return explicit, explicit.name == "main_generic_check.pdf"
    candidates = [venue_root / "main.pdf", venue_root / "paper.pdf"]
    for candidate in candidates:
        if candidate.exists():
            return candidate, False
    generic = venue_root / "main_generic_check.pdf"
    if allow_generic_check and generic.exists():
        return generic, True
    return None, False


def default_post_audit(venue_root: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    for candidate in [
        venue_root / "post_compile_pdf_audit.json",
        venue_root / "POST_COMPILE_PDF_AUDIT.json",
        venue_root / "venue_workspace_post_compile_pdf_audit.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def default_layout_audit(venue_root: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    for candidate in [
        venue_root / "pdf_layout_audit.json",
        venue_root / "PDF_LAYOUT_AUDIT.json",
        venue_root / "venue_workspace_pdf_layout_audit.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def official_template_present(venue_root: Path) -> bool:
    template_root = venue_root / "venue_template_original"
    if not template_root.exists():
        return False
    files = [p for p in template_root.rglob("*") if p.is_file()]
    if not files:
        return False
    placeholder = template_root / "PUT_OFFICIAL_TEMPLATE_HERE.md"
    return any(path != placeholder for path in files)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    venue_root = args.venue_root
    venue_pdf, generic_check_only = find_pdf(venue_root, args.venue_pdf, args.allow_generic_check)
    post_audit_path = default_post_audit(venue_root, args.post_compile_audit)
    layout_audit_path = default_layout_audit(venue_root, args.layout_audit)

    venue_decision = load_json(args.venue_decision_audit)
    post_audit = load_json(post_audit_path) if post_audit_path else {"_missing": True, "_path": None}
    layout_audit = load_json(layout_audit_path) if layout_audit_path else {"_missing": True, "_path": None}

    checks = {
        "venue_decision_ready": bool(venue_decision.get("ready_for_target_conversion")),
        "venue_root_exists": venue_root.exists(),
        "official_template_present": official_template_present(venue_root),
        "venue_pdf_exists": venue_pdf is not None and venue_pdf.exists() and venue_pdf.stat().st_size > 0,
        "not_generic_check_pdf": not generic_check_only,
        "post_compile_audit_pass": bool(post_audit.get("passed")),
        "layout_audit_pass": bool(layout_audit.get("passed")),
    }

    blockers: list[str] = []
    if not checks["venue_decision_ready"]:
        blockers.append("venue decision audit is not ready; fill VENUE_DECISION_FORM.md and pass --require-ready")
    if not checks["official_template_present"]:
        blockers.append("official target venue template is not present in venue_template_original/")
    if not checks["venue_pdf_exists"]:
        blockers.append("venue-formatted PDF is missing")
    if generic_check_only:
        blockers.append("only the generic compile-check PDF is present; this is not a target-venue PDF")
    if not checks["post_compile_audit_pass"]:
        blockers.append("venue PDF post-compile content audit has not passed")
    if not checks["layout_audit_pass"]:
        blockers.append("venue PDF layout audit has not passed")

    final_ready = all(checks.values()) and not blockers
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "venue_root": str(venue_root),
        "venue_decision_audit": str(args.venue_decision_audit),
        "venue_pdf": file_info(venue_pdf),
        "generic_check_only": generic_check_only,
        "post_compile_audit": str(post_audit_path) if post_audit_path else None,
        "layout_audit": str(layout_audit_path) if layout_audit_path else None,
        "checks": checks,
        "blockers": blockers,
        "final_submission_ready": final_ready,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Final Submission Gate\n\n")
    lines.append("This gate is for the venue-converted paper, not for the generic source package. It should pass only after the target venue/template has been chosen, the venue PDF has been compiled, and the rendered PDF audits pass.\n\n")
    lines.append(f"- created_at_utc: `{report['created_at_utc']}`\n")
    lines.append(f"- final submission ready: `{report['final_submission_ready']}`\n")
    lines.append(f"- venue root: `{report['venue_root']}`\n")
    lines.append(f"- venue PDF: `{report['venue_pdf']['path']}`\n")
    lines.append(f"- generic check only: `{report['generic_check_only']}`\n")
    lines.append(f"- venue decision audit: `{report['venue_decision_audit']}`\n")
    lines.append(f"- post-compile audit: `{report['post_compile_audit']}`\n")
    lines.append(f"- layout audit: `{report['layout_audit']}`\n\n")

    lines.append("## Checks\n\n")
    lines.append("| check | passed |\n")
    lines.append("|---|---:|\n")
    for key, value in report["checks"].items():
        lines.append(f"| {key} | `{value}` |\n")
    lines.append("\n")

    lines.append("## Blockers\n\n")
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- {blocker}\n")
    else:
        lines.append("- none\n")
    lines.append("\n")

    lines.append("## Required Final Command\n\n")
    lines.append("After producing the target-venue PDF, run this gate with `--require-ready` before upload.\n\n")
    lines.append("```bash\n")
    lines.append("python scripts/dagig_paper_main/52_audit_final_submission_gate.py --require-ready\n")
    lines.append("```\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue_root", type=Path, default=None)
    parser.add_argument("--venue_decision_audit", type=Path, default=None)
    parser.add_argument("--venue_pdf", type=Path, default=None)
    parser.add_argument("--post_compile_audit", type=Path, default=None)
    parser.add_argument("--layout_audit", type=Path, default=None)
    parser.add_argument("--allow-generic-check", action="store_true")
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    args.venue_root = args.venue_root or default_venue_root()
    args.venue_decision_audit = args.venue_decision_audit or default_venue_decision_audit()
    default_json, default_md = default_output_paths(args.venue_decision_audit)
    args.output_json = args.output_json or default_json
    args.output_md = args.output_md or default_md

    report = audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.write_text(build_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"final_submission_ready={report['final_submission_ready']}")
    if args.require_ready and not report["final_submission_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
