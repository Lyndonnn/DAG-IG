#!/usr/bin/env python3
"""Build a compact final readiness dashboard for paper handoff.

This dashboard aggregates the existing authoritative audits. It is not a new
experiment and does not replace the underlying reports; it is a one-page index
for deciding what remains before venue submission.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
OUT_JSON = ASSETS / "SUBMISSION_READINESS_DASHBOARD.json"
OUT_MD = ASSETS / "SUBMISSION_READINESS_DASHBOARD.md"

INPUTS = {
    "goal": ASSETS / "goal_completion_audit.json",
    "asset": ASSETS / "paper_asset_audit.json",
    "package_index": ASSETS / "SUBMISSION_PACKAGE_INDEX.json",
    "package_extract": ASSETS / "package_extract_verification.json",
    "source_compile": ASSETS / "source_bundle_compile_verification.json",
    "review_clean_compile": ASSETS / "review_clean_compile_verification.json",
    "review_clean_anonymity": ASSETS / "REVIEW_CLEAN_ANONYMITY_AUDIT.json",
    "post_compile_pdf": ASSETS / "post_compile_pdf_audit.json",
    "pdf_layout": ASSETS / "pdf_layout_audit.json",
    "artifact_checksums": ASSETS / "ARTIFACT_CHECKSUMS.json",
    "text_finalization": ASSETS / "text_finalization_audit.json",
    "venue_workspace": ASSETS / "venue_workspace_audit.json",
    "submission_payload": ASSETS / "submission_payload_index.json",
    "venue_decision": ASSETS / "venue_decision_audit.json",
    "final_submission_gate": ASSETS / "final_submission_gate.json",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def file_info(path: Path) -> dict[str, Any]:
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256,
    }


def bool_path(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def build_dashboard() -> dict[str, Any]:
    reports = {name: load_json(path) for name, path in INPUTS.items()}
    package_index = reports["package_index"]
    pdf_layout = reports["pdf_layout"]
    post_pdf = reports["post_compile_pdf"]
    venue_decision = reports["venue_decision"]
    final_submission_gate = reports["final_submission_gate"]

    packages: dict[str, Any] = {}
    for key in ["tarball", "zipfile", "review_clean_zip"]:
        item = package_index.get(key)
        if isinstance(item, dict) and item.get("path"):
            packages[key] = {
                "path": item.get("path"),
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
    compiled_pdf = file_info(ASSETS / "submission_bundle/main.pdf")
    if isinstance(package_index.get("compiled_pdf"), dict):
        compiled_pdf.update(
            {
                "path": package_index["compiled_pdf"].get("path", compiled_pdf["path"]),
                "bytes": package_index["compiled_pdf"].get("bytes", compiled_pdf["bytes"]),
                "sha256": package_index["compiled_pdf"].get("sha256", compiled_pdf["sha256"]),
            }
        )

    gates = {
        "experimental_mainline_complete": bool_path(reports["goal"], "experimental_mainline_complete"),
        "paper_package_ready_for_template": bool_path(reports["goal"], "paper_package_ready_for_template"),
        "compiled_source_pdf_verified": bool_path(reports["goal"], "compiled_source_pdf_verified"),
        "final_paper_complete": bool_path(reports["goal"], "final_paper_complete"),
        "paper_asset_audit_pass": bool_path(reports["asset"], "overall_pass"),
        "package_extract_verification_pass": bool_path(reports["package_extract"], "overall_pass"),
        "source_bundle_clean_extract_compile_pass": bool_path(reports["source_compile"], "overall_pass"),
        "review_clean_compile_pass": bool_path(reports["review_clean_compile"], "passed"),
        "review_clean_anonymity_pass": bool_path(reports["review_clean_anonymity"], "overall_pass"),
        "post_compile_pdf_audit_pass": bool_path(post_pdf, "passed"),
        "pdf_layout_audit_pass": bool_path(pdf_layout, "passed"),
        "artifact_checksums_pass": bool_path(reports["artifact_checksums"], "passed"),
        "text_finalization_pass": bool_path(reports["text_finalization"], "overall_pass"),
        "venue_workspace_audit_pass": bool_path(reports["venue_workspace"], "overall_pass"),
        "submission_payload_index_pass": bool_path(reports["submission_payload"], "overall_pass"),
    }

    ready_for_venue_conversion = all(
        bool(gates[key])
        for key in [
            "experimental_mainline_complete",
            "paper_package_ready_for_template",
            "compiled_source_pdf_verified",
            "paper_asset_audit_pass",
            "package_extract_verification_pass",
            "source_bundle_clean_extract_compile_pass",
            "review_clean_compile_pass",
            "review_clean_anonymity_pass",
            "post_compile_pdf_audit_pass",
            "pdf_layout_audit_pass",
            "artifact_checksums_pass",
            "text_finalization_pass",
            "venue_workspace_audit_pass",
            "submission_payload_index_pass",
        ]
    )

    dashboard = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_venue_conversion": ready_for_venue_conversion,
        "ready_for_final_submission": bool_path(final_submission_gate, "final_submission_ready"),
        "gates": gates,
        "packages": packages,
        "compiled_pdf": compiled_pdf,
        "pdf_metadata": {
            "pages": bool_path(pdf_layout, "pdfinfo", "pages"),
            "page_size": bool_path(pdf_layout, "pdfinfo", "page_size"),
            "encrypted": bool_path(pdf_layout, "pdfinfo", "encrypted"),
            "layout_pass": bool_path(pdf_layout, "passed"),
            "content_pass": bool_path(post_pdf, "passed"),
        },
        "main_result": {
            "format_sft_dev_strict": "42.9%",
            "dagig_seed42_dev_strict": "49.0%",
            "format_sft_test_strict": "34.4%",
            "dagig_seed42_test_strict": "40.6%",
            "seed42_dev_gain": "6.1 pts",
            "seed42_test_gain": "6.2 pts",
        },
        "remaining_external_inputs": [
            "target venue/template",
            "anonymous vs author-visible metadata policy",
            "page limit and whether appendix/supplement counts",
        ],
        "venue_decision_form": str(ASSETS / "VENUE_DECISION_FORM.md"),
        "submission_route_guide": str(ASSETS / "SUBMISSION_ROUTE_GUIDE.md"),
        "submission_payload_index": str(ASSETS / "SUBMISSION_PAYLOAD_INDEX.md"),
        "venue_decision_audit": str(ASSETS / "VENUE_DECISION_AUDIT.md"),
        "final_submission_gate": str(ASSETS / "FINAL_SUBMISSION_GATE.md"),
        "venue_decision_ready_for_target_conversion": bool_path(venue_decision, "ready_for_target_conversion"),
        "final_submission_gate_ready": bool_path(final_submission_gate, "final_submission_ready"),
        "do_not_reopen_mainline": [
            "DAG-SFT trace imitation as the main method",
            "query reranking/switching as the main method",
            "no-teacher fusion as the main method",
            "broad answer repair as the main method",
            "same-recipe GRPO reruns without a new mechanism",
        ],
        "source_reports": {name: str(path) for name, path in INPUTS.items()},
    }
    return dashboard


def build_markdown(dashboard: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Submission Readiness Dashboard\n\n")
    lines.append(f"- created_at_utc: `{dashboard['created_at_utc']}`\n")
    lines.append(f"- ready for venue conversion: `{dashboard['ready_for_venue_conversion']}`\n")
    lines.append(f"- ready for final submission: `{dashboard['ready_for_final_submission']}`\n")
    lines.append("\n")

    lines.append("## Main Result\n\n")
    lines.append("| comparison | value |\n")
    lines.append("|---|---:|\n")
    for key, value in dashboard["main_result"].items():
        lines.append(f"| {key} | {value} |\n")
    lines.append("\n")

    lines.append("## Gates\n\n")
    lines.append("| gate | status |\n")
    lines.append("|---|---:|\n")
    for key, value in dashboard["gates"].items():
        lines.append(f"| {key} | `{value}` |\n")
    lines.append("\n")

    lines.append("## Packages\n\n")
    lines.append("| artifact | bytes | sha256 | path |\n")
    lines.append("|---|---:|---|---|\n")
    for key, item in dashboard["packages"].items():
        lines.append(f"| {key} | {item.get('bytes')} | `{item.get('sha256')}` | `{item.get('path')}` |\n")
    pdf = dashboard["compiled_pdf"]
    lines.append(f"| compiled_pdf | {pdf['bytes']} | `{pdf.get('sha256')}` | `{pdf['path']}` |\n")
    lines.append("\n")

    meta = dashboard["pdf_metadata"]
    lines.append("## PDF Metadata\n\n")
    lines.append(f"- pages: `{meta.get('pages')}`\n")
    lines.append(f"- page size: `{meta.get('page_size')}`\n")
    lines.append(f"- encrypted: `{meta.get('encrypted')}`\n")
    lines.append(f"- content audit pass: `{meta.get('content_pass')}`\n")
    lines.append(f"- layout audit pass: `{meta.get('layout_pass')}`\n")
    lines.append("\n")

    lines.append("## Remaining External Inputs\n\n")
    lines.append(f"Fill this form before venue conversion: `{dashboard['venue_decision_form']}`\n\n")
    lines.append(f"Current venue decision audit: `{dashboard['venue_decision_audit']}`\n\n")
    lines.append(f"- venue decision ready for target conversion: `{dashboard['venue_decision_ready_for_target_conversion']}`\n\n")
    lines.append(f"Final submission gate: `{dashboard['final_submission_gate']}`\n\n")
    lines.append(f"- final submission gate ready: `{dashboard['final_submission_gate_ready']}`\n\n")
    lines.append(f"Use this route guide from the source bundle/full repository: `{dashboard['submission_route_guide']}`\n\n")
    lines.append(f"Use this full-repository payload index for exact artifact hashes: `{dashboard['submission_payload_index']}`\n\n")
    for item in dashboard["remaining_external_inputs"]:
        lines.append(f"- {item}\n")
    lines.append("\n")

    lines.append("## Do Not Reopen As Mainline\n\n")
    for item in dashboard["do_not_reopen_mainline"]:
        lines.append(f"- {item}\n")
    lines.append("\n")

    lines.append("## Source Reports\n\n")
    for key, path in dashboard["source_reports"].items():
        lines.append(f"- {key}: `{path}`\n")
    return "".join(lines)


def main() -> None:
    dashboard = build_dashboard()
    OUT_JSON.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_markdown(dashboard), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not dashboard["ready_for_venue_conversion"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
