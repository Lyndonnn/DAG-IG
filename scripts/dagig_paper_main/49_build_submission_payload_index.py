#!/usr/bin/env python3
"""Build a route-specific submission payload index for paper handoff."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
OUT_JSON = ASSETS / "submission_payload_index.json"
OUT_MD = ASSETS / "SUBMISSION_PAYLOAD_INDEX.md"

ARTIFACTS = {
    "generic_pdf": ASSETS / "submission_bundle/main.pdf",
    "full_source_zip": ASSETS / "DAGIG_Pix2Fact_overleaf_source_bundle.zip",
    "full_source_tarball": ASSETS / "DAGIG_Pix2Fact_paper_source_bundle.tar.gz",
    "review_clean_source_zip": ASSETS / "DAGIG_Pix2Fact_review_clean_source_bundle.zip",
    "sha256sums": ASSETS / "SHA256SUMS.txt",
    "package_index": ASSETS / "SUBMISSION_PACKAGE_INDEX.md",
    "final_handoff_prompt": ASSETS / "FINAL_HANDOFF_PROMPT.md",
    "submission_route_guide": ASSETS / "SUBMISSION_ROUTE_GUIDE.md",
    "mainline_schema_contract_audit": ASSETS / "MAINLINE_SCHEMA_CONTRACT_AUDIT.md",
    "mainline_schema_contract_audit_json": ASSETS / "MAINLINE_SCHEMA_CONTRACT_AUDIT.json",
    "venue_decision_audit": ASSETS / "VENUE_DECISION_AUDIT.md",
    "venue_decision_form": ASSETS / "VENUE_DECISION_FORM.md",
    "final_submission_gate": ASSETS / "FINAL_SUBMISSION_GATE.md",
}

ROUTES = {
    "anonymous_review_generic": {
        "when_to_use": "Anonymous review route before target template conversion, if the venue accepts a generic article-format PDF/source.",
        "upload_or_share": ["generic_pdf", "review_clean_source_zip"],
        "do_not_upload": ["full_source_zip", "full_source_tarball"],
        "required_external_decisions": ["venue accepts generic article format", "anonymous review mode"],
    },
    "full_handoff_or_overleaf": {
        "when_to_use": "Internal handoff, collaborator handoff, or Overleaf editing where docs/scripts/audits are useful.",
        "upload_or_share": ["full_source_zip", "full_source_tarball", "generic_pdf", "sha256sums"],
        "do_not_upload": [],
        "required_external_decisions": ["recipient is allowed to see internal audit/reproducibility docs"],
    },
    "target_venue_conversion": {
        "when_to_use": "Final submission path after a target venue/template is chosen.",
        "upload_or_share": ["venue_decision_form", "venue_decision_audit", "final_submission_gate", "full_source_zip"],
        "do_not_upload": [],
        "required_external_decisions": ["target venue/template", "anonymous vs author-visible policy", "page and supplement rules"],
    },
    "generic_preprint": {
        "when_to_use": "Public preprint route if author-visible metadata and preprint policy are approved.",
        "upload_or_share": ["generic_pdf", "full_source_zip"],
        "do_not_upload": ["review_clean_source_zip"],
        "required_external_decisions": ["author-visible metadata", "preprint permission", "whether to retain appendix"],
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_info(label: str, path: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def inspect_zip(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "entry_count": 0, "has_docs": None, "has_scripts": None, "disallowed_review_entries": []}
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    disallowed = [name for name in names if name.startswith("docs/") or name.startswith("scripts/") or name.endswith(".json")]
    return {
        "exists": True,
        "entry_count": len(names),
        "has_docs": any(name.startswith("docs/") for name in names),
        "has_scripts": any(name.startswith("scripts/") for name in names),
        "disallowed_review_entries": disallowed,
    }


def build_index() -> dict[str, Any]:
    artifacts = {label: artifact_info(label, path) for label, path in ARTIFACTS.items()}
    review_zip_audit = inspect_zip(ARTIFACTS["review_clean_source_zip"])
    full_zip_audit = inspect_zip(ARTIFACTS["full_source_zip"])
    problems: list[str] = []
    for label, item in artifacts.items():
        if not item["exists"]:
            problems.append(f"{label} missing: {item['path']}")
        elif item["bytes"] == 0:
            problems.append(f"{label} empty: {item['path']}")
    if review_zip_audit["disallowed_review_entries"]:
        problems.append(f"review-clean zip contains disallowed entries: {review_zip_audit['disallowed_review_entries'][:10]}")
    if not full_zip_audit["has_docs"]:
        problems.append("full source zip does not contain docs/")

    resolved_routes: dict[str, Any] = {}
    for route_name, route in ROUTES.items():
        upload_items = [artifacts[label] for label in route["upload_or_share"]]
        route_problems = [f"{item['label']} missing" for item in upload_items if not item["exists"]]
        resolved_routes[route_name] = {
            **route,
            "upload_or_share_artifacts": upload_items,
            "route_problems": route_problems,
            "ready_if_external_decisions_are_filled": not route_problems,
        }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Route-specific index for choosing the correct already-audited paper artifacts.",
        "artifacts": artifacts,
        "zip_audits": {
            "review_clean_source_zip": review_zip_audit,
            "full_source_zip": full_zip_audit,
        },
        "routes": resolved_routes,
        "problems": problems,
        "overall_pass": not problems,
    }


def build_markdown(index: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Submission Payload Index\n\n")
    lines.append("This index maps submission routes to the already-audited artifacts. It does not create new experimental outputs.\n\n")
    lines.append(f"- created_at_utc: `{index['created_at_utc']}`\n")
    lines.append(f"- overall pass: `{index['overall_pass']}`\n\n")

    lines.append("## Artifacts\n\n")
    lines.append("| artifact | bytes | sha256 | path |\n")
    lines.append("|---|---:|---|---|\n")
    for label, item in index["artifacts"].items():
        lines.append(f"| {label} | {item['bytes']} | `{item['sha256']}` | `{item['path']}` |\n")
    lines.append("\n")

    lines.append("## Routes\n\n")
    for route_name, route in index["routes"].items():
        lines.append(f"### {route_name}\n\n")
        lines.append(f"- when to use: {route['when_to_use']}\n")
        lines.append(f"- ready after external decisions: `{route['ready_if_external_decisions_are_filled']}`\n")
        lines.append("- upload/share:\n")
        for item in route["upload_or_share_artifacts"]:
            lines.append(f"  - `{item['path']}`\n")
        if route["do_not_upload"]:
            lines.append("- do not upload for this route:\n")
            for label in route["do_not_upload"]:
                lines.append(f"  - `{index['artifacts'][label]['path']}`\n")
        lines.append("- external decisions required:\n")
        for decision in route["required_external_decisions"]:
            lines.append(f"  - {decision}\n")
        if route["route_problems"]:
            lines.append(f"- route problems: `{route['route_problems']}`\n")
        lines.append("\n")

    lines.append("## Review-Clean Boundary\n\n")
    review = index["zip_audits"]["review_clean_source_zip"]
    lines.append(f"- review-clean zip entry count: `{review['entry_count']}`\n")
    lines.append(f"- contains docs/: `{review['has_docs']}`\n")
    lines.append(f"- contains scripts/: `{review['has_scripts']}`\n")
    lines.append(f"- disallowed review entries: `{review['disallowed_review_entries']}`\n\n")

    if index["problems"]:
        lines.append("## Problems\n\n")
        for problem in index["problems"]:
            lines.append(f"- {problem}\n")
    else:
        lines.append("## Problems\n\n- none\n")
    return "".join(lines)


def main() -> None:
    index = build_index()
    OUT_JSON.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_markdown(index), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not index["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
