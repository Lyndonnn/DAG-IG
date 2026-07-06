#!/usr/bin/env python3
"""Audit whether the venue decision form has enough information for final conversion.

This is intentionally not a release-blocking check by default. The generic paper
package can be ready for venue conversion before the target venue is known. Pass
--require-ready only after the user has filled the form and wants to treat venue
decisions as a hard gate.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
DEFAULT_FORM = ASSETS / "VENUE_DECISION_FORM.md"
OUT_JSON = ASSETS / "venue_decision_audit.json"
OUT_MD = ASSETS / "VENUE_DECISION_AUDIT.md"


FIELD_PATTERNS = {
    "venue_name": r"^- Venue name:[ \t]*([^\n]*)$",
    "venue_year_cycle": r"^- Venue year/cycle:[ \t]*([^\n]*)$",
    "track": r"^- Track:[ \t]*([^\n]*)$",
    "submission_system": r"^- Submission system:[ \t]*([^\n]*)$",
    "template_path_or_url": r"^- Template path or URL:[ \t]*([^\n]*)$",
    "main_paper_page_limit": r"^- Main paper page limit:[ \t]*([^\n]*)$",
}

CHECKBOX_GROUPS = {
    "submission_route": [
        "Target venue/conference/journal template conversion.",
        "Generic article-format preprint.",
        "Internal draft only for now.",
    ],
    "payload_route": [
        "`anonymous_review_generic`: submit generic PDF plus review-clean source zip.",
        "`full_handoff_or_overleaf`: share full source bundle, tarball, PDF, and checksums with a collaborator.",
        "`target_venue_conversion`: use full source bundle and this decision form to convert into a target venue template.",
        "`generic_preprint`: use author-visible generic PDF plus full source bundle only if preprint policy allows it.",
    ],
    "official_template_source": [
        "Template zip/directory attached locally.",
        "Official URL provided.",
        "Permission granted to fetch official template.",
    ],
    "review_mode": [
        "Anonymous review.",
        "Author-visible submission.",
        "Camera-ready/final version.",
    ],
    "references_count_toward_limit": [
        "Yes",
        "No",
        "Unknown",
    ],
    "appendix_allowed_in_main_pdf": [
        "Yes",
        "No",
        "Unknown",
    ],
    "supplementary_material_allowed": [
        "Yes",
        "No",
        "Unknown",
    ],
    "appendix_decision": [
        "Keep diagnostic appendix in main PDF.",
        "Move diagnostic appendix to supplement.",
        "Remove diagnostic appendix from submitted version.",
    ],
    "artifact_to_submit": [
        "Venue-formatted PDF only.",
        "Venue-formatted source zip.",
        "Anonymous review-clean source zip.",
        "Supplementary zip.",
    ],
}

GROUP_MARKERS = {
    "references_count_toward_limit": "- References count toward limit:",
    "appendix_allowed_in_main_pdf": "- Appendix allowed in main PDF:",
    "supplementary_material_allowed": "- Supplementary material allowed:",
}

OPTIONAL_AUTHOR_FIELDS = {
    "author_list": r"^- Author list:[ \t]*([^\n]*)$",
    "affiliations": r"^- Affiliations:[ \t]*([^\n]*)$",
    "corresponding_author": r"^- Corresponding author:[ \t]*([^\n]*)$",
}

OPTIONAL_NOTE_FIELDS = {
    "figure_table_placement_constraints": r"^- Figure/table placement constraints:[ \t]*([^\n]*)$",
    "font_format_constraints": r"^- Font/format constraints beyond template defaults:[ \t]*([^\n]*)$",
}


def extract_field(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return ""
    value = match.group(1).strip()
    return value


def checkbox_state(text: str, label: str) -> bool:
    escaped = re.escape(label)
    return bool(re.search(rf"^- \[[xX]\]\s+{escaped}\s*$", text, re.MULTILINE))


def checkbox_state_after_marker(text: str, marker: str, label: str) -> bool:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == marker:
            start = idx + 1
            break
    if start is None:
        return False
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ["):
            if re.match(rf"^- \[[xX]\]\s+{re.escape(label)}\s*$", stripped):
                return True
            continue
        break
    return False


def audit_form(form: Path) -> dict[str, Any]:
    text = form.read_text(encoding="utf-8") if form.exists() else ""
    fields = {name: extract_field(text, pattern) for name, pattern in FIELD_PATTERNS.items()}
    optional_author_fields = {name: extract_field(text, pattern) for name, pattern in OPTIONAL_AUTHOR_FIELDS.items()}
    optional_note_fields = {name: extract_field(text, pattern) for name, pattern in OPTIONAL_NOTE_FIELDS.items()}

    checkboxes: dict[str, Any] = {}
    for group, labels in CHECKBOX_GROUPS.items():
        marker = GROUP_MARKERS.get(group)
        if marker:
            selected = [label for label in labels if checkbox_state_after_marker(text, marker, label)]
        else:
            selected = [label for label in labels if checkbox_state(text, label)]
        checkboxes[group] = {
            "selected": selected,
            "selected_count": len(selected),
            "valid_single_choice": len(selected) == 1,
        }

    required_blank = [name for name, value in fields.items() if not value]
    checkbox_problems = [
        group for group, info in checkboxes.items() if not info["valid_single_choice"]
    ]

    review_selected = checkboxes["review_mode"]["selected"]
    author_visible = bool(
        review_selected
        and review_selected[0] in {"Author-visible submission.", "Camera-ready/final version."}
    )
    author_blanks = [name for name, value in optional_author_fields.items() if not value] if author_visible else []

    template_source_selected = checkboxes["official_template_source"]["selected"]
    template_reference_missing = not fields["template_path_or_url"] and bool(template_source_selected)

    ready_for_target_conversion = (
        form.exists()
        and not required_blank
        and not checkbox_problems
        and not author_blanks
        and not template_reference_missing
    )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "form_path": str(form),
        "form_exists": form.exists(),
        "fields": fields,
        "optional_author_fields": optional_author_fields,
        "optional_note_fields": optional_note_fields,
        "checkboxes": checkboxes,
        "required_blank_fields": required_blank,
        "checkbox_groups_not_single_choice": checkbox_problems,
        "author_visible_or_camera_ready": author_visible,
        "author_visible_blank_fields": author_blanks,
        "template_reference_missing": template_reference_missing,
        "ready_for_target_conversion": ready_for_target_conversion,
        "ready_for_final_submission": False,
        "final_submission_note": (
            "Final submission still requires a venue-converted PDF/source that compiles and passes "
            "post-compile content/layout audits."
        ),
    }


def default_form_path() -> Path:
    if DEFAULT_FORM.exists():
        return DEFAULT_FORM
    bundle_form = Path("docs/VENUE_DECISION_FORM.md")
    if bundle_form.exists():
        return bundle_form
    local_form = Path("VENUE_DECISION_FORM.md")
    if local_form.exists():
        return local_form
    return DEFAULT_FORM


def default_output_paths(form: Path) -> tuple[Path, Path]:
    if form == DEFAULT_FORM or str(form).startswith(str(ASSETS)):
        return OUT_JSON, OUT_MD
    if form.parent.exists():
        return form.parent / "venue_decision_audit.json", form.parent / "VENUE_DECISION_AUDIT.md"
    return Path("venue_decision_audit.json"), Path("VENUE_DECISION_AUDIT.md")


def build_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Venue Decision Audit\n\n")
    lines.append("This audit checks whether `VENUE_DECISION_FORM.md` has enough external venue information to proceed to target-template conversion. It does not inspect a venue-converted PDF.\n\n")
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- form: `{audit['form_path']}`\n")
    lines.append(f"- form exists: `{audit['form_exists']}`\n")
    lines.append(f"- ready for target conversion: `{audit['ready_for_target_conversion']}`\n")
    lines.append(f"- ready for final submission: `{audit['ready_for_final_submission']}`\n")
    lines.append(f"- final submission note: {audit['final_submission_note']}\n\n")

    lines.append("## Required Fields\n\n")
    lines.append("| field | value | filled |\n")
    lines.append("|---|---|---:|\n")
    for name, value in audit["fields"].items():
        lines.append(f"| {name} | `{value}` | `{bool(value)}` |\n")
    lines.append("\n")

    lines.append("## Optional Note Fields\n\n")
    lines.append("| field | value |\n")
    lines.append("|---|---|\n")
    for name, value in audit["optional_note_fields"].items():
        lines.append(f"| {name} | `{value}` |\n")
    lines.append("\n")

    lines.append("## Checkbox Groups\n\n")
    lines.append("| group | selected_count | valid_single_choice | selected |\n")
    lines.append("|---|---:|---:|---|\n")
    for group, info in audit["checkboxes"].items():
        lines.append(
            f"| {group} | {info['selected_count']} | `{info['valid_single_choice']}` | `{info['selected']}` |\n"
        )
    lines.append("\n")

    lines.append("## Problems\n\n")
    problems = False
    if audit["required_blank_fields"]:
        problems = True
        lines.append(f"- blank required fields: `{audit['required_blank_fields']}`\n")
    if audit["checkbox_groups_not_single_choice"]:
        problems = True
        lines.append(f"- checkbox groups needing exactly one choice: `{audit['checkbox_groups_not_single_choice']}`\n")
    if audit["author_visible_blank_fields"]:
        problems = True
        lines.append(f"- author-visible/camera-ready blank fields: `{audit['author_visible_blank_fields']}`\n")
    if audit["template_reference_missing"]:
        problems = True
        lines.append("- template source is selected but `Template path or URL` is blank.\n")
    if not problems:
        lines.append("- none\n")

    lines.append("\n## Next Action\n\n")
    if audit["ready_for_target_conversion"]:
        lines.append("Proceed to target-template conversion, then compile and run post-compile PDF/layout audits.\n")
    else:
        lines.append("Fill the missing venue/template/review/page-rule fields before treating this as final-submission ready.\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--form", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    form = args.form or default_form_path()
    default_json, default_md = default_output_paths(form)
    output_json = args.output_json or default_json
    output_md = args.output_md or default_md

    audit = audit_form(form)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(build_markdown(audit), encoding="utf-8")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    print(f"ready_for_target_conversion={audit['ready_for_target_conversion']}")
    if args.require_ready and not audit["ready_for_target_conversion"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
