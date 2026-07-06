#!/usr/bin/env python3
"""Build a self-contained LaTeX source bundle for paper handoff/submission."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
REPORTS = ROOT / "reports"
LATEX = ASSETS / "latex"
FIGURES = ASSETS / "figures"
BUNDLE = ASSETS / "submission_bundle"
TARBALL = ASSETS / "DAGIG_Pix2Fact_paper_source_bundle.tar.gz"
ZIPFILE = ASSETS / "DAGIG_Pix2Fact_overleaf_source_bundle.zip"
PACKAGE_INDEX_JSON = ASSETS / "SUBMISSION_PACKAGE_INDEX.json"
PACKAGE_INDEX_MD = ASSETS / "SUBMISSION_PACKAGE_INDEX.md"

LATEX_BUILD_ARTIFACTS = {
    "main.aux",
    "main.bbl",
    "main.blg",
    "main.log",
    "main.out",
    "main.pdf",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_text_with_replacements(src: Path, dst: Path, replacements: dict[str, str]) -> None:
    text = src.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    write_text(dst, text)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def source_package_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(p for p in BUNDLE.rglob("*") if p.is_file()):
        rel = path.relative_to(BUNDLE)
        if len(rel.parts) == 1 and rel.name in LATEX_BUILD_ARTIFACTS:
            continue
        files.append(path)
    return files


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def build_bundle() -> None:
    venue_parts_script = Path("scripts/dagig_paper_main/34_build_venue_body_files.py")
    if venue_parts_script.exists():
        subprocess.run(["python", str(venue_parts_script)], check=True)
    length_audit_script = Path("scripts/dagig_paper_main/35_paper_length_audit.py")
    if length_audit_script.exists():
        subprocess.run(["python", str(length_audit_script)], check=True)

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True, exist_ok=True)

    path_replacements = {
        "../figures/dagig_method_diagram.tex": "figures/dagig_method_diagram.tex",
        "../figures/dagig_reward_equations.tex": "figures/dagig_reward_equations.tex",
        "../main_results_table.tex": "tables/main_results_table.tex",
        "../node_credit_diagnostic_table.tex": "tables/node_credit_diagnostic_table.tex",
    }
    make_replacements = {
        "../main_results_table.tex": "tables/main_results_table.tex",
        "../node_credit_diagnostic_table.tex": "tables/node_credit_diagnostic_table.tex",
        "../figures/dagig_method_diagram.tex": "figures/dagig_method_diagram.tex",
        "../figures/dagig_reward_equations.tex": "figures/dagig_reward_equations.tex",
    }

    copy_text_with_replacements(LATEX / "main.tex", BUNDLE / "main.tex", path_replacements)
    copy_file(LATEX / "appendix.tex", BUNDLE / "appendix.tex")
    copy_file(LATEX / "diagnostic_branches_table.tex", BUNDLE / "diagnostic_branches_table.tex")
    copy_file(LATEX / "algorithm_dagig_grpo.tex", BUNDLE / "algorithm_dagig_grpo.tex")
    copy_file(LATEX / "references.bib", BUNDLE / "references.bib")
    copy_text_with_replacements(LATEX / "Makefile", BUNDLE / "Makefile", make_replacements)

    copy_file(ASSETS / "main_results_table.tex", BUNDLE / "tables/main_results_table.tex")
    copy_file(ASSETS / "node_credit_diagnostic_table.tex", BUNDLE / "tables/node_credit_diagnostic_table.tex")
    copy_file(FIGURES / "dagig_method_diagram.tex", BUNDLE / "figures/dagig_method_diagram.tex")
    copy_file(FIGURES / "dagig_reward_equations.tex", BUNDLE / "figures/dagig_reward_equations.tex")

    docs = [
        "HANDOFF_README.md",
        "FINAL_HANDOFF_PROMPT.md",
        "MAINLINE_EVIDENCE_CHAIN.md",
        "MAINLINE_SCHEMA_CONTRACT_AUDIT.md",
        "MAINLINE_SCHEMA_CONTRACT_AUDIT.json",
        "GOAL_COMPLETION_AUDIT.md",
        "PAPER_MAIN_EVIDENCE_BRIEF.md",
        "PAPER_DRAFT_V0.md",
        "PAPER_COMPLETION_CHECKLIST.md",
        "REPRODUCIBILITY_APPENDIX.md",
        "PAPER_ASSET_AUDIT_REPORT.md",
        "CLAIMS_EVIDENCE_MATRIX.md",
        "REVIEWER_RISK_REGISTER.md",
        "SUBMISSION_READINESS_REPORT.md",
        "PDF_BUILD_PREFLIGHT_REPORT.md",
        "POST_COMPILE_PDF_AUDIT.md",
        "PDF_LAYOUT_AUDIT.md",
        "TEXT_FINALIZATION_AUDIT.md",
        "PAPER_LENGTH_AUDIT.md",
        "SUBMISSION_ROUTE_GUIDE.md",
        "VENUE_DECISION_AUDIT.md",
        "FINAL_SUBMISSION_GATE.md",
        "VENUE_TEMPLATE_CONVERSION_GUIDE.md",
        "VENUE_DECISION_FORM.md",
        "APPENDIX_DIAGNOSTIC_RESULTS_INDEX.md",
        "RELATED_WORK_DRAFT.md",
        "CITATION_SOURCE_NOTE.md",
    ]
    for name in docs:
        src = ASSETS / name
        if src.exists():
            copy_file(src, BUNDLE / "docs" / name)
    status = REPORTS / "PAPER_MAIN_V1_CURRENT_STATUS.md"
    if status.exists():
        copy_file(status, BUNDLE / "docs/PAPER_MAIN_V1_CURRENT_STATUS.md")

    copy_file(ASSETS / "reproduce_main_commands.sh", BUNDLE / "scripts/reproduce_main_commands.sh")
    release_checks = ASSETS / "run_release_checks.sh"
    if release_checks.exists():
        copy_file(release_checks, BUNDLE / "scripts/run_release_checks.sh")
    post_compile_audit = Path("scripts/dagig_paper_main/38_post_compile_pdf_audit.py")
    if post_compile_audit.exists():
        copy_file(post_compile_audit, BUNDLE / "scripts/post_compile_pdf_audit.py")
    prepare_venue = Path("scripts/dagig_paper_main/39_prepare_venue_workspace.py")
    if prepare_venue.exists():
        copy_file(prepare_venue, BUNDLE / "scripts/prepare_venue_workspace.py")
    pdf_layout_audit = Path("scripts/dagig_paper_main/40_pdf_layout_audit.py")
    if pdf_layout_audit.exists():
        copy_file(pdf_layout_audit, BUNDLE / "scripts/pdf_layout_audit.py")
    venue_decision_audit = Path("scripts/dagig_paper_main/50_audit_venue_decision_form.py")
    if venue_decision_audit.exists():
        copy_file(venue_decision_audit, BUNDLE / "scripts/audit_venue_decision_form.py")
    final_submission_gate = Path("scripts/dagig_paper_main/52_audit_final_submission_gate.py")
    if final_submission_gate.exists():
        copy_file(final_submission_gate, BUNDLE / "scripts/audit_final_submission_gate.py")
    venue_parts = ASSETS / "venue_template_parts"
    if venue_parts.exists():
        for src in sorted(path for path in venue_parts.rglob("*") if path.is_file()):
            copy_file(src, BUNDLE / "venue_template_parts" / src.relative_to(venue_parts))

    readme = """# DAG-IG Pix2Fact Paper Source Bundle

This is a self-contained LaTeX source bundle for the current DAG-IG / Pix2Fact paper draft.

## Build

If LaTeX is installed:

```bash
make check
make all
```

`make check` verifies required source files and runs lightweight source hygiene checks for local path leaks and common TeX linebreak risks. `make all` runs `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.

## Main Position

The main method is DAG-IG node-level GRPO for a two-stage multimodal search agent. DAG-SFT, query reranking, fusion, and broad answer repair are diagnostic/appendix material, not the main claim.

## Next Step

Read `docs/FINAL_HANDOFF_PROMPT.md` and `docs/SUBMISSION_ROUTE_GUIDE.md` before continuing. The next work should be target venue template conversion, venue-specific PDF compilation, and rendered layout inspection only, not new experiments.

From the full repository, run `outputs/dagig_paper_main_v1/paper_assets/run_release_checks.sh` after any source/package edit.

After compiling a PDF in a TeX-enabled environment, run:

```bash
python scripts/post_compile_pdf_audit.py --pdf main.pdf --output_json post_compile_pdf_audit.json --output_md POST_COMPILE_PDF_AUDIT.md --require-pass
python scripts/pdf_layout_audit.py --pdf main.pdf --log main.log --output_json pdf_layout_audit.json --output_md PDF_LAYOUT_AUDIT.md --require-pass
```

To prepare a clean target-venue conversion workspace from this bundle, run:

```bash
python scripts/prepare_venue_workspace.py --source_bundle . --output_dir venue_workspace --force
```

## Contents

- `main.tex`: paper draft.
- `appendix.tex`: appendix sections.
- `references.bib`: bibliography.
- `tables/`: main result and reward diagnostic tables.
- `figures/`: TikZ method diagram and reward equation snippet.
- `venue_template_parts/`: generated snippets for moving the paper into a target venue template.
- `docs/`: final handoff prompt, evidence brief, reproducibility appendix, audit report, and diagnostic notes.
- `docs/MAINLINE_EVIDENCE_CHAIN.md`: single-entry audit trail for data, rollout schema, node credit, reward audit, selected checkpoint, and final result.
- `docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.md`: machine-checkable contract for split/corpus isolation, four-node rollout schema, node credits, and two-stage prediction files.
- `docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.json`: JSON form of the same schema contract audit.
- `docs/POST_COMPILE_PDF_AUDIT.md`: rendered-PDF content audit from the latest compiled generic PDF.
- `docs/PDF_LAYOUT_AUDIT.md`: rendered-PDF page/log/layout audit from the latest compiled generic PDF.
- `docs/TEXT_FINALIZATION_AUDIT.md`: source/rendered text audit for unresolved placeholders, citations, references, and local path leaks.
- `docs/SUBMISSION_ROUTE_GUIDE.md`: route guide for anonymous review, full handoff, target venue conversion, and preprint use.
- `docs/VENUE_DECISION_AUDIT.md`: current audit of missing venue/template/review/page-rule decisions.
- `docs/FINAL_SUBMISSION_GATE.md`: current final-upload gate status for the venue-converted paper.
- `scripts/reproduce_main_commands.sh`: training/evaluation command template.
- `scripts/run_release_checks.sh`: full repository release-check wrapper.
- `scripts/post_compile_pdf_audit.py`: rendered-PDF text audit for the compiled paper.
- `scripts/prepare_venue_workspace.py`: helper for creating a target-venue conversion workspace.
- `scripts/pdf_layout_audit.py`: rendered-PDF metadata and LaTeX-log audit.
- `scripts/audit_final_submission_gate.py`: final upload gate after venue conversion.
"""
    write_text(BUNDLE / "README.md", readme)


def validate_bundle() -> dict[str, object]:
    required = [
        "main.tex",
        "appendix.tex",
        "diagnostic_branches_table.tex",
        "algorithm_dagig_grpo.tex",
        "references.bib",
        "Makefile",
        "tables/main_results_table.tex",
        "tables/node_credit_diagnostic_table.tex",
        "figures/dagig_method_diagram.tex",
        "figures/dagig_reward_equations.tex",
        "README.md",
        "docs/HANDOFF_README.md",
        "docs/SUBMISSION_ROUTE_GUIDE.md",
        "docs/MAINLINE_EVIDENCE_CHAIN.md",
        "docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.md",
        "docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.json",
        "scripts/reproduce_main_commands.sh",
        "scripts/post_compile_pdf_audit.py",
        "scripts/prepare_venue_workspace.py",
        "scripts/pdf_layout_audit.py",
        "scripts/audit_venue_decision_form.py",
        "scripts/audit_final_submission_gate.py",
    ]
    missing = [path for path in required if not (BUNDLE / path).exists()]
    empty = [path for path in required if (BUNDLE / path).exists() and (BUNDLE / path).stat().st_size == 0]
    parent_refs: list[str] = []
    for tex in BUNDLE.rglob("*.tex"):
        text = tex.read_text(encoding="utf-8")
        if "../" in text:
            parent_refs.append(str(tex.relative_to(BUNDLE)))
    make = subprocess.run(["make", "check"], cwd=str(BUNDLE), text=True, capture_output=True)
    return {
        "required": required,
        "missing": missing,
        "empty": empty,
        "tex_files_with_parent_refs": parent_refs,
        "make_check_returncode": make.returncode,
        "make_check_stdout": make.stdout.strip(),
        "make_check_stderr": make.stderr.strip(),
        "passed": not missing and not empty and not parent_refs and make.returncode == 0,
    }


def write_manifest(validation: dict[str, object]) -> None:
    files = []
    for path in source_package_files():
        files.append(
            {
                "path": str(path.relative_to(BUNDLE)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "bundle_root": str(BUNDLE),
        "tarball": str(TARBALL),
        "zipfile": str(ZIPFILE),
        "validation": validation,
        "files": files,
    }
    write_text(BUNDLE / "SUBMISSION_BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))


def write_tarball() -> None:
    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as tar:
        for path in source_package_files():
            tar.add(path, arcname=str(Path("DAGIG_Pix2Fact_paper_source_bundle") / path.relative_to(BUNDLE)))


def write_zipfile() -> None:
    if ZIPFILE.exists():
        ZIPFILE.unlink()
    with zipfile.ZipFile(ZIPFILE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_package_files():
            zf.write(path, arcname=str(path.relative_to(BUNDLE)))


def write_package_index() -> None:
    bundle_manifest = json.loads((BUNDLE / "SUBMISSION_BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    package = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "bundle_root": str(BUNDLE),
        "bundle_manifest": str(BUNDLE / "SUBMISSION_BUNDLE_MANIFEST.json"),
        "bundle_file_count": len(bundle_manifest.get("files", [])),
        "bundle_validation_passed": bool(bundle_manifest.get("validation", {}).get("passed")),
        "tarball": {
            "path": str(TARBALL),
            "bytes": TARBALL.stat().st_size,
            "sha256": sha256_file(TARBALL),
        },
        "zipfile": {
            "path": str(ZIPFILE),
            "bytes": ZIPFILE.stat().st_size,
            "sha256": sha256_file(ZIPFILE),
        },
        "review_clean_zip": None,
        "compiled_pdf": None,
        "core_entrypoints": [
            "main.tex",
            "appendix.tex",
            "references.bib",
            "tables/main_results_table.tex",
            "tables/node_credit_diagnostic_table.tex",
            "figures/dagig_method_diagram.tex",
            "figures/dagig_reward_equations.tex",
        "docs/GOAL_COMPLETION_AUDIT.md",
            "docs/FINAL_HANDOFF_PROMPT.md",
            "docs/MAINLINE_EVIDENCE_CHAIN.md",
            "docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.md",
            "docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.json",
            "docs/SUBMISSION_READINESS_REPORT.md",
            "docs/PDF_BUILD_PREFLIGHT_REPORT.md",
            "docs/POST_COMPILE_PDF_AUDIT.md",
            "docs/PDF_LAYOUT_AUDIT.md",
            "docs/TEXT_FINALIZATION_AUDIT.md",
            "docs/PAPER_LENGTH_AUDIT.md",
            "docs/SUBMISSION_ROUTE_GUIDE.md",
            "docs/VENUE_DECISION_AUDIT.md",
            "docs/FINAL_SUBMISSION_GATE.md",
            "docs/VENUE_TEMPLATE_CONVERSION_GUIDE.md",
            "docs/VENUE_DECISION_FORM.md",
            "scripts/post_compile_pdf_audit.py",
            "scripts/prepare_venue_workspace.py",
            "scripts/pdf_layout_audit.py",
            "scripts/audit_venue_decision_form.py",
            "scripts/audit_final_submission_gate.py",
        ],
        "verification_commands": [
            "cd outputs/dagig_paper_main_v1/paper_assets/submission_bundle && make check",
            "python scripts/dagig_paper_main/46_build_mainline_evidence_chain.py",
            "python scripts/dagig_paper_main/53_audit_mainline_schema_contract.py",
            "python scripts/dagig_paper_main/50_audit_venue_decision_form.py",
            "python scripts/dagig_paper_main/52_audit_final_submission_gate.py",
            "python scripts/dagig_paper_main/32_pdf_build_preflight.py --compile --require-pdf",
            "python scripts/dagig_paper_main/38_post_compile_pdf_audit.py --pdf outputs/dagig_paper_main_v1/paper_assets/submission_bundle/main.pdf --output_json outputs/dagig_paper_main_v1/paper_assets/post_compile_pdf_audit.json --output_md outputs/dagig_paper_main_v1/paper_assets/POST_COMPILE_PDF_AUDIT.md --require-pass",
            "python scripts/dagig_paper_main/40_pdf_layout_audit.py --pdf outputs/dagig_paper_main_v1/paper_assets/submission_bundle/main.pdf --log outputs/dagig_paper_main_v1/paper_assets/submission_bundle/main.log --output_json outputs/dagig_paper_main_v1/paper_assets/pdf_layout_audit.json --output_md outputs/dagig_paper_main_v1/paper_assets/PDF_LAYOUT_AUDIT.md --require-pass",
            "python scripts/dagig_paper_main/47_paper_text_finalization_audit.py",
            "python scripts/dagig_paper_main/41_sync_pdf_audits_to_bundle.py",
            "bash outputs/dagig_paper_main_v1/paper_assets/run_release_checks.sh",
            "python scripts/dagig_paper_main/42_verify_source_bundle_compile.py",
            "python scripts/dagig_paper_main/43_verify_review_clean_compile.py",
            "python scripts/dagig_paper_main/39_prepare_venue_workspace.py --force --compile_check",
            "python scripts/dagig_paper_main/48_audit_venue_workspace.py",
            "python scripts/dagig_paper_main/52_audit_final_submission_gate.py",
            "python scripts/dagig_paper_main/51_sync_goal_audit_to_bundle.py",
            "python scripts/dagig_paper_main/33_verify_submission_package.py",
            "python scripts/dagig_paper_main/42_verify_source_bundle_compile.py",
            "python scripts/dagig_paper_main/45_write_artifact_checksums.py",
            "sha256sum -c outputs/dagig_paper_main_v1/paper_assets/SHA256SUMS.txt",
            "python scripts/dagig_paper_main/49_build_submission_payload_index.py",
            "python scripts/dagig_paper_main/29_audit_paper_assets.py",
            "python scripts/dagig_paper_main/31_goal_completion_audit.py",
            "python scripts/dagig_paper_main/51_sync_goal_audit_to_bundle.py",
            "python scripts/dagig_paper_main/33_verify_submission_package.py",
            "python scripts/dagig_paper_main/42_verify_source_bundle_compile.py",
            "python scripts/dagig_paper_main/45_write_artifact_checksums.py",
            "sha256sum -c outputs/dagig_paper_main_v1/paper_assets/SHA256SUMS.txt",
            "python scripts/dagig_paper_main/49_build_submission_payload_index.py",
            "python scripts/dagig_paper_main/29_audit_paper_assets.py",
            "python scripts/dagig_paper_main/31_goal_completion_audit.py",
            "python scripts/dagig_paper_main/44_build_submission_readiness_dashboard.py",
        ],
        "post_compile_verification_commands": [
            "python scripts/dagig_paper_main/38_post_compile_pdf_audit.py --pdf outputs/dagig_paper_main_v1/paper_assets/submission_bundle/main.pdf --require-pass",
            "python scripts/dagig_paper_main/40_pdf_layout_audit.py --pdf outputs/dagig_paper_main_v1/paper_assets/submission_bundle/main.pdf --log outputs/dagig_paper_main_v1/paper_assets/submission_bundle/main.log --require-pass",
            "cd outputs/dagig_paper_main_v1/paper_assets/submission_bundle && python scripts/post_compile_pdf_audit.py --pdf main.pdf --output_json post_compile_pdf_audit.json --output_md POST_COMPILE_PDF_AUDIT.md --require-pass",
            "cd outputs/dagig_paper_main_v1/paper_assets/submission_bundle && python scripts/pdf_layout_audit.py --pdf main.pdf --log main.log --output_json pdf_layout_audit.json --output_md PDF_LAYOUT_AUDIT.md --require-pass",
        ],
        "remaining_external_verification": [
            "Convert source to the target venue template.",
            "Set author/anonymous metadata according to the target venue.",
            "Recompile and inspect the rendered PDF after venue-template conversion.",
        ],
    }
    clean_zip = ASSETS / "DAGIG_Pix2Fact_review_clean_source_bundle.zip"
    if clean_zip.exists():
        package["review_clean_zip"] = {
            "path": str(clean_zip),
            "bytes": clean_zip.stat().st_size,
            "sha256": sha256_file(clean_zip),
        }
    compiled_pdf = BUNDLE / "main.pdf"
    if compiled_pdf.exists() and compiled_pdf.stat().st_size > 0:
        package["compiled_pdf"] = {
            "path": str(compiled_pdf),
            "bytes": compiled_pdf.stat().st_size,
            "sha256": sha256_file(compiled_pdf),
        }
    write_text(PACKAGE_INDEX_JSON, json.dumps(package, indent=2, ensure_ascii=False))

    lines: list[str] = []
    lines.append("# Submission Package Index\n\n")
    lines.append("This file indexes the outer paper source packages. It is intentionally kept outside the zip/tarball because it records their checksums.\n\n")
    lines.append("## Packages\n\n")
    lines.append("| package | bytes | sha256 |\n")
    lines.append("|---|---:|---|\n")
    lines.append(f"| `{TARBALL}` | {package['tarball']['bytes']} | `{package['tarball']['sha256']}` |\n")
    lines.append(f"| `{ZIPFILE}` | {package['zipfile']['bytes']} | `{package['zipfile']['sha256']}` |\n")
    if package["review_clean_zip"]:
        clean = package["review_clean_zip"]
        lines.append(f"| `{clean['path']}` | {clean['bytes']} | `{clean['sha256']}` |\n")
    if package["compiled_pdf"]:
        pdf = package["compiled_pdf"]
        lines.append(f"| `{pdf['path']}` | {pdf['bytes']} | `{pdf['sha256']}` |\n")
    lines.append("\n")
    lines.append("## Bundle\n\n")
    lines.append(f"- bundle root: `{BUNDLE}`\n")
    lines.append(f"- bundle file count: `{package['bundle_file_count']}`\n")
    lines.append(f"- bundle validation passed: `{package['bundle_validation_passed']}`\n")
    lines.append("\n")
    lines.append("## Core Entrypoints\n\n")
    for entry in package["core_entrypoints"]:
        lines.append(f"- `{entry}`\n")
    lines.append("\n")
    lines.append("## Verification Commands\n\n")
    lines.append("```bash\n")
    for command in package["verification_commands"]:
        lines.append(f"{command}\n")
    lines.append("```\n\n")
    lines.append("## Post-Compile Verification Commands\n\n")
    lines.append("Run one of these only after `main.pdf` exists:\n\n")
    lines.append("```bash\n")
    for command in package["post_compile_verification_commands"]:
        lines.append(f"{command}\n")
    lines.append("```\n\n")
    lines.append("## Remaining External Verification\n\n")
    for item in package["remaining_external_verification"]:
        lines.append(f"- {item}\n")
    write_text(PACKAGE_INDEX_MD, "".join(lines))


def main() -> None:
    build_bundle()
    validation = validate_bundle()
    write_manifest(validation)
    # Revalidate after manifest is written so the final bundle includes it.
    validation = validate_bundle()
    write_manifest(validation)
    write_tarball()
    write_zipfile()
    clean_script = Path("scripts/dagig_paper_main/36_build_review_clean_bundle.py")
    if clean_script.exists():
        subprocess.run(["python", str(clean_script)], check=True)
    clean_anon_script = Path("scripts/dagig_paper_main/37_audit_review_clean_anonymity.py")
    if clean_anon_script.exists():
        subprocess.run(["python", str(clean_anon_script)], check=True)
    write_package_index()
    print(f"wrote {BUNDLE}")
    print(f"wrote {TARBALL}")
    print(f"wrote {ZIPFILE}")
    print(f"wrote {PACKAGE_INDEX_MD}")
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
