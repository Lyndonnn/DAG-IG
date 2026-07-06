#!/usr/bin/env python3
"""Build a minimal review/Overleaf-clean LaTeX source bundle.

The existing submission bundle is intentionally rich: it includes handoff docs,
audits, scripts, and local reproducibility paths. This script creates a smaller
bundle containing only paper source files and venue-template snippets, avoiding
internal docs/scripts that are not needed for review compilation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
FULL_BUNDLE = ASSETS / "submission_bundle"
CLEAN_BUNDLE = ASSETS / "review_clean_bundle"
CLEAN_ZIP = ASSETS / "DAGIG_Pix2Fact_review_clean_source_bundle.zip"
CLEAN_MANIFEST = ASSETS / "REVIEW_CLEAN_BUNDLE_MANIFEST.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    return {
        "cmd": " ".join(cmd),
        "cwd": str(cwd) if cwd else None,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def build_clean_bundle() -> None:
    if CLEAN_BUNDLE.exists():
        shutil.rmtree(CLEAN_BUNDLE)
    CLEAN_BUNDLE.mkdir(parents=True, exist_ok=True)

    root_files = [
        "main.tex",
        "appendix.tex",
        "diagnostic_branches_table.tex",
        "algorithm_dagig_grpo.tex",
        "references.bib",
        "Makefile",
    ]
    for name in root_files:
        copy_file(FULL_BUNDLE / name, CLEAN_BUNDLE / name)

    for subdir in ["tables", "figures"]:
        for src in sorted((FULL_BUNDLE / subdir).glob("*")):
            if src.is_file():
                copy_file(src, CLEAN_BUNDLE / subdir / src.name)

    venue_src = FULL_BUNDLE / "venue_template_parts"
    if venue_src.exists():
        for src in sorted(venue_src.glob("*.tex")):
            copy_file(src, CLEAN_BUNDLE / "venue_template_parts" / src.name)
        readme = venue_src / "README.md"
        if readme.exists():
            copy_file(readme, CLEAN_BUNDLE / "venue_template_parts/README.md")

    readme = """# DAG-IG Pix2Fact Review-Clean Source Bundle

This is a minimal LaTeX source bundle for the DAG-IG / Pix2Fact paper draft.

It contains only files needed for paper compilation or target-template conversion.
It intentionally excludes internal handoff reports, experiment logs, scripts, and
machine-local paths.

## Build

```bash
make check
make all
```

`make check` verifies required source files and lightweight TeX hygiene. `make all`
requires `pdflatex` and `bibtex`.

## Contents

- `main.tex`: full standalone paper draft.
- `appendix.tex`: appendix sections.
- `references.bib`: bibliography.
- `tables/`: audited result tables.
- `figures/`: TikZ method diagram and reward equations.
- `venue_template_parts/`: snippets for moving the paper into another venue template.

Main method: DAG-IG node-level GRPO for a two-stage multimodal search agent.
"""
    write_text(CLEAN_BUNDLE / "README.md", readme)


def validate_clean_bundle() -> dict[str, Any]:
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
        "venue_template_parts/body_sections.tex",
        "venue_template_parts/abstract.tex",
        "README.md",
    ]
    missing = [path for path in required if not (CLEAN_BUNDLE / path).exists()]
    empty = [path for path in required if (CLEAN_BUNDLE / path).exists() and (CLEAN_BUNDLE / path).stat().st_size == 0]
    forbidden_patterns = ["/root/", "outputs/", "scripts/"]
    leaks: list[dict[str, Any]] = []
    for path in sorted(p for p in CLEAN_BUNDLE.rglob("*") if p.is_file()):
        if path.suffix not in {".tex", ".md", ".bib"}:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            hits = [pattern for pattern in forbidden_patterns if pattern in line]
            if hits:
                leaks.append({"file": str(path.relative_to(CLEAN_BUNDLE)), "line": lineno, "patterns": hits})
    disallowed_dirs = [name for name in ["docs", "scripts"] if (CLEAN_BUNDLE / name).exists()]
    make_check = run(["make", "check"], cwd=CLEAN_BUNDLE)
    return {
        "required": required,
        "missing": missing,
        "empty": empty,
        "forbidden_patterns": forbidden_patterns,
        "path_leaks": leaks,
        "disallowed_dirs": disallowed_dirs,
        "make_check": make_check,
        "passed": not missing and not empty and not leaks and not disallowed_dirs and make_check["returncode"] == 0,
    }


def write_manifest(validation: dict[str, Any]) -> None:
    files = []
    for path in sorted(p for p in CLEAN_BUNDLE.rglob("*") if p.is_file()):
        files.append(
            {
                "path": str(path.relative_to(CLEAN_BUNDLE)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(CLEAN_BUNDLE),
        "zipfile": str(CLEAN_ZIP),
        "validation": validation,
        "files": files,
    }
    write_text(CLEAN_MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False))


def write_zip() -> None:
    if CLEAN_ZIP.exists():
        CLEAN_ZIP.unlink()
    with zipfile.ZipFile(CLEAN_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in CLEAN_BUNDLE.rglob("*") if p.is_file()):
            zf.write(path, arcname=str(path.relative_to(CLEAN_BUNDLE)))


def verify_zip(validation: dict[str, Any]) -> dict[str, Any]:
    extract_root = ASSETS / "review_clean_bundle_extract_check"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(CLEAN_ZIP) as zf:
        bad = zf.testzip()
        if bad:
            return {"passed": False, "error": f"zip test failed at {bad}"}
        names = zf.namelist()
        if any(name.endswith(".json") for name in names):
            return {"passed": False, "error": "review-clean zip contains JSON metadata"}
        if any(name.startswith("docs/") or name.startswith("scripts/") for name in names):
            return {"passed": False, "error": "review-clean zip contains docs/ or scripts/"}
        zf.extractall(extract_root)
    make_check = run(["make", "check"], cwd=extract_root)
    return {
        "extract_root": str(extract_root),
        "zip_sha256": sha256_file(CLEAN_ZIP),
        "make_check": make_check,
        "passed": validation["passed"] and make_check["returncode"] == 0,
    }


def main() -> None:
    build_clean_bundle()
    validation = validate_clean_bundle()
    write_manifest(validation)
    validation = validate_clean_bundle()
    write_manifest(validation)
    write_zip()
    zip_verification = verify_zip(validation)
    index = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_bundle": str(CLEAN_BUNDLE),
        "clean_zip": str(CLEAN_ZIP),
        "clean_manifest": str(CLEAN_MANIFEST),
        "clean_zip_bytes": CLEAN_ZIP.stat().st_size,
        "clean_zip_sha256": sha256_file(CLEAN_ZIP),
        "validation": validation,
        "zip_verification": zip_verification,
    }
    write_text(ASSETS / "REVIEW_CLEAN_BUNDLE_REPORT.json", json.dumps(index, indent=2, ensure_ascii=False))
    report = [
        "# Review-Clean Bundle Report\n\n",
        f"- clean bundle: `{CLEAN_BUNDLE}`\n",
        f"- clean zip: `{CLEAN_ZIP}`\n",
        f"- clean zip sha256: `{index['clean_zip_sha256']}`\n",
        f"- validation passed: `{validation['passed']}`\n",
        f"- zip verification passed: `{zip_verification['passed']}`\n",
        "\n",
        "This package excludes internal docs and scripts. Use the richer source bundle for handoff/reproducibility notes.\n",
    ]
    write_text(ASSETS / "REVIEW_CLEAN_BUNDLE_REPORT.md", "".join(report))
    print(f"wrote {CLEAN_BUNDLE}")
    print(f"wrote {CLEAN_ZIP}")
    print(f"wrote {ASSETS / 'REVIEW_CLEAN_BUNDLE_REPORT.md'}")
    if not validation["passed"] or not zip_verification["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
