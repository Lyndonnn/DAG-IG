#!/usr/bin/env python3
"""Sync final PDF audit reports into the submission bundle and repackage.

`30_build_submission_bundle.py` creates the source bundle before PDF compilation.
The post-compile audits are generated after compilation, so the bundle's
`docs/` copies can drift by one run unless they are explicitly refreshed. This
script copies the latest root audit reports into `submission_bundle/docs/`,
rewrites the bundle manifest, and regenerates the zip/tarball/package index.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
BUNDLE = ASSETS / "submission_bundle"
BUILD_SCRIPT = Path("scripts/dagig_paper_main/30_build_submission_bundle.py")

REPORTS_TO_SYNC = [
    "PDF_BUILD_PREFLIGHT_REPORT.md",
    "pdf_build_preflight.json",
    "POST_COMPILE_PDF_AUDIT.md",
    "post_compile_pdf_audit.json",
    "PDF_LAYOUT_AUDIT.md",
    "pdf_layout_audit.json",
    "TEXT_FINALIZATION_AUDIT.md",
    "text_finalization_audit.json",
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_build_module():
    spec = importlib.util.spec_from_file_location("dagig_build_submission_bundle", BUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BUILD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not BUNDLE.exists():
        raise FileNotFoundError(BUNDLE)

    for name in REPORTS_TO_SYNC:
        src = ASSETS / name
        dst = BUNDLE / "docs" / name
        if src.exists():
            copy_file(src, dst)

    # Keep bundle-root audit outputs aligned too when they exist.
    for name in [
        "POST_COMPILE_PDF_AUDIT.md",
        "post_compile_pdf_audit.json",
        "PDF_LAYOUT_AUDIT.md",
        "pdf_layout_audit.json",
    ]:
        src = ASSETS / name
        if src.exists():
            copy_file(src, BUNDLE / name)

    build = load_build_module()
    validation = build.validate_bundle()
    build.write_manifest(validation)
    validation = build.validate_bundle()
    build.write_manifest(validation)
    build.write_tarball()
    build.write_zipfile()
    build.write_package_index()

    if not validation["passed"]:
        raise SystemExit(1)
    print(f"synced PDF audits into {BUNDLE}")
    print(f"rewrote {build.TARBALL}")
    print(f"rewrote {build.ZIPFILE}")
    print(f"rewrote {build.PACKAGE_INDEX_MD}")


if __name__ == "__main__":
    main()
