#!/usr/bin/env python3
"""Sync late-stage gate reports into the source bundle and repackage.

Some gate reports are generated after the initial source bundle is created.
This step refreshes their bundle copies and repackages the source artifacts.
It is safe to run more than once in the release sequence.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
BUNDLE = ASSETS / "submission_bundle"
BUILD_SCRIPT = Path("scripts/dagig_paper_main/30_build_submission_bundle.py")

FILES_TO_SYNC = [
    "MAINLINE_SCHEMA_CONTRACT_AUDIT.md",
    "MAINLINE_SCHEMA_CONTRACT_AUDIT.json",
    "GOAL_COMPLETION_AUDIT.md",
    "FINAL_SUBMISSION_GATE.md",
    "final_submission_gate.json",
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
    for name in FILES_TO_SYNC:
        copy_file(ASSETS / name, BUNDLE / "docs" / name)

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
    print(f"synced late-stage audits into {BUNDLE}")
    print(f"rewrote {build.TARBALL}")
    print(f"rewrote {build.ZIPFILE}")
    print(f"rewrote {build.PACKAGE_INDEX_MD}")


if __name__ == "__main__":
    main()
