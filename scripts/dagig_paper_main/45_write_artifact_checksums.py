#!/usr/bin/env python3
"""Write standard checksums for paper handoff artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSETS = Path("outputs/dagig_paper_main_v1/paper_assets")
OUT_SHA256SUMS = ASSETS / "SHA256SUMS.txt"
OUT_JSON = ASSETS / "ARTIFACT_CHECKSUMS.json"
OUT_MD = ASSETS / "ARTIFACT_CHECKSUMS.md"

ARTIFACTS = [
    ("compiled_pdf", ASSETS / "submission_bundle/main.pdf"),
    ("full_source_zip", ASSETS / "DAGIG_Pix2Fact_overleaf_source_bundle.zip"),
    ("full_source_tarball", ASSETS / "DAGIG_Pix2Fact_paper_source_bundle.tar.gz"),
    ("review_clean_zip", ASSETS / "DAGIG_Pix2Fact_review_clean_source_bundle.zip"),
    ("package_index_json", ASSETS / "SUBMISSION_PACKAGE_INDEX.json"),
    ("package_index_md", ASSETS / "SUBMISSION_PACKAGE_INDEX.md"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    empty: list[str] = []
    for label, path in ARTIFACTS:
        if not path.exists():
            missing.append(str(path))
            continue
        if path.stat().st_size == 0:
            empty.append(str(path))
        artifacts.append(
            {
                "label": label,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256sums_path": str(OUT_SHA256SUMS),
        "verification_command": f"sha256sum -c {OUT_SHA256SUMS}",
        "artifacts": artifacts,
        "missing": missing,
        "empty": empty,
        "passed": not missing and not empty and len(artifacts) == len(ARTIFACTS),
    }
    return manifest


def write_sha256sums(manifest: dict[str, Any]) -> None:
    lines = [f"{item['sha256']}  {item['path']}\n" for item in manifest["artifacts"]]
    OUT_SHA256SUMS.write_text("".join(lines), encoding="utf-8")


def write_markdown(manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Artifact Checksums\n\n")
    lines.append("This file records checksums for the final paper handoff artifacts. Run the verification command from the repository root.\n\n")
    lines.append(f"- created_at_utc: `{manifest['created_at_utc']}`\n")
    lines.append(f"- passed: `{manifest['passed']}`\n")
    lines.append(f"- standard checksum file: `{manifest['sha256sums_path']}`\n")
    lines.append(f"- verification command: `{manifest['verification_command']}`\n\n")
    lines.append("| artifact | bytes | sha256 | path |\n")
    lines.append("|---|---:|---|---|\n")
    for item in manifest["artifacts"]:
        lines.append(f"| {item['label']} | {item['bytes']} | `{item['sha256']}` | `{item['path']}` |\n")
    if manifest["missing"] or manifest["empty"]:
        lines.append("\n## Problems\n\n")
        if manifest["missing"]:
            lines.append(f"- missing: `{manifest['missing']}`\n")
        if manifest["empty"]:
            lines.append(f"- empty: `{manifest['empty']}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    manifest = build_manifest()
    OUT_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_sha256sums(manifest)
    write_markdown(manifest)
    print(f"wrote {OUT_SHA256SUMS}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
