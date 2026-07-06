#!/usr/bin/env python3
"""Prepare a venue-conversion workspace from the verified paper source bundle.

This script does not perform venue-specific rewriting. It creates a clean
workspace that contains the official venue template (if provided), the audited
DAG-IG paper parts, and a generic compile-check wrapper. The goal is to make the
remaining venue conversion mechanical and auditable without reopening
experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCE_BUNDLE = Path("outputs/dagig_paper_main_v1/paper_assets/submission_bundle")
DEFAULT_OUTPUT = Path("outputs/dagig_paper_main_v1/paper_assets/venue_workspace")


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


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        copy_file(path, dst / path.relative_to(src))


def run(cmd: list[str], cwd: Path) -> dict[str, object]:
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def require_source_bundle(source: Path) -> None:
    required = [
        "main.tex",
        "references.bib",
        "venue_template_parts/body_sections.tex",
        "venue_template_parts/abstract_environment.tex",
        "venue_template_parts/appendix_sections.tex",
        "tables/main_results_table.tex",
        "tables/node_credit_diagnostic_table.tex",
        "figures/dagig_method_diagram.tex",
        "figures/dagig_reward_equations.tex",
        "algorithm_dagig_grpo.tex",
        "diagnostic_branches_table.tex",
    ]
    missing = [path for path in required if not (source / path).exists()]
    if missing:
        raise FileNotFoundError(f"source bundle is missing required files: {missing}")


def build_generic_wrapper(source: Path, output: Path) -> None:
    title = (source / "venue_template_parts/title.tex").read_text(encoding="utf-8").strip()
    tex = rf"""\documentclass[11pt]{{article}}

\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{graphicx}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta,positioning,fit,calc}}
\usepackage[hidelinks]{{hyperref}}

\title{{{title}}}
\author{{Anonymous Authors}}
\date{{}}

\begin{{document}}
\maketitle

\input{{paper_parts/abstract_environment.tex}}
\input{{paper_parts/body_sections.tex}}

\bibliographystyle{{plain}}
\bibliography{{references}}

\appendix
\input{{paper_parts/appendix_sections.tex}}

\end{{document}}
"""
    write_text(output / "main_generic_check.tex", tex)


def build_workspace(
    source: Path,
    output: Path,
    venue_template: Path | None,
    force: bool,
    compile_check: bool,
) -> dict[str, object]:
    require_source_bundle(source)
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    if venue_template:
        if not venue_template.exists():
            raise FileNotFoundError(f"venue template not found: {venue_template}")
        copy_tree(venue_template, output / "venue_template_original")
    else:
        write_text(
            output / "venue_template_original/PUT_OFFICIAL_TEMPLATE_HERE.md",
            "Place the official venue LaTeX template files in this directory, or rerun "
            "`prepare_venue_workspace.py --venue_template /path/to/template --force`.\n",
        )

    copy_tree(source / "venue_template_parts", output / "paper_parts")
    copy_tree(source / "tables", output / "tables")
    copy_tree(source / "figures", output / "figures")
    for name in [
        "algorithm_dagig_grpo.tex",
        "diagnostic_branches_table.tex",
        "references.bib",
        "main.tex",
        "appendix.tex",
        "Makefile",
    ]:
        src = source / name
        if src.exists():
            copy_file(src, output / "generic_source_reference" / name)
    copy_tree(source / "docs", output / "docs")
    if (source / "scripts/post_compile_pdf_audit.py").exists():
        copy_file(source / "scripts/post_compile_pdf_audit.py", output / "scripts/post_compile_pdf_audit.py")

    # Keep the files needed by paper_parts/body_sections.tex at workspace root.
    copy_file(source / "algorithm_dagig_grpo.tex", output / "algorithm_dagig_grpo.tex")
    copy_file(source / "diagnostic_branches_table.tex", output / "diagnostic_branches_table.tex")
    copy_file(source / "references.bib", output / "references.bib")
    build_generic_wrapper(source, output)

    compile_result: dict[str, object] | None = None
    if compile_check:
        compile_result = run(["pdflatex", "-interaction=nonstopmode", "main_generic_check.tex"], output)
        if compile_result["returncode"] == 0:
            compile_result = run(["bibtex", "main_generic_check"], output)
            if compile_result["returncode"] == 0:
                run(["pdflatex", "-interaction=nonstopmode", "main_generic_check.tex"], output)
                compile_result = run(["pdflatex", "-interaction=nonstopmode", "main_generic_check.tex"], output)

    readme = f"""# DAG-IG Venue Conversion Workspace

This workspace is generated from the verified DAG-IG Pix2Fact paper source bundle.
It is for venue/template conversion only. Do not change experimental numbers here.

## Current Inputs

- source bundle: `{source}`
- venue template provided: `{bool(venue_template)}`
- generated at UTC: `{datetime.now(timezone.utc).isoformat()}`

## Directory Layout

- `venue_template_original/`: official venue template files, if provided.
- `paper_parts/`: extracted title, abstract, body sections, and appendix sections.
- `tables/`: audited main-result and reward-diagnostic tables.
- `figures/`: TikZ method diagram and reward equations.
- `generic_source_reference/`: the already-verified generic article source.
- `docs/`: audits, claim boundaries, readiness reports, and handoff notes.
- `main_generic_check.tex`: generic compile-check wrapper assembled from the venue parts.

## Conversion Procedure

1. Start from the venue template's main `.tex` file under `venue_template_original/`.
2. Insert `paper_parts/abstract_environment.tex` or the contents of `paper_parts/abstract.tex` into the venue abstract block.
3. Insert `paper_parts/body_sections.tex` after the venue `\\maketitle`/abstract area.
4. Copy or keep these workspace-root paths available to the venue source:
   - `tables/`
   - `figures/`
   - `algorithm_dagig_grpo.tex`
   - `diagnostic_branches_table.tex`
   - `references.bib`
5. Add `\\appendix` and `\\input{{paper_parts/appendix_sections.tex}}` only if the venue allows appendix in the main PDF.
6. Compile the venue PDF, then run:

```bash
python scripts/post_compile_pdf_audit.py --pdf main.pdf --output_json post_compile_pdf_audit.json --output_md POST_COMPILE_PDF_AUDIT.md --require-pass
```

## Invariants

- Main method: DAG-IG node-level GRPO, not DAG-SFT.
- Reported numbers must match `tables/main_results_table.tex`.
- Reward diagnostics must match `tables/node_credit_diagnostic_table.tex`.
- Evaluation is frozen offline BM25, not live web search.
- Remaining bottlenecks and limitations must stay visible.
"""
    write_text(output / "README_VENUE_WORKSPACE.md", readme)

    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        files.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(source),
        "output_dir": str(output),
        "venue_template": str(venue_template) if venue_template else None,
        "compile_check_requested": compile_check,
        "compile_check": compile_result,
        "files": files,
    }
    write_text(output / "VENUE_WORKSPACE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--venue_template", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Replace output_dir if it already exists.")
    parser.add_argument("--compile_check", action="store_true", help="Compile main_generic_check.tex in the workspace.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_workspace(
        source=args.source_bundle,
        output=args.output_dir,
        venue_template=args.venue_template,
        force=args.force,
        compile_check=args.compile_check,
    )
    print(f"wrote {manifest['output_dir']}")
    print(f"wrote {args.output_dir / 'VENUE_WORKSPACE_MANIFEST.json'}")
    if args.compile_check:
        result = manifest.get("compile_check") or {}
        print(f"compile_check_returncode={result.get('returncode')}")
        if result.get("returncode") != 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
