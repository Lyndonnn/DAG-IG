#!/usr/bin/env python3
"""Generate venue-template-ready LaTeX snippets from the current paper source."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
LATEX = ASSETS / "latex"
OUT = ASSETS / "venue_template_parts"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"could not extract {label}")
    return match.group(1).strip() + "\n"


def bundle_paths(text: str) -> str:
    replacements = {
        "../figures/dagig_method_diagram.tex": "figures/dagig_method_diagram.tex",
        "../figures/dagig_reward_equations.tex": "figures/dagig_reward_equations.tex",
        "../main_results_table.tex": "tables/main_results_table.tex",
        "../node_credit_diagnostic_table.tex": "tables/node_credit_diagnostic_table.tex",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main() -> None:
    main_tex = read(LATEX / "main.tex")
    appendix = read(LATEX / "appendix.tex").strip() + "\n"
    title = extract(r"\\title\{(.+?)\}\s*\\author", main_tex, "title")
    abstract = extract(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", main_tex, "abstract")
    body = extract(r"(\\section\{Introduction\}.+?)\\appendix", main_tex, "body sections")
    body = bundle_paths(body)

    if "\\begin{document}" in body or "\\bibliography" in body:
        raise ValueError("body extraction included document wrapper or bibliography")

    write(OUT / "title.tex", title)
    write(OUT / "abstract.tex", abstract)
    write(OUT / "abstract_environment.tex", "\\begin{abstract}\n" + abstract + "\\end{abstract}\n")
    write(OUT / "body_sections.tex", body)
    write(OUT / "appendix_sections.tex", appendix)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(LATEX / "main.tex"),
        "appendix_source": str(LATEX / "appendix.tex"),
        "outputs": [
            "title.tex",
            "abstract.tex",
            "abstract_environment.tex",
            "body_sections.tex",
            "appendix_sections.tex",
            "README.md",
        ],
        "path_convention": "bundle-root relative paths for figures/tables",
        "do_not_edit_numbers_here": True,
    }
    write(OUT / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

    readme = """# Venue Template Parts

These files are generated from `latex/main.tex` and `latex/appendix.tex` to make venue-template conversion less error-prone.

Use these when moving the paper into a conference/journal template:

- `title.tex`: title text only.
- `abstract.tex`: abstract body only.
- `abstract_environment.tex`: abstract wrapped in a generic `abstract` environment.
- `body_sections.tex`: main sections from Introduction through Conclusion, without documentclass, preamble, bibliography, or appendix marker.
- `appendix_sections.tex`: appendix sections.

The body uses bundle-root relative paths such as `figures/...` and `tables/...`. Put `figures/`, `tables/`, `algorithm_dagig_grpo.tex`, `diagnostic_branches_table.tex`, and `references.bib` at the venue project root, or adjust paths once in the venue template.

Do not change experimental numbers in these snippets. If a number must change, regenerate the audited tables and rerun the paper asset audit.
"""
    write(OUT / "README.md", readme)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
