#!/usr/bin/env python3
"""Static length and structure audit for venue-template planning."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
LATEX = ASSETS / "latex"
OUT_JSON = ASSETS / "paper_length_audit.json"
OUT_MD = ASSETS / "PAPER_LENGTH_AUDIT.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_latex(text: str) -> str:
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\cite\{[^}]+\}", " ", text)
    text = re.sub(r"\\(?:ref|pageref|autoref)\{[^}]+\}", " ", text)
    text = re.sub(r"\\input\{[^}]+\}", " ", text)
    text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?(?:\{([^{}]*)\})?", r" \1 ", text)
    text = re.sub(r"[$_^{}\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    clean = strip_latex(text)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", clean))


def extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip() if match else ""


def section_word_counts(text: str) -> list[dict[str, Any]]:
    parts = re.split(r"(\\section\{[^}]+\})", text)
    rows: list[dict[str, Any]] = []
    for i in range(1, len(parts), 2):
        title_cmd = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        title_match = re.search(r"\\section\{([^}]+)\}", title_cmd)
        title = title_match.group(1) if title_match else title_cmd
        rows.append({"section": title, "words": word_count(body)})
    return rows


def main() -> None:
    main_tex = read(LATEX / "main.tex")
    appendix_tex = read(LATEX / "appendix.tex")
    abstract = extract(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", main_tex)
    body = extract(r"(\\section\{Introduction\}.+?)\\appendix", main_tex)
    references = re.findall(r"\\cite\{([^}]+)\}", main_tex + "\n" + appendix_tex)
    cite_keys = sorted({key.strip() for group in references for key in group.split(",") if key.strip()})

    counts = {
        "abstract_words": word_count(abstract),
        "main_body_words": word_count(body),
        "appendix_words": word_count(appendix_tex),
        "total_words_excluding_bib": word_count(abstract) + word_count(body) + word_count(appendix_tex),
        "main_sections": len(re.findall(r"\\section\{", body)),
        "appendix_sections": len(re.findall(r"\\section\{", appendix_tex)),
        "figures": len(re.findall(r"\\begin\{figure\}", main_tex + "\n" + appendix_tex)),
        "tables": len(re.findall(r"\\begin\{table\}", main_tex + "\n" + appendix_tex)),
        "citations": len(cite_keys),
        "labels": len(re.findall(r"\\label\{", main_tex + "\n" + appendix_tex)),
    }
    # Conservative rough estimate for 11pt single-column article-like drafts.
    counts["rough_text_pages_500wpp"] = round(counts["total_words_excluding_bib"] / 500.0, 2)
    counts["rough_main_text_pages_500wpp"] = round((counts["abstract_words"] + counts["main_body_words"]) / 500.0, 2)

    section_rows = section_word_counts(body)
    appendix_rows = section_word_counts(appendix_tex)
    warnings: list[str] = []
    if counts["abstract_words"] > 250:
        warnings.append("Abstract is longer than 250 words; many venues require a shorter abstract.")
    if counts["main_body_words"] > 4500:
        warnings.append("Main body exceeds 4500 words before bibliography; check target venue page limit.")
    if counts["figures"] + counts["tables"] > 5:
        warnings.append("Figure/table count may be high for short-format venues.")
    if counts["appendix_words"] > 1200:
        warnings.append("Appendix is substantial; verify whether supplementary material is allowed.")

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(LATEX / "main.tex"),
        "appendix_source": str(LATEX / "appendix.tex"),
        "counts": counts,
        "main_section_word_counts": section_rows,
        "appendix_section_word_counts": appendix_rows,
        "citation_keys": cite_keys,
        "warnings": warnings,
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Paper Length Audit\n\n")
    lines.append("This is a static approximation for venue-template planning. It does not replace PDF compilation.\n\n")
    lines.append("## Summary\n\n")
    for key, value in counts.items():
        lines.append(f"- {key}: `{value}`\n")
    lines.append("\n")
    lines.append("## Main Section Word Counts\n\n")
    lines.append("| section | approximate words |\n")
    lines.append("|---|---:|\n")
    for row in section_rows:
        lines.append(f"| {row['section']} | {row['words']} |\n")
    lines.append("\n")
    lines.append("## Appendix Section Word Counts\n\n")
    lines.append("| section | approximate words |\n")
    lines.append("|---|---:|\n")
    for row in appendix_rows:
        lines.append(f"| {row['section']} | {row['words']} |\n")
    lines.append("\n")
    lines.append("## Warnings\n\n")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}\n")
    else:
        lines.append("- No static length warnings.\n")
    lines.append("\n")
    lines.append("## Interpretation\n\n")
    lines.append(
        "The current draft is compact enough for most full-paper templates, but final page count "
        "depends on the target style file, bibliography style, and figure/table placement.\n"
    )
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
