#!/usr/bin/env python3
"""Build a train BM25 corpus with missing gold flags repaired.

Uniform rule:
- keep existing is_gold=true docs;
- for train samples with no gold doc, mark same-sample corpus docs whose URL is
  listed in the train row evidence_urls as is_gold=true;
- record fix metadata and do not overwrite the original corpus.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import answer_match_details, load_corpus, read_jsonl, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "derived_assets"
REPORT_DIR = ROOT / "reports/train_gold_corpus_coverage"
TRAIN = Path("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl")
CORPUS = Path("outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl")
FIXED = OUT / "bm25_train_corpus_goldfixed.jsonl"


def evidence_urls(row: dict[str, Any]) -> set[str]:
    urls = set()
    for item in row.get("evidence_urls") or []:
        if isinstance(item, dict) and item.get("url"):
            urls.add(str(item.get("url")))
        elif isinstance(item, str):
            urls.add(item)
    return urls


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(TRAIN)
    row_by_sample = {str(row.get("sample_id")): row for row in rows}
    docs = load_corpus(CORPUS)
    gold_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    docs_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        sid = str(doc.get("sample_id"))
        docs_by_sample[sid].append(doc)
        if doc.get("is_gold"):
            gold_by_sample[sid].append(doc)

    fixed_docs = []
    fixed_cases = []
    reason_counts = Counter()
    for doc in docs:
        new_doc = dict(doc)
        sid = str(doc.get("sample_id"))
        row = row_by_sample.get(sid)
        if row and not gold_by_sample.get(sid):
            urls = evidence_urls(row)
            if str(doc.get("url")) in urls:
                match = answer_match_details(
                    " ".join(str(doc.get(k, "")) for k in ("title", "url", "text")),
                    str(row.get("gold_answer") or ""),
                )
                new_doc["is_gold"] = True
                new_doc["gold_fix_applied"] = True
                new_doc["gold_fix_reason"] = (
                    "answer_match_evidence_url" if match.get("answer_correct") else "evidence_url_missing_gold_flag"
                )
                new_doc["gold_fix_match_type"] = match.get("answer_match_type")
                reason_counts[new_doc["gold_fix_reason"]] += 1
                fixed_cases.append(
                    {
                        "sample_id": sid,
                        "doc_id": doc.get("doc_id"),
                        "url": doc.get("url"),
                        "old_is_gold": doc.get("is_gold"),
                        "new_is_gold": True,
                        "gold_fix_reason": new_doc["gold_fix_reason"],
                        "gold_fix_match_type": new_doc["gold_fix_match_type"],
                    }
                )
        fixed_docs.append(new_doc)

    fixed_gold_by_sample: dict[str, int] = defaultdict(int)
    for doc in fixed_docs:
        if doc.get("is_gold"):
            fixed_gold_by_sample[str(doc.get("sample_id"))] += 1
    missing_after = [str(row.get("sample_id")) for row in rows if not fixed_gold_by_sample.get(str(row.get("sample_id")))]
    summary = {
        "input_corpus": str(CORPUS),
        "output_corpus": str(FIXED),
        "train_samples": len(rows),
        "docs": len(docs),
        "samples_with_gold_before": sum(1 for row in rows if gold_by_sample.get(str(row.get("sample_id")))),
        "samples_with_gold_after": sum(1 for row in rows if fixed_gold_by_sample.get(str(row.get("sample_id")))),
        "missing_gold_after": len(missing_after),
        "docs_fixed": len(fixed_cases),
        "fix_reason_counts": dict(reason_counts),
        "missing_after_sample_ids": missing_after,
    }
    write_jsonl(FIXED, fixed_docs)
    write_jsonl(REPORT_DIR / "goldfixed_doc_changes.jsonl", fixed_cases)
    write_json(REPORT_DIR / "goldfixed_train_corpus_summary.json", summary)

    lines: list[str] = []
    lines.append("# Gold-Fixed Train Corpus Manifest\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This builds a fixed train BM25 corpus without overwriting the original. "
        "For train samples with no `is_gold=true` doc, same-sample docs whose URL appears in the row's `evidence_urls` are marked as gold. "
        "The rule is uniform and train-only.\n\n"
    )
    lines.append("## Summary\n\n")
    lines.append(f"- input corpus: `{CORPUS}`\n")
    lines.append(f"- output corpus: `{FIXED}`\n")
    lines.append(f"- train samples: `{len(rows)}`\n")
    lines.append(f"- samples with gold before: `{summary['samples_with_gold_before']}`\n")
    lines.append(f"- samples with gold after: `{summary['samples_with_gold_after']}`\n")
    lines.append(f"- missing gold after: `{summary['missing_gold_after']}`\n")
    lines.append(f"- docs fixed: `{len(fixed_cases)}`\n")
    lines.append(f"- fix reason counts: `{dict(reason_counts)}`\n\n")
    lines.append("## Decision\n\n")
    lines.append(
        "Use this fixed corpus for future train-side reward/coverage audits and any future GRPO runs. "
        "Do not compare future train-side reward statistics to old runs without noting the corpus fix. "
        "Dev/test corpora are not modified by this script.\n\n"
    )
    lines.append("## Artifacts\n\n")
    lines.append(f"- fixed corpus: `{FIXED}`\n")
    lines.append(f"- changed docs: `{REPORT_DIR / 'goldfixed_doc_changes.jsonl'}`\n")
    lines.append(f"- machine summary: `{REPORT_DIR / 'goldfixed_train_corpus_summary.json'}`\n")
    (REPORT_DIR / "GOLDFIXED_TRAIN_CORPUS_MANIFEST.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", FIXED)
    print("wrote", REPORT_DIR / "GOLDFIXED_TRAIN_CORPUS_MANIFEST.md")


if __name__ == "__main__":
    main()
