#!/usr/bin/env python3
"""Audit train gold/support document coverage in the BM25 train corpus."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import load_corpus, read_jsonl, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "reports/train_gold_corpus_coverage"
TRAIN = Path("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl")
CORPUS = Path("outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl")


def evidence_urls(row: dict[str, Any]) -> list[str]:
    urls = []
    for item in row.get("evidence_urls") or []:
        if isinstance(item, dict) and item.get("url"):
            urls.append(str(item.get("url")))
        elif isinstance(item, str):
            urls.append(item)
    return urls


def classify(row: dict[str, Any], sample_docs: list[dict[str, Any]], gold_docs: list[dict[str, Any]]) -> str:
    if gold_docs:
        return "has_gold_doc"
    urls = evidence_urls(row)
    if not sample_docs and urls:
        return "evidence_urls_present_but_no_sample_docs_in_corpus"
    if sample_docs and urls:
        return "sample_docs_present_but_none_marked_gold"
    if not urls:
        return "no_evidence_urls_in_train_row"
    return "unknown_missing_gold_doc"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path, default=TRAIN)
    parser.add_argument("--corpus_path", type=Path, default=CORPUS)
    parser.add_argument("--out_dir", type=Path, default=OUT)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.train_file)
    docs = load_corpus(args.corpus_path)
    docs_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gold_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        sid = str(doc.get("sample_id"))
        docs_by_sample[sid].append(doc)
        if doc.get("is_gold"):
            gold_by_sample[sid].append(doc)

    cases = []
    class_counts = Counter()
    for row in rows:
        sid = str(row.get("sample_id"))
        sample_docs = docs_by_sample.get(sid, [])
        gold_docs = gold_by_sample.get(sid, [])
        cls = classify(row, sample_docs, gold_docs)
        class_counts[cls] += 1
        if cls != "has_gold_doc":
            cases.append(
                {
                    "sample_id": sid,
                    "classification": cls,
                    "question": row.get("question"),
                    "gold_answer": row.get("gold_answer"),
                    "semantic_anchor": row.get("semantic_anchor"),
                    "hf_search_query": row.get("hf_search_query"),
                    "evidence_urls": evidence_urls(row),
                    "positive_doc_ids": row.get("positive_doc_ids") or [],
                    "n_sample_docs_in_corpus": len(sample_docs),
                    "sample_docs": [
                        {
                            "doc_id": doc.get("doc_id"),
                            "title": doc.get("title"),
                            "url": doc.get("url"),
                            "is_gold": doc.get("is_gold"),
                            "source": doc.get("source"),
                        }
                        for doc in sample_docs[:5]
                    ],
                }
            )

    summary = {
        "train_samples": len(rows),
        "corpus_docs": len(docs),
        "samples_with_any_corpus_doc": sum(1 for row in rows if docs_by_sample.get(str(row.get("sample_id")))),
        "samples_with_gold_doc": sum(1 for row in rows if gold_by_sample.get(str(row.get("sample_id")))),
        "samples_missing_gold_doc": len(cases),
        "classification_counts": dict(class_counts),
        "outputs": {
            "cases": str(out_dir / "train_missing_gold_doc_cases.jsonl"),
            "report": str(out_dir / "TRAIN_GOLD_CORPUS_COVERAGE_REPORT.md"),
        },
    }
    write_jsonl(out_dir / "train_missing_gold_doc_cases.jsonl", cases)
    write_json(out_dir / "train_gold_corpus_coverage_summary.json", summary)

    lines: list[str] = []
    lines.append("# Train Gold Corpus Coverage Report\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This audits whether each train sample has at least one `is_gold=true` support document in the BM25 train corpus. "
        "This matters because query/evidence credit and GRPO rewards cannot learn retrieval for samples whose support doc is absent or not marked gold.\n\n"
    )
    lines.append(f"- train file: `{args.train_file}`\n")
    lines.append(f"- corpus path: `{args.corpus_path}`\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"- train samples: `{len(rows)}`\n")
    lines.append(f"- corpus docs: `{len(docs)}`\n")
    lines.append(f"- samples with any corpus doc: `{summary['samples_with_any_corpus_doc']}`\n")
    lines.append(f"- samples with gold doc: `{summary['samples_with_gold_doc']}`\n")
    lines.append(f"- samples missing gold doc: `{summary['samples_missing_gold_doc']}`\n\n")
    lines.append("## Classification Counts\n\n")
    lines.append("| class | count |\n|---|---:|\n")
    for cls, count in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {cls} | {count} |\n")
    lines.append("\n## Decision\n\n")
    if cases:
        lines.append(
            "This is a data/corpus construction issue for the affected samples. Before more GRPO, rebuild or patch the train BM25 corpus so every train row with evidence URLs has at least one corresponding `is_gold=true` support doc, then rerun reward/coverage audits. "
            "Do not treat missing-gold samples as model failures.\n"
        )
    else:
        lines.append("Train corpus gold coverage is complete; remaining retrieval misses are model/query issues rather than corpus coverage issues.\n")
    lines.append("\n## Artifacts\n\n")
    lines.append(f"- cases: `{out_dir / 'train_missing_gold_doc_cases.jsonl'}`\n")
    lines.append(f"- summary: `{out_dir / 'train_gold_corpus_coverage_summary.json'}`\n")
    (out_dir / "TRAIN_GOLD_CORPUS_COVERAGE_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", out_dir / "TRAIN_GOLD_CORPUS_COVERAGE_REPORT.md")


if __name__ == "__main__":
    main()
