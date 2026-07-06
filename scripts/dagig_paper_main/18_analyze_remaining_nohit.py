#!/usr/bin/env python3
"""Analyze no-hit train samples still unrecovered after query mining."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import load_corpus, read_jsonl, tokenize, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "reports/remaining_nohit_analysis"
GROUPS = ROOT / "reports/hard_retrieval_mining/train_hard_retrieval_groups.jsonl"
TRAIN_FILE = Path("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl")
CORPUS = Path("outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl")
SIMPLE_BEST = ROOT / "reports/nohit_query_candidate_mining/nohit_best_queries.jsonl"
SIMPLE_CAND = ROOT / "reports/nohit_query_candidate_mining/nohit_query_candidates.jsonl"
SUPPORT_BEST = ROOT / "reports/support_doc_query_candidate_mining/support_doc_best_queries.jsonl"
SUPPORT_CAND = ROOT / "reports/support_doc_query_candidate_mining/support_doc_query_candidates.jsonl"


def short(text: Any, n: int = 220) -> str:
    value = " ".join(str(text or "").split())
    return value[:n]


def semantic_anchor(row: dict[str, Any]) -> str:
    grounding = row.get("grounding")
    if isinstance(grounding, dict) and grounding.get("semantic_anchor"):
        return str(grounding.get("semantic_anchor") or "")
    return str(row.get("semantic_anchor") or "")


def overlap(a: str, b: str) -> list[str]:
    stop = {"the", "and", "for", "with", "from", "that", "this", "what", "where", "when", "which"}
    aa = {t for t in tokenize(a) if len(t) >= 3 and t not in stop}
    bb = {t for t in tokenize(b) if len(t) >= 3 and t not in stop}
    return sorted(aa & bb)


def group_by_sample(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id"))].append(row)
    return grouped


def best_wrong_doc(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    for candidate in candidates:
        docs = candidate.get("top_docs") or []
        if not docs:
            continue
        top = docs[0]
        score = float(top.get("score") or 0.0)
        if best is None or score > best["score"]:
            best = {
                "query": candidate.get("query"),
                "source": candidate.get("source"),
                "score": score,
                "top_doc_sample_id": top.get("sample_id"),
                "top_doc_title": top.get("title"),
                "top_doc_url": top.get("url"),
            }
    return best


def classify(row: dict[str, Any], gold_docs: list[dict[str, Any]], simple_cands: list[dict[str, Any]], support_cands: list[dict[str, Any]]) -> str:
    if not gold_docs:
        return "missing_gold_doc_in_train_corpus"
    if not support_cands and not simple_cands:
        return "candidate_generation_empty_or_all_filtered"
    anchor = semantic_anchor(row)
    gold_text = " ".join(" ".join(str(doc.get(k, "")) for k in ("title", "domain", "url", "text")) for doc in gold_docs)
    if anchor and not overlap(anchor, gold_text):
        return "semantic_anchor_not_in_support_doc_text"
    if support_cands:
        wrong = best_wrong_doc(support_cands)
        if wrong and wrong.get("top_doc_sample_id") != row.get("sample_id"):
            return "bm25_wrong_cluster_even_with_support_terms"
    return "unrecovered_query_wording_or_support_text_mismatch"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_rows = {str(row.get("sample_id")): row for row in read_jsonl(TRAIN_FILE)}
    corpus = load_corpus(CORPUS)
    gold_docs_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in corpus:
        if doc.get("is_gold"):
            gold_docs_by_sample[str(doc.get("sample_id"))].append(doc)

    nohit_ids = {
        str(row.get("sample_id"))
        for row in read_jsonl(GROUPS)
        if row.get("status") == "candidate_insufficient_no_hit_rollout"
    }
    simple_recovered = {str(row.get("sample_id")) for row in read_jsonl(SIMPLE_BEST)}
    support_recovered = {str(row.get("sample_id")) for row in read_jsonl(SUPPORT_BEST)}
    remaining = sorted(nohit_ids - simple_recovered - support_recovered)
    simple_candidates = group_by_sample(read_jsonl(SIMPLE_CAND))
    support_candidates = group_by_sample(read_jsonl(SUPPORT_CAND))

    cases = []
    class_counts = Counter()
    for sid in remaining:
        row = train_rows.get(sid, {"sample_id": sid})
        gold_docs = gold_docs_by_sample.get(sid, [])
        simple_cands = simple_candidates.get(sid, [])
        support_cands = support_candidates.get(sid, [])
        cls = classify(row, gold_docs, simple_cands, support_cands)
        class_counts[cls] += 1
        wrong = best_wrong_doc(support_cands) or best_wrong_doc(simple_cands)
        cases.append(
            {
                "sample_id": sid,
                "classification": cls,
                "question": row.get("question"),
                "gold_answer": row.get("gold_answer"),
                "semantic_anchor": semantic_anchor(row),
                "hf_search_query": row.get("hf_search_query"),
                "n_gold_docs": len(gold_docs),
                "gold_docs": [
                    {
                        "title": doc.get("title"),
                        "domain": doc.get("domain"),
                        "url": doc.get("url"),
                        "text": short(doc.get("text")),
                    }
                    for doc in gold_docs[:3]
                ],
                "n_simple_candidates": len(simple_cands),
                "n_support_doc_candidates": len(support_cands),
                "best_wrong_retrieval": wrong,
            }
        )

    summary = {
        "original_nohit_samples": len(nohit_ids),
        "simple_recovered": len(simple_recovered),
        "support_doc_recovered": len(support_recovered),
        "remaining_unrecovered": len(remaining),
        "classification_counts": dict(class_counts),
        "outputs": {
            "cases": str(OUT / "remaining_nohit_cases.jsonl"),
            "report": str(OUT / "REMAINING_NOHIT_ANALYSIS_REPORT.md"),
        },
    }
    write_jsonl(OUT / "remaining_nohit_cases.jsonl", cases)
    write_json(OUT / "remaining_nohit_summary.json", summary)

    lines: list[str] = []
    lines.append("# Remaining No-Hit Analysis Report\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This analyzes train samples that had no support-hit rollout and were still not recovered by simple clean query recipes or support-document lexical mining. "
        "It does not use dev/test labels and does not train a model.\n\n"
    )
    lines.append("## Counts\n\n")
    lines.append(f"- original no-hit samples: `{len(nohit_ids)}`\n")
    lines.append(f"- recovered by simple recipes: `{len(simple_recovered)}`\n")
    lines.append(f"- recovered by support-doc lexical mining: `{len(support_recovered)}`\n")
    lines.append(f"- remaining unrecovered: `{len(remaining)}`\n\n")
    lines.append("## Failure Classes\n\n")
    lines.append("| class | count |\n|---|---:|\n")
    for cls, count in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {cls} | {count} |\n")
    lines.append("\n## Decision\n\n")
    lines.append(
        "The remaining unrecovered set should not trigger another GRPO run. First inspect these cases for support/corpus mismatch and semantic-anchor mismatch. "
        "If most are support-text mismatch, the fix is data/corpus cleanup. If most are wrong-cluster retrieval, the fix is stronger query generation or retrieval scoring. "
        "Keep this separate from the corrected paper-facing KL-fixed two-seed result.\n\n"
    )
    lines.append("## Artifacts\n\n")
    lines.append(f"- cases: `{OUT / 'remaining_nohit_cases.jsonl'}`\n")
    lines.append(f"- summary: `{OUT / 'remaining_nohit_summary.json'}`\n")
    (OUT / "REMAINING_NOHIT_ANALYSIS_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "REMAINING_NOHIT_ANALYSIS_REPORT.md")


if __name__ == "__main__":
    main()
