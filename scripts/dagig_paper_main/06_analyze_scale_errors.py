#!/usr/bin/env python3
"""Analyze remaining errors for the current paper-main checkpoint."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import read_jsonl, tokenize, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "reports/scale60_error_analysis"


def infer_answer_type(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(phone|telephone|contact number|call|hotline|tel)\b", q):
        return "phone"
    if re.search(r"\b(address|located|location|where|mailing address|store in|branch in)\b", q):
        return "address"
    if re.search(r"\b(opening time|opening hours|business hours|hours|closing|what time|check-?out)\b", q):
        return "time"
    if re.search(r"\b(price|cost|how much|pay|fee)\b", q):
        return "price"
    if re.search(r"\b(email|e-?mail)\b", q):
        return "email"
    if re.search(r"\b(how many|number of|population|gdp|rank|ranking|percent|percentage|value|revenue)\b", q):
        return "numeric"
    return "entity"


def compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def token_overlap(a: str, b: str) -> float:
    ta = {tok for tok in tokenize(a) if len(tok) >= 3}
    tb = {tok for tok in tokenize(b) if len(tok) >= 3}
    return len(ta & tb) / max(1, len(ta)) if ta else 0.0


def load_data(split: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in read_jsonl(Path(f"outputs/dagig_grpo_main/derived_assets/grpo_{split}.jsonl"))}


def classify_retrieval_miss(pred: dict[str, Any], data_row: dict[str, Any]) -> str:
    query = compact(pred.get("search_query", ""))
    q_tokens = tokenize(query)
    semantic_anchor = compact(data_row.get("semantic_anchor", ""))
    ground_expression = compact(data_row.get("ground_expression", ""))
    hf_query = compact(data_row.get("hf_search_query", ""))
    top_docs = pred.get("retrieved_docs") or []
    if not query:
        return "empty_query"
    if len(q_tokens) <= 3:
        return "query_too_short_or_generic"
    if semantic_anchor and token_overlap(semantic_anchor, query) == 0 and token_overlap(ground_expression, query) < 0.2:
        return "missing_semantic_anchor"
    if hf_query and token_overlap(hf_query, query) < 0.25:
        return "query_drift_from_teacher_search_intent"
    if top_docs:
        top_sample = str(top_docs[0].get("sample_id", ""))
        if top_sample and top_sample != str(pred.get("sample_id")):
            return "retrieves_wrong_sample_cluster"
    return "retrieval_miss_other"


def classify_answer_error(pred: dict[str, Any]) -> str:
    answer = compact(pred.get("final_answer", ""))
    question = compact(pred.get("question", ""))
    answer_type = infer_answer_type(question)
    if not answer:
        return "empty_answer"
    low = answer.lower()
    if any(x in low for x in ["don't have enough", "cannot determine", "unknown", "无法", "not enough"]):
        return "abstention_or_unknown"
    if answer_type == "phone" and not re.search(r"\d", answer):
        return "wrong_type_phone"
    if answer_type in {"numeric", "price", "time"} and not re.search(r"\d", answer):
        return f"wrong_type_{answer_type}"
    if answer_type == "address" and len(answer) < 8:
        return "too_short_address"
    if answer.startswith("http") and answer_type not in {"entity"}:
        return "url_instead_of_answer"
    return "answer_extraction_wrong"


def analyze_split(split: str) -> dict[str, Any]:
    data = load_data(split)
    pred_path = ROOT / f"two_stage_predictions/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_{split}.jsonl"
    rows = read_jsonl(pred_path)
    error_rows = []
    counts = Counter()
    answer_types = Counter()
    for row in rows:
        sid = str(row.get("sample_id"))
        data_row = data.get(sid, {})
        answer_type = infer_answer_type(str(row.get("question", "")))
        answer_types[answer_type] += 1
        if row.get("strict_success"):
            counts["strict_success"] += 1
            continue
        if not row.get("retrieval_top5_hit"):
            bottleneck = "retrieval_miss"
            subtype = classify_retrieval_miss(row, data_row)
        elif not row.get("answer_correct"):
            bottleneck = "hit_answer_wrong"
            subtype = classify_answer_error(row)
        else:
            bottleneck = "answer_correct_without_support"
            subtype = "unsupported_or_wrong_evidence"
        counts[bottleneck] += 1
        counts[f"{bottleneck}:{subtype}"] += 1
        top_docs = row.get("retrieved_docs") or []
        support_rank = None
        for doc in top_docs:
            if str(doc.get("sample_id")) == sid and doc.get("is_gold"):
                support_rank = doc.get("rank")
                break
        error_rows.append(
            {
                "sample_id": sid,
                "split": split,
                "bottleneck": bottleneck,
                "subtype": subtype,
                "answer_type": answer_type,
                "question": row.get("question"),
                "gold_answer": row.get("gold_answer"),
                "search_query": row.get("search_query"),
                "final_answer": row.get("final_answer"),
                "retrieval_top5_hit": bool(row.get("retrieval_top5_hit")),
                "answer_correct": bool(row.get("answer_correct")),
                "strict_success": bool(row.get("strict_success")),
                "support_rank": support_rank,
                "semantic_anchor": data_row.get("semantic_anchor"),
                "ground_expression": data_row.get("ground_expression"),
                "hf_search_query": data_row.get("hf_search_query"),
                "top_docs": [
                    {
                        "rank": doc.get("rank"),
                        "sample_id": doc.get("sample_id"),
                        "title": doc.get("title"),
                        "url": doc.get("url"),
                        "text": compact(doc.get("text", ""))[:240],
                    }
                    for doc in top_docs[:5]
                ],
            }
        )
    write_jsonl(OUT / f"{split}_errors.jsonl", error_rows)
    return {
        "split": split,
        "n": len(rows),
        "strict_success": counts["strict_success"],
        "errors": len(error_rows),
        "counts": dict(counts),
        "answer_type_counts": dict(answer_types),
        "error_path": str(OUT / f"{split}_errors.jsonl"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {split: analyze_split(split) for split in ("dev", "test")}
    write_json(OUT / "scale60_error_analysis.json", summaries)
    lines = ["# Scale60 Error Analysis\n\n"]
    lines.append("This analyzes the current main checkpoint `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60` without changing predictions or metrics.\n\n")
    lines.append("## Summary\n\n")
    lines.append("| split | n | strict | retrieval miss | hit-answer-wrong | answer correct without support | error file |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---|\n")
    for split, s in summaries.items():
        counts = s["counts"]
        lines.append(
            f"| {split} | {s['n']} | {s['strict_success']} | {counts.get('retrieval_miss', 0)} | "
            f"{counts.get('hit_answer_wrong', 0)} | {counts.get('answer_correct_without_support', 0)} | `{s['error_path']}` |\n"
        )
    lines.append("\n## Retrieval Miss Subtypes\n\n")
    for split, s in summaries.items():
        lines.append(f"### {split}\n\n")
        items = [(k, v) for k, v in s["counts"].items() if k.startswith("retrieval_miss:")]
        for key, value in sorted(items, key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{key.split(':', 1)[1]}`: `{value}`\n")
        lines.append("\n")
    lines.append("## Hit-Answer-Wrong Subtypes\n\n")
    for split, s in summaries.items():
        lines.append(f"### {split}\n\n")
        items = [(k, v) for k, v in s["counts"].items() if k.startswith("hit_answer_wrong:")]
        for key, value in sorted(items, key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{key.split(':', 1)[1]}`: `{value}`\n")
        lines.append("\n")
    lines.append("## Decision\n\n")
    lines.append("The dominant remaining bottleneck is still retrieval miss, not answer extraction: ckpt60 has 42 dev and 31 test retrieval misses versus 8 dev and 7 test hit-answer-wrong cases. The next paper-facing work should therefore prioritize query/evidence node training and hard retrieval-miss analysis. Reader training should be revisited only after building a larger hard-context answer dataset.\n")
    (OUT / "SCALE60_ERROR_ANALYSIS.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "SCALE60_ERROR_ANALYSIS.md")


if __name__ == "__main__":
    main()
