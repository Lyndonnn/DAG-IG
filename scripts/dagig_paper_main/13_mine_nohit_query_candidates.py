#!/usr/bin/env python3
"""Generate train-only query candidates for samples with no hit in existing rollouts.

This is mining, not training. It uses only train rows, clean grounding/query fields,
and the train BM25 corpus to estimate whether missing retrieval coverage can be
recovered by better stage-1 query targets.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import (  # noqa: E402
    BM25Index,
    answer_leaks_in_query,
    load_corpus,
    read_jsonl,
    support_rank,
    tokenize,
    write_json,
    write_jsonl,
)


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "reports/nohit_query_candidate_mining"
GROUPS = ROOT / "reports/hard_retrieval_mining/train_hard_retrieval_groups.jsonl"
TRAIN_FILE = Path("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl")
CORPUS = Path("outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl")
ASSET_ROOT = Path("data/Pix2Fact_DAGIG_Clean_GRPO_ASSET")

STAGE1_PROMPT = """You are a multimodal evidence-search agent.
Given an image and a question, return JSON only with exactly:
{
  "visual_observation": "brief visual evidence you used",
  "search_query": "one concise search query for retrieving supporting evidence"
}
Do not output the final answer. Do not include reasoning, evidence lists, markdown, or extra text.
Do not include the final answer inside the search_query unless it is unavoidable from the question itself."""


STOP = {
    "about",
    "after",
    "also",
    "answer",
    "based",
    "been",
    "before",
    "being",
    "could",
    "current",
    "currently",
    "does",
    "from",
    "have",
    "image",
    "into",
    "know",
    "like",
    "looking",
    "many",
    "name",
    "need",
    "only",
    "photo",
    "picture",
    "please",
    "question",
    "show",
    "shown",
    "tell",
    "that",
    "this",
    "want",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def image_abs_path(image_path: str) -> str:
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    return str((ASSET_ROOT / path).resolve())


def uniq_keep_order(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    out = []
    for source, query in items:
        query = clean(query)
        key = " ".join(tokenize(query))
        if not query or key in seen:
            continue
        seen.add(key)
        out.append((source, query))
    return out


def visible_name(row: dict[str, Any]) -> str:
    grounding = row.get("grounding")
    if isinstance(grounding, dict):
        return clean(grounding.get("visible_text_or_name"))
    return ""


def semantic_anchor(row: dict[str, Any]) -> str:
    grounding = row.get("grounding")
    if isinstance(grounding, dict) and grounding.get("semantic_anchor"):
        return clean(grounding.get("semantic_anchor"))
    return clean(row.get("semantic_anchor"))


def answer_intent(question: str) -> str:
    q = question.lower()
    pieces: list[str] = []
    if any(t in q for t in ("phone", "contact number", "call", "telephone", "hotline")):
        pieces.append("contact phone number")
    if any(t in q for t in ("email", "e-mail")):
        pieces.append("email contact")
    if any(t in q for t in ("address", "street", "located", "location", "closest", "boutique")):
        pieces.append("address location")
    if any(t in q for t in ("opening", "hours", "closing", "open ", "close ", "time")):
        pieces.append("opening hours")
    if any(t in q for t in ("price", "cost", "how much", "fee")):
        pieces.append("price")
    if any(t in q for t in ("revenue", "sales", "market cap")):
        pieces.append("revenue")
    if any(t in q for t in ("album", "song", "charity", "donated")):
        pieces.append("album charity")
    if any(t in q for t in ("population", "gdp", "percentage", "percent", "how many", "number of", "count")):
        pieces.append("number")
    if any(t in q for t in ("latest", "as of", "2024", "2025", "2026")):
        years = " ".join(sorted(set(re.findall(r"\b20\d{2}\b", q))))
        pieces.append(("latest " + years).strip())
    if pieces:
        return " ".join(dict.fromkeys(" ".join(pieces).split()))
    toks = [t for t in tokenize(question) if len(t) >= 4 and t not in STOP]
    return " ".join(toks[:6])


def question_keywords(question: str) -> str:
    toks = [t for t in tokenize(question) if len(t) >= 4 and t not in STOP]
    return " ".join(toks[:10])


def candidates_for(row: dict[str, Any]) -> list[tuple[str, str]]:
    q = clean(row.get("question"))
    anchor = semantic_anchor(row)
    hf = clean(row.get("hf_search_query"))
    visible = visible_name(row)
    ground_expr = clean(row.get("ground_expression"))
    intent = answer_intent(q)
    qkw = question_keywords(q)
    raw: list[tuple[str, str]] = []

    for source, base in [
        ("semantic_anchor", anchor),
        ("hf_search_query", hf),
        ("visible_text_or_name", visible),
    ]:
        if base:
            raw.append((source, base))
            if intent:
                raw.append((source + "+intent", f"{base} {intent}"))
                raw.append((source + "+official_intent", f"{base} official website {intent}"))
            if qkw:
                raw.append((source + "+question_keywords", f"{base} {qkw}"))

    if ground_expr and intent:
        raw.append(("ground_expression+intent", f"{ground_expr} {intent}"))
    if hf and anchor and hf.lower() != anchor.lower():
        raw.append(("anchor+hf", f"{anchor} {hf}"))
        if intent:
            raw.append(("anchor+hf+intent", f"{anchor} {hf} {intent}"))
    return uniq_keep_order(raw)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_rows = {str(r.get("sample_id")): r for r in read_jsonl(TRAIN_FILE)}
    nohit_ids = [
        str(r.get("sample_id"))
        for r in read_jsonl(GROUPS)
        if r.get("status") == "candidate_insufficient_no_hit_rollout"
    ]
    corpus = load_corpus(CORPUS)
    bm25 = BM25Index.from_docs(corpus)

    candidate_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    source_counts = Counter()
    hit_source_counts = Counter()
    missing_train_rows = []

    for sid in nohit_ids:
        row = train_rows.get(sid)
        if not row:
            missing_train_rows.append(sid)
            continue
        sample_candidates = candidates_for(row)
        best: dict[str, Any] | None = None
        for source, query in sample_candidates:
            if answer_leaks_in_query(query, str(row.get("gold_answer", ""))):
                continue
            docs = bm25.search(query, top_k=5)
            rank = support_rank(docs, sid, 5)
            source_counts[source] += 1
            if rank is not None:
                hit_source_counts[source] += 1
            out = {
                "sample_id": sid,
                "source": source,
                "query": query,
                "support_rank5": rank,
                "hit5": rank is not None,
                "top_docs": [
                    {
                        "rank": d.get("rank"),
                        "score": d.get("score"),
                        "sample_id": d.get("sample_id"),
                        "is_gold": d.get("is_gold"),
                        "title": d.get("title"),
                        "url": d.get("url"),
                    }
                    for d in docs
                ],
            }
            candidate_rows.append(out)
            if rank is not None:
                if best is None or rank < int(best["support_rank5"]):
                    best = out
        if best:
            best_row = {
                "sample_id": sid,
                "question": row.get("question"),
                "image_path": row.get("image_path"),
                "semantic_anchor": semantic_anchor(row),
                "hf_search_query": row.get("hf_search_query"),
                "chosen_query": best["query"],
                "chosen_source": best["source"],
                "support_rank5": best["support_rank5"],
            }
            best_rows.append(best_row)
            visual = clean(row.get("ground_expression")) or semantic_anchor(row) or visible_name(row)
            image_path = image_abs_path(str(row.get("image_path") or ""))
            prompt = f"{STAGE1_PROMPT.strip()}\n\nQuestion: {row.get('question')}"
            training_rows.append(
                {
                    "sample_id": sid,
                    "image_path": image_path,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": prompt},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "visual_observation": visual,
                                    "search_query": best["query"],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    "setting": "train_nohit_query_recovery_sft",
                    "query_source": best["source"],
                    "support_rank5": best["support_rank5"],
                }
            )

    recovered = len(best_rows)
    total = len(nohit_ids)
    rank_counts = Counter(str(r["support_rank5"]) for r in best_rows)
    summary = {
        "scope": "train-only no-hit query candidate mining",
        "nohit_samples": total,
        "missing_train_rows": missing_train_rows,
        "candidate_rows": len(candidate_rows),
        "recovered_samples": recovered,
        "recovery_rate": recovered / total if total else 0.0,
        "support_rank_counts": dict(rank_counts),
        "candidate_source_counts": dict(source_counts),
        "hit_source_counts": dict(hit_source_counts),
        "outputs": {
            "candidates": str(OUT / "nohit_query_candidates.jsonl"),
            "best": str(OUT / "nohit_best_queries.jsonl"),
            "sft": str(OUT / "nohit_query_recovery_sft.jsonl"),
        },
    }

    write_jsonl(OUT / "nohit_query_candidates.jsonl", candidate_rows)
    write_jsonl(OUT / "nohit_best_queries.jsonl", best_rows)
    write_jsonl(OUT / "nohit_query_recovery_sft.jsonl", training_rows)
    write_json(OUT / "nohit_query_recovery_summary.json", summary)

    lines = []
    lines.append("# No-Hit Query Candidate Mining Report\n")
    lines.append("## Scope\n")
    lines.append(
        "This is train-only mining for the 124 samples whose existing rollout pool has no top-5 support hit. "
        "It does not use dev/test labels and does not train a model.\n"
    )
    lines.append("## Results\n")
    lines.append(f"- no-hit train samples: `{total}`\n")
    lines.append(f"- generated candidate rows: `{len(candidate_rows)}`\n")
    lines.append(f"- recovered samples with at least one top-5 support hit: `{recovered}` (`{100.0 * summary['recovery_rate']:.1f}%`)\n")
    lines.append(f"- support-rank counts among recovered: `{dict(rank_counts)}`\n")
    lines.append("\n## Candidate Source Counts\n")
    lines.append("| source | candidates | hit candidates |\n|---|---:|---:|\n")
    for source, count in sorted(source_counts.items(), key=lambda kv: (-hit_source_counts[kv[0]], kv[0])):
        lines.append(f"| {source} | {count} | {hit_source_counts[source]} |\n")
    lines.append("\n## Decision\n")
    if recovered:
        lines.append(
            "The no-hit gap is partially recoverable with train-only query recipes. "
            "Use `nohit_query_recovery_sft.jsonl` as an auxiliary query-node dataset or merge it with existing hit-vs-miss query-node supervision before the next GRPO run. "
            "Keep it separate from dev/test and do not treat it as a final paper result until a controlled evaluation improves dev strict success.\n"
        )
    else:
        lines.append(
            "The no-hit gap is not recoverable with simple train-only query recipes. "
            "The next step would be additional rollout sampling or stronger visual/entity extraction, not another GRPO run over the same candidate space.\n"
        )
    lines.append("\n## Artifacts\n")
    lines.append(f"- candidates: `{OUT / 'nohit_query_candidates.jsonl'}`\n")
    lines.append(f"- best recovered queries: `{OUT / 'nohit_best_queries.jsonl'}`\n")
    lines.append(f"- auxiliary SFT rows: `{OUT / 'nohit_query_recovery_sft.jsonl'}`\n")
    lines.append(f"- machine summary: `{OUT / 'nohit_query_recovery_summary.json'}`\n")
    (OUT / "NOHIT_QUERY_CANDIDATE_MINING_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "NOHIT_QUERY_CANDIDATE_MINING_REPORT.md")


if __name__ == "__main__":
    main()
