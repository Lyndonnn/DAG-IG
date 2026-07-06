#!/usr/bin/env python3
"""Build evidence-conditioned reader SFT data from clean two-stage rollouts.

The reader target is deliberately narrow:
image + question + generated search query + retrieved top-k evidence -> final_answer.

No dev/test labels are used. No oracle/teacher query is injected. We keep only
train rollout contexts whose generated query retrieves supporting evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import (  # noqa: E402
    BM25Index,
    load_corpus,
    read_jsonl,
    support_rank,
    write_json,
    write_jsonl,
)


READER_TRAIN_PROMPT = """You are an evidence-grounded answer reader.
Given an image, a question, and retrieved evidence documents, return JSON only with exactly:
{
  "final_answer": "short final answer"
}
Use the retrieved evidence when it supports an answer. Keep the answer concise.
Do not output reasoning, citations, search queries, markdown, or extra text."""


def format_docs(docs: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for doc in docs:
        text = str(doc.get("text", "")).strip()
        if len(text) > 1200:
            text = text[:1200].rstrip() + " ..."
        blocks.append(
            "\n".join(
                [
                    f"[Doc {doc.get('rank')}]",
                    f"Title: {str(doc.get('title', '')).strip()}",
                    f"URL: {str(doc.get('url', '')).strip()}",
                    f"Text: {text}",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "[No retrieved documents]"


def build_messages(row: dict[str, Any], query: str, docs: list[dict[str, Any]], image_path: str, include_query: bool) -> list[dict[str, Any]]:
    parts = [
        READER_TRAIN_PROMPT.strip(),
        f"Question: {str(row.get('question', '')).strip()}",
    ]
    if include_query:
        parts.append(f"Generated search query: {query}")
    parts.extend(["Retrieved evidence:", format_docs(docs)])
    user_text = "\n\n".join(parts)
    assistant_text = json.dumps({"final_answer": str(row.get("gold_answer", "")).strip()}, ensure_ascii=False, separators=(",", ":"))
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_root", type=Path, default=Path("data/Pix2Fact_DAGIG_Clean_GRPO_ASSET"))
    parser.add_argument("--train_data", type=Path, default=Path("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl"))
    parser.add_argument("--train_rollouts", type=Path, default=Path("outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_medium30/reward_rollouts.jsonl"))
    parser.add_argument("--train_corpus", type=Path, default=Path("outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dagig_paper_main_v1/reader_sft"))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_contexts_per_sample", type=int, default=4)
    parser.add_argument("--include_query_in_prompt", action="store_true")
    args = parser.parse_args()

    rows_by_id = {str(row.get("sample_id")): row for row in read_jsonl(args.train_data)}
    rollouts = read_jsonl(args.train_rollouts)
    index = BM25Index.from_docs(load_corpus(args.train_corpus))

    per_sample: dict[str, int] = defaultdict(int)
    seen_contexts: set[tuple[str, str, tuple[str, ...]]] = set()
    out_rows: list[dict[str, Any]] = []
    skipped = Counter()
    support_ranks = Counter()

    for rollout in rollouts:
        sample_id = str(rollout.get("sample_id", ""))
        row = rows_by_id.get(sample_id)
        if not row:
            skipped["missing_train_row"] += 1
            continue
        parsed = rollout.get("parsed") or {}
        query = str(parsed.get("search_query") or "").strip()
        if not query:
            skipped["missing_query"] += 1
            continue
        docs = index.search(query, top_k=args.top_k)
        rank = support_rank(docs, sample_id, args.top_k)
        if rank is None:
            skipped["no_support_in_topk"] += 1
            continue
        doc_ids = tuple(str(doc.get("doc_id", "")) for doc in docs)
        key = (sample_id, query.lower(), doc_ids)
        if key in seen_contexts:
            skipped["duplicate_context"] += 1
            continue
        if per_sample[sample_id] >= args.max_contexts_per_sample:
            skipped["max_contexts_per_sample"] += 1
            continue
        seen_contexts.add(key)
        per_sample[sample_id] += 1
        support_ranks[str(rank)] += 1
        image_path = str(row.get("image_abs_path") or (args.asset_root / row.get("image_path", "")))
        out_rows.append(
            {
                "sample_id": sample_id,
                "image_path": image_path,
                "messages": build_messages(row, query, docs, image_path, include_query=args.include_query_in_prompt),
                "setting": "paper_main_v1_reader_sft",
                "source": "ckpt30_train_rollout_query_with_support",
                "search_query": query,
                "support_rank": rank,
                "retrieved_doc_ids": list(doc_ids),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "reader_sft_train.jsonl"
    report_path = args.output_dir / "reader_sft_data_report.json"
    md_path = args.output_dir / "READER_SFT_DATA_REPORT.md"
    write_jsonl(out_path, out_rows)
    report = {
        "train_rows": len(rows_by_id),
        "train_rollouts": len(rollouts),
        "reader_sft_rows": len(out_rows),
        "unique_samples": len({row["sample_id"] for row in out_rows}),
        "top_k": args.top_k,
        "max_contexts_per_sample": args.max_contexts_per_sample,
        "skipped": dict(skipped),
        "support_rank_counts": dict(support_ranks),
        "output_path": str(out_path),
        "oracle_teacher_policy": {
            "uses_oracle_query": False,
            "uses_dev_or_test_labels": False,
            "uses_train_gold_answer_as_sft_target": True,
            "keeps_only_train_rollout_queries_with_supporting_evidence": True,
            "include_query_in_prompt": bool(args.include_query_in_prompt),
        },
    }
    write_json(report_path, report)
    lines = ["# Reader SFT Data Report\n\n"]
    lines.append("This data trains only the answer reader node: image + question + generated query + retrieved top-k evidence -> compact final answer JSON.\n\n")
    lines.append(f"- train rollouts scanned: `{len(rollouts)}`\n")
    lines.append(f"- reader SFT rows: `{len(out_rows)}`\n")
    lines.append(f"- unique samples: `{report['unique_samples']}`\n")
    lines.append(f"- support rank counts: `{json.dumps(dict(support_ranks), ensure_ascii=False)}`\n")
    lines.append(f"- skipped: `{json.dumps(dict(skipped), ensure_ascii=False)}`\n\n")
    lines.append("No dev/test labels or teacher/oracle queries are used. Train gold answers are used only as supervised reader targets.\n")
    md_path.write_text("".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
