#!/usr/bin/env python3
"""Build train-only query-node SFT warmup data from hit-vs-miss rollout pairs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import read_jsonl, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
ASSET_ROOT = Path("data/Pix2Fact_DAGIG_Clean_GRPO_ASSET")
PAIRS = ROOT / "reports/hard_retrieval_mining/train_query_hit_vs_miss_pairs.jsonl"
OUT = ROOT / "query_node_sft"

STAGE1_PROMPT = """You are a multimodal evidence-search agent.
Given an image and a question, return JSON only with exactly:
{
  "visual_observation": "brief visual evidence you used",
  "search_query": "one concise search query for retrieving supporting evidence"
}
Do not output the final answer. Do not include reasoning, evidence lists, markdown, or extra text.
Do not include the final answer inside the search_query unless it is unavoidable from the question itself."""


def image_abs_path(image_path: str) -> str:
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    return str((ASSET_ROOT / path).resolve())


def image_message(image_path: str, text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_abs_path(image_path)},
                {"type": "text", "text": text},
            ],
        }
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(PAIRS)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for pair in pairs:
        sid = str(pair.get("sample_id"))
        chosen = pair.get("chosen") or {}
        image_path = str(pair.get("image_path") or "")
        question = str(pair.get("question") or "").strip()
        visual = str(chosen.get("visual_observation") or "").strip()
        query = str(chosen.get("search_query") or "").strip()
        if not sid or not image_path or not question or not query:
            skipped.append({"sample_id": sid, "reason": "missing_required_field"})
            continue
        if sid in seen:
            # The mining script emits one best pair per sample, but keep this guard
            # because duplicate sample supervision would overweight one prompt.
            skipped.append({"sample_id": sid, "reason": "duplicate_sample_id"})
            continue
        seen.add(sid)
        if not visual:
            visual = "relevant visual evidence in the image"
        answer = json.dumps(
            {"visual_observation": visual, "search_query": query},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = f"{STAGE1_PROMPT.strip()}\n\nQuestion: {question}"
        messages = image_message(image_path, prompt)
        messages.append({"role": "assistant", "content": answer})
        rows.append(
            {
                "sample_id": sid,
                "image_path": image_abs_path(image_path),
                "messages": messages,
                "setting": "query_node_sft_from_train_hit_vs_miss",
                "source_pair_type": pair.get("pair_type"),
                "chosen_source_run": chosen.get("source_run"),
                "chosen_support_rank5": chosen.get("support_rank5"),
                "chosen_query_credit": chosen.get("query_credit"),
                "margin_query_credit": pair.get("margin_query_credit"),
                "margin_total_reward": pair.get("margin_total_reward"),
            }
        )

    out_file = OUT / "query_node_sft_train.jsonl"
    write_jsonl(out_file, rows)
    write_jsonl(OUT / "query_node_sft_skipped.jsonl", skipped)
    source_counts: dict[str, int] = {}
    rank_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("chosen_source_run") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        rank = str(row.get("chosen_support_rank5") or "none")
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    summary = {
        "input_pairs": len(pairs),
        "rows": len(rows),
        "skipped": len(skipped),
        "output": str(out_file),
        "source_counts": source_counts,
        "support_rank5_counts": rank_counts,
        "schema": {
            "assistant_json_fields": ["visual_observation", "search_query"],
            "contains_final_answer": False,
            "contains_gold_answer": False,
            "uses_dev_or_test_labels": False,
        },
    }
    write_json(OUT / "query_node_sft_summary.json", summary)

    lines = ["# Query Node SFT Warmup Data\n\n"]
    lines.append("## Scope\n\n")
    lines.append(
        "This data is built only from train hit-vs-miss rollout pairs. It supervises stage-1 query generation only and does not include final answers, retrieved documents, dev/test labels, or oracle queries.\n\n"
    )
    lines.append("## Counts\n\n")
    lines.append(f"- input pairs: `{len(pairs)}`\n")
    lines.append(f"- output rows: `{len(rows)}`\n")
    lines.append(f"- skipped: `{len(skipped)}`\n")
    lines.append(f"- output file: `{out_file}`\n\n")
    lines.append("## Chosen Source Runs\n\n")
    for source, count in sorted(source_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{source}`: `{count}`\n")
    lines.append("\n## Decision\n\n")
    lines.append(
        "Use this as a query-node warmup smoke from Format-SFT before another GRPO run. Evaluation should use a fixed Format-SFT reader first, so any movement is attributable to query/retrieval behavior rather than answer-reader drift.\n"
    )
    (OUT / "QUERY_NODE_SFT_DATA_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print(f"wrote {out_file}")
    print(f"wrote {OUT / 'QUERY_NODE_SFT_DATA_REPORT.md'}")


if __name__ == "__main__":
    main()
