#!/usr/bin/env python3
"""Build augmented query-node SFT data from hit-vs-miss and no-hit recovery rows."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import read_jsonl, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
BASE = ROOT / "query_node_sft/query_node_sft_train.jsonl"
NOHIT = ROOT / "reports/nohit_query_candidate_mining/nohit_query_recovery_sft.jsonl"
OUT = ROOT / "query_node_sft_aug"


def assistant_json(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                return json.loads(content)
    raise ValueError(f"missing assistant content for {row.get('sample_id')}")


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    sid = str(row.get("sample_id") or "")
    if not sid:
        errors.append("missing_sample_id")
    image_path = Path(str(row.get("image_path") or ""))
    if not image_path.exists():
        errors.append("missing_image")
    try:
        obj = assistant_json(row)
    except Exception:  # noqa: BLE001
        errors.append("assistant_not_json")
        return errors
    if "search_query" not in obj or not str(obj.get("search_query") or "").strip():
        errors.append("missing_search_query")
    if "visual_observation" not in obj or not str(obj.get("visual_observation") or "").strip():
        errors.append("missing_visual_observation")
    forbidden = {"final_answer", "answer", "gold_answer", "retrieved_docs", "evidence"}
    if any(key in obj for key in forbidden):
        errors.append("contains_forbidden_answer_or_evidence_field")
    return errors


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [
        ("hit_vs_miss", read_jsonl(BASE)),
        ("nohit_recovery", read_jsonl(NOHIT)),
    ]
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_counts = Counter()
    rank_counts = Counter()

    for source_name, source_rows in sources:
        for row in source_rows:
            sid = str(row.get("sample_id") or "")
            if sid in seen:
                skipped.append({"sample_id": sid, "source": source_name, "reason": "duplicate_sample_id"})
                continue
            errors = validate_row(row)
            if errors:
                skipped.append({"sample_id": sid, "source": source_name, "reason": ",".join(errors)})
                continue
            seen.add(sid)
            new_row = dict(row)
            new_row["augmented_source"] = source_name
            rows.append(new_row)
            source_counts[source_name] += 1
            rank = (
                row.get("chosen_support_rank5")
                or row.get("support_rank5")
                or "none"
            )
            rank_counts[str(rank)] += 1

    out_file = OUT / "query_node_sft_aug_train.jsonl"
    write_jsonl(out_file, rows)
    write_jsonl(OUT / "query_node_sft_aug_skipped.jsonl", skipped)
    summary = {
        "base_rows": len(sources[0][1]),
        "nohit_recovery_rows": len(sources[1][1]),
        "output_rows": len(rows),
        "skipped": len(skipped),
        "source_counts": dict(source_counts),
        "support_rank5_counts": dict(rank_counts),
        "schema": {
            "assistant_json_fields": ["visual_observation", "search_query"],
            "contains_final_answer": False,
            "contains_gold_answer": False,
            "uses_dev_or_test_labels": False,
        },
        "output": str(out_file),
    }
    write_json(OUT / "query_node_sft_aug_summary.json", summary)

    lines = []
    lines.append("# Augmented Query Node SFT Data\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This combines train-only hit-vs-miss query supervision with train-only no-hit recovery queries. "
        "It remains stage1-only: assistant targets contain `visual_observation` and `search_query`, not final answers or evidence lists.\n\n"
    )
    lines.append("## Counts\n\n")
    lines.append(f"- hit-vs-miss rows: `{len(sources[0][1])}`\n")
    lines.append(f"- no-hit recovery rows: `{len(sources[1][1])}`\n")
    lines.append(f"- output rows: `{len(rows)}`\n")
    lines.append(f"- skipped: `{len(skipped)}`\n")
    lines.append(f"- support-rank counts: `{dict(rank_counts)}`\n\n")
    lines.append("## Decision\n\n")
    lines.append(
        "Use this for one short query-node warmup smoke initialized from Format-SFT, evaluated with a fixed Format-SFT reader. "
        "Promote it only if it improves dev strict or clearly improves retrieval without increasing hit-answer-wrong.\n\n"
    )
    lines.append("## Artifacts\n\n")
    lines.append(f"- train file: `{out_file}`\n")
    lines.append(f"- skipped rows: `{OUT / 'query_node_sft_aug_skipped.jsonl'}`\n")
    lines.append(f"- summary: `{OUT / 'query_node_sft_aug_summary.json'}`\n")
    (OUT / "QUERY_NODE_SFT_AUG_DATA_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", out_file)


if __name__ == "__main__":
    main()
