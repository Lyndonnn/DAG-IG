#!/usr/bin/env python3
"""Mine train query hit-vs-miss pairs using gold-fixed retrieval labels."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import read_jsonl, tokenize, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "reports/hard_retrieval_mining_goldfixed"
ROLLOUTS = ROOT / "rollouts/train_rollouts_unified_scored.jsonl"
RESCORE = ROOT / "reports/goldfixed_rollout_rescore/train_rollouts_goldfixed_retrieval_rescore.jsonl"


def compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def query_key(query: str) -> str:
    return " ".join(tokenize(query))


def rollout_query(row: dict[str, Any]) -> str:
    return compact((row.get("rollout") or {}).get("search_query"))


def rollout_visual(row: dict[str, Any]) -> str:
    return compact((row.get("rollout") or {}).get("visual_observation"))


def source_run(row: dict[str, Any]) -> str:
    return compact(row.get("source_run")) or "unknown"


def query_credit(row: dict[str, Any]) -> float:
    rank = row.get("new_support_rank10")
    if rank:
        return 1.0 / float(rank)
    return 0.0


def best_hit(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    hits = [row for row in rows if row.get("new_hit5") and rollout_query(row)]
    if not hits:
        return None
    return sorted(
        hits,
        key=lambda row: (
            bool(row.get("new_strict_approx")),
            query_credit(row),
            -(row.get("new_support_rank5") or 99),
            -len(rollout_query(row)),
        ),
        reverse=True,
    )[0]


def worst_miss(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    misses = [row for row in rows if not row.get("new_hit5") and rollout_query(row)]
    if not misses:
        return None
    return sorted(misses, key=lambda row: (query_credit(row), len(rollout_query(row))))[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rollout_rows = read_jsonl(ROLLOUTS)
    rescore_rows = read_jsonl(RESCORE)
    if len(rollout_rows) != len(rescore_rows):
        raise ValueError(f"row count mismatch: {len(rollout_rows)} vs {len(rescore_rows)}")

    merged = []
    for old, new in zip(rollout_rows, rescore_rows, strict=True):
        if str(old.get("sample_id")) != str(new.get("sample_id")):
            raise ValueError("sample_id order mismatch")
        row = dict(old)
        row.update(
            {
                "new_hit5": bool(new.get("new_hit5")),
                "new_support_rank5": new.get("new_support_rank5"),
                "new_support_rank10": new.get("new_support_rank10"),
                "new_strict_approx": bool(new.get("new_strict_approx")),
            }
        )
        merged.append(row)

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        by_sample[str(row.get("sample_id"))].append(row)

    groups = []
    pairs = []
    counters = Counter()
    source_counts = Counter()
    source_hit_counts = Counter()
    source_strict_counts = Counter()

    for sid, rows in sorted(by_sample.items()):
        hits = [row for row in rows if row.get("new_hit5")]
        stricts = [row for row in rows if row.get("new_strict_approx")]
        for row in rows:
            source_counts[source_run(row)] += 1
            if row.get("new_hit5"):
                source_hit_counts[source_run(row)] += 1
            if row.get("new_strict_approx"):
                source_strict_counts[source_run(row)] += 1
        chosen = best_hit(rows)
        rejected = worst_miss(rows)
        if hits and rejected:
            status = "learnable_from_existing_rollouts"
        elif hits:
            status = "already_hit_no_miss_pair"
        else:
            status = "candidate_insufficient_no_hit_rollout"
        counters[status] += 1
        group = {
            "sample_id": sid,
            "n_rollouts": len(rows),
            "n_unique_queries": len({query_key(rollout_query(row)) for row in rows if rollout_query(row)}),
            "n_hit5_rollouts": len(hits),
            "n_strict_rollouts": len(stricts),
            "status": status,
            "question": rows[0].get("question"),
            "gold_answer": rows[0].get("gold_answer"),
            "best_hit_query": rollout_query(chosen) if chosen else "",
            "best_hit_visual": rollout_visual(chosen) if chosen else "",
            "best_hit_source": source_run(chosen) if chosen else "",
            "best_hit_rank": chosen.get("new_support_rank5") if chosen else None,
            "best_hit_query_credit": query_credit(chosen) if chosen else None,
            "worst_miss_query": rollout_query(rejected) if rejected else "",
            "worst_miss_visual": rollout_visual(rejected) if rejected else "",
            "worst_miss_source": source_run(rejected) if rejected else "",
            "worst_miss_query_credit": query_credit(rejected) if rejected else None,
        }
        groups.append(group)
        if chosen and rejected:
            pairs.append(
                {
                    "sample_id": sid,
                    "pair_type": "train_query_retrieval_hit_vs_miss_goldfixed",
                    "question": rows[0].get("question"),
                    "image_path": rows[0].get("image_path"),
                    "chosen": {
                        "visual_observation": rollout_visual(chosen),
                        "search_query": rollout_query(chosen),
                        "source_run": source_run(chosen),
                        "support_rank5": chosen.get("new_support_rank5"),
                        "query_credit": query_credit(chosen),
                    },
                    "rejected": {
                        "visual_observation": rollout_visual(rejected),
                        "search_query": rollout_query(rejected),
                        "source_run": source_run(rejected),
                        "support_rank5": rejected.get("new_support_rank5"),
                        "query_credit": query_credit(rejected),
                    },
                    "margin_query_credit": query_credit(chosen) - query_credit(rejected),
                }
            )

    summary = {
        "train_rollouts": len(merged),
        "train_samples": len(by_sample),
        "status_counts": dict(counters),
        "train_pairs": len(pairs),
        "source_rollouts": dict(source_counts),
        "source_hit5_rollouts": dict(source_hit_counts),
        "source_strict_rollouts_approx": dict(source_strict_counts),
        "outputs": {
            "groups": str(OUT / "train_hard_retrieval_groups_goldfixed.jsonl"),
            "pairs": str(OUT / "train_query_hit_vs_miss_pairs_goldfixed.jsonl"),
        },
    }
    write_jsonl(OUT / "train_hard_retrieval_groups_goldfixed.jsonl", groups)
    write_jsonl(OUT / "train_query_hit_vs_miss_pairs_goldfixed.jsonl", pairs)
    write_json(OUT / "hard_retrieval_mining_goldfixed_summary.json", summary)

    lines = ["# Gold-Fixed Hard Retrieval Mining Report\n\n"]
    lines.append("## Scope\n\n")
    lines.append(
        "This rebuilds train query hit-vs-miss mining after fixing train corpus gold labels. "
        "It uses the same existing rollout text; only retrieval hit labels are recomputed with the fixed train corpus.\n\n"
    )
    lines.append("## Train Rollout Coverage\n\n")
    lines.append(f"- train rollouts: `{len(merged)}`\n")
    lines.append(f"- train samples: `{len(by_sample)}`\n")
    lines.append(f"- train query hit-vs-miss pairs: `{len(pairs)}`\n")
    for key, value in sorted(counters.items()):
        lines.append(f"- `{key}`: `{value}`\n")
    lines.append("\n## Source Distribution\n\n")
    lines.append("| source | rollouts | hit@5 rollouts | strict rollouts approx |\n")
    lines.append("|---|---:|---:|---:|\n")
    for source in sorted(source_counts):
        lines.append(f"| {source} | {source_counts[source]} | {source_hit_counts[source]} | {source_strict_counts[source]} |\n")
    lines.append("\n## Decision\n\n")
    lines.append(
        "Use these goldfixed pairs for future query-node warmup or candidate analysis. "
        "The previous pair file is still useful for reproducing old runs, but future train-side work should use this fixed version.\n\n"
    )
    lines.append("## Artifacts\n\n")
    lines.append(f"- groups: `{OUT / 'train_hard_retrieval_groups_goldfixed.jsonl'}`\n")
    lines.append(f"- pairs: `{OUT / 'train_query_hit_vs_miss_pairs_goldfixed.jsonl'}`\n")
    lines.append(f"- summary: `{OUT / 'hard_retrieval_mining_goldfixed_summary.json'}`\n")
    (OUT / "HARD_RETRIEVAL_MINING_GOLDFIXED_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "HARD_RETRIEVAL_MINING_GOLDFIXED_REPORT.md")


if __name__ == "__main__":
    main()
