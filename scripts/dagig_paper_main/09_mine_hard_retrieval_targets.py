#!/usr/bin/env python3
"""Mine hard retrieval targets for the next paper-main iteration.

This is analysis/data-mining only. It does not train a model and does not use
dev/test labels to create train pairs.
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

from scripts.dagig_grpo.grpo_utils import read_jsonl, tokenize, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "reports/hard_retrieval_mining"
TRAIN_ROLLOUTS = ROOT / "rollouts/train_rollouts_unified_scored.jsonl"
DEV_ERRORS = ROOT / "reports/scale60_error_analysis/dev_errors.jsonl"
TEST_ERRORS = ROOT / "reports/scale60_error_analysis/test_errors.jsonl"


def compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def query_key(text: str) -> str:
    return " ".join(tokenize(text.lower()))


def rollout_query(row: dict[str, Any]) -> str:
    return compact((row.get("rollout") or {}).get("search_query", ""))


def rollout_visual(row: dict[str, Any]) -> str:
    return compact((row.get("rollout") or {}).get("visual_observation", ""))


def total_reward(row: dict[str, Any]) -> float:
    return float((row.get("node_credits") or {}).get("total_reward", 0.0))


def query_credit(row: dict[str, Any]) -> float:
    return float((row.get("node_credits") or {}).get("query_credit", 0.0))


def support_rank(row: dict[str, Any]) -> int | None:
    rank = (row.get("retrieval") or {}).get("support_rank5")
    if rank is None:
        return None
    try:
        return int(rank)
    except (TypeError, ValueError):
        return None


def is_hit(row: dict[str, Any]) -> bool:
    return bool((row.get("retrieval") or {}).get("hit5"))


def is_strict(row: dict[str, Any]) -> bool:
    return bool((row.get("metrics") or {}).get("strict_success"))


def source_run(row: dict[str, Any]) -> str:
    return compact(row.get("source_run", "unknown")) or "unknown"


def best_hit(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    hits = [r for r in rows if is_hit(r) and rollout_query(r)]
    if not hits:
        return None
    return sorted(
        hits,
        key=lambda r: (
            is_strict(r),
            query_credit(r),
            total_reward(r),
            -(support_rank(r) or 99),
            -len(rollout_query(r)),
        ),
        reverse=True,
    )[0]


def worst_miss(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    misses = [r for r in rows if (not is_hit(r)) and rollout_query(r)]
    if not misses:
        return None
    return sorted(
        misses,
        key=lambda r: (
            query_credit(r),
            total_reward(r),
            len(rollout_query(r)),
        ),
    )[0]


def load_dev_test_miss_cases(path: Path, split: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for row in read_jsonl(path):
        if row.get("bottleneck") != "retrieval_miss":
            continue
        query = compact(row.get("search_query", ""))
        semantic_anchor = compact(row.get("semantic_anchor", ""))
        hf_query = compact(row.get("hf_search_query", ""))
        q_tokens = set(tokenize(query.lower()))
        anchor_tokens = {t for t in tokenize(semantic_anchor.lower()) if len(t) >= 3}
        hf_tokens = {t for t in tokenize(hf_query.lower()) if len(t) >= 3}
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "split": split,
                "subtype": row.get("subtype"),
                "answer_type": row.get("answer_type"),
                "question": row.get("question"),
                "gold_answer": row.get("gold_answer"),
                "search_query": query,
                "semantic_anchor": semantic_anchor,
                "hf_search_query": hf_query,
                "query_token_count": len(q_tokens),
                "anchor_overlap": sorted(q_tokens & anchor_tokens),
                "teacher_query_overlap": sorted(q_tokens & hf_tokens),
                "top_docs": row.get("top_docs", [])[:3],
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(TRAIN_ROLLOUTS)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row.get("sample_id"))].append(row)

    groups = []
    pairs = []
    counters = Counter()
    source_counter = Counter()
    hit_source_counter = Counter()
    strict_source_counter = Counter()

    for sid, sample_rows in sorted(by_sample.items()):
        n = len(sample_rows)
        hit_rows = [r for r in sample_rows if is_hit(r)]
        strict_rows = [r for r in sample_rows if is_strict(r)]
        unique_queries = {query_key(rollout_query(r)) for r in sample_rows if rollout_query(r)}
        for r in sample_rows:
            source_counter[source_run(r)] += 1
            if is_hit(r):
                hit_source_counter[source_run(r)] += 1
            if is_strict(r):
                strict_source_counter[source_run(r)] += 1
        chosen = best_hit(sample_rows)
        rejected = worst_miss(sample_rows)
        if hit_rows and rejected:
            status = "learnable_from_existing_rollouts"
            counters[status] += 1
        elif hit_rows:
            status = "already_hit_no_miss_pair"
            counters[status] += 1
        else:
            status = "candidate_insufficient_no_hit_rollout"
            counters[status] += 1
        group = {
            "sample_id": sid,
            "n_rollouts": n,
            "n_unique_queries": len(unique_queries),
            "n_hit5_rollouts": len(hit_rows),
            "n_strict_rollouts": len(strict_rows),
            "status": status,
            "question": sample_rows[0].get("question"),
            "gold_answer": sample_rows[0].get("gold_answer"),
            "best_hit_query": rollout_query(chosen) if chosen else "",
            "best_hit_visual": rollout_visual(chosen) if chosen else "",
            "best_hit_source": source_run(chosen) if chosen else "",
            "best_hit_rank": support_rank(chosen) if chosen else None,
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
                    "pair_type": "train_query_retrieval_hit_vs_miss",
                    "question": sample_rows[0].get("question"),
                    "image_path": sample_rows[0].get("image_path"),
                    "chosen": {
                        "visual_observation": rollout_visual(chosen),
                        "search_query": rollout_query(chosen),
                        "source_run": source_run(chosen),
                        "support_rank5": support_rank(chosen),
                        "query_credit": query_credit(chosen),
                        "total_reward": total_reward(chosen),
                    },
                    "rejected": {
                        "visual_observation": rollout_visual(rejected),
                        "search_query": rollout_query(rejected),
                        "source_run": source_run(rejected),
                        "support_rank5": support_rank(rejected),
                        "query_credit": query_credit(rejected),
                        "total_reward": total_reward(rejected),
                    },
                    "margin_query_credit": query_credit(chosen) - query_credit(rejected),
                    "margin_total_reward": total_reward(chosen) - total_reward(rejected),
                }
            )

    dev_misses = load_dev_test_miss_cases(DEV_ERRORS, "dev")
    test_misses = load_dev_test_miss_cases(TEST_ERRORS, "test")
    miss_subtypes = Counter([r["subtype"] for r in dev_misses + test_misses])
    no_anchor_overlap = sum(1 for r in dev_misses + test_misses if not r["anchor_overlap"] and r["semantic_anchor"])
    no_teacher_overlap = sum(1 for r in dev_misses + test_misses if not r["teacher_query_overlap"] and r["hf_search_query"])

    write_jsonl(OUT / "train_hard_retrieval_groups.jsonl", groups)
    write_jsonl(OUT / "train_query_hit_vs_miss_pairs.jsonl", pairs)
    write_jsonl(OUT / "dev_test_retrieval_miss_cases.jsonl", dev_misses + test_misses)

    summary = {
        "train_rollouts": len(rows),
        "train_samples": len(by_sample),
        "status_counts": dict(counters),
        "train_pairs": len(pairs),
        "source_rollouts": dict(source_counter),
        "source_hit5_rollouts": dict(hit_source_counter),
        "source_strict_rollouts": dict(strict_source_counter),
        "dev_retrieval_misses": len(dev_misses),
        "test_retrieval_misses": len(test_misses),
        "dev_test_miss_subtypes": dict(miss_subtypes),
        "dev_test_no_anchor_overlap": no_anchor_overlap,
        "dev_test_no_teacher_query_overlap": no_teacher_overlap,
        "outputs": {
            "groups": str(OUT / "train_hard_retrieval_groups.jsonl"),
            "pairs": str(OUT / "train_query_hit_vs_miss_pairs.jsonl"),
            "dev_test_misses": str(OUT / "dev_test_retrieval_miss_cases.jsonl"),
        },
    }
    write_json(OUT / "hard_retrieval_mining_summary.json", summary)

    lines = ["# Hard Retrieval Mining Report\n\n"]
    lines.append("## Scope\n\n")
    lines.append(
        "This is train-only mining for future query/evidence-node supervision plus dev/test diagnostics. It does not train a model and does not use dev/test labels to create train pairs.\n\n"
    )
    lines.append("## Train Rollout Coverage\n\n")
    lines.append(f"- train rollouts: `{len(rows)}`\n")
    lines.append(f"- train samples: `{len(by_sample)}`\n")
    lines.append(f"- train query hit-vs-miss pairs: `{len(pairs)}`\n")
    for key, value in sorted(counters.items()):
        lines.append(f"- `{key}`: `{value}`\n")
    lines.append("\n")
    lines.append("## Source Distribution\n\n")
    lines.append("| source | rollouts | hit@5 rollouts | strict rollouts |\n")
    lines.append("|---|---:|---:|---:|\n")
    for source in sorted(source_counter):
        lines.append(
            f"| {source} | {source_counter[source]} | {hit_source_counter[source]} | {strict_source_counter[source]} |\n"
        )
    lines.append("\n")
    lines.append("## Dev/Test Retrieval Miss Diagnostics\n\n")
    lines.append(f"- dev retrieval misses: `{len(dev_misses)}`\n")
    lines.append(f"- test retrieval misses: `{len(test_misses)}`\n")
    lines.append(f"- misses with no semantic-anchor overlap: `{no_anchor_overlap}`\n")
    lines.append(f"- misses with no teacher-query overlap: `{no_teacher_overlap}`\n\n")
    lines.append("| subtype | count |\n|---|---:|\n")
    for subtype, count in sorted(miss_subtypes.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {subtype} | {count} |\n")
    lines.append("\n")
    lines.append("## Decision\n\n")
    lines.append(
        "If most train samples are learnable from existing hit-vs-miss rollouts, the efficient next step is a query/evidence-node preference warmup or GRPO sampling bias using `train_query_hit_vs_miss_pairs.jsonl`. "
        "If many samples have no hit rollout, the next step is to generate more diverse non-oracle query candidates before another GRPO run.\n\n"
    )
    lines.append(
        "For the current state, use this mining output to focus the next iteration on retrieval miss reduction. Do not continue broad answer repair unless retrieval-hit-answer-wrong becomes the dominant error class.\n"
    )
    (OUT / "HARD_RETRIEVAL_MINING_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "HARD_RETRIEVAL_MINING_REPORT.md")
    print("wrote", OUT / "hard_retrieval_mining_summary.json")


if __name__ == "__main__":
    main()
