#!/usr/bin/env python3
"""Audit paper_main_v1 rewards after fixing train corpus gold labels."""

from __future__ import annotations

import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import BM25Index, compute_reward, load_corpus, read_jsonl, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "reports/reward_audit_goldfixed"
TRAIN = Path("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl")
ROLLOUTS = ROOT / "rollouts/train_rollouts_unified_scored.jsonl"
FIXED_CORPUS = ROOT / "derived_assets/bm25_train_corpus_goldfixed.jsonl"


def auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [(s, i) for i, (s, y) in enumerate(zip(scores, labels, strict=True)) if y]
    negatives = [(s, i) for i, (s, y) in enumerate(zip(scores, labels, strict=True)) if not y]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def raw_output(row: dict[str, Any]) -> str:
    rollout = row.get("rollout") or {}
    raw = rollout.get("raw")
    if raw:
        return str(raw)
    # Fallback should rarely be needed, but preserves parsability.
    import json

    return json.dumps(
        {
            "visual_observation": rollout.get("visual_observation", ""),
            "search_query": rollout.get("search_query", ""),
            "final_answer": rollout.get("final_answer", ""),
        },
        ensure_ascii=False,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_rows = {str(row.get("sample_id")): row for row in read_jsonl(TRAIN)}
    rollouts = read_jsonl(ROLLOUTS)
    bm25 = BM25Index.from_docs(load_corpus(FIXED_CORPUS))

    rescored = []
    old_rewards: list[float] = []
    new_rewards: list[float] = []
    new_hits: list[bool] = []
    new_stricts: list[bool] = []
    old_hits = []
    old_stricts = []
    source_counts = Counter()
    source_hit_counts = Counter()
    source_strict_counts = Counter()
    group_rewards: dict[tuple[str, Any], list[float]] = defaultdict(list)

    for rollout in rollouts:
        sid = str(rollout.get("sample_id"))
        train_row = train_rows[sid]
        result = compute_reward(train_row, raw_output(rollout), bm25, "paper_main_v1", top_k=5)
        old_reward = float((rollout.get("node_credits") or {}).get("total_reward", 0.0))
        new_reward = float(result["reward"])
        source = str(rollout.get("source_run") or "unknown")
        old_hit = bool((rollout.get("retrieval") or {}).get("hit5"))
        old_strict = bool((rollout.get("metrics") or {}).get("strict_success"))
        new_hit = bool(result["retrieval_hit"])
        new_strict = bool(result["strict_success"])
        old_rewards.append(old_reward)
        new_rewards.append(new_reward)
        old_hits.append(old_hit)
        old_stricts.append(old_strict)
        new_hits.append(new_hit)
        new_stricts.append(new_strict)
        source_counts[source] += 1
        if new_hit:
            source_hit_counts[source] += 1
        if new_strict:
            source_strict_counts[source] += 1
        group_key = (source, rollout.get("micro_step"))
        group_rewards[group_key].append(new_reward)
        rescored.append(
            {
                "sample_id": sid,
                "source_run": source,
                "micro_step": rollout.get("micro_step"),
                "generation_index": rollout.get("generation_index"),
                "old_reward": old_reward,
                "new_reward": new_reward,
                "old_hit5": old_hit,
                "new_hit5": new_hit,
                "old_strict": old_strict,
                "new_strict": new_strict,
                "new_support_rank5": result.get("support_rank5"),
                "new_components": result.get("components"),
            }
        )

    constant_groups = 0
    valid_groups = 0
    for rewards in group_rewards.values():
        if len(rewards) < 2:
            continue
        valid_groups += 1
        if max(rewards) - min(rewards) <= 1e-9:
            constant_groups += 1
    source_rows = {}
    for source in sorted(source_counts):
        source_rows[source] = {
            "rollouts": source_counts[source],
            "new_hit_rollouts": source_hit_counts[source],
            "new_strict_rollouts": source_strict_counts[source],
        }
    summary = {
        "rollouts": len(rollouts),
        "samples": len(train_rows),
        "old_reward_mean": mean(old_rewards),
        "old_reward_std": stdev(old_rewards),
        "new_reward_mean": mean(new_rewards),
        "new_reward_std": stdev(new_rewards),
        "old_hit_rollouts": sum(old_hits),
        "new_hit_rollouts": sum(new_hits),
        "old_strict_rollouts": sum(old_stricts),
        "new_strict_rollouts": sum(new_stricts),
        "reward_auc_new_hit": auc(new_rewards, new_hits),
        "reward_auc_new_strict": auc(new_rewards, new_stricts),
        "constant_reward_groups": constant_groups,
        "valid_reward_groups": valid_groups,
        "constant_reward_group_rate": constant_groups / max(1, valid_groups),
        "source_summary": source_rows,
        "decision": {
            "reward_audit_go": bool(
                stdev(new_rewards) > 0.05
                and (auc(new_rewards, new_hits) or 0.0) > 0.80
                and constant_groups / max(1, valid_groups) < 0.10
            )
        },
    }
    write_jsonl(OUT / "train_rollouts_paper_main_v1_goldfixed_rescored.jsonl", rescored)
    write_json(OUT / "paper_main_v1_goldfixed_reward_audit_summary.json", summary)

    lines: list[str] = []
    lines.append("# Paper Main v1 Gold-Fixed Reward Audit\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This rescoring audit uses the fixed train BM25 corpus and recomputes paper_main_v1 node credits for existing train rollouts. "
        "It checks whether the reward remains discriminative before any future GRPO run.\n\n"
    )
    lines.append("## Summary\n\n")
    lines.append(f"- rollouts: `{len(rollouts)}`\n")
    lines.append(f"- old reward mean/std: `{summary['old_reward_mean']:.4f}` / `{summary['old_reward_std']:.4f}`\n")
    lines.append(f"- goldfixed reward mean/std: `{summary['new_reward_mean']:.4f}` / `{summary['new_reward_std']:.4f}`\n")
    lines.append(f"- old/new hit rollouts: `{summary['old_hit_rollouts']}` / `{summary['new_hit_rollouts']}`\n")
    lines.append(f"- old/new strict rollouts: `{summary['old_strict_rollouts']}` / `{summary['new_strict_rollouts']}`\n")
    lines.append(f"- AUC(reward, hit): `{summary['reward_auc_new_hit']:.3f}`\n")
    lines.append(f"- AUC(reward, strict): `{summary['reward_auc_new_strict']:.3f}`\n")
    lines.append(f"- constant reward groups: `{constant_groups}` / `{valid_groups}` (`{100.0 * summary['constant_reward_group_rate']:.1f}%`)\n\n")
    lines.append("## Source Summary\n\n")
    lines.append("| source | rollouts | hit rollouts | strict rollouts |\n|---|---:|---:|---:|\n")
    for source, row in source_rows.items():
        lines.append(f"| {source} | {row['rollouts']} | {row['new_hit_rollouts']} | {row['new_strict_rollouts']} |\n")
    lines.append("\n## Decision\n\n")
    if summary["decision"]["reward_audit_go"]:
        lines.append(
            "GO for reward health under the fixed train corpus: reward variance is non-trivial, hit/strict AUC are high, and constant-reward groups are low. "
            "This does not mean immediate GRPO is required; it means future GRPO should use the fixed corpus and updated mining files.\n"
        )
    else:
        lines.append(
            "NO-GO for immediate GRPO under the fixed corpus; inspect reward components before training.\n"
        )
    lines.append("\n## Artifacts\n\n")
    lines.append(f"- rescored rollouts: `{OUT / 'train_rollouts_paper_main_v1_goldfixed_rescored.jsonl'}`\n")
    lines.append(f"- summary: `{OUT / 'paper_main_v1_goldfixed_reward_audit_summary.json'}`\n")
    (OUT / "PAPER_MAIN_V1_GOLDFIXED_REWARD_AUDIT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "PAPER_MAIN_V1_GOLDFIXED_REWARD_AUDIT.md")


if __name__ == "__main__":
    main()
