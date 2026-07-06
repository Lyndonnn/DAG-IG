#!/usr/bin/env python3
"""Audit paper_main_v2 reward before any training."""

from __future__ import annotations

import json
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


OUT = Path("outputs/dagig_paper_main_v1/reports/reward_v2_audit")


def auc_binary(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = ties = total = 0
    for ps in pos:
        for ns in neg:
            wins += int(ps > ns)
            ties += int(abs(ps - ns) < 1e-12)
            total += 1
    return (wins + 0.5 * ties) / total


def safe_std(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def group_constant(groups: dict[tuple[str, int, str], list[dict[str, Any]]], key: str) -> int:
    out = 0
    for rows in groups.values():
        vals = [round(float(row[key]), 8) for row in rows]
        if len(set(vals)) <= 1:
            out += 1
    return out


def group_top_stats(groups: dict[tuple[str, int, str], list[dict[str, Any]]], key: str) -> dict[str, Any]:
    selected = []
    for rows in groups.values():
        selected.append(max(rows, key=lambda row: float(row[key])))
    n = max(1, len(selected))
    return {
        "groups": len(selected),
        "selected_hit5": sum(1 for row in selected if row["hit5"]) / n,
        "selected_strict": sum(1 for row in selected if row["strict"]) / n,
        "selected_answer": sum(1 for row in selected if row["answer_correct"]) / n,
        "selected_anchor_overlap": sum(1 for row in selected if row["query_anchor_overlap"] > 0) / n,
        "selected_missing_anchor_penalty": sum(1 for row in selected if row["missing_anchor_penalty"] < 0) / n,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_rows = {str(row.get("sample_id")): row for row in read_jsonl(Path("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl"))}
    rollouts = read_jsonl(Path("outputs/dagig_paper_main_v1/rollouts/train_rollouts_unified_scored.jsonl"))
    bm25 = BM25Index.from_docs(load_corpus(Path("outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl")))
    rescored = []
    skipped = Counter()
    for old in rollouts:
        sid = str(old.get("sample_id"))
        row = train_rows.get(sid)
        if not row:
            skipped["missing_train_row"] += 1
            continue
        raw = ((old.get("rollout") or {}).get("raw") or "").strip()
        if not raw:
            skipped["missing_raw"] += 1
            continue
        v1 = compute_reward(row, raw, bm25, variant="paper_main_v1", top_k=5)
        v2 = compute_reward(row, raw, bm25, variant="paper_main_v2", top_k=5)
        rescored.append(
            {
                "sample_id": sid,
                "source_run": old.get("source_run"),
                "micro_step": old.get("micro_step"),
                "generation_index": old.get("generation_index"),
                "v1_reward": v1["reward"],
                "v2_reward": v2["reward"],
                "reward_delta": v2["reward"] - v1["reward"],
                "hit5": bool(v1["retrieval_hit"]),
                "strict": bool(v1["strict_success"]),
                "answer_correct": bool(v1["answer_correct"]),
                "query": v1["parsed"].get("search_query", ""),
                "query_mrr": v2["components"].get("query_mrr", 0.0),
                "query_anchor": v2["components"].get("query_anchor", 0.0),
                "query_anchor_overlap": v2.get("query_anchor_overlap", 0),
                "query_anchor_term_count": v2.get("query_anchor_term_count", 0),
                "missing_anchor_penalty": v2["components"].get("missing_anchor_penalty", 0.0),
                "v1_components": v1["components"],
                "v2_components": v2["components"],
            }
        )
    write_jsonl(OUT / "train_rollouts_v1_v2_rescored.jsonl", rescored)
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rescored:
        groups[(str(row["source_run"]), int(row.get("micro_step") or -1), str(row["sample_id"]))].append(row)

    v1_rewards = [float(row["v1_reward"]) for row in rescored]
    v2_rewards = [float(row["v2_reward"]) for row in rescored]
    hit_labels = [int(row["hit5"]) for row in rescored]
    strict_labels = [int(row["strict"]) for row in rescored]
    anchor_labels = [int(row["query_anchor_overlap"] > 0) for row in rescored]
    miss_rows = [row for row in rescored if not row["hit5"]]
    changed_top = []
    for key, rows in groups.items():
        top_v1 = max(rows, key=lambda row: float(row["v1_reward"]))
        top_v2 = max(rows, key=lambda row: float(row["v2_reward"]))
        if top_v1 is not top_v2:
            changed_top.append(
                {
                    "group": {"source_run": key[0], "micro_step": key[1], "sample_id": key[2]},
                    "v1_query": top_v1["query"],
                    "v2_query": top_v2["query"],
                    "v1_hit5": top_v1["hit5"],
                    "v2_hit5": top_v2["hit5"],
                    "v1_strict": top_v1["strict"],
                    "v2_strict": top_v2["strict"],
                    "v1_anchor_overlap": top_v1["query_anchor_overlap"],
                    "v2_anchor_overlap": top_v2["query_anchor_overlap"],
                    "v1_reward": top_v1["v1_reward"],
                    "v2_reward": top_v2["v2_reward"],
                }
            )
    write_jsonl(OUT / "changed_group_top_candidates.jsonl", changed_top)
    summary = {
        "n": len(rescored),
        "groups": len(groups),
        "skipped": dict(skipped),
        "v1_reward_mean": sum(v1_rewards) / max(1, len(v1_rewards)),
        "v2_reward_mean": sum(v2_rewards) / max(1, len(v2_rewards)),
        "v1_reward_std": safe_std(v1_rewards),
        "v2_reward_std": safe_std(v2_rewards),
        "v1_constant_groups": group_constant(groups, "v1_reward"),
        "v2_constant_groups": group_constant(groups, "v2_reward"),
        "v1_auc_hit5": auc_binary(v1_rewards, hit_labels),
        "v2_auc_hit5": auc_binary(v2_rewards, hit_labels),
        "v1_auc_strict": auc_binary(v1_rewards, strict_labels),
        "v2_auc_strict": auc_binary(v2_rewards, strict_labels),
        "query_anchor_auc_hit5": auc_binary([float(row["query_anchor"]) for row in rescored], hit_labels),
        "query_anchor_auc_anchor_overlap": auc_binary([float(row["query_anchor"]) for row in rescored], anchor_labels),
        "v1_top_stats": group_top_stats(groups, "v1_reward"),
        "v2_top_stats": group_top_stats(groups, "v2_reward"),
        "changed_top_groups": len(changed_top),
        "changed_top_v2_hit_gain": sum(1 for row in changed_top if (not row["v1_hit5"]) and row["v2_hit5"]),
        "changed_top_v2_hit_harm": sum(1 for row in changed_top if row["v1_hit5"] and (not row["v2_hit5"])),
        "changed_top_v2_strict_gain": sum(1 for row in changed_top if (not row["v1_strict"]) and row["v2_strict"]),
        "changed_top_v2_strict_harm": sum(1 for row in changed_top if row["v1_strict"] and (not row["v2_strict"])),
        "miss_rows": len(miss_rows),
        "miss_rows_missing_anchor_penalty": sum(1 for row in miss_rows if row["missing_anchor_penalty"] < 0),
        "mean_v2_delta_on_misses": sum(float(row["reward_delta"]) for row in miss_rows) / max(1, len(miss_rows)),
    }
    write_json(OUT / "reward_v2_audit_summary.json", summary)
    lines = ["# Paper Main v2 Reward Audit\n\n"]
    lines.append("This is an audit only. No model is trained here.\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"- rescored rollouts: `{summary['n']}`\n")
    lines.append(f"- groups: `{summary['groups']}`\n")
    lines.append(f"- v1 constant groups: `{summary['v1_constant_groups']}`\n")
    lines.append(f"- v2 constant groups: `{summary['v2_constant_groups']}`\n")
    lines.append(f"- v1/v2 reward std: `{summary['v1_reward_std']:.4f}` / `{summary['v2_reward_std']:.4f}`\n")
    lines.append(f"- v1/v2 AUC hit5: `{summary['v1_auc_hit5']:.3f}` / `{summary['v2_auc_hit5']:.3f}`\n")
    lines.append(f"- v1/v2 AUC strict: `{summary['v1_auc_strict']:.3f}` / `{summary['v2_auc_strict']:.3f}`\n")
    lines.append(f"- changed top groups: `{summary['changed_top_groups']}`\n")
    lines.append(f"- changed-top hit gain/harm: `{summary['changed_top_v2_hit_gain']}` / `{summary['changed_top_v2_hit_harm']}`\n")
    lines.append(f"- changed-top strict gain/harm: `{summary['changed_top_v2_strict_gain']}` / `{summary['changed_top_v2_strict_harm']}`\n")
    lines.append(f"- miss rows with missing-anchor penalty: `{summary['miss_rows_missing_anchor_penalty']}` / `{summary['miss_rows']}`\n\n")
    lines.append("## Group Top Selection\n\n")
    lines.append("| reward | selected hit5 | selected strict | selected answer | selected anchor overlap | selected missing-anchor penalty |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for key, label in [("v1_top_stats", "v1"), ("v2_top_stats", "v2")]:
        stats = summary[key]
        lines.append(
            f"| {label} | {100*stats['selected_hit5']:.1f}% | {100*stats['selected_strict']:.1f}% | "
            f"{100*stats['selected_answer']:.1f}% | {100*stats['selected_anchor_overlap']:.1f}% | "
            f"{100*stats['selected_missing_anchor_penalty']:.1f}% |\n"
        )
    lines.append("\n## Decision\n\n")
    if summary["changed_top_v2_hit_gain"] > summary["changed_top_v2_hit_harm"] and summary["v2_constant_groups"] <= summary["v1_constant_groups"]:
        lines.append("v2 is safe enough for a small controlled training pilot, but only if we keep the same two-stage/stage1-only/KL=0.1 recipe and compare against scale60_s320 ckpt60. The audit suggests the anchor-aware query reward changes some top candidates in the right direction.\n")
    else:
        lines.append("v2 is not clearly better than v1 as a reward formulation. Do not train v2 yet; instead improve query candidate generation/data before changing reward.\n")
    (OUT / "PAPER_MAIN_V2_REWARD_AUDIT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "PAPER_MAIN_V2_REWARD_AUDIT.md")


if __name__ == "__main__":
    main()
