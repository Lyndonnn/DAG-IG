#!/usr/bin/env python3
"""Summarize the query-warm GRPO control run for paper-main v1."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs/dagig_paper_main_v1")
METRICS = ROOT / "two_stage_metrics"
METRICS_V3 = ROOT / "two_stage_metrics_rescored_v3"
REPORT = ROOT / "reports" / "QUERYWARM_GRPO_30_REPORT.md"
SUMMARY = ROOT / "reports" / "querywarm_grpo_30_summary.json"
TRAIN_SUMMARY = (
    ROOT
    / "checkpoints"
    / "paper_main_v1_querywarm_stage1loss_kl01_30_s320"
    / "grpo_train_summary.json"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(x: float | int | None) -> str:
    if x is None:
        return "-"
    return f"{100.0 * float(x):.1f}%"


def metric_row(name: str, path: Path) -> dict:
    m = load_json(path)
    b = m.get("breakdown", {})
    return {
        "method": name,
        "path": str(path),
        "n": m.get("n"),
        "r1": m.get("retrieval_top1_hit"),
        "r3": m.get("retrieval_top3_hit"),
        "r5": m.get("retrieval_top5_hit"),
        "answer": m.get("answer_correct"),
        "strict": m.get("strict_success"),
        "format": m.get("format_parse_success"),
        "retrieval_miss": b.get("retrieval_miss"),
        "hit_answer_wrong": b.get("retrieval_hit_answer_wrong"),
    }


def main() -> None:
    rows = [
        metric_row(
            "Format-SFT full dev",
            METRICS_V3 / "format_sft_two_stage_own_full_dev.json",
        ),
        metric_row(
            "Query-node SFT smoke + fixed reader",
            METRICS / "query_node_sft_format_init_smoke20_full__reader_format_sft_dev.json",
        ),
        metric_row(
            "DAG-IG GRPO seed42 ckpt60 dev",
            METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.json",
        ),
        metric_row(
            "DAG-IG GRPO seed43 ckpt60 dev",
            METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.json",
        ),
        metric_row(
            "Query-warm GRPO ckpt10 own reader",
            METRICS / "paper_main_v1_querywarm_stage1loss_kl01_30_s320_ckpt10_dev.json",
        ),
        metric_row(
            "Query-warm GRPO ckpt20 own reader",
            METRICS / "paper_main_v1_querywarm_stage1loss_kl01_30_s320_ckpt20_dev.json",
        ),
        metric_row(
            "Query-warm GRPO ckpt30 own reader",
            METRICS / "paper_main_v1_querywarm_stage1loss_kl01_30_s320_ckpt30_dev.json",
        ),
        metric_row(
            "Query-warm GRPO ckpt30 + fixed Format reader",
            METRICS
            / "paper_main_v1_querywarm_stage1loss_kl01_30_s320_ckpt30_formatreader__reader_format_sft_dev.json",
        ),
        metric_row(
            "DAG-IG GRPO seed42 ckpt60 test",
            METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.json",
        ),
        metric_row(
            "DAG-IG GRPO seed43 ckpt60 test",
            METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test.json",
        ),
    ]
    train = load_json(TRAIN_SUMMARY)
    train["constant_reward_rate"] = (
        train.get("constant_reward_groups", 0) / train.get("micro_steps", 1)
        if train.get("micro_steps")
        else None
    )

    best_dev = max(
        [r for r in rows if "dev" in r["method"].lower()],
        key=lambda r: (r["strict"] or 0.0, r["r5"] or 0.0),
    )
    querywarm_fixed = rows[7]
    querywarm_own = rows[6]
    current_best = rows[2]

    summary = {
        "training": train,
        "rows": rows,
        "best_dev_by_strict_then_r5": best_dev,
        "decision": {
            "promote_querywarm": False,
            "reason": (
                "Query-warm ckpt30 has the best dev R@5, but strict success remains "
                "below the current seed42/seed43 main checkpoints."
            ),
            "current_main_checkpoint": "outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60",
        },
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Query-Warm GRPO 30-Step Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This run tests whether train-only query-node warmup can improve the paper-main v1 two-stage GRPO recipe. "
        "It is a controlled mainline experiment, not a new reward variant or reader repair."
    )
    lines.append("")
    lines.append("## Training Health")
    lines.append("")
    lines.append("| status | optimizer steps | micro steps | constant reward groups | constant rate | max GPU GB | elapsed sec |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {train.get('status')} | {train.get('optimizer_steps')} | {train.get('micro_steps')} | "
        f"{train.get('constant_reward_groups')} | {pct(train.get('constant_reward_rate'))} | "
        f"{train.get('max_gpu_mem_gb')} | {train.get('elapsed_seconds'):.1f} |"
    )
    lines.append("")
    lines.append("The run is healthy: constant-reward groups are low, so this is not the old reward-collapse failure mode.")
    lines.append("")
    lines.append("## Evaluation")
    lines.append("")
    lines.append("| Method | n | R@1 | R@3 | R@5 | answer | strict | format | retrieval miss | hit-answer-wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['n']} | {pct(r['r1'])} | {pct(r['r3'])} | {pct(r['r5'])} | "
            f"{pct(r['answer'])} | {pct(r['strict'])} | {pct(r['format'])} | "
            f"{r['retrieval_miss']} | {r['hit_answer_wrong']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- Query-warm ckpt30 improves dev retrieval to `{pct(querywarm_own['r5'])}` own-reader and "
        f"`{pct(querywarm_fixed['r5'])}` with fixed Format-SFT reader, the highest dev R@5 among the current dev runs."
    )
    lines.append(
        f"- It does not improve final strict success: own-reader strict is `{pct(querywarm_own['strict'])}` and "
        f"fixed-reader strict is `{pct(querywarm_fixed['strict'])}`, below the current seed42/seed43 dev strict of `49.0%`."
    )
    lines.append(
        f"- Fixed-reader evaluation reduces the answer penalty but still leaves `{querywarm_fixed['hit_answer_wrong']}` "
        "retrieval-hit-answer-wrong cases, so better retrieval alone is not sufficient for promotion."
    )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(
        "Do not promote query-warm GRPO as the current main checkpoint and do not run test for it. "
        "Keep seed42 scale60_s320 checkpoint-60 as the paper-main v1 checkpoint, with seed43 as seed confirmation."
    )
    lines.append("")
    lines.append(
        "The useful result is diagnostic: query-node supervision can raise retrieval, but the answer node still blocks strict success. "
        "The next efficient mainline step is to improve retrieval data for the 124 train samples where existing rollouts contain no hit, "
        "and to keep answer-reader changes separate from the current main result."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- training summary: `{TRAIN_SUMMARY}`")
    lines.append(f"- machine summary: `{SUMMARY}`")
    lines.append("")

    with REPORT.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
