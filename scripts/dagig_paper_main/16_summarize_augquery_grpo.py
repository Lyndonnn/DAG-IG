#!/usr/bin/env python3
"""Summarize the augmented-query initialized GRPO gated run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs/dagig_paper_main_v1")
METRICS = ROOT / "two_stage_metrics"
METRICS_V3 = ROOT / "two_stage_metrics_rescored_v3"
REPORT = ROOT / "reports/AUGQUERY_GRPO_30_REPORT.md"
SUMMARY = ROOT / "reports/augquery_grpo_30_summary.json"
TRAIN = ROOT / "checkpoints/paper_main_v1_augquery_stage1loss_kl01_30_s320/grpo_train_summary.json"


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}%"


def metric_row(name: str, path: Path) -> dict:
    metrics = load(path)
    breakdown = metrics.get("breakdown", {})
    return {
        "method": name,
        "path": str(path),
        "n": metrics.get("n"),
        "r1": metrics.get("retrieval_top1_hit"),
        "r3": metrics.get("retrieval_top3_hit"),
        "r5": metrics.get("retrieval_top5_hit"),
        "answer": metrics.get("answer_correct"),
        "strict": metrics.get("strict_success"),
        "format": metrics.get("format_parse_success"),
        "retrieval_miss": breakdown.get("retrieval_miss"),
        "hit_answer_wrong": breakdown.get("retrieval_hit_answer_wrong"),
    }


def main() -> None:
    train = load(TRAIN)
    rows = [
        metric_row("Format-SFT full dev", METRICS_V3 / "format_sft_two_stage_own_full_dev.json"),
        metric_row(
            "Augmented query-node SFT smoke + fixed reader",
            METRICS / "query_node_sft_aug_format_init_smoke20_full__reader_format_sft_dev.json",
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
            "Query-warm GRPO ckpt30 + fixed Format reader",
            METRICS / "paper_main_v1_querywarm_stage1loss_kl01_30_s320_ckpt30_formatreader__reader_format_sft_dev.json",
        ),
        metric_row(
            "Aug-query GRPO ckpt10 dev",
            METRICS / "paper_main_v1_augquery_stage1loss_kl01_30_s320_ckpt10_dev.json",
        ),
        metric_row(
            "Aug-query GRPO ckpt20 dev",
            METRICS / "paper_main_v1_augquery_stage1loss_kl01_30_s320_ckpt20_dev.json",
        ),
        metric_row(
            "Aug-query GRPO ckpt30 dev",
            METRICS / "paper_main_v1_augquery_stage1loss_kl01_30_s320_ckpt30_dev.json",
        ),
    ]
    aug_rows = rows[-3:]
    best_aug = max(aug_rows, key=lambda row: (row["strict"] or 0.0, row["r5"] or 0.0))
    current_main = rows[2]
    decision = {
        "promote": False,
        "run_test": False,
        "best_aug_checkpoint": best_aug["method"],
        "reason": (
            "All augmented-init GRPO checkpoints are below current main dev strict "
            "and do not improve dev retrieval enough to justify test evaluation."
        ),
    }
    summary = {"training": train, "rows": rows, "decision": decision}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines: list[str] = []
    lines.append("# Augmented-Query GRPO 30-Step Report\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This is the dev-gated GRPO run initialized from augmented query-node SFT. "
        "It uses the same paper-main v1 two-stage stage1-loss recipe as the current mainline. "
        "It is evaluated on dev only unless it beats the current dev gate.\n\n"
    )
    lines.append("## Training Health\n\n")
    lines.append("| status | optimizer steps | micro steps | constant reward groups | constant rate | max GPU GB | elapsed sec |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    constant_rate = train["constant_reward_groups"] / max(1, train["micro_steps"])
    lines.append(
        f"| {train['status']} | {train['optimizer_steps']} | {train['micro_steps']} | "
        f"{train['constant_reward_groups']} | {pct(constant_rate)} | "
        f"{train['max_gpu_mem_gb']} | {train['elapsed_seconds']:.1f} |\n\n"
    )
    lines.append("The run is healthy: `constant_reward_groups=0`, so this is not reward collapse.\n\n")
    lines.append("## Dev Evaluation\n\n")
    lines.append("| Method | n | R@1 | R@3 | R@5 | answer | strict | format | retrieval miss | hit-answer-wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['n']} | {pct(row['r1'])} | {pct(row['r3'])} | {pct(row['r5'])} | "
            f"{pct(row['answer'])} | {pct(row['strict'])} | {pct(row['format'])} | "
            f"{row['retrieval_miss']} | {row['hit_answer_wrong']} |\n"
        )
    lines.append("\n## Interpretation\n\n")
    lines.append(
        f"- Best augmented-init GRPO checkpoint is `{best_aug['method']}` with dev strict `{pct(best_aug['strict'])}` and R@5 `{pct(best_aug['r5'])}`.\n"
    )
    lines.append(
        f"- Current main seed42 ckpt60 remains better: dev strict `{pct(current_main['strict'])}`, R@5 `{pct(current_main['r5'])}`.\n"
    )
    lines.append(
        "- Augmented SFT data was useful as a warmup, but GRPO from that warmup degraded final dev strict rather than improving it.\n"
    )
    lines.append(
        "- Because the dev gate failed, do not run test for this checkpoint and do not promote it to the paper-main result.\n\n"
    )
    lines.append("## Decision\n\n")
    lines.append(
        "No-go for augmented-init GRPO. Keep `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60` as the current main checkpoint, with seed43 as confirmation. "
        "The next mainline action should target stronger non-oracle query candidate generation for the remaining no-hit train samples, not another GRPO run from the same augmented warmup.\n\n"
    )
    lines.append("## Artifacts\n\n")
    lines.append(f"- training summary: `{TRAIN}`\n")
    lines.append("- dev metrics: `outputs/dagig_paper_main_v1/two_stage_metrics/paper_main_v1_augquery_stage1loss_kl01_30_s320_ckpt{10,20,30}_dev.json`\n")
    lines.append(f"- machine summary: `{SUMMARY}`\n")
    REPORT.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
