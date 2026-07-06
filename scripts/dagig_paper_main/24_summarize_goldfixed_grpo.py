#!/usr/bin/env python3
"""Summarize the gold-fixed paper-main GRPO rerun."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
METRICS = ROOT / "two_stage_metrics"
METRICS_V3 = ROOT / "two_stage_metrics_rescored_v3"
REPORT = ROOT / "reports/GOLDFIXED_GRPO_60_REPORT.md"
SUMMARY = ROOT / "reports/goldfixed_grpo_60_summary.json"
TRAIN = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/grpo_train_summary.json"
AUDIT = ROOT / "reports/reward_audit_goldfixed/paper_main_v1_goldfixed_reward_audit_summary.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.1f}%"


def metric(path: Path) -> dict[str, Any]:
    data = load_json(path)
    breakdown = data.get("breakdown") or {}
    return {
        "path": str(path),
        "r5": data.get("retrieval_top5_hit"),
        "answer": data.get("answer_correct"),
        "strict": data.get("strict_success"),
        "format": data.get("format_parse_success"),
        "retrieval_miss": breakdown.get("retrieval_miss"),
        "hit_answer_wrong": breakdown.get("retrieval_hit_answer_wrong"),
    }


def row(name: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {pct(m.get('r5'))} | {pct(m.get('answer'))} | {pct(m.get('strict'))} | "
        f"{pct(m.get('format'))} | {m.get('retrieval_miss', '-')} | {m.get('hit_answer_wrong', '-')} |"
    )


def main() -> None:
    train = load_json(TRAIN)
    audit = load_json(AUDIT)
    metrics = {
        "format_dev": metric(METRICS_V3 / "format_sft_two_stage_own_full_dev.json"),
        "format_test": metric(METRICS_V3 / "format_sft_two_stage_own_full_test.json"),
        "seed42_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.json"),
        "seed42_test": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.json"),
        "seed43_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.json"),
        "seed43_test": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test.json"),
        "goldfixed20_dev": metric(METRICS / "paper_main_v1_goldfixed_scale60_s320_ckpt20_dev.json"),
        "goldfixed40_dev": metric(METRICS / "paper_main_v1_goldfixed_scale60_s320_ckpt40_dev.json"),
        "goldfixed60_dev": metric(METRICS / "paper_main_v1_goldfixed_scale60_s320_ckpt60_dev.json"),
        "goldfixed60_test": metric(METRICS / "paper_main_v1_goldfixed_scale60_s320_ckpt60_test.json"),
    }
    decision = {
        "promote": False,
        "reason": "dev improves over current seed42, but test strict/R@5 are below current seed42",
        "current_main": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60",
        "goldfixed_checkpoint": "paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/checkpoint-60",
    }
    summary = {"train": train, "audit": audit, "metrics": metrics, "decision": decision}
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Gold-Fixed GRPO 60 Report\n\n")
    lines.append("## 1. Motivation\n\n")
    lines.append(
        "The train BM25 corpus had missing `is_gold=true` labels for 41 train samples. "
        "After a uniform train-only gold-label fix, the reward was re-audited and passed. "
        "This run tests the same stable paper-main recipe with the fixed train corpus as the only intended protocol change.\n\n"
    )
    lines.append("## 2. Reward Health\n\n")
    lines.append(f"- reward AUC vs fixed-corpus hit: `{audit.get('reward_auc_new_hit'):.3f}`\n")
    lines.append(f"- reward AUC vs fixed-corpus strict: `{audit.get('reward_auc_new_strict'):.3f}`\n")
    lines.append(
        f"- audit constant reward groups: `{audit.get('constant_reward_groups')} / {audit.get('valid_reward_groups')}` "
        f"(`{100.0 * audit.get('constant_reward_group_rate', 0):.1f}%`)\n\n"
    )
    lines.append("## 3. Training\n\n")
    lines.append(f"- checkpoint root: `{train.get('output_dir')}`\n")
    lines.append(f"- optimizer steps: `{train.get('optimizer_steps')}`\n")
    lines.append(f"- micro steps: `{train.get('micro_steps')}`\n")
    lines.append(
        f"- training constant reward groups: `{train.get('constant_reward_groups')} / {train.get('micro_steps')}` "
        f"(`{100.0 * train.get('constant_reward_groups', 0) / max(1, train.get('micro_steps', 1)):.2f}%`)\n"
    )
    lines.append(f"- max GPU memory: `{train.get('max_gpu_mem_gb')}` GB\n\n")
    lines.append("## 4. Dev Sweep\n\n")
    lines.append("| Method | Dev R@5 | Dev answer | Dev strict | Format | Retrieval miss | Hit-answer-wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    lines.append(row("Format-SFT", metrics["format_dev"]) + "\n")
    lines.append(row("Current seed42 ckpt60", metrics["seed42_dev"]) + "\n")
    lines.append(row("Seed43 ckpt60", metrics["seed43_dev"]) + "\n")
    lines.append(row("Goldfixed ckpt20", metrics["goldfixed20_dev"]) + "\n")
    lines.append(row("Goldfixed ckpt40", metrics["goldfixed40_dev"]) + "\n")
    lines.append(row("Goldfixed ckpt60", metrics["goldfixed60_dev"]) + "\n\n")
    lines.append("## 5. Test Check\n\n")
    lines.append("| Method | Test R@5 | Test answer | Test strict | Format | Retrieval miss | Hit-answer-wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    lines.append(row("Format-SFT", metrics["format_test"]) + "\n")
    lines.append(row("Current seed42 ckpt60", metrics["seed42_test"]) + "\n")
    lines.append(row("Seed43 ckpt60", metrics["seed43_test"]) + "\n")
    lines.append(row("Goldfixed ckpt60", metrics["goldfixed60_test"]) + "\n\n")
    lines.append("## 6. Decision\n\n")
    lines.append(
        "NO PROMOTION. The fixed-corpus rerun is train-healthy and improves dev strict to `50.0%`, "
        "but test strict is `39.1%`, below the current seed42 main checkpoint's `40.6%`. "
        "Test R@5 also drops from `51.6%` to `50.0%`. Keep the existing seed42 scale60_s320 checkpoint as the paper-main checkpoint, "
        "and treat the fixed-corpus run as a useful robustness/control run rather than the new main result.\n\n"
    )
    lines.append("## 7. Next Mainline Action\n\n")
    lines.append(
        "Do not run another same-recipe GRPO immediately. The useful signal from the fixed corpus is that reward health is solid, "
        "but the generalization bottleneck remains retrieval coverage and answer extraction. The next paper-facing step should be a targeted comparison/report section: "
        "main seed42 result, seed43 confirmation, fixed-corpus control, and failure categories. Only after that should another method change be attempted.\n"
    )
    REPORT.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
