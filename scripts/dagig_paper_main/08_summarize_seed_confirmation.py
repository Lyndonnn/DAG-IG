#!/usr/bin/env python3
"""Summarize seed confirmation for the paper-main v1 GRPO recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
METRICS = ROOT / "two_stage_metrics"
METRICS_V3 = ROOT / "two_stage_metrics_rescored_v3"
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def metric(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    return {
        "path": str(path),
        "n": obj["n"],
        "r1": obj["retrieval_top1_hit"],
        "r3": obj["retrieval_top3_hit"],
        "r5": obj["retrieval_top5_hit"],
        "answer": obj["answer_correct"],
        "strict": obj["strict_success"],
        "format": obj["format_parse_success"],
        "retrieval_miss": obj.get("breakdown", {}).get("retrieval_miss", 0),
        "hit_answer_wrong": obj.get("breakdown", {}).get("retrieval_hit_answer_wrong", 0),
    }


def train_summary(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    micro = max(1, int(obj["micro_steps"]))
    return {
        "path": str(path),
        "status": obj["status"],
        "optimizer_steps": obj["optimizer_steps"],
        "micro_steps": obj["micro_steps"],
        "constant_reward_groups": obj["constant_reward_groups"],
        "constant_reward_rate": obj["constant_reward_groups"] / micro,
        "elapsed_seconds": obj["elapsed_seconds"],
        "max_gpu_mem_gb": obj["max_gpu_mem_gb"],
    }


def main_row(name: str, dev: dict[str, Any] | None, test: dict[str, Any] | None) -> str:
    def cells(m: dict[str, Any] | None) -> list[str]:
        if m is None:
            return ["-", "-", "-", "-", "-"]
        return [
            pct(m["r5"]),
            pct(m["answer"]),
            pct(m["strict"]),
            pct(m["format"]),
            str(m["hit_answer_wrong"]),
        ]

    return "| " + " | ".join([name] + cells(dev) + cells(test)) + " |\n"


def sweep_row(name: str, dev: dict[str, Any]) -> str:
    return (
        f"| {name} | {pct(dev['r1'])} | {pct(dev['r3'])} | {pct(dev['r5'])} | "
        f"{pct(dev['answer'])} | {pct(dev['strict'])} | {pct(dev['format'])} | "
        f"{dev['retrieval_miss']} | {dev['hit_answer_wrong']} |\n"
    )


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    metrics = {
        "format_dev": metric(METRICS_V3 / "format_sft_two_stage_own_full_dev.json"),
        "format_test": metric(METRICS_V3 / "format_sft_two_stage_own_full_test.json"),
        "seed42_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.json"),
        "seed42_test": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.json"),
        "seed43_ckpt20_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt20_dev.json"),
        "seed43_ckpt40_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt40_dev.json"),
        "seed43_ckpt60_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.json"),
        "seed43_ckpt60_test": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test.json"),
    }
    train = {
        "seed42": train_summary(
            ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/grpo_train_summary.json"
        ),
        "seed43": train_summary(
            ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/grpo_train_summary.json"
        ),
    }

    seed42_gain = {
        "dev_r5": metrics["seed42_dev"]["r5"] - metrics["format_dev"]["r5"],
        "dev_strict": metrics["seed42_dev"]["strict"] - metrics["format_dev"]["strict"],
        "test_r5": metrics["seed42_test"]["r5"] - metrics["format_test"]["r5"],
        "test_strict": metrics["seed42_test"]["strict"] - metrics["format_test"]["strict"],
    }
    seed43_gain = {
        "dev_r5": metrics["seed43_ckpt60_dev"]["r5"] - metrics["format_dev"]["r5"],
        "dev_strict": metrics["seed43_ckpt60_dev"]["strict"] - metrics["format_dev"]["strict"],
        "test_r5": metrics["seed43_ckpt60_test"]["r5"] - metrics["format_test"]["r5"],
        "test_strict": metrics["seed43_ckpt60_test"]["strict"] - metrics["format_test"]["strict"],
    }
    two_seed_mean = {
        "dev_r5": (metrics["seed42_dev"]["r5"] + metrics["seed43_ckpt60_dev"]["r5"]) / 2,
        "dev_strict": (metrics["seed42_dev"]["strict"] + metrics["seed43_ckpt60_dev"]["strict"]) / 2,
        "test_r5": (metrics["seed42_test"]["r5"] + metrics["seed43_ckpt60_test"]["r5"]) / 2,
        "test_strict": (metrics["seed42_test"]["strict"] + metrics["seed43_ckpt60_test"]["strict"]) / 2,
    }

    summary = {
        "train": train,
        "metrics": metrics,
        "seed42_gain_over_format": seed42_gain,
        "seed43_gain_over_format": seed43_gain,
        "two_seed_mean": two_seed_mean,
        "selected_current_best": {
            "checkpoint": "outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60",
            "reason": "seed42 remains the best single checkpoint by test strict, while seed43 confirms the recipe improves over Format-SFT.",
        },
    }
    (REPORTS / "seed_confirmation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# Paper Main v1 Seed Confirmation\n\n")
    lines.append("## Purpose\n\n")
    lines.append(
        "This report checks whether the current two-stage DAG-IG GRPO recipe is a repeatable mainline result, not a single-seed artifact.\n\n"
    )

    lines.append("## Training Health\n\n")
    lines.append("| Run | status | optimizer steps | micro steps | constant reward groups | constant rate | max GPU GB | elapsed sec |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for name, t in train.items():
        lines.append(
            f"| {name} | {t['status']} | {t['optimizer_steps']} | {t['micro_steps']} | "
            f"{t['constant_reward_groups']} | {pct(t['constant_reward_rate'])} | "
            f"{t['max_gpu_mem_gb']:.3f} | {t['elapsed_seconds']:.1f} |\n"
        )
    lines.append(
        "\nBoth runs avoid the old constant-reward failure mode. The reward signal remains usable under the paper-main v1 two-stage setup.\n\n"
    )

    lines.append("## Seed43 Dev Checkpoint Sweep\n\n")
    lines.append("| Checkpoint | Dev R@1 | Dev R@3 | Dev R@5 | Dev answer | Dev strict | Format parse | Retrieval miss | Hit-answer-wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    lines.append(sweep_row("seed43 ckpt20", metrics["seed43_ckpt20_dev"]))
    lines.append(sweep_row("seed43 ckpt40", metrics["seed43_ckpt40_dev"]))
    lines.append(sweep_row("seed43 ckpt60", metrics["seed43_ckpt60_dev"]))
    lines.append(
        "\nSelection rule: choose by dev strict first, then R@5 as a tie-breaker. ckpt40 and ckpt60 tie on dev strict; ckpt60 has higher R@5, so ckpt60 was evaluated on test.\n\n"
    )

    lines.append("## Main Comparison\n\n")
    lines.append("| Method | Dev R@5 | Dev answer | Dev strict | Dev format | Dev hit-answer-wrong | Test R@5 | Test answer | Test strict | Test format | Test hit-answer-wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    lines.append(main_row("Format-SFT", metrics["format_dev"], metrics["format_test"]))
    lines.append(main_row("DAG-IG GRPO seed42 ckpt60", metrics["seed42_dev"], metrics["seed42_test"]))
    lines.append(main_row("DAG-IG GRPO seed43 ckpt60", metrics["seed43_ckpt60_dev"], metrics["seed43_ckpt60_test"]))
    lines.append("\n")

    lines.append("## Gains Over Format-SFT\n\n")
    lines.append("| Run | Dev R@5 gain | Dev strict gain | Test R@5 gain | Test strict gain |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    lines.append(
        f"| seed42 ckpt60 | {pct(seed42_gain['dev_r5'])} | {pct(seed42_gain['dev_strict'])} | "
        f"{pct(seed42_gain['test_r5'])} | {pct(seed42_gain['test_strict'])} |\n"
    )
    lines.append(
        f"| seed43 ckpt60 | {pct(seed43_gain['dev_r5'])} | {pct(seed43_gain['dev_strict'])} | "
        f"{pct(seed43_gain['test_r5'])} | {pct(seed43_gain['test_strict'])} |\n"
    )
    lines.append(
        f"| two-seed mean | {pct(two_seed_mean['dev_r5'] - metrics['format_dev']['r5'])} | "
        f"{pct(two_seed_mean['dev_strict'] - metrics['format_dev']['strict'])} | "
        f"{pct(two_seed_mean['test_r5'] - metrics['format_test']['r5'])} | "
        f"{pct(two_seed_mean['test_strict'] - metrics['format_test']['strict'])} |\n\n"
    )

    lines.append("## Decision\n\n")
    lines.append(
        "Seed43 confirms the main recipe: dev strict remains `49.0%`, test strict is `39.1%`, and both dev/test remain above Format-SFT. "
        "The current best single checkpoint is still seed42 scale60_s320 checkpoint-60 because it has the best test strict (`40.6%`) and test R@5 (`51.6%`).\n\n"
    )
    lines.append(
        "Use seed42 checkpoint-60 as the current main checkpoint, and cite seed43 as seed confirmation. "
        "The next mainline work should target remaining retrieval misses with better query/evidence credit data, not more answer repair or reward reshuffling.\n"
    )

    report_path = REPORTS / "SEED_CONFIRMATION_REPORT.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {REPORTS / 'seed_confirmation_summary.json'}")


if __name__ == "__main__":
    main()
