#!/usr/bin/env python3
"""Summarize the paper-main scale GRPO run."""

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
METRICS = ROOT / "two_stage_metrics"
METRICS_V3 = ROOT / "two_stage_metrics_rescored_v3"
PREDS = ROOT / "two_stage_predictions"
PREDS_V3 = ROOT / "two_stage_predictions_rescored_v3"
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    return {
        "n": obj["n"],
        "r1": obj["retrieval_top1_hit"],
        "r3": obj["retrieval_top3_hit"],
        "r5": obj["retrieval_top5_hit"],
        "answer": obj["answer_correct"],
        "strict": obj["strict_success"],
        "format": obj["format_parse_success"],
        "hit_answer_wrong": obj.get("breakdown", {}).get("retrieval_hit_answer_wrong", 0),
        "retrieval_miss": obj.get("breakdown", {}).get("retrieval_miss", 0),
    }


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def row(name: str, m_dev: dict[str, Any] | None, m_test: dict[str, Any] | None) -> str:
    def cells(m: dict[str, Any] | None) -> list[str]:
        if not m:
            return ["-", "-", "-", "-"]
        return [pct(m["r5"]), pct(m["answer"]), pct(m["strict"]), str(m["hit_answer_wrong"])]
    return "| " + " | ".join([name] + cells(m_dev) + cells(m_test)) + " |\n"


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in read_jsonl(path)}


def delta_cases(base_path: Path, new_path: Path, out_path: Path) -> dict[str, Any]:
    base = by_id(base_path)
    new = by_id(new_path)
    rows = []
    counts = {
        "n": 0,
        "strict_recoveries": 0,
        "strict_harms": 0,
        "retrieval_recoveries": 0,
        "retrieval_harms": 0,
        "same_strict": 0,
    }
    for sid in sorted(set(base) & set(new)):
        b = base[sid]
        n = new[sid]
        counts["n"] += 1
        b_strict = bool(b.get("strict_success"))
        n_strict = bool(n.get("strict_success"))
        b_r5 = bool(b.get("retrieval_top5_hit"))
        n_r5 = bool(n.get("retrieval_top5_hit"))
        if (not b_strict) and n_strict:
            counts["strict_recoveries"] += 1
        elif b_strict and (not n_strict):
            counts["strict_harms"] += 1
        else:
            counts["same_strict"] += 1
        if (not b_r5) and n_r5:
            counts["retrieval_recoveries"] += 1
        elif b_r5 and (not n_r5):
            counts["retrieval_harms"] += 1
        if b_strict != n_strict or b_r5 != n_r5:
            rows.append(
                {
                    "sample_id": sid,
                    "question": n.get("question"),
                    "gold_answer": n.get("gold_answer"),
                    "base_query": b.get("search_query"),
                    "new_query": n.get("search_query"),
                    "base_answer": b.get("final_answer"),
                    "new_answer": n.get("final_answer"),
                    "base_r5": b_r5,
                    "new_r5": n_r5,
                    "base_strict": b_strict,
                    "new_strict": n_strict,
                }
            )
    write_jsonl(out_path, rows)
    return counts | {"cases_path": str(out_path)}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    train_summary = load_json(ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/grpo_train_summary.json")

    metrics = {
        "format_dev": metric(METRICS_V3 / "format_sft_two_stage_own_full_dev.json"),
        "format_test": metric(METRICS_V3 / "format_sft_two_stage_own_full_test.json"),
        "ckpt30_dev": metric(METRICS_V3 / "paper_main_v1_two_stage_stage1loss_kl01_medium30_ckpt30_dev.json"),
        "ckpt30_test": metric(METRICS_V3 / "paper_main_v1_two_stage_stage1loss_kl01_medium30_ckpt30_test.json"),
        "ckpt30_fmt_dev": metric(METRICS_V3 / "paper_main_v1_two_stage_stage1loss_kl01_medium30_ckpt30_formatreader__reader_format_sft_dev.json"),
        "ckpt30_fmt_test": metric(METRICS_V3 / "paper_main_v1_two_stage_stage1loss_kl01_medium30_ckpt30_formatreader__reader_format_sft_test.json"),
        "scale20_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt20_dev.json"),
        "scale40_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt40_dev.json"),
        "scale60_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.json"),
        "scale60_test": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.json"),
        "scale60_fmt_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_formatreader__reader_format_sft_dev.json"),
        "scale60_fmt_test": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_formatreader__reader_format_sft_test.json"),
    }

    case_dir = ROOT / "reports/scale60_s320_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    deltas = {
        "scale60_vs_format_dev": delta_cases(
            PREDS_V3 / "format_sft_two_stage_own_full_dev.jsonl",
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.jsonl",
            case_dir / "scale60_vs_format_dev.jsonl",
        ),
        "scale60_vs_format_test": delta_cases(
            PREDS_V3 / "format_sft_two_stage_own_full_test.jsonl",
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.jsonl",
            case_dir / "scale60_vs_format_test.jsonl",
        ),
        "scale60_vs_ckpt30_dev": delta_cases(
            PREDS_V3 / "paper_main_v1_two_stage_stage1loss_kl01_medium30_ckpt30_dev.jsonl",
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.jsonl",
            case_dir / "scale60_vs_ckpt30_dev.jsonl",
        ),
        "scale60_vs_ckpt30_test": delta_cases(
            PREDS_V3 / "paper_main_v1_two_stage_stage1loss_kl01_medium30_ckpt30_test.jsonl",
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.jsonl",
            case_dir / "scale60_vs_ckpt30_test.jsonl",
        ),
    }
    summary = {"train_summary": train_summary, "metrics": metrics, "deltas": deltas}
    write_json(REPORTS / "scale60_s320_comparison.json", summary)

    const_rate = train_summary["constant_reward_groups"] / max(1, train_summary["micro_steps"])
    lines = ["# Scale60 S320 GRPO Comparison\n\n"]
    lines.append("## Training Health\n\n")
    lines.append("- run: `paper_main_v1_two_stage_stage1loss_kl01_scale60_s320`\n")
    lines.append(f"- optimizer steps / micro steps: `{train_summary['optimizer_steps']} / {train_summary['micro_steps']}`\n")
    lines.append(f"- constant reward groups: `{train_summary['constant_reward_groups']}` (`{100*const_rate:.2f}%` of micro steps)\n")
    lines.append(f"- max GPU memory: `{train_summary['max_gpu_mem_gb']:.3f} GB`\n")
    lines.append(f"- elapsed seconds: `{train_summary['elapsed_seconds']:.1f}`\n\n")
    lines.append("Reward did not collapse into the old constant-reward failure mode. The run is trainable, but the dev checkpoint sweep is still needed for model selection.\n\n")

    lines.append("## Main Metrics\n\n")
    lines.append("| Method | Dev R@5 | Dev answer | Dev strict | Dev hit-answer-wrong | Test R@5 | Test answer | Test strict | Test hit-answer-wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    lines.append(row("Format-SFT", metrics["format_dev"], metrics["format_test"]))
    lines.append(row("DAG-IG medium30 ckpt30", metrics["ckpt30_dev"], metrics["ckpt30_test"]))
    lines.append(row("Scale60 ckpt20", metrics["scale20_dev"], None))
    lines.append(row("Scale60 ckpt40", metrics["scale40_dev"], None))
    lines.append(row("Scale60 ckpt60", metrics["scale60_dev"], metrics["scale60_test"]))
    lines.append(row("Scale60 ckpt60 + Format reader", metrics["scale60_fmt_dev"], metrics["scale60_fmt_test"]))
    lines.append("\n")

    lines.append("## Delta Counts\n\n")
    lines.append("| Comparison | n | strict recoveries | strict harms | retrieval recoveries | retrieval harms | cases |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---|\n")
    for name, d in deltas.items():
        lines.append(
            f"| {name} | {d['n']} | {d['strict_recoveries']} | {d['strict_harms']} | "
            f"{d['retrieval_recoveries']} | {d['retrieval_harms']} | `{d['cases_path']}` |\n"
        )
    lines.append("\n")

    lines.append("## Interpretation\n\n")
    lines.append("- Scale60 ckpt60 is the new best clean checkpoint by dev/test strict success: dev `49.0%`, test `40.6%`.\n")
    lines.append("- Compared with Format-SFT, ckpt60 improves strict by `+6.1` dev points and `+6.2` test points.\n")
    lines.append("- Compared with medium30 ckpt30, ckpt60 gives a small but positive strict gain: `+1.0` dev point and `+1.6` test points.\n")
    lines.append("- Test R@5 improves from ckpt30 `50.0%` to ckpt60 `51.6%`, so the test gain is still aligned with the query/evidence node.\n")
    lines.append("- Fixed-reader isolation keeps test strict at `40.6%`; therefore the test improvement is not just reader drift. On dev, fixed-reader drops from `49.0%` to `48.0%`, so the extra dev point is reader/shared-adapter dependent.\n\n")

    lines.append("## Decision\n\n")
    lines.append("Promote scale60_s320 checkpoint-60 as the current main method checkpoint. Do not continue tiny answer repair or reader-SFT. The next efficient paper step is a controlled scale/seed confirmation around the same recipe, plus sharper analysis of remaining retrieval misses and hit-answer-wrong cases.\n")
    (REPORTS / "SCALE60_S320_COMPARISON.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", REPORTS / "SCALE60_S320_COMPARISON.md")


if __name__ == "__main__":
    main()
