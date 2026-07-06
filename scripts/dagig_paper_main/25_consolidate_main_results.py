#!/usr/bin/env python3
"""Build a paper-main consolidated result and failure report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
METRICS = ROOT / "two_stage_metrics"
METRICS_V3 = ROOT / "two_stage_metrics_rescored_v3"
PREDS = ROOT / "two_stage_predictions"
PREDS_V3 = ROOT / "two_stage_predictions_rescored_v3"
REPORT = ROOT / "reports/PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md"
SUMMARY = ROOT / "reports/paper_main_v1_consolidated_results.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.1f}%"


def metric(path: Path) -> dict[str, Any]:
    data = load_json(path)
    breakdown = data.get("breakdown") or {}
    return {
        "path": str(path),
        "n": data.get("n"),
        "r1": data.get("retrieval_top1_hit"),
        "r3": data.get("retrieval_top3_hit"),
        "r5": data.get("retrieval_top5_hit"),
        "answer": data.get("answer_correct"),
        "strict": data.get("strict_success"),
        "format": data.get("format_parse_success"),
        "answer_in_query": data.get("answer_in_query_rate"),
        "retrieval_miss": breakdown.get("retrieval_miss"),
        "hit_answer_wrong": breakdown.get("retrieval_hit_answer_wrong"),
        "stage1_format_failure": breakdown.get("stage1_format_failure"),
        "reader_format_failure": breakdown.get("reader_format_failure"),
    }


def prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in read_jsonl(path)}


def compare_predictions(base_path: Path, method_path: Path) -> dict[str, Any]:
    base = prediction_map(base_path)
    method = prediction_map(method_path)
    common = sorted(set(base) & set(method))
    counts = {
        "common": len(common),
        "both_strict": 0,
        "method_only_strict": 0,
        "base_only_strict": 0,
        "both_fail_strict": 0,
        "method_retrieval_gain": 0,
        "method_retrieval_loss": 0,
        "same_retrieval": 0,
    }
    examples = {"method_only_strict": [], "base_only_strict": [], "method_retrieval_gain": [], "method_retrieval_loss": []}
    for sid in common:
        b = base[sid]
        m = method[sid]
        b_strict = bool(b.get("strict_success"))
        m_strict = bool(m.get("strict_success"))
        if b_strict and m_strict:
            counts["both_strict"] += 1
        elif m_strict and not b_strict:
            counts["method_only_strict"] += 1
            if len(examples["method_only_strict"]) < 5:
                examples["method_only_strict"].append(sid)
        elif b_strict and not m_strict:
            counts["base_only_strict"] += 1
            if len(examples["base_only_strict"]) < 5:
                examples["base_only_strict"].append(sid)
        else:
            counts["both_fail_strict"] += 1
        b_r5 = bool(b.get("retrieval_top5_hit"))
        m_r5 = bool(m.get("retrieval_top5_hit"))
        if m_r5 and not b_r5:
            counts["method_retrieval_gain"] += 1
            if len(examples["method_retrieval_gain"]) < 5:
                examples["method_retrieval_gain"].append(sid)
        elif b_r5 and not m_r5:
            counts["method_retrieval_loss"] += 1
            if len(examples["method_retrieval_loss"]) < 5:
                examples["method_retrieval_loss"].append(sid)
        else:
            counts["same_retrieval"] += 1
    counts["examples"] = examples
    return counts


def method_row(name: str, split: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {split} | {pct(m.get('r1'))} | {pct(m.get('r3'))} | {pct(m.get('r5'))} | "
        f"{pct(m.get('answer'))} | {pct(m.get('strict'))} | {pct(m.get('format'))} | "
        f"{m.get('retrieval_miss', '-')} | {m.get('hit_answer_wrong', '-')} |"
    )


def compare_row(name: str, c: dict[str, Any]) -> str:
    return (
        f"| {name} | {c['common']} | {c['method_only_strict']} | {c['base_only_strict']} | "
        f"{c['both_strict']} | {c['both_fail_strict']} | {c['method_retrieval_gain']} | {c['method_retrieval_loss']} |"
    )


def main() -> None:
    metrics = {
        "format_dev": metric(METRICS_V3 / "format_sft_two_stage_own_full_dev.json"),
        "format_test": metric(METRICS_V3 / "format_sft_two_stage_own_full_test.json"),
        "seed42_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.json"),
        "seed42_test": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.json"),
        "seed43_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.json"),
        "seed43_test": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test.json"),
        "goldfixed_dev": metric(METRICS / "paper_main_v1_goldfixed_scale60_s320_ckpt60_dev.json"),
        "goldfixed_test": metric(METRICS / "paper_main_v1_goldfixed_scale60_s320_ckpt60_test.json"),
    }
    comparisons = {
        "seed42_vs_format_dev": compare_predictions(
            PREDS_V3 / "format_sft_two_stage_own_full_dev.jsonl",
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.jsonl",
        ),
        "seed42_vs_format_test": compare_predictions(
            PREDS_V3 / "format_sft_two_stage_own_full_test.jsonl",
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.jsonl",
        ),
        "seed43_vs_format_dev": compare_predictions(
            PREDS_V3 / "format_sft_two_stage_own_full_dev.jsonl",
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.jsonl",
        ),
        "seed43_vs_format_test": compare_predictions(
            PREDS_V3 / "format_sft_two_stage_own_full_test.jsonl",
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test.jsonl",
        ),
        "goldfixed_vs_format_dev": compare_predictions(
            PREDS_V3 / "format_sft_two_stage_own_full_dev.jsonl",
            PREDS / "paper_main_v1_goldfixed_scale60_s320_ckpt60_dev.jsonl",
        ),
        "goldfixed_vs_format_test": compare_predictions(
            PREDS_V3 / "format_sft_two_stage_own_full_test.jsonl",
            PREDS / "paper_main_v1_goldfixed_scale60_s320_ckpt60_test.jsonl",
        ),
        "goldfixed_vs_seed42_dev": compare_predictions(
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.jsonl",
            PREDS / "paper_main_v1_goldfixed_scale60_s320_ckpt60_dev.jsonl",
        ),
        "goldfixed_vs_seed42_test": compare_predictions(
            PREDS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.jsonl",
            PREDS / "paper_main_v1_goldfixed_scale60_s320_ckpt60_test.jsonl",
        ),
    }
    summary = {"metrics": metrics, "comparisons": comparisons}
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Paper Main v1 Consolidated Results\n\n")
    lines.append(
        "> Status: superseded old-KL diagnostic. This report consolidates the pre-audit mainline before the KL/checker fixes. Use `KLFIXED_GRPO_60_REPORT.md` for the corrected paper-facing result.\n\n"
    )
    lines.append("## 1. Scope\n\n")
    lines.append(
        "This report consolidates earlier paper-main evidence: Format-SFT baseline, old-KL seed42, old-KL seed43, and fixed-corpus control. "
        "It does not reflect the corrected k3 KL reruns or checker v4 headline.\n\n"
    )
    lines.append("## 2. Main Table\n\n")
    lines.append("| Method | Split | R@1 | R@3 | R@5 | Answer correct | Strict success | Format | Retrieval miss | Hit-answer-wrong |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = [
        ("Format-SFT", "dev", metrics["format_dev"]),
        ("Format-SFT", "test", metrics["format_test"]),
        ("DAG-IG seed42 main", "dev", metrics["seed42_dev"]),
        ("DAG-IG seed42 main", "test", metrics["seed42_test"]),
        ("DAG-IG seed43 confirm", "dev", metrics["seed43_dev"]),
        ("DAG-IG seed43 confirm", "test", metrics["seed43_test"]),
        ("Goldfixed control", "dev", metrics["goldfixed_dev"]),
        ("Goldfixed control", "test", metrics["goldfixed_test"]),
    ]
    for name, split, m in rows:
        lines.append(method_row(name, split, m) + "\n")
    lines.append("\n")
    lines.append("## 3. Strict Gain/Loss Counts\n\n")
    lines.append("| Comparison | Common samples | Method-only strict | Baseline-only strict | Both strict | Both fail | Retrieval gain | Retrieval loss |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for key in [
        "seed42_vs_format_dev",
        "seed42_vs_format_test",
        "seed43_vs_format_dev",
        "seed43_vs_format_test",
        "goldfixed_vs_format_dev",
        "goldfixed_vs_format_test",
        "goldfixed_vs_seed42_dev",
        "goldfixed_vs_seed42_test",
    ]:
        lines.append(compare_row(key, comparisons[key]) + "\n")
    lines.append("\n")
    lines.append("## 4. Main Claim Status\n\n")
    lines.append(
        "These old-KL numbers are diagnostic only. The corrected stable claim is stated in `KLFIXED_GRPO_60_REPORT.md`: "
        "KL-fixed two-seed mean strict success is `45.9%` dev and `39.1%` test, versus Format-SFT v4 `40.8%` dev and `34.4%` test.\n\n"
    )
    lines.append("## 5. Bottleneck\n\n")
    lines.append(
        "The old-KL bottleneck pattern was retrieval misses plus retrieval-hit-answer-wrong cases. "
        "For corrected failure counts, use `KLFIXED_GRPO_60_REPORT.md` and `klfixed_grpo_60_summary.json`.\n\n"
    )
    lines.append("## 6. Decision\n\n")
    lines.append(
        "Do not keep the old-KL seed42 checkpoint as the current main checkpoint. "
        "The corrected paper-facing result is the KL-fixed seed42/seed43 mean with checker v4 and fixed-reader control.\n"
    )
    REPORT.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
