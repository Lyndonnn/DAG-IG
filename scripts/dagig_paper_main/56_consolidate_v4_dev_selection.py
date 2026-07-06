#!/usr/bin/env python3
"""Consolidate checker-v4 results with dev-only checkpoint selection.

This report corrects two audit findings for already-generated predictions:
- answer checker v4 is used consistently;
- test strict is not used to choose or promote a checkpoint.

It does not claim final paper validity for the old GRPO checkpoints because the
training-time KL implementation was fixed after those checkpoints were trained.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
METRICS_V4 = ROOT / "two_stage_metrics_rescored_v4"
PREDS_V4 = ROOT / "two_stage_predictions_rescored_v4"
REPORT = ROOT / "reports/PAPER_MAIN_V1_CORRECTED_V4_DEV_SELECTION.md"
SUMMARY = ROOT / "reports/paper_main_v1_corrected_v4_dev_selection.json"


RUNS = {
    "format": {
        "label": "Format-SFT baseline",
        "dev": "format_sft_two_stage_own_full_dev",
        "test": "format_sft_two_stage_own_full_test",
    },
    "seed42": {
        "label": "DAG-IG old-KL seed42",
        "dev": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev",
        "test": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test",
    },
    "seed43": {
        "label": "DAG-IG old-KL seed43",
        "dev": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev",
        "test": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test",
    },
    "goldfixed": {
        "label": "Goldfixed train-corpus control",
        "dev": "paper_main_v1_goldfixed_scale60_s320_ckpt60_dev",
        "test": "paper_main_v1_goldfixed_scale60_s320_ckpt60_test",
    },
    "seed42_fixed_reader": {
        "label": "DAG-IG seed42 with fixed Format-SFT reader",
        "dev": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_formatreader__reader_format_sft_dev",
        "test": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_formatreader__reader_format_sft_test",
    },
}


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


def metric(stem: str) -> dict[str, Any]:
    data = load_json(METRICS_V4 / f"{stem}.json")
    return {
        "stem": stem,
        "n": data.get("n"),
        "r1": data.get("retrieval_top1_hit"),
        "r3": data.get("retrieval_top3_hit"),
        "r5": data.get("retrieval_top5_hit"),
        "answer": data.get("answer_correct"),
        "strict": data.get("strict_success"),
        "format": data.get("format_parse_success"),
        "retrieval_miss": (data.get("breakdown") or {}).get("retrieval_miss"),
        "hit_answer_wrong": (data.get("breakdown") or {}).get("retrieval_hit_answer_wrong"),
        "strict_count": data.get("new_strict_count"),
        "answer_count": data.get("new_answer_correct_count"),
        "checker_delta": data.get("strict_delta_count"),
    }


def prediction_map(stem: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in read_jsonl(PREDS_V4 / f"{stem}.jsonl")}


def exact_mcnemar_p(method_only: int, base_only: int) -> float:
    n = method_only + base_only
    if n == 0:
        return 1.0
    k = min(method_only, base_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def compare(base_stem: str, method_stem: str) -> dict[str, Any]:
    base = prediction_map(base_stem)
    method = prediction_map(method_stem)
    common = sorted(set(base) & set(method))
    counts = {
        "common": len(common),
        "both_strict": 0,
        "method_only_strict": 0,
        "base_only_strict": 0,
        "both_fail": 0,
        "method_r5_gain": 0,
        "method_r5_loss": 0,
    }
    for sid in common:
        b = bool(base[sid].get("strict_success"))
        m = bool(method[sid].get("strict_success"))
        if b and m:
            counts["both_strict"] += 1
        elif m and not b:
            counts["method_only_strict"] += 1
        elif b and not m:
            counts["base_only_strict"] += 1
        else:
            counts["both_fail"] += 1
        b_r5 = bool(base[sid].get("retrieval_top5_hit"))
        m_r5 = bool(method[sid].get("retrieval_top5_hit"))
        counts["method_r5_gain"] += int(m_r5 and not b_r5)
        counts["method_r5_loss"] += int(b_r5 and not m_r5)
    counts["mcnemar_exact_p"] = exact_mcnemar_p(counts["method_only_strict"], counts["base_only_strict"])
    return counts


def row(label: str, split: str, m: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    strict_gain = ""
    if baseline is not None:
        strict_gain = f"{100.0 * (float(m['strict']) - float(baseline['strict'])):+.1f}"
    else:
        strict_gain = "-"
    return (
        f"| {label} | {split} | {m['n']} | {pct(m['r5'])} | {pct(m['answer'])} | "
        f"{pct(m['strict'])} | {strict_gain} | {m['retrieval_miss']} | {m['hit_answer_wrong']} |"
    )


def main() -> None:
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for key, spec in RUNS.items():
        metrics[key] = {"dev": metric(spec["dev"]), "test": metric(spec["test"])}

    baseline = metrics["format"]
    comparisons: dict[str, dict[str, Any]] = {}
    for key in ["seed42", "seed43", "goldfixed", "seed42_fixed_reader"]:
        comparisons[f"{key}_dev_vs_format"] = compare(RUNS["format"]["dev"], RUNS[key]["dev"])
        comparisons[f"{key}_test_vs_format"] = compare(RUNS["format"]["test"], RUNS[key]["test"])

    candidates = ["seed42", "seed43", "goldfixed"]
    dev_selected = max(candidates, key=lambda key: (metrics[key]["dev"]["strict"], metrics[key]["dev"]["r5"]))
    seed_mean = {
        split: {
            "strict": (metrics["seed42"][split]["strict"] + metrics["seed43"][split]["strict"]) / 2.0,
            "r5": (metrics["seed42"][split]["r5"] + metrics["seed43"][split]["r5"]) / 2.0,
            "answer": (metrics["seed42"][split]["answer"] + metrics["seed43"][split]["answer"]) / 2.0,
        }
        for split in ["dev", "test"]
    }
    summary = {
        "checker_version": "v4",
        "dev_selected_without_test": dev_selected,
        "metrics": metrics,
        "seed42_seed43_mean": seed_mean,
        "comparisons": comparisons,
        "note": "Old checkpoints remain diagnostic until KL-fixed GRPO is rerun.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Paper Main v1 Corrected v4 Dev-Selection Report\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This report corrects existing paper-main predictions for answer checker v4 and removes test-set checkpoint selection. "
        "It does not retrain models and it does not repair the old training-time KL bug. Therefore these old-KL checkpoints are diagnostic only until KL-fixed GRPO is rerun.\n\n"
    )
    lines.append("## v4 Main Table\n\n")
    lines.append("| Method | Split | n | R@5 | Answer correct | Strict success | Strict gain vs Format | Retrieval miss | Hit-answer-wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for key in ["format", "seed42", "seed43", "goldfixed", "seed42_fixed_reader"]:
        for split in ["dev", "test"]:
            base = None if key == "format" else baseline[split]
            lines.append(row(RUNS[key]["label"], split, metrics[key][split], base) + "\n")
    lines.append("\n## Dev-Only Selection\n\n")
    lines.append(
        f"- Candidate runs considered for single-checkpoint selection: `seed42`, `seed43`, `goldfixed`.\n"
        f"- Selection rule: highest dev strict, then dev R@5. Test metrics are not used.\n"
        f"- Selected by this rule: `{dev_selected}` ({RUNS[dev_selected]['label']}).\n"
        f"- Selected dev/test strict: `{pct(metrics[dev_selected]['dev']['strict'])}` / `{pct(metrics[dev_selected]['test']['strict'])}`.\n"
        f"- The old headline that chose seed42 because of higher test strict is not protocol-clean and should not be used.\n\n"
    )
    lines.append("## Two-Seed Mean\n\n")
    lines.append(
        f"- seed42/seed43 mean dev strict: `{pct(seed_mean['dev']['strict'])}`; mean dev R@5: `{pct(seed_mean['dev']['r5'])}`.\n"
        f"- seed42/seed43 mean test strict: `{pct(seed_mean['test']['strict'])}`; mean test R@5: `{pct(seed_mean['test']['r5'])}`.\n"
        f"- Mean strict gain vs Format-SFT: dev `{100*(seed_mean['dev']['strict'] - baseline['dev']['strict']):+.1f}` points, "
        f"test `{100*(seed_mean['test']['strict'] - baseline['test']['strict']):+.1f}` points.\n\n"
    )
    lines.append("## Paired Strict Comparisons vs Format-SFT\n\n")
    lines.append("| Comparison | Split | common | method-only strict | format-only strict | McNemar exact p | R@5 gain | R@5 loss |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for key in ["seed42", "seed43", "goldfixed", "seed42_fixed_reader"]:
        for split in ["dev", "test"]:
            c = comparisons[f"{key}_{split}_vs_format"]
            lines.append(
                f"| {RUNS[key]['label']} | {split} | {c['common']} | {c['method_only_strict']} | "
                f"{c['base_only_strict']} | {c['mcnemar_exact_p']:.3f} | {c['method_r5_gain']} | {c['method_r5_loss']} |\n"
            )
    lines.append("\n## Decision\n\n")
    lines.append(
        "Use this report as the corrected status for old predictions only. "
        "The answer-checker false positives are fixed, and the test-set selection issue is removed. "
        "However, because the old GRPO checkpoints were trained with the incorrect signed KL penalty, they are not final paper-main results. "
        "The next required experiment is a KL-fixed rerun of the same recipe, reported with checker v4 and dev-only selection/two-seed mean.\n"
    )
    REPORT.write_text("".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
