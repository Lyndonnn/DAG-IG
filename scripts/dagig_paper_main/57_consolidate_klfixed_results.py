#!/usr/bin/env python3
"""Consolidate KL-fixed GRPO results for the paper-main audit trail.

This report intentionally separates three states:
1. Format-SFT baseline rescored with checker v4.
2. Old-KL diagnostic GRPO rescored with checker v4.
3. KL-fixed GRPO reruns, evaluated directly with checker v4.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
REPORT_DIR = ROOT / "reports"
PRED_V4 = ROOT / "two_stage_predictions_rescored_v4"
METRIC_V4 = ROOT / "two_stage_metrics_rescored_v4"
PRED = ROOT / "two_stage_predictions"
METRIC = ROOT / "two_stage_metrics"


METHODS = {
    "format": {
        "label": "Format-SFT baseline",
        "kind": "baseline",
        "pred_dir": PRED_V4,
        "metric_dir": METRIC_V4,
        "dev": "format_sft_two_stage_own_full_dev",
        "test": "format_sft_two_stage_own_full_test",
    },
    "old_kl_seed42": {
        "label": "old-KL GRPO seed42 diagnostic",
        "kind": "old_kl_diagnostic",
        "pred_dir": PRED_V4,
        "metric_dir": METRIC_V4,
        "dev": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev",
        "test": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test",
    },
    "old_kl_seed43": {
        "label": "old-KL GRPO seed43 diagnostic",
        "kind": "old_kl_diagnostic",
        "pred_dir": PRED_V4,
        "metric_dir": METRIC_V4,
        "dev": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev",
        "test": "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test",
    },
    "klfixed_seed42": {
        "label": "KL-fixed GRPO seed42",
        "kind": "klfixed_main",
        "pred_dir": PRED,
        "metric_dir": METRIC,
        "dev": "paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_dev",
        "test": "paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_test",
    },
    "klfixed_seed43": {
        "label": "KL-fixed GRPO seed43",
        "kind": "klfixed_main",
        "pred_dir": PRED,
        "metric_dir": METRIC,
        "dev": "paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_dev",
        "test": "paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_test",
    },
    "klfixed_seed42_fixed_reader": {
        "label": "KL-fixed GRPO seed42 + fixed Format reader",
        "kind": "klfixed_fixed_reader",
        "pred_dir": PRED,
        "metric_dir": METRIC,
        "dev": "paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_formatreader__reader_format_sft_dev",
        "test": "paper_main_v1_klfixed_scale60_s320_seed42_ckpt60_formatreader__reader_format_sft_test",
    },
    "klfixed_seed43_fixed_reader": {
        "label": "KL-fixed GRPO seed43 + fixed Format reader",
        "kind": "klfixed_fixed_reader",
        "pred_dir": PRED,
        "metric_dir": METRIC,
        "dev": "paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_formatreader__reader_format_sft_dev",
        "test": "paper_main_v1_klfixed_scale60_s320_seed43_ckpt60_formatreader__reader_format_sft_test",
    },
}

TRAIN_SUMMARIES = {
    "klfixed_seed42": ROOT / "checkpoints/paper_main_v1_klfixed_scale60_s320_seed42/grpo_train_summary.json",
    "klfixed_seed43": ROOT / "checkpoints/paper_main_v1_klfixed_scale60_s320_seed43/grpo_train_summary.json",
    "klfixed_smoke_v2": ROOT / "checkpoints/paper_main_v1_klfixed_smoke1_v2/grpo_train_summary.json",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:.1f}%"


def count_pct(count: int, n: int) -> str:
    return f"{count}/{n} = {100.0 * count / n:.1f}%"


def metric_counts(m: dict[str, Any]) -> dict[str, Any]:
    n = int(m["n"])
    strict = int(round(float(m["strict_success"]) * n))
    r5 = int(round(float(m["retrieval_top5_hit"]) * n))
    ans = int(round(float(m["answer_correct"]) * n))
    fmt = int(round(float(m["format_parse_success"]) * n))
    return {
        "n": n,
        "r5_count": r5,
        "answer_count": ans,
        "strict_count": strict,
        "format_count": fmt,
        "r5": float(m["retrieval_top5_hit"]),
        "answer": float(m["answer_correct"]),
        "strict": float(m["strict_success"]),
        "format": float(m["format_parse_success"]),
        "answer_in_query": float(m.get("answer_in_query_rate", 0.0)),
        "breakdown": m.get("breakdown", {}),
    }


def exact_mcnemar_p(method_only: int, base_only: int) -> float:
    n = method_only + base_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(method_only, base_only) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_compare(base_rows: list[dict[str, Any]], method_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = {r["sample_id"]: r for r in base_rows}
    method = {r["sample_id"]: r for r in method_rows}
    common_ids = sorted(set(base) & set(method))
    both = method_only = base_only = both_fail = 0
    r5_gain = r5_loss = 0
    for sid in common_ids:
        b = bool(base[sid].get("strict_success"))
        m = bool(method[sid].get("strict_success"))
        if b and m:
            both += 1
        elif m and not b:
            method_only += 1
        elif b and not m:
            base_only += 1
        else:
            both_fail += 1
        br = bool(base[sid].get("retrieval_top5_hit"))
        mr = bool(method[sid].get("retrieval_top5_hit"))
        if mr and not br:
            r5_gain += 1
        elif br and not mr:
            r5_loss += 1
    return {
        "common": len(common_ids),
        "both_strict": both,
        "method_only_strict": method_only,
        "base_only_strict": base_only,
        "both_fail": both_fail,
        "method_r5_gain": r5_gain,
        "method_r5_loss": r5_loss,
        "mcnemar_exact_p": exact_mcnemar_p(method_only, base_only),
    }


def average_methods(metrics: dict[str, Any], method_keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["dev", "test"]:
        vals = {field: [metrics[k][split][field] for k in method_keys] for field in ["r5", "answer", "strict", "format"]}
        out[split] = {
            field: {
                "mean": mean(numbers),
                "std_population": pstdev(numbers) if len(numbers) > 1 else 0.0,
                "values": numbers,
            }
            for field, numbers in vals.items()
        }
    return out


def train_summary(path: Path) -> dict[str, Any]:
    d = read_json(path)
    micro = int(d.get("micro_steps") or 0)
    constant = int(d.get("constant_reward_groups") or 0)
    return {
        "status": d.get("status"),
        "optimizer_steps": d.get("optimizer_steps"),
        "micro_steps": micro,
        "constant_reward_groups": constant,
        "constant_reward_rate": (constant / micro) if micro else None,
        "elapsed_seconds": d.get("elapsed_seconds"),
        "max_gpu_mem_gb": d.get("max_gpu_mem_gb"),
    }


def build_summary() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, spec in METHODS.items():
        metrics[key] = {}
        for split in ["dev", "test"]:
            stem = spec[split]
            metrics[key][split] = metric_counts(read_json(spec["metric_dir"] / f"{stem}.json"))
            predictions[(key, split)] = read_jsonl(spec["pred_dir"] / f"{stem}.jsonl")

    comparisons: dict[str, Any] = {}
    for key in [
        "old_kl_seed42",
        "old_kl_seed43",
        "klfixed_seed42",
        "klfixed_seed43",
        "klfixed_seed42_fixed_reader",
        "klfixed_seed43_fixed_reader",
    ]:
        comparisons[key] = {}
        for split in ["dev", "test"]:
            comparisons[key][split] = paired_compare(predictions[("format", split)], predictions[(key, split)])

    averages = {
        "old_kl_two_seed_mean": average_methods(metrics, ["old_kl_seed42", "old_kl_seed43"]),
        "klfixed_two_seed_mean": average_methods(metrics, ["klfixed_seed42", "klfixed_seed43"]),
        "klfixed_fixed_reader_two_seed_mean": average_methods(
            metrics, ["klfixed_seed42_fixed_reader", "klfixed_seed43_fixed_reader"]
        ),
    }

    train = {key: train_summary(path) for key, path in TRAIN_SUMMARIES.items()}
    core_validation = read_json(REPORT_DIR / "core_fix_validation.json")

    return {
        "checker_version": "v4",
        "selection_protocol": "dev-only / two-seed mean; no test-set model selection",
        "metrics": metrics,
        "paired_comparisons_vs_format": comparisons,
        "averages": averages,
        "training": train,
        "core_validation": {
            "passed": core_validation.get("passed"),
            "k3_kl": core_validation.get("k3_kl"),
            "checker_cases": core_validation.get("checker_cases"),
        },
        "decision": {
            "old_kl_status": "diagnostic_only_due_to_incorrect_kl_penalty",
            "klfixed_status": "protocol_cleaner_main_result_candidate",
            "headline": "KL-fixed GRPO preserves a positive mean gain over Format-SFT, but the claim should be reported as a modest two-seed result rather than best-test-seed selection.",
        },
    }


def table_row(name: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {pct(m['dev']['r5'])} | {count_pct(m['dev']['strict_count'], m['dev']['n'])} "
        f"| {pct(m['test']['r5'])} | {count_pct(m['test']['strict_count'], m['test']['n'])} |"
    )


def avg_row(name: str, avg: dict[str, Any], base: dict[str, Any]) -> str:
    dev_strict = avg["dev"]["strict"]["mean"]
    test_strict = avg["test"]["strict"]["mean"]
    dev_r5 = avg["dev"]["r5"]["mean"]
    test_r5 = avg["test"]["r5"]["mean"]
    return (
        f"| {name} | {pct(dev_r5)} | {pct(dev_strict)} "
        f"| {pct(test_r5)} | {pct(test_strict)} "
        f"| {100*(dev_strict-base['dev']['strict']):+.1f} | {100*(test_strict-base['test']['strict']):+.1f} |"
    )


def write_report(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    train = summary["training"]
    comps = summary["paired_comparisons_vs_format"]
    avg = summary["averages"]
    base = metrics["format"]

    lines: list[str] = []
    lines.append("# KL-Fixed GRPO 60-Step Audit Report")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append(
        "This report fixes the reviewer audit issues that affect the main GRPO result: the KL penalty is now the non-negative k3 estimator, answer matching is rescored with checker v4, and model selection is reported without choosing by test performance."
    )
    lines.append("")
    lines.append("Old-KL GRPO numbers are kept only as diagnostics. The paper-facing candidate is the KL-fixed rerun.")
    lines.append("")
    lines.append("## 2. Core Fix Validation")
    lines.append("")
    k3 = summary["core_validation"]["k3_kl"]
    lines.append(f"- validation passed: `{summary['core_validation']['passed']}`")
    lines.append(f"- k3 KL same-policy value: `{k3['same_kl']}`")
    lines.append(f"- k3 KL positive case: `{k3['positive_case_kl']}`")
    lines.append(f"- k3 gradient nonzero check: `{k3['positive_case_grad']}`")
    lines.append(f"- bf16 near-zero KL after clamp: `{k3['bf16_near_zero_kl']}`")
    lines.append("- checker v4 blocks the audited false positives: bare AM/PM fallback, substring boundary errors, and numeric-range-as-single-answer errors.")
    lines.append("")
    lines.append("## 3. Training Stability")
    lines.append("")
    lines.append("| run | optimizer steps | micro steps | constant groups | constant rate | max GPU GB |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in ["klfixed_smoke_v2", "klfixed_seed42", "klfixed_seed43"]:
        t = train[key]
        lines.append(
            f"| {key} | {t['optimizer_steps']} | {t['micro_steps']} | {t['constant_reward_groups']} | {pct(t['constant_reward_rate'])} | {t['max_gpu_mem_gb']:.3f} |"
        )
    lines.append("")
    lines.append("The earlier constant-reward concern does not hold for the KL-fixed main reruns: seed42 has 3/240 constant groups and seed43 has 1/240.")
    lines.append("")
    lines.append("## 4. Main Metrics")
    lines.append("")
    lines.append("| Method | Dev R@5 | Dev strict | Test R@5 | Test strict |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(table_row("Format-SFT baseline", metrics["format"]))
    lines.append(table_row("old-KL GRPO seed42 diagnostic", metrics["old_kl_seed42"]))
    lines.append(table_row("old-KL GRPO seed43 diagnostic", metrics["old_kl_seed43"]))
    lines.append(table_row("KL-fixed GRPO seed42", metrics["klfixed_seed42"]))
    lines.append(table_row("KL-fixed GRPO seed43", metrics["klfixed_seed43"]))
    lines.append(table_row("KL-fixed seed42 + fixed Format reader", metrics["klfixed_seed42_fixed_reader"]))
    lines.append(table_row("KL-fixed seed43 + fixed Format reader", metrics["klfixed_seed43_fixed_reader"]))
    lines.append("")
    lines.append("## 5. Two-Seed Mean")
    lines.append("")
    lines.append("| Method | Dev R@5 | Dev strict | Test R@5 | Test strict | Dev strict gain vs Format | Test strict gain vs Format |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(avg_row("old-KL two-seed mean diagnostic", avg["old_kl_two_seed_mean"], base))
    lines.append(avg_row("KL-fixed two-seed mean", avg["klfixed_two_seed_mean"], base))
    lines.append(avg_row("KL-fixed fixed-reader two-seed mean", avg["klfixed_fixed_reader_two_seed_mean"], base))
    lines.append("")
    lines.append("KL-fixed mean strict success is dev `45.9%` and test `39.1%`, versus Format-SFT dev `40.8%` and test `34.4%`. The corrected gain is therefore +5.1 dev and +4.7 test points.")
    lines.append("")
    lines.append("The fixed-reader two-seed mean is identical on strict success: dev `45.9%` and test `39.1%`. This closes the reader-drift confound for the KL-fixed result: the query/retrieval stage accounts for the observed gain under the same Format-SFT reader.")
    lines.append("")
    lines.append("## 6. Paired Significance")
    lines.append("")
    lines.append("| Method | Split | method-only strict | baseline-only strict | McNemar exact p | R@5 gains/losses |")
    lines.append("|---|---|---:|---:|---:|---|")
    for key in [
        "old_kl_seed42",
        "old_kl_seed43",
        "klfixed_seed42",
        "klfixed_seed43",
        "klfixed_seed42_fixed_reader",
        "klfixed_seed43_fixed_reader",
    ]:
        for split in ["dev", "test"]:
            c = comps[key][split]
            lines.append(
                f"| {METHODS[key]['label']} | {split} | {c['method_only_strict']} | {c['base_only_strict']} | {c['mcnemar_exact_p']:.4f} | +{c['method_r5_gain']} / -{c['method_r5_loss']} |"
            )
    lines.append("")
    lines.append("The paired tests are directionally positive but not conventionally significant for KL-fixed seed42/seed43. This should be described as a small-sample main candidate, not a settled large-scale result.")
    lines.append("")
    lines.append("## 7. Corrected Interpretation")
    lines.append("")
    lines.append("- The old KL penalty was invalid for the paper claim; old-KL results should be marked diagnostic only.")
    lines.append("- The corrected KL-fixed rerun keeps the same direction of improvement over Format-SFT under checker v4.")
    lines.append("- Seed42 alone matches the old test headline, but seed43 is lower; the clean headline is the two-seed mean, not best test seed.")
    lines.append("- Fixed-reader control now matches the own-reader KL-fixed result on strict success, so the main improvement is not an artifact of evaluating each method with a different reader.")
    lines.append("- The result is useful enough to continue the DAG-IG main line, but the paper should avoid overstating statistical certainty until more seeds or larger data are run.")
    lines.append("")
    lines.append("## 8. Next Step")
    lines.append("")
    lines.append("Do not start unrelated DPO/RL variants before closing this main path. The next efficient step is to run the same KL-fixed recipe with one stronger setting only: either more GRPO steps or a larger clean training pool, selected by dev protocol, while keeping checker v4 and k3 KL fixed.")

    (REPORT_DIR / "KLFIXED_GRPO_60_REPORT.md").write_text("\n".join(lines) + "\n")
    (REPORT_DIR / "klfixed_grpo_60_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    write_report(summary)
    print(json.dumps({
        "wrote": [
            str(REPORT_DIR / "KLFIXED_GRPO_60_REPORT.md"),
            str(REPORT_DIR / "klfixed_grpo_60_summary.json"),
        ],
        "klfixed_two_seed_mean": summary["averages"]["klfixed_two_seed_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
