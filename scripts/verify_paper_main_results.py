#!/usr/bin/env python3
"""Verify the corrected KL-fixed paper-main metrics shipped with this repo.

This script does not rerun model inference. It checks internal consistency of
the exported tables, corrected summary JSON, fixed-reader control, core-fix
validation, and training health records.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results" / "tables" / "main_results_table.csv"
SUMMARY = ROOT / "results" / "metrics" / "klfixed_grpo_60_summary.json"
CORE_FIX = ROOT / "results" / "metrics" / "core_fix_validation.json"
CORPUS_AUDIT = ROOT / "results" / "metrics" / "corpus_reality_audit.json"


EXPECTED_TABLE = {
    ("Format-SFT v4", "dev"): {"r5": 52.0, "strict_success": 40.8},
    ("Format-SFT v4", "test"): {"r5": 46.9, "strict_success": 34.4},
    ("KL-fixed GRPO seed42", "dev"): {"r5": 56.1, "strict_success": 45.9},
    ("KL-fixed GRPO seed42", "test"): {"r5": 51.6, "strict_success": 40.6},
    ("KL-fixed GRPO seed43", "dev"): {"r5": 56.1, "strict_success": 45.9},
    ("KL-fixed GRPO seed43", "test"): {"r5": 48.4, "strict_success": 37.5},
    ("KL-fixed GRPO two-seed mean", "dev"): {"r5": 56.1, "strict_success": 45.9},
    ("KL-fixed GRPO two-seed mean", "test"): {"r5": 50.0, "strict_success": 39.1},
}


def close(a: float, b: float, eps: float = 0.06) -> bool:
    return abs(a - b) <= eps


def pct(x: float) -> float:
    return round(100.0 * x, 1)


def load_table() -> dict[tuple[str, str], dict[str, str]]:
    if not TABLE.exists():
        raise FileNotFoundError(TABLE)
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with TABLE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["method"], row["split"])] = row
    return rows


def assert_table(rows: dict[tuple[str, str], dict[str, str]]) -> None:
    for key, expected in EXPECTED_TABLE.items():
        if key not in rows:
            raise AssertionError(f"Missing table row: {key}")
        row = rows[key]
        for field, expected_value in expected.items():
            value = float(row[field])
            if not close(value, expected_value):
                raise AssertionError(f"{key} {field}: got {value}, expected {expected_value}")


def assert_summary(rows: dict[tuple[str, str], dict[str, str]]) -> None:
    if not SUMMARY.exists():
        raise FileNotFoundError(SUMMARY)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    checks = {
        ("Format-SFT v4", "dev"): metrics["format"]["dev"],
        ("Format-SFT v4", "test"): metrics["format"]["test"],
        ("KL-fixed GRPO seed42", "dev"): metrics["klfixed_seed42"]["dev"],
        ("KL-fixed GRPO seed42", "test"): metrics["klfixed_seed42"]["test"],
        ("KL-fixed GRPO seed43", "dev"): metrics["klfixed_seed43"]["dev"],
        ("KL-fixed GRPO seed43", "test"): metrics["klfixed_seed43"]["test"],
    }
    for key, metric in checks.items():
        row = rows[key]
        if not close(float(row["r5"]), pct(metric["r5"])):
            raise AssertionError(f"{key} R@5 table/JSON mismatch")
        if not close(float(row["strict_success"]), pct(metric["strict"])):
            raise AssertionError(f"{key} strict table/JSON mismatch")

    avg = summary["averages"]["klfixed_two_seed_mean"]
    fixed_avg = summary["averages"]["klfixed_fixed_reader_two_seed_mean"]
    for split in ("dev", "test"):
        row = rows[("KL-fixed GRPO two-seed mean", split)]
        if not close(float(row["r5"]), pct(avg[split]["r5"]["mean"])):
            raise AssertionError(f"{split} mean R@5 mismatch")
        if not close(float(row["strict_success"]), pct(avg[split]["strict"]["mean"])):
            raise AssertionError(f"{split} mean strict mismatch")
        if not close(avg[split]["strict"]["mean"], fixed_avg[split]["strict"]["mean"], eps=1e-9):
            raise AssertionError(f"{split} fixed-reader strict mean differs from own-reader mean")
        if not close(avg[split]["r5"]["mean"], fixed_avg[split]["r5"]["mean"], eps=1e-9):
            raise AssertionError(f"{split} fixed-reader R@5 mean differs from own-reader mean")

    train = summary["training"]
    expected_constant = {"klfixed_seed42": 3, "klfixed_seed43": 1}
    for seed_name, constant_groups in expected_constant.items():
        t = train[seed_name]
        if t["status"] != "success":
            raise AssertionError(f"{seed_name} training status is not success")
        if t["optimizer_steps"] != 60 or t["micro_steps"] != 240:
            raise AssertionError(f"{seed_name} step count mismatch: {t}")
        if t["constant_reward_groups"] != constant_groups:
            raise AssertionError(f"{seed_name} constant reward groups mismatch: {t}")


def assert_core_fix() -> None:
    if not CORE_FIX.exists():
        raise FileNotFoundError(CORE_FIX)
    core = json.loads(CORE_FIX.read_text(encoding="utf-8"))
    if not core.get("passed"):
        raise AssertionError("Core fix validation did not pass")
    if not core.get("no_top_level_7b_imports"):
        raise AssertionError("Top-level 7B extension imports remain")
    if not core.get("no_hardcoded_local_model_paths"):
        raise AssertionError("Hard-coded local model/cache paths remain")
    k3 = core["k3_kl"]
    if k3["same_kl"] != 0.0:
        raise AssertionError(f"same-policy k3 KL should be zero: {k3}")
    if k3["positive_case_kl"] <= 0.0 or k3["positive_case_grad"] == 0.0:
        raise AssertionError(f"k3 KL positive/gradient check failed: {k3}")
    if k3["bf16_near_zero_kl"] < 0.0:
        raise AssertionError(f"bf16 near-zero KL should be nonnegative: {k3}")


def assert_corpus_audit() -> None:
    if not CORPUS_AUDIT.exists():
        raise FileNotFoundError(CORPUS_AUDIT)
    audit = json.loads(CORPUS_AUDIT.read_text(encoding="utf-8"))
    if audit.get("claim_boundary") != "offline frozen evidence-note BM25 corpus, not live web search":
        raise AssertionError("Corpus claim boundary is missing or too broad")
    eval_corpus = audit["corpora"]["eval_devtest"]
    if eval_corpus["docs"] != 201:
        raise AssertionError(f"Unexpected eval corpus size: {eval_corpus['docs']}")
    if eval_corpus["lengths"]["token_median"] != 6:
        raise AssertionError(f"Unexpected eval corpus median token length: {eval_corpus['lengths']}")
    if eval_corpus["gold_doc_answer_embedded_rate"] < 0.80:
        raise AssertionError("Expected audit to record high answer-string embedding in gold notes")
    per_split = eval_corpus.get("per_split") or {}
    expected_coverage = {
        "dev": {"expected_samples": 98, "samples_with_gold_doc": 92, "coverage": 92 / 98},
        "test": {"expected_samples": 64, "samples_with_gold_doc": 58, "coverage": 58 / 64},
    }
    for split, expected in expected_coverage.items():
        if split not in per_split:
            raise AssertionError(f"Missing per-split corpus coverage for {split}")
        row = per_split[split]
        if row["expected_samples"] != expected["expected_samples"]:
            raise AssertionError(f"{split} expected sample count mismatch: {row}")
        if row["samples_with_gold_doc"] != expected["samples_with_gold_doc"]:
            raise AssertionError(f"{split} gold-doc sample count mismatch: {row}")
        if not close(row["sample_gold_doc_coverage"], expected["coverage"], eps=1e-9):
            raise AssertionError(f"{split} gold-doc coverage mismatch: {row}")
        if not close(row["strict_upper_bound_from_gold_doc_coverage"], expected["coverage"], eps=1e-9):
            raise AssertionError(f"{split} strict upper bound mismatch: {row}")


def main() -> None:
    rows = load_table()
    assert_table(rows)
    assert_summary(rows)
    assert_core_fix()
    assert_corpus_audit()
    print("Corrected KL-fixed paper-main verification passed.")
    print("Two-seed KL-fixed strict gain over Format-SFT: dev +5.1, test +4.7.")
    print("Core fixes passed: k3 KL, checker v4, training health, fixed reader, and corpus boundary.")


if __name__ == "__main__":
    main()
