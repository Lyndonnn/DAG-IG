#!/usr/bin/env python3
"""Verify the paper-main result table shipped with this core repo.

This script does not rerun model inference. It checks the internal consistency of
the exported paper-facing metrics and confirms the main claim numbers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results" / "tables" / "main_results_table.csv"
CONSOLIDATED = ROOT / "results" / "metrics" / "paper_main_v1_consolidated_results.json"
SEED_CONFIRM = ROOT / "results" / "metrics" / "seed_confirmation_summary.json"


EXPECTED_TABLE = {
    ("Format-SFT", "dev"): {"r5": 52.0, "strict_success": 42.9},
    ("Format-SFT", "test"): {"r5": 46.9, "strict_success": 34.4},
    ("DAG-IG seed42 main", "dev"): {"r5": 57.1, "strict_success": 49.0},
    ("DAG-IG seed42 main", "test"): {"r5": 51.6, "strict_success": 40.6},
    ("DAG-IG seed43 confirm", "dev"): {"r5": 58.2, "strict_success": 49.0},
    ("DAG-IG seed43 confirm", "test"): {"r5": 50.0, "strict_success": 39.1},
}


def close(a: float, b: float, eps: float = 0.05) -> bool:
    return abs(a - b) <= eps


def pct(x: float) -> float:
    return round(100.0 * x, 1)


def main() -> None:
    if not TABLE.exists():
        raise FileNotFoundError(TABLE)
    if not CONSOLIDATED.exists():
        raise FileNotFoundError(CONSOLIDATED)
    if not SEED_CONFIRM.exists():
        raise FileNotFoundError(SEED_CONFIRM)

    rows = {}
    with TABLE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["method"], row["split"])] = row

    for key, expected in EXPECTED_TABLE.items():
        if key not in rows:
            raise AssertionError(f"Missing table row: {key}")
        row = rows[key]
        for field, expected_value in expected.items():
            value = float(row[field])
            if not close(value, expected_value):
                raise AssertionError(f"{key} {field}: got {value}, expected {expected_value}")

    consolidated = json.loads(CONSOLIDATED.read_text(encoding="utf-8"))
    metrics = consolidated["metrics"]
    json_checks = {
        ("Format-SFT", "dev"): metrics["format_dev"],
        ("Format-SFT", "test"): metrics["format_test"],
        ("DAG-IG seed42 main", "dev"): metrics["seed42_dev"],
        ("DAG-IG seed42 main", "test"): metrics["seed42_test"],
        ("DAG-IG seed43 confirm", "dev"): metrics["seed43_dev"],
        ("DAG-IG seed43 confirm", "test"): metrics["seed43_test"],
    }
    for key, metric in json_checks.items():
        row = rows[key]
        if not close(float(row["r5"]), pct(metric["r5"])):
            raise AssertionError(f"{key} R@5 table/JSON mismatch")
        if not close(float(row["strict_success"]), pct(metric["strict"])):
            raise AssertionError(f"{key} strict table/JSON mismatch")

    seed = json.loads(SEED_CONFIRM.read_text(encoding="utf-8"))
    for seed_name in ("seed42", "seed43"):
        train = seed["train"][seed_name]
        if train["status"] != "success":
            raise AssertionError(f"{seed_name} training status is not success")
        if train["optimizer_steps"] != 60 or train["micro_steps"] != 240:
            raise AssertionError(f"{seed_name} step count mismatch: {train}")
        if train["constant_reward_groups"] != 2:
            raise AssertionError(f"{seed_name} constant reward groups mismatch: {train}")

    seed42_dev_gain = float(rows[("DAG-IG seed42 main", "dev")]["strict_success"]) - float(
        rows[("Format-SFT", "dev")]["strict_success"]
    )
    seed42_test_gain = float(rows[("DAG-IG seed42 main", "test")]["strict_success"]) - float(
        rows[("Format-SFT", "test")]["strict_success"]
    )

    print("Paper-main result verification passed.")
    print(f"Seed42 strict gain over Format-SFT: dev +{seed42_dev_gain:.1f}, test +{seed42_test_gain:.1f}.")
    print("Seed42 and seed43 training health checks passed.")


if __name__ == "__main__":
    main()
