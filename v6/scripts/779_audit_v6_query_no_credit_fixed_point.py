#!/usr/bin/env python3
"""Require deterministic query No-credit to remain an exact fixed point."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--smoke_audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    smoke_path = args.smoke_audit.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze = read_json(freeze_path)
    smoke = read_json(smoke_path)
    if freeze.get("protocol_version") != "dagig_v6_backward_fixed_descendants_equal_query_training_deterministic_v2":
        raise ValueError("fixed-point audit requires deterministic query v2")
    if sha256(Path(__file__).resolve()) != freeze["runner_hashes"]["fixed_point_auditor"]:
        raise ValueError("query fixed-point auditor differs from frozen runner")
    if smoke.get("decision") != "DAGIG_V6_BACKWARD_QUERY_POLICY_SMOKE_READY" or smoke.get("method") != "no_credit":
        raise ValueError("invalid No-credit smoke audit")
    if smoke["input_hashes"].get("freeze") != sha256(freeze_path):
        raise ValueError("smoke belongs to another freeze")
    log_path = Path(smoke["output_paths"]["training_log"])
    if sha256(log_path) != smoke["output_hashes"]["training_log"]:
        raise ValueError("smoke training log changed")
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tv = mean(float(row["policy_target_tv"]) for row in rows)
    shift = mean(float(row["mean_abs_policy_shift"]) for row in rows)
    grad = max(float(row["grad_norm"]) for row in rows)
    expected = freeze["required_preflight_gate"]
    gates = {
        "exact_32_groups": smoke.get("trained_groups") == int(expected["groups"]),
        "deterministic_policy_logits": bool(smoke.get("deterministic_policy_logits")),
        "dropout_disabled": smoke.get("dropout_active_during_optimization") is False,
        "mean_tv_at_most_1e_5": tv <= float(expected["max_mean_tv"]),
        "mean_shift_at_most_1e_5": shift <= float(expected["max_mean_abs_policy_shift"]),
        "max_grad_at_most_1e_4": grad <= float(expected["max_grad_norm"]),
        "internal_holdout_unused": not bool(smoke.get("internal_holdout_used")),
        "dev_sealed": not bool(smoke.get("dev_used")),
        "test_sealed": not bool(smoke.get("test_used")),
    }
    decision = "DAGIG_V6_QUERY_NO_CREDIT_FIXED_POINT_GO" if all(gates.values()) else "DAGIG_V6_QUERY_NO_CREDIT_FIXED_POINT_NO_GO"
    result = {
        "decision": decision,
        "gates": gates,
        "metrics": {"groups": smoke["trained_groups"], "optimizer_steps": len(rows), "mean_tv": tv, "mean_abs_policy_shift": shift, "max_grad_norm": grad},
        "input_paths": {"freeze": str(freeze_path), "smoke_audit": str(smoke_path), "training_log": str(log_path)},
        "input_hashes": {"freeze": sha256(freeze_path), "smoke_audit": sha256(smoke_path), "training_log": sha256(log_path)},
        "internal_holdout_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
