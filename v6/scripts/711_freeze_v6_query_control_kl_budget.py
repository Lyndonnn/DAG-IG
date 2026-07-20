#!/usr/bin/env python3
"""Calibrate matched query-control update strengths on policy-train groups only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(candidate: dict[str, Any], method: str) -> float:
    if method == "dagig_exact":
        return float(candidate["dagig_nats"])
    if method == "local_fixed_descendant":
        return math.log(float(candidate["local_fixed_descendant_value"]))
    if method == "true_outcome_grpo":
        return float(candidate["outcome_mean_advantage"])
    raise ValueError(method)


def policy(candidates: list[dict[str, Any]], method: str, beta: float) -> list[float]:
    logits = [
        math.log(float(candidate["behavior_probability"])) + beta * score(candidate, method)
        for candidate in candidates
    ]
    offset = max(logits)
    masses = [math.exp(value - offset) for value in logits]
    total = sum(masses)
    return [value / total for value in masses]


def mean_kl(groups: list[dict[str, Any]], method: str, beta: float) -> float:
    values = []
    for group in groups:
        candidates = group["candidates"]
        posterior = policy(candidates, method, beta)
        behavior = [float(candidate["behavior_probability"]) for candidate in candidates]
        values.append(
            sum(value * math.log(value / base) for value, base in zip(posterior, behavior) if value > 0.0)
        )
    return mean(values)


def solve_beta(groups: list[dict[str, Any]], method: str, target: float) -> float:
    low, high = 0.0, 1.0
    while mean_kl(groups, method, high) < target:
        high *= 2.0
        if high > 1024.0:
            raise ValueError(f"cannot reach shared KL target for {method}")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if mean_kl(groups, method, middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    control_path = args.control_audit.resolve()
    control = read_json(control_path)
    if control.get("decision") != "DAGIG_V6_NO_GOLD_QUERY_CONTROLS_FROZEN":
        raise ValueError("matched no-gold query controls are not frozen")
    targets_path = Path(control["output_paths"]["query_targets"])
    if sha256(targets_path) != control["output_hashes"]["query_targets"]:
        raise ValueError("frozen query targets changed")
    groups = read_jsonl(targets_path)
    if len(groups) != 158 or any(row["partition"] != "policy_train" for row in groups):
        raise ValueError("KL calibration must use exactly the 158 policy-train groups")

    target_kl = mean_kl(groups, "dagig_exact", 1.0)
    beta = {
        "dagig_exact": 1.0,
        "local_fixed_descendant": solve_beta(groups, "local_fixed_descendant", target_kl),
        "true_outcome_grpo": solve_beta(groups, "true_outcome_grpo", target_kl),
        "no_credit": 0.0,
    }
    calibrated_kl = {
        method: (0.0 if method == "no_credit" else mean_kl(groups, method, value))
        for method, value in beta.items()
    }
    uncalibrated_kl = {
        method: mean_kl(groups, method, 1.0)
        for method in ("dagig_exact", "local_fixed_descendant", "true_outcome_grpo")
    }
    max_error = max(
        abs(calibrated_kl[method] - target_kl)
        for method in ("dagig_exact", "local_fixed_descendant", "true_outcome_grpo")
    )
    gates = {
        "policy_train_groups_complete": len(groups) == 158,
        "dagig_beta_fixed_at_one": beta["dagig_exact"] == 1.0,
        "matched_mean_kl": max_error <= 1e-10,
        "local_update_reduced": beta["local_fixed_descendant"] < 1.0,
        "outcome_update_reduced": beta["true_outcome_grpo"] < 1.0,
        "internal_development_unopened": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = (
        "DAGIG_V6_QUERY_CONTROL_KL_BUDGET_FROZEN"
        if all(gates.values())
        else "DAGIG_V6_QUERY_CONTROL_KL_BUDGET_NO_GO"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = {
        "decision": decision,
        "protocol_version": "dagig_v6_train_only_matched_query_control_kl_budget_v1",
        "rationale": "Credit quality is compared under the same mean KL update budget; otherwise Local and Outcome receive substantially larger policy shifts than exact DAG-IG.",
        "target_mean_kl": target_kl,
        "policy_beta": beta,
        "calibrated_mean_kl": calibrated_kl,
        "uncalibrated_mean_kl": uncalibrated_kl,
        "max_calibration_error": max_error,
        "gates": gates,
        "input_paths": {
            "control_audit": str(control_path),
            "query_targets": str(targets_path),
        },
        "input_hashes": {
            "control_audit": sha256(control_path),
            "query_targets": sha256(targets_path),
        },
        "internal_development_used": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    path = output / "DAGIG_V6_QUERY_CONTROL_KL_BUDGET_AUDIT.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "target_mean_kl": target_kl, "policy_beta": beta, "gates": gates, "audit": str(path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
