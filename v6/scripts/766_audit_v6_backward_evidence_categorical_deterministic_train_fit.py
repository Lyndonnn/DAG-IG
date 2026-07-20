#!/usr/bin/env python3
"""Policy-train fit audit for deterministic categorical evidence controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


METHODS = ("no_credit", "local_ig", "outcome", "dagig")


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


def parse_mappings(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        method, separator, path = value.partition("=")
        if not separator or method not in METHODS or method in result:
            raise ValueError(f"invalid method score mapping: {value}")
        result[method] = Path(path).resolve()
    if set(result) != set(METHODS):
        raise ValueError("all four method scores are required")
    return result


def summarize(policies: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, Any]:
    tvs: list[float] = []
    agreements: list[int] = []
    margins: list[float] = []
    kls: list[float] = []
    for policy, target in zip(policies, targets):
        tvs.append(float(0.5 * np.abs(policy - target).sum()))
        agreements.append(int(np.argmax(policy) == np.argmax(target)))
        ordered = np.sort(target)
        margins.append(float(ordered[-1] - ordered[-2]))
        kls.append(float(np.sum(target * (np.log(target.clip(1e-12)) - np.log(policy.clip(1e-12))))))
    high = [index for index, margin in enumerate(margins) if margin >= 0.05]
    return {
        "groups": len(policies),
        "mean_policy_target_tv": mean(tvs),
        "mean_target_policy_kl": mean(kls),
        "top_action_agreement": mean(agreements),
        "margin_ge_0p05_groups": len(high),
        "margin_ge_0p05_top_action_agreement": mean(agreements[index] for index in high) if high else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--reference_scores", type=Path, required=True)
    parser.add_argument("--method_scores", action="append", required=True, help="method=/path/to/score_audit.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.training_freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN":
        raise ValueError("backward evidence training is not frozen")
    if freeze.get("protocol_version") != "dagig_v6_backward_evidence_explicit_categorical_deterministic_v2":
        raise ValueError("train-fit audit requires deterministic categorical v2")
    reference_audit_path = args.reference_scores.resolve()
    reference_audit = read_json(reference_audit_path)
    if (
        reference_audit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORES_READY"
        or reference_audit.get("method") != "reference"
        or reference_audit.get("partition") != "policy_train"
    ):
        raise ValueError("reference train scores are invalid")
    score_paths = parse_mappings(args.method_scores)
    train_path = Path(freeze["input_paths"]["categorical_train_data"])
    if sha256(train_path) != freeze["input_hashes"]["categorical_train_data"]:
        raise ValueError("categorical train rows changed")
    rows = read_jsonl(train_path)
    row_by_id = {row["parent_group_id"]: row for row in rows}
    reference_path = Path(reference_audit["output_paths"]["scores"])
    if sha256(reference_path) != reference_audit["output_hashes"]["scores"]:
        raise ValueError("reference score rows changed")
    reference = {row["parent_group_id"]: row for row in read_jsonl(reference_path)}
    inputs = {"training_freeze": str(freeze_path), "reference_score_audit": str(reference_audit_path)}
    metrics: dict[str, Any] = {}
    beta = float(freeze["training"]["beta"])
    for method in METHODS:
        audit = read_json(score_paths[method])
        if (
            audit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORES_READY"
            or audit.get("method") != method
            or audit.get("partition") != "policy_train"
        ):
            raise ValueError(f"invalid method train scores: {method}")
        scores_path = Path(audit["output_paths"]["scores"])
        if sha256(scores_path) != audit["output_hashes"]["scores"]:
            raise ValueError(f"method score rows changed: {method}")
        current = {row["parent_group_id"]: row for row in read_jsonl(scores_path)}
        if set(current) != set(reference) or set(current) != set(row_by_id):
            raise ValueError(f"train score universe mismatch: {method}")
        policies: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        target_key = freeze["target_keys"][method]
        for group_id in sorted(row_by_id):
            row = row_by_id[group_id]
            behavior = np.asarray(row["behavior_probabilities"], dtype=np.float64)
            delta = np.asarray(current[group_id]["field_logprob_scores"]) - np.asarray(reference[group_id]["field_logprob_scores"])
            logits = np.log(behavior) + beta * delta
            policy = np.exp(logits - logits.max())
            policy /= policy.sum()
            policies.append(policy)
            targets.append(np.asarray(row[target_key], dtype=np.float64))
        metrics[method] = summarize(policies, targets)
        inputs[f"{method}_score_audit"] = str(score_paths[method])
    gates = {
        "complete_equal_groups": all(metrics[method]["groups"] == len(rows) for method in METHODS),
        "no_credit_mean_tv_at_most_0p03": metrics["no_credit"]["mean_policy_target_tv"] <= 0.03,
        "trained_method_mean_tv_at_most_0p10": all(metrics[method]["mean_policy_target_tv"] <= 0.10 for method in ("local_ig", "outcome", "dagig")),
        "trained_method_top_agreement_at_least_0p65": all(metrics[method]["top_action_agreement"] >= 0.65 for method in ("local_ig", "outcome", "dagig")),
        "trained_method_high_margin_agreement_at_least_0p85": all(
            metrics[method]["margin_ge_0p05_top_action_agreement"] is None
            or metrics[method]["margin_ge_0p05_top_action_agreement"] >= 0.85
            for method in ("local_ig", "outcome", "dagig")
        ),
        "policy_train_groups_only": True,
        "no_gold_or_qrels_loaded": True,
        "internal_holdout_unused": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_BACKWARD_EVIDENCE_TRAIN_FIT_GO" if all(gates.values()) else "DAGIG_V6_BACKWARD_EVIDENCE_TRAIN_FIT_NO_GO"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "decision": decision,
        "metrics": metrics,
        "action_representation": "one_of_five_single_token_labels_A_to_E",
        "deterministic_policy_logits": True,
        "gates": gates,
        "input_paths": inputs,
        "input_hashes": {key: sha256(Path(path)) for key, path in inputs.items()},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_TRAIN_FIT_AUDIT.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
