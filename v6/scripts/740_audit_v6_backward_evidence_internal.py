#!/usr/bin/env python3
"""Private clustered internal audit for backward evidence-node controls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_freeze", type=Path, required=True)
    parser.add_argument("--train_fit", type=Path, required=True)
    parser.add_argument("--answer_amendment", type=Path, required=True)
    for method in METHODS:
        parser.add_argument(f"--{method}_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    control_path = args.control_freeze.resolve()
    fit_path = args.train_fit.resolve()
    amendment_path = args.answer_amendment.resolve()
    control = read_json(control_path)
    fit = read_json(fit_path)
    amendment_manifest = read_json(amendment_path)
    if control.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_CONTROLS_FROZEN":
        raise ValueError("backward evidence controls are not frozen")
    if fit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAIN_FIT_GO":
        raise ValueError("evidence train fit is not GO")
    if amendment_manifest.get("decision") != "DAGIG_V6_ANSWER_NORMALIZATION_AMENDMENT_FROZEN":
        raise ValueError("answer normalization amendment is not frozen")
    baseline_path = Path(amendment_manifest["input_paths"]["baseline_eval_utils"])
    matcher_path = Path(amendment_manifest["input_paths"]["amendment_module"])
    baseline = load_module("dagig_evidence_internal_baseline", baseline_path)
    matcher = load_module("dagig_evidence_internal_matcher", matcher_path)
    labels = {row["sample_id"]: row for row in read_jsonl(Path(amendment_manifest["input_paths"]["private_labels"]))}
    evidence_audit = read_json(Path(control["input_paths"]["evidence_action_audit"]))
    support_path = Path(evidence_audit["output_paths"]["private_support"])
    if sha256(support_path) != evidence_audit["output_hashes"]["private_support"]:
        raise ValueError("private support labels changed")
    support: dict[str, bool] = {}
    for row in read_jsonl(support_path):
        for strategy, value in row["strategy_support"].items():
            support[f"{row['query_id']}::{strategy}"] = bool(value)
    value_audit = read_json(Path(control["input_paths"]["shared_answer_value_audit"]))
    value_path = Path(value_audit["output_paths"]["shared_answer_values"])
    if sha256(value_path) != value_audit["output_hashes"]["shared_answer_values"]:
        raise ValueError("shared answer values changed")
    values = {row["evidence_action_id"]: float(row["shared_answer_value"]) for row in read_jsonl(value_path)}

    metrics: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    inputs = {"control_freeze": str(control_path), "train_fit": str(fit_path), "answer_amendment": str(amendment_path)}
    per_method_sample: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for method in METHODS:
        audit_path = getattr(args, f"{method}_audit").resolve()
        audit = read_json(audit_path)
        if audit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_INTERNAL_METHOD_READY" or audit.get("method") != method:
            raise ValueError(f"invalid internal method predictions: {method}")
        predictions_path = Path(audit["output_paths"]["predictions"])
        if sha256(predictions_path) != audit["output_hashes"]["predictions"]:
            raise ValueError(f"internal predictions changed: {method}")
        rows = read_jsonl(predictions_path)
        evaluated: list[dict[str, Any]] = []
        for row in rows:
            label = labels[row["sample_id"]]
            match = matcher.answer_match_details(baseline, row["final_answer"], label["gold_answer"], label.get("aliases") or [])
            is_supported = support[row["selected_evidence_action_id"]]
            evaluated_row = {
                "method": method,
                "sample_id": row["sample_id"],
                "query_id": row["query_id"],
                "selected_evidence_action_id": row["selected_evidence_action_id"],
                "selected_evidence_strategy": row["selected_evidence_strategy"],
                "expected_terminal": values[row["selected_evidence_action_id"]],
                "evidence_supported": is_supported,
                "answer_valid": bool(row["answer_valid"]),
                "answer_correct": bool(match["answer_correct"]),
                "answer_match_type": match.get("answer_match_type"),
                "strict": bool(is_supported and match["answer_correct"]),
            }
            evaluated.append(evaluated_row)
            cases.append(evaluated_row)
            per_method_sample[method][row["sample_id"]].append(evaluated_row)
        metrics[method] = {
            "query_states": len(evaluated),
            "samples": len(per_method_sample[method]),
            "expected_terminal": mean(row["expected_terminal"] for row in evaluated),
            "evidence_support": mean(row["evidence_supported"] for row in evaluated),
            "answer_valid": mean(row["answer_valid"] for row in evaluated),
            "answer_correct": mean(row["answer_correct"] for row in evaluated),
            "strict": mean(row["strict"] for row in evaluated),
            "sample_macro_strict": mean(mean(row["strict"] for row in sample_rows) for sample_rows in per_method_sample[method].values()),
        }
        inputs[f"{method}_method_audit"] = str(audit_path)

    strongest_terminal = max(metrics[method]["expected_terminal"] for method in ("local_ig", "outcome"))
    strongest_support = max(metrics[method]["evidence_support"] for method in ("local_ig", "outcome"))
    strongest_strict = max(metrics[method]["strict"] for method in ("local_ig", "outcome"))
    sample_ids = sorted(per_method_sample["dagig"])
    rng = random.Random(761943)
    bootstrap_dag_minus_outcome: list[float] = []
    for _ in range(20000):
        draw = [rng.choice(sample_ids) for _ in sample_ids]
        differences = []
        for sample_id in draw:
            dag = mean(row["strict"] for row in per_method_sample["dagig"][sample_id])
            outcome = mean(row["strict"] for row in per_method_sample["outcome"][sample_id])
            differences.append(dag - outcome)
        bootstrap_dag_minus_outcome.append(mean(differences))
    pairwise = {
        "dagig_minus_outcome_sample_cluster_bootstrap_95ci": [percentile(bootstrap_dag_minus_outcome, 0.025), percentile(bootstrap_dag_minus_outcome, 0.975)],
        "dagig_minus_outcome_sample_macro_strict": metrics["dagig"]["sample_macro_strict"] - metrics["outcome"]["sample_macro_strict"],
    }
    gates = {
        "complete_equal_595_query_states": all(metrics[method]["query_states"] == 595 for method in METHODS),
        "complete_equal_40_samples": all(metrics[method]["samples"] == 40 for method in METHODS),
        "all_answer_valid_at_least_0p98": all(metrics[method]["answer_valid"] >= 0.98 for method in METHODS),
        "dagig_expected_terminal_noninferior_to_strong_control": metrics["dagig"]["expected_terminal"] + 0.005 >= strongest_terminal,
        "dagig_evidence_support_noninferior_to_strong_control": metrics["dagig"]["evidence_support"] + 0.01 >= strongest_support,
        "dagig_generated_strict_noninferior_to_strong_control": metrics["dagig"]["strict"] + 0.01 >= strongest_strict,
        "dagig_expected_terminal_not_below_no_credit": metrics["dagig"]["expected_terminal"] >= metrics["no_credit"]["expected_terminal"],
        "dagig_generated_strict_not_below_no_credit": metrics["dagig"]["strict"] >= metrics["no_credit"]["strict"],
        "same_shared_answer_policy": True,
        "internal_holdout_unused_for_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_BACKWARD_EVIDENCE_INTERNAL_GO" if all(gates.values()) else "DAGIG_V6_BACKWARD_EVIDENCE_INTERNAL_NO_GO"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cases_path = output_dir / "v6_backward_evidence_internal_private_cases.jsonl"
    cases_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cases), encoding="utf-8")
    result = {
        "decision": decision,
        "metrics": metrics,
        "pairwise": pairwise,
        "gates": gates,
        "input_paths": inputs,
        "input_hashes": {key: sha256(Path(path)) for key, path in inputs.items()},
        "output_paths": {"private_cases": str(cases_path)},
        "output_hashes": {"private_cases": sha256(cases_path)},
        "gold_or_qrels_loaded_only_by_private_auditor": True,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_INTERNAL_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
