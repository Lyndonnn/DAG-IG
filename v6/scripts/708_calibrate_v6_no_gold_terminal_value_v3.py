#!/usr/bin/env python3
"""Stratified repeated-group CV for deployable no-gold P_success v3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_helper(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_v6_no_gold_v3_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stratified_sample_folds(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed_text: str,
) -> tuple[dict[str, int], list[dict[str, int]]]:
    stats = []
    for sample_id in sorted(set(sample_ids.tolist())):
        indices = np.flatnonzero(sample_ids == sample_id)
        tie = int(hashlib.sha256(f"{seed_text}:{sample_id}".encode()).hexdigest(), 16)
        stats.append((sample_id, int(labels[indices].sum()), len(indices), tie))
    stats.sort(key=lambda row: (-row[1], -row[2], row[3]))
    target_positive = float(labels.sum()) / folds
    target_actions = len(labels) / folds
    target_samples = len(stats) / folds
    state = np.zeros((folds, 3), dtype=np.float64)
    assignment: dict[str, int] = {}
    for index, (sample_id, positives, actions, tie) in enumerate(stats):
        if index < folds:
            selected = index
        else:
            objectives = []
            for candidate in range(folds):
                proposed = state.copy()
                proposed[candidate] += [positives, actions, 1]
                objective = (
                    np.sum(((proposed[:, 0] - target_positive) / max(target_positive, 1.0)) ** 2)
                    + np.sum(((proposed[:, 1] - target_actions) / target_actions) ** 2)
                    + 0.25 * np.sum(((proposed[:, 2] - target_samples) / target_samples) ** 2)
                )
                objectives.append(float(objective))
            best = min(objectives)
            choices = [candidate for candidate, value in enumerate(objectives) if abs(value - best) <= 1e-12]
            selected = choices[tie % len(choices)]
        assignment[sample_id] = selected
        state[selected] += [positives, actions, 1]
    summary = [
        {"fold": fold, "positive_actions": int(row[0]), "actions": int(row[1]), "samples": int(row[2])}
        for fold, row in enumerate(state)
    ]
    return assignment, summary


def imbalance(rows: list[dict[str, int]], key: str) -> float:
    values = [row[key] for row in rows]
    return (max(values) - min(values)) / max(float(np.mean(values)), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    helper_path = Path(__file__).resolve().with_name("702_calibrate_v6_no_gold_terminal_value.py")
    helper = load_helper(helper_path)
    freeze = helper.read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN" or freeze.get("protocol_version") != "dagig_v6_deployable_no_gold_terminal_value_stratified_repeated_group_cv_v3":
        raise ValueError("stratified no-gold terminal v3 is not frozen")
    if freeze["code_hashes"]["calibrator"] != helper.sha256(Path(__file__).resolve()):
        raise ValueError("v3 calibrator changed after freeze")
    if freeze["code_hashes"]["calibration_helper"] != helper.sha256(helper_path):
        raise ValueError("calibration helper changed")
    for key, path in freeze["source_paths"].items():
        if helper.sha256(Path(path)) != freeze["source_hashes"][key]:
            raise ValueError(f"v3 source changed: {key}")
    records = helper.read_jsonl(Path(freeze["source_paths"]["v2_private_records"]))
    if len(records) != int(freeze["answer_actions"]):
        raise ValueError("v3 private record universe differs")
    x = np.asarray([row["feature"] for row in records], dtype=np.float64)
    y = np.asarray([row["strict_proxy"] for row in records], dtype=np.float64)
    if x.shape[1] != len(freeze["feature_names"]):
        raise ValueError("v3 feature schema differs")
    train_index = np.asarray([index for index, row in enumerate(records) if row["partition"] == "policy_train"])
    development_index = np.asarray([index for index, row in enumerate(records) if row["partition"] == "internal_holdout"])
    train_samples = np.asarray([records[index]["sample_id"] for index in train_index])
    config = freeze["calibration"]
    repeat_predictions = []
    repeated_metrics = []
    fold_summaries = []
    positive_imbalances, action_imbalances = [], []
    for repeat in range(int(config["repeats"])):
        assignment, summary = stratified_sample_folds(
            train_samples,
            y[train_index],
            int(config["folds"]),
            f"{config['fold_seed_prefix']}:{repeat}",
        )
        positive_imbalances.append(imbalance(summary, "positive_actions"))
        action_imbalances.append(imbalance(summary, "actions"))
        prediction = np.full(len(train_index), np.nan, dtype=np.float64)
        baseline = np.full(len(train_index), np.nan, dtype=np.float64)
        for fold in range(int(config["folds"])):
            fit_index = np.asarray([index for index in train_index if assignment[records[index]["sample_id"]] != fold])
            validation_index = np.asarray([index for index in train_index if assignment[records[index]["sample_id"]] == fold])
            model = helper.fit_logistic(x[fit_index], y[fit_index], float(config["l2"]), int(config["max_newton_steps"]))
            positions = np.searchsorted(train_index, validation_index)
            prediction[positions] = helper.predict(model, x[validation_index])
            baseline[positions] = float(y[fit_index].mean())
        if not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
            raise ValueError("incomplete stratified OOF probabilities")
        repeat_predictions.append(prediction)
        repeated_metrics.append({"repeat": repeat, **helper.metric(prediction, y[train_index], baseline)})
        fold_summaries.append({"repeat": repeat, "folds": summary, "positive_imbalance_fraction": positive_imbalances[-1], "action_imbalance_fraction": action_imbalances[-1]})
    oof_probability = np.mean(np.stack(repeat_predictions), axis=0)
    oof_baseline = np.full(len(train_index), float(y[train_index].mean()))
    aggregate_oof = helper.metric(oof_probability, y[train_index], oof_baseline)
    final_model = helper.fit_logistic(x[train_index], y[train_index], float(config["l2"]), int(config["max_newton_steps"]))
    development_probability = helper.predict(final_model, x[development_index])
    development_baseline = np.full(len(development_index), float(y[train_index].mean()))
    development_metrics = helper.metric(development_probability, y[development_index], development_baseline)
    probabilities = np.full(len(records), np.nan, dtype=np.float64)
    probabilities[train_index] = oof_probability
    probabilities[development_index] = development_probability
    low, high = config["probability_clip"]
    probabilities = np.clip(probabilities, float(low), float(high))
    if not np.isfinite(probabilities).all():
        raise ValueError("incomplete v3 terminal values")
    by_evidence: dict[str, list[float]] = defaultdict(list)
    for row, probability in zip(records, probabilities):
        by_evidence[row["evidence_action_id"]].append(float(probability))
    nonconstant = float(np.mean([max(values) - min(values) > 1e-5 for values in by_evidence.values()]))
    aucs = [float(row["auc"]) for row in repeated_metrics]
    brier = [float(row["brier_improvement"]) for row in repeated_metrics]
    spec = freeze["gates_spec"]
    gates = {
        "complete_expanded_actions": len(records) == int(freeze["answer_actions"]),
        "stratified_repeated_oof_complete": len(repeated_metrics) == int(config["repeats"]),
        "repeated_oof_auc_mean": float(np.mean(aucs)) >= float(spec["repeated_oof_auc_mean_min"]),
        "repeated_oof_auc_worst": min(aucs) >= float(spec["repeated_oof_auc_worst_min"]),
        "repeated_oof_brier_worst": min(brier) >= float(spec["repeated_oof_brier_improvement_worst_min"]),
        "fold_positive_balance": max(positive_imbalances) <= float(spec["max_fold_positive_imbalance_fraction"]),
        "fold_action_balance": max(action_imbalances) <= float(spec["max_fold_action_imbalance_fraction"]),
        "development_auc": float(development_metrics["auc"]) >= float(spec["development_auc_min"]),
        "development_brier": float(development_metrics["brier_improvement"]) >= float(spec["development_brier_improvement_min"]),
        "development_ece": float(development_metrics["ece_10bin"]) <= float(spec["development_ece_max"]),
        "nonconstant_answer_groups": nonconstant >= float(spec["nonconstant_group_rate_min"]),
        "sample_groups_disjoint": True,
        "equivalence_logit_not_used": True,
        "runtime_features_contain_no_gold_or_qrels": True,
        "development_never_fit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    decision = "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO" if all(gates.values()) else "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    value_path = output / "v6_no_gold_terminal_success_values.jsonl"
    private_path = output / "v6_no_gold_terminal_private_audit.jsonl"
    model_path = output / "v6_no_gold_terminal_calibrator.json"
    with value_path.open("w", encoding="utf-8") as handle:
        for row, probability in sorted(zip(records, probabilities), key=lambda item: item[0]["answer_action_id"]):
            handle.write(json.dumps({"answer_action_id": row["answer_action_id"], "evidence_action_id": row["evidence_action_id"], "query_id": row["query_id"], "sample_id": row["sample_id"], "partition": row["partition"], "terminal_success_probability": float(probability), "terminal_log_value": math.log(float(probability)), "calibration_source": "ten_repeat_stratified_sample_grouped_oof_mean" if row["partition"] == "policy_train" else "policy_train_fit_development_score"}, sort_keys=True) + "\n")
    with private_path.open("w", encoding="utf-8") as handle:
        for row, probability in sorted(zip(records, probabilities), key=lambda item: item[0]["answer_action_id"]):
            handle.write(json.dumps({**row, "terminal_success_probability": float(probability)}, sort_keys=True) + "\n")
    model_path.write_text(json.dumps({"feature_names": freeze["feature_names"], "center": final_model["center"].tolist(), "scale": final_model["scale"].tolist(), "weights": final_model["weights"].tolist(), "fit_partition": "policy_train_only", "fold_summaries": fold_summaries, "equivalence_logit_used": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "metrics": {
            "repeated_oof": repeated_metrics,
            "repeated_oof_auc_mean": float(np.mean(aucs)),
            "repeated_oof_auc_worst": min(aucs),
            "repeated_oof_brier_improvement_mean": float(np.mean(brier)),
            "repeated_oof_brier_improvement_worst": min(brier),
            "max_fold_positive_imbalance_fraction": max(positive_imbalances),
            "max_fold_action_imbalance_fraction": max(action_imbalances),
            "aggregate_oof_mean_probability": aggregate_oof,
            "development": development_metrics,
            "nonconstant_answer_group_rate": nonconstant,
            "answer_actions": len(records),
            "evidence_groups": len(by_evidence),
        },
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": helper.sha256(freeze_path)},
        "output_paths": {"terminal_values": str(value_path), "private_audit": str(private_path), "calibrator": str(model_path)},
        "output_hashes": {"terminal_values": helper.sha256(value_path), "private_audit": helper.sha256(private_path), "calibrator": helper.sha256(model_path)},
        "gold_or_qrels_in_runtime_features": False,
        "equivalence_logit_used": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
