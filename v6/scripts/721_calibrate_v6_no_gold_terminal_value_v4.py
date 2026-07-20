#!/usr/bin/env python3
"""Calibrate and audit the compact no-gold terminal value v4."""

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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    x, y = rankdata(left), rankdata(right)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    helper_path = Path(__file__).with_name("702_calibrate_v6_no_gold_terminal_value.py")
    fold_path = Path(__file__).with_name("708_calibrate_v6_no_gold_terminal_value_v3.py")
    helper = load_module("dagig_v6_compact_terminal_helper", helper_path)
    fold_helper = load_module("dagig_v6_compact_fold_helper", fold_path)
    freeze = helper.read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN":
        raise ValueError("compact no-gold terminal protocol is not frozen")
    if freeze.get("protocol_version") != "dagig_v6_compact_monotone_no_gold_terminal_value_v4":
        raise ValueError("unexpected compact terminal protocol")
    if freeze["code_hashes"]["calibrator"] != helper.sha256(Path(__file__).resolve()):
        raise ValueError("v4 calibrator changed after freeze")
    if freeze["code_hashes"]["calibration_helper"] != helper.sha256(helper_path):
        raise ValueError("calibration helper changed")
    if freeze["code_hashes"]["fold_helper"] != helper.sha256(fold_path):
        raise ValueError("fold helper changed")
    for key, path in freeze["source_paths"].items():
        if helper.sha256(Path(path)) != freeze["source_hashes"][key]:
            raise ValueError(f"compact terminal source changed: {key}")

    records = helper.read_jsonl(Path(freeze["source_paths"]["v3_private_records"]))
    if len(records) != int(freeze["answer_actions"]):
        raise ValueError("compact terminal record universe differs")
    x = np.asarray([[float(row["feature"][0]), float(row["feature"][1])] for row in records], dtype=np.float64)
    y = np.asarray([row["strict_proxy"] for row in records], dtype=np.float64)
    train_index = np.asarray([index for index, row in enumerate(records) if row["partition"] == "policy_train"])
    development_index = np.asarray([index for index, row in enumerate(records) if row["partition"] == "internal_holdout"])
    train_samples = np.asarray([records[index]["sample_id"] for index in train_index])
    config = freeze["calibration"]
    repeat_predictions, repeated_metrics = [], []
    for repeat in range(int(config["repeats"])):
        assignment, _ = fold_helper.stratified_sample_folds(
            train_samples,
            y[train_index],
            int(config["folds"]),
            f"{config['fold_seed_prefix']}:{repeat}",
        )
        prediction = np.full(len(train_index), np.nan, dtype=np.float64)
        baseline = np.full(len(train_index), np.nan, dtype=np.float64)
        for fold in range(int(config["folds"])):
            fit_index = np.asarray([index for index in train_index if assignment[records[index]["sample_id"]] != fold])
            validation_index = np.asarray([index for index in train_index if assignment[records[index]["sample_id"]] == fold])
            model = helper.fit_logistic(x[fit_index], y[fit_index], float(config["l2"]), int(config["max_newton_steps"]))
            positions = np.searchsorted(train_index, validation_index)
            prediction[positions] = helper.predict(model, x[validation_index])
            baseline[positions] = float(y[fit_index].mean())
        if not np.isfinite(prediction).all():
            raise ValueError("incomplete compact OOF probabilities")
        repeat_predictions.append(prediction)
        repeated_metrics.append({"repeat": repeat, **helper.metric(prediction, y[train_index], baseline)})

    oof_probability = np.mean(np.stack(repeat_predictions), axis=0)
    aggregate_oof = helper.metric(oof_probability, y[train_index], np.full(len(train_index), float(y[train_index].mean())))
    final_model = helper.fit_logistic(x[train_index], y[train_index], float(config["l2"]), int(config["max_newton_steps"]))
    development_probability = helper.predict(final_model, x[development_index])
    development = helper.metric(
        development_probability,
        y[development_index],
        np.full(len(development_index), float(y[train_index].mean())),
    )
    probabilities = np.full(len(records), np.nan, dtype=np.float64)
    probabilities[train_index] = oof_probability
    probabilities[development_index] = development_probability
    low, high = config["probability_clip"]
    probabilities = np.clip(probabilities, float(low), float(high))

    answer_actions = {
        row["answer_action_id"]: row
        for row in helper.read_jsonl(Path(freeze["input_paths"]["answer_actions"]))
    }
    evidence_values: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row, probability in zip(records, probabilities):
        if row["partition"] != "internal_holdout":
            continue
        action = answer_actions[row["answer_action_id"]]
        evidence_values[row["evidence_action_id"]].append(
            (float(action["behavior_weight"]), float(probability), float(row["strict_proxy"]))
        )
    backed_values, backed_strict = [], []
    for rows in evidence_values.values():
        weights = np.asarray([row[0] for row in rows], dtype=np.float64)
        weights /= weights.sum()
        backed_values.append(float(weights @ np.asarray([row[1] for row in rows])))
        backed_strict.append(float(weights @ np.asarray([row[2] for row in rows])))
    evidence_spearman = spearman(backed_values, backed_strict)
    by_evidence: dict[str, list[float]] = defaultdict(list)
    for row, probability in zip(records, probabilities):
        by_evidence[row["evidence_action_id"]].append(float(probability))
    nonconstant = float(np.mean([max(values) - min(values) > 1e-5 for values in by_evidence.values()]))
    aucs = [float(row["auc"]) for row in repeated_metrics]
    brier = [float(row["brier_improvement"]) for row in repeated_metrics]
    spec = freeze["gates_spec"]
    feature_weights = final_model["weights"][1:]
    gates = {
        "complete_expanded_actions": len(records) == int(freeze["answer_actions"]),
        "stratified_repeated_oof_complete": len(repeated_metrics) == int(config["repeats"]),
        "repeated_oof_auc_mean": float(np.mean(aucs)) >= float(spec["repeated_oof_auc_mean_min"]),
        "repeated_oof_auc_worst": min(aucs) >= float(spec["repeated_oof_auc_worst_min"]),
        "repeated_oof_brier_worst": min(brier) >= float(spec["repeated_oof_brier_improvement_worst_min"]),
        "development_auc": float(development["auc"]) >= float(spec["development_auc_min"]),
        "development_brier": float(development["brier_improvement"]) >= float(spec["development_brier_improvement_min"]),
        "development_ece": float(development["ece_10bin"]) <= float(spec["development_ece_max"]),
        "development_evidence_value_strict_spearman": evidence_spearman >= float(spec["development_evidence_value_strict_spearman_min"]),
        "monotone_feature_coefficients": bool(np.all(feature_weights > 0.0)),
        "nonconstant_answer_groups": nonconstant >= float(spec["nonconstant_group_rate_min"]),
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
            handle.write(json.dumps({
                "answer_action_id": row["answer_action_id"],
                "evidence_action_id": row["evidence_action_id"],
                "query_id": row["query_id"],
                "sample_id": row["sample_id"],
                "partition": row["partition"],
                "terminal_success_probability": float(probability),
                "terminal_log_value": math.log(float(probability)),
                "calibration_source": "compact_v4_repeated_group_oof" if row["partition"] == "policy_train" else "compact_v4_policy_train_fit_development_score",
            }, sort_keys=True) + "\n")
    with private_path.open("w", encoding="utf-8") as handle:
        for row, probability in sorted(zip(records, probabilities), key=lambda item: item[0]["answer_action_id"]):
            compact = [float(row["feature"][0]), float(row["feature"][1])]
            handle.write(json.dumps({**row, "feature": compact, "terminal_success_probability": float(probability)}, sort_keys=True) + "\n")
    model_path.write_text(json.dumps({
        "feature_names": freeze["feature_names"],
        "center": final_model["center"].tolist(),
        "scale": final_model["scale"].tolist(),
        "weights": final_model["weights"].tolist(),
        "fit_partition": "policy_train_only",
        "equivalence_logit_used": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "metrics": {
            "repeated_oof": repeated_metrics,
            "repeated_oof_auc_mean": float(np.mean(aucs)),
            "repeated_oof_auc_worst": min(aucs),
            "repeated_oof_brier_improvement_mean": float(np.mean(brier)),
            "repeated_oof_brier_improvement_worst": min(brier),
            "aggregate_oof_mean_probability": aggregate_oof,
            "development": development,
            "development_evidence_value_strict_spearman": evidence_spearman,
            "final_standardized_feature_weights": feature_weights.tolist(),
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
