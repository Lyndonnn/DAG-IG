#!/usr/bin/env python3
"""Fit the train-only Platt map for the frozen hybrid evidence value v3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_v6_hybrid_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deserialize(model: dict[str, Any]) -> dict[str, Any]:
    return {key: np.asarray(value, dtype=np.float64) if isinstance(value, list) else value for key, value in model.items()}


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_FROZEN":
        raise ValueError("hybrid evidence value v3 is not frozen")
    if freeze["input_hashes"]["fitter"] != sha256(Path(__file__).resolve()):
        raise ValueError("v3 fitter changed after freeze")
    for key, raw_path in freeze["input_paths"].items():
        if sha256(Path(raw_path)) != freeze["input_hashes"][key]:
            raise ValueError(f"v3 frozen input changed: {key}")
    for key, raw_path in freeze["component_output_paths"].items():
        if sha256(Path(raw_path)) != freeze["component_output_hashes"][key]:
            raise ValueError(f"v3 component changed: {key}")
    if sha256(Path(freeze["feature_path"])) != freeze["feature_hash"]:
        raise ValueError("v3 runtime features changed")
    for key, raw_path in freeze["label_and_control_paths"].items():
        if sha256(Path(raw_path)) != freeze["label_and_control_hashes"][key]:
            raise ValueError(f"v3 labels/controls changed: {key}")

    helper_path = Path(__file__).with_name("806_fit_v6_pairwise_evidence_state_critic_v2.py")
    helper = load_module(helper_path)
    records = read_jsonl(Path(freeze["feature_path"]))
    x = np.asarray([row["features"] for row in records], dtype=np.float64)
    v1_predictions = {row["evidence_action_id"]: row for row in read_jsonl(Path(freeze["component_output_paths"]["v1_predictions"]))}
    v2_predictions = {row["evidence_action_id"]: row for row in read_jsonl(Path(freeze["component_output_paths"]["v2_predictions"]))}
    v1_models = read_json(Path(freeze["component_output_paths"]["v1_models"]))
    v2_models = read_json(Path(freeze["component_output_paths"]["v2_models"]))
    shared = {row["evidence_action_id"]: row for row in read_jsonl(Path(freeze["label_and_control_paths"]["shared_answer_values"]))}
    support_map = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(Path(freeze["label_and_control_paths"]["private_support"]))
        if row["partition"] == "policy_train"
    }
    terminal = {
        row["answer_action_id"]: row
        for row in read_jsonl(Path(freeze["label_and_control_paths"]["terminal_private"]))
        if row["partition"] == "policy_train"
    }
    train = np.asarray([index for index, row in enumerate(records) if row["partition"] == "policy_train"])
    internal = np.asarray([index for index, row in enumerate(records) if row["partition"] == "internal_holdout"])
    y = np.full(len(records), np.nan)
    support = np.full(len(records), np.nan)
    for index in train:
        row = records[index]
        value = shared[row["evidence_action_id"]]
        probabilities = np.asarray(value["answer_policy_probabilities"], dtype=np.float64)
        y[index] = probabilities @ np.asarray([terminal[answer_id]["strict_proxy"] for answer_id in value["answer_action_ids"]], dtype=np.float64)
        support[index] = float(support_map[row["query_id"]][row["evidence_strategy"]])

    alpha = float(freeze["alpha"])
    oof_base = np.asarray([float(v1_predictions[records[index]["evidence_action_id"]]["evidence_success_probability"]) for index in train])
    oof_rank = np.asarray([float(v2_predictions[records[index]["evidence_action_id"]]["pairwise_rank_score"]) for index in train])
    oof_score = np.log(np.clip(oof_base, 1e-5, 1 - 1e-5) / (1 - np.clip(oof_base, 1e-5, 1 - 1e-5))) + alpha * oof_rank
    samples = [records[index]["sample_id"] for index in train]
    config = freeze["platt"]
    repeated = []
    for repeat in range(int(config["repeats"])):
        assignment = helper.folds_for_samples(samples, int(config["folds"]), f"{config['seed_prefix']}:{repeat}")
        probability = np.full(len(train), np.nan)
        for fold in range(int(config["folds"])):
            fit = np.asarray([position for position in range(len(train)) if assignment[samples[position]] != fold])
            valid = np.asarray([position for position in range(len(train)) if assignment[samples[position]] == fold])
            model = helper.fit_logistic(oof_score[fit, None], y[train][fit], float(config["l2"]), int(config["newton_steps"]))
            probability[valid] = helper.predict_logistic(model, oof_score[valid, None])
        if not np.isfinite(probability).all():
            raise ValueError("incomplete hybrid Platt OOF")
        repeated.append(probability)
    train_probability = np.mean(np.stack(repeated), axis=0)

    support_model = deserialize(v1_models["support_head"])
    conditional_model = deserialize(v1_models["conditional_answer_head"])
    base_full = helper.predict_logistic(support_model, x) * helper.predict_logistic(conditional_model, x)
    ranker = deserialize(v2_models["pairwise_ranker"])
    rank_full = helper.rank_score(ranker, x)
    full_score = np.log(np.clip(base_full, 1e-5, 1 - 1e-5) / (1 - np.clip(base_full, 1e-5, 1 - 1e-5))) + alpha * rank_full
    final_platt = helper.fit_logistic(full_score[train, None], y[train], float(config["l2"]), int(config["newton_steps"]))
    internal_probability = helper.predict_logistic(final_platt, full_score[internal, None])
    prediction = np.full(len(records), np.nan)
    prediction[train] = train_probability
    prediction[internal] = internal_probability
    score = np.full(len(records), np.nan)
    score[train] = oof_score
    score[internal] = full_score[internal]

    old = np.asarray([float(shared[row["evidence_action_id"]]["shared_answer_value"]) for row in records])
    query_ids = [records[index]["query_id"] for index in train]
    pair = helper.pair_order(train_probability, y[train], query_ids)
    old_pair = helper.pair_order(old[train], y[train], query_ids)
    groups: dict[str, list[int]] = defaultdict(list)
    for position, query_id in enumerate(query_ids):
        groups[query_id].append(position)
    nonconstant = mean(float(max(train_probability[group]) - min(train_probability[group]) > 1e-8) for group in groups.values())

    categorical = read_jsonl(Path(freeze["label_and_control_paths"]["categorical_train"]))
    index_by_id = {records[index]["evidence_action_id"]: index for index in train}
    selected_groups = []
    for group in categorical:
        indices = [index_by_id[action_id] for action_id in group["action_ids"]]
        posterior = normalize([0.2 * max(float(prediction[index]), 1e-8) for index in indices])
        methods = {
            "no_credit": group["behavior_probabilities"],
            "local_ig_m": group["local_target_probabilities"],
            "outcome": group["outcome_target_probabilities"],
            "old_dagig": group["dagig_target_probabilities"],
            "hybrid_dagig": posterior,
        }
        selected = {}
        for method, probabilities in methods.items():
            choice = max(range(5), key=lambda index: (float(probabilities[index]), -index))
            absolute = indices[choice]
            selected[method] = {
                "strategy": records[absolute]["evidence_strategy"],
                "support": float(support[absolute]),
                "expected_strict": float(y[absolute]),
                "old_terminal_value": float(old[absolute]),
            }
        selected_groups.append(selected)
    selector = {}
    for method in ("no_credit", "local_ig_m", "outcome", "old_dagig", "hybrid_dagig"):
        rows = [group[method] for group in selected_groups]
        selector[method] = {
            "states": len(rows),
            "support": mean(row["support"] for row in rows),
            "expected_strict": mean(row["expected_strict"] for row in rows),
            "old_terminal_value": mean(row["old_terminal_value"] for row in rows),
            "strategy_distribution": dict(sorted(Counter(row["strategy"] for row in rows).items())),
        }

    old_brier = float(np.mean((old[train] - y[train]) ** 2))
    brier = float(np.mean((train_probability - y[train]) ** 2))
    metrics = {
        "train_actions": len(train),
        "internal_actions_scored_without_labels": len(internal),
        "alpha": alpha,
        "strict_oof_brier": brier,
        "old_value_strict_brier": old_brier,
        "strict_brier_improvement_vs_old": old_brier - brier,
        "strict_oof_spearman": helper.spearman(train_probability, y[train]),
        "old_value_strict_spearman": helper.spearman(old[train], y[train]),
        "strict_spearman_delta_vs_old": helper.spearman(train_probability, y[train]) - helper.spearman(old[train], y[train]),
        "pairwise_order": pair,
        "old_value_pairwise_order": old_pair,
        "pair_order_delta_vs_old": float(pair["accuracy"]) - float(old_pair["accuracy"]),
        "nonconstant_query_group_rate": nonconstant,
        "final_platt_slope": float(final_platt["weights"][1]),
        "selector_train_oof": selector,
    }
    threshold = freeze["train_oof_gates"]
    dag, outcome = selector["hybrid_dagig"], selector["outcome"]
    gates = {
        "complete_train_oof": len(train) == 11795 and np.isfinite(train_probability).all(),
        "internal_scored_from_train_fit_only": len(internal) == 2975 and np.isfinite(internal_probability).all(),
        "sample_grouped_repeated_platt_cv": True,
        "strict_brier_improves_old": metrics["strict_brier_improvement_vs_old"] >= threshold["strict_brier_improvement_vs_old_min"],
        "strict_spearman_improves_old": metrics["strict_spearman_delta_vs_old"] >= threshold["strict_spearman_delta_vs_old_min"],
        "pair_order_improves_old": metrics["pair_order_delta_vs_old"] >= threshold["pair_order_delta_vs_old_min"],
        "selected_support_noninferior_outcome": dag["support"] >= outcome["support"] - threshold["selected_support_noninferiority_vs_outcome_tolerance"],
        "selected_strict_noninferior_outcome": dag["expected_strict"] >= outcome["expected_strict"] - threshold["selected_strict_noninferiority_vs_outcome_tolerance"],
        "query_groups_nonconstant": nonconstant >= threshold["nonconstant_query_group_rate_min"],
        "positive_platt_slope": metrics["final_platt_slope"] > threshold["platt_slope_min"],
        "runtime_predictions_use_no_gold": True,
        "internal_labels_not_loaded": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {key: bool(value) for key, value in gates.items()}
    decision = "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_TRAIN_OOF_GO" if all(gates.values()) else "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_TRAIN_OOF_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "v6_hybrid_evidence_state_values_no_eval_labels.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, raw_score, value in sorted(zip(records, score, prediction), key=lambda item: item[0]["evidence_action_id"]):
            handle.write(json.dumps({
                "evidence_action_id": row["evidence_action_id"],
                "query_id": row["query_id"],
                "sample_id": row["sample_id"],
                "partition": row["partition"],
                "evidence_strategy": row["evidence_strategy"],
                "hybrid_rank_score": float(raw_score),
                "evidence_success_probability": float(value),
                "prediction_source": "component_oof_plus_grouped_platt_oof" if row["partition"] == "policy_train" else "policy_train_full_fit",
            }, sort_keys=True) + "\n")
    model_path = output / "v6_hybrid_evidence_state_platt.json"
    model_path.write_text(json.dumps({
        "alpha": alpha,
        "platt_center": final_platt["center"].tolist(),
        "platt_scale": final_platt["scale"].tolist(),
        "platt_weights": final_platt["weights"].tolist(),
        "fit_partition": "policy_train_only",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "metrics": metrics,
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": sha256(freeze_path)},
        "output_paths": {"predictions": str(prediction_path), "model": str(model_path)},
        "output_hashes": {"predictions": sha256(prediction_path), "model": sha256(model_path)},
        "internal_private_labels_loaded": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": True,
    }
    audit_path = output / "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_TRAIN_OOF_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
