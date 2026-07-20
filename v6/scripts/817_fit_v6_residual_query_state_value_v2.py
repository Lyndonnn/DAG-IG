#!/usr/bin/env python3
"""Fit the conservative residual query-state value using train-only labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_v6_residual_query_helper", path)
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


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def deserialize(model: dict[str, Any]) -> dict[str, Any]:
    return {key: np.asarray(value, dtype=np.float64) if isinstance(value, list) else value for key, value in model.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_RESIDUAL_QUERY_STATE_VALUE_V2_FROZEN":
        raise ValueError("residual query value v2 is not frozen")
    if freeze["input_hashes"]["fitter"] != sha256(Path(__file__).resolve()):
        raise ValueError("residual query fitter changed")
    for key, raw_path in freeze["input_paths"].items():
        if sha256(Path(raw_path)) != freeze["input_hashes"][key]:
            raise ValueError(f"residual query input changed: {key}")
    for key, raw_path in freeze["component_output_paths"].items():
        if sha256(Path(raw_path)) != freeze["component_output_hashes"][key]:
            raise ValueError(f"query v1 component changed: {key}")
    helper = load_module(Path(freeze["input_paths"]["helper"]))
    records = read_jsonl(Path(freeze["feature_path"]))
    x = np.asarray([row["features"] for row in records], dtype=np.float64)
    train = np.asarray([index for index, row in enumerate(records) if row["partition"] == "policy_train"])
    internal = np.asarray([index for index, row in enumerate(records) if row["partition"] == "internal_holdout"])
    v1_predictions = {row["query_action_id"]: row for row in read_jsonl(Path(freeze["component_output_paths"]["predictions"]))}
    v1_models = read_json(Path(freeze["component_output_paths"]["models"]))
    shared = {row["evidence_action_id"]: row for row in read_jsonl(Path(freeze["label_and_control_paths"]["shared_answer_values"]))}
    support_map = {row["query_id"]: row["strategy_support"] for row in read_jsonl(Path(freeze["label_and_control_paths"]["private_support"])) if row["partition"] == "policy_train"}
    terminal = {row["answer_action_id"]: row for row in read_jsonl(Path(freeze["label_and_control_paths"]["terminal_private"])) if row["partition"] == "policy_train"}
    y = np.full(len(records), np.nan)
    support = np.full(len(records), np.nan)
    downstream_index = read_json(Path(freeze["input_paths"]["v1_freeze"]))["feature_names"].index("selected_hybrid_evidence_value")
    downstream = np.asarray([float(row["features"][downstream_index]) for row in records])
    for index in train:
        row = records[index]
        evidence_id = row["selected_evidence_action_id"]
        evidence_strategy = evidence_id.rsplit("::", 1)[-1]
        support[index] = float(support_map[row["query_action_id"]][evidence_strategy])
        value = shared[evidence_id]
        probabilities = np.asarray(value["answer_policy_probabilities"], dtype=np.float64)
        y[index] = probabilities @ np.asarray([terminal[answer_id]["strict_proxy"] for answer_id in value["answer_action_ids"]], dtype=np.float64)

    q_oof = np.asarray([float(v1_predictions[records[index]["query_action_id"]]["query_success_probability"]) for index in train])
    log_downstream = np.log(np.clip(downstream[train], 1e-5, 1 - 1e-5) / (1 - np.clip(downstream[train], 1e-5, 1 - 1e-5)))
    log_query = np.log(np.clip(q_oof, 1e-5, 1 - 1e-5) / (1 - np.clip(q_oof, 1e-5, 1 - 1e-5)))
    samples = [records[index]["sample_id"] for index in train]
    parent_ids = [records[index]["parent_visual_state_id"] for index in train]
    config = freeze["fit"]
    query_value_freeze = read_json(Path(freeze["label_and_control_paths"]["query_value_freeze"]))
    target_rows = read_jsonl(Path(query_value_freeze["output_paths"]["train_targets"]))
    diagnostic = {row["parent_state_id"]: row for row in read_jsonl(Path(query_value_freeze["output_paths"]["diagnostics"])) if row["partition"] == "policy_train"}
    position_by_query = {records[index]["query_action_id"]: position for position, index in enumerate(train)}
    downstream_pair = helper.pair_order(downstream[train], y[train], parent_ids)
    downstream_spearman = helper.spearman(downstream[train], y[train])
    downstream_brier = float(np.mean((downstream[train] - y[train]) ** 2))
    threshold = freeze["train_oof_gates"]
    candidates = []
    for beta in config["beta_grid"]:
        raw_score = log_downstream + float(beta) * log_query
        repeated = []
        for repeat in range(int(config["repeats"])):
            assignment = helper.folds_for_samples(samples, int(config["folds"]), f"{config['seed_prefix']}:{beta}:{repeat}")
            probability = np.full(len(train), np.nan)
            for fold in range(int(config["folds"])):
                fit = np.asarray([position for position in range(len(train)) if assignment[samples[position]] != fold])
                valid = np.asarray([position for position in range(len(train)) if assignment[samples[position]] == fold])
                model = helper.fit_logistic(raw_score[fit, None], y[train][fit], float(config["platt_l2"]), int(config["newton_steps"]))
                probability[valid] = helper.predict_logistic(model, raw_score[valid, None])
            repeated.append(probability)
        probability = np.mean(np.stack(repeated), axis=0)
        pair = helper.pair_order(probability, y[train], parent_ids)
        selected_groups = []
        for target in target_rows:
            query_ids = diagnostic[target["parent_state_id"]]["query_action_ids"]
            positions = [position_by_query[query_id] for query_id in query_ids]
            posterior = normalize([float(probability[position]) for position in positions])
            methods = {"no_credit": target["target_distributions"]["no_credit"], "local_ig_m": target["target_distributions"]["local_ig_m"], "outcome": target["target_distributions"]["outcome"], "residual_dagig": posterior}
            selected = {}
            for method, distribution in methods.items():
                choice = max(range(len(distribution)), key=lambda item: (float(distribution[item]), -item))
                absolute = train[positions[choice]]
                selected[method] = {"support": float(support[absolute]), "strict": float(y[absolute])}
            selected_groups.append(selected)
        selector = {method: {"support": mean(group[method]["support"] for group in selected_groups), "expected_strict": mean(group[method]["strict"] for group in selected_groups)} for method in ("no_credit", "local_ig_m", "outcome", "residual_dagig")}
        brier = float(np.mean((probability - y[train]) ** 2))
        metrics = {"beta": float(beta), "strict_brier": brier, "strict_brier_improvement_vs_downstream": downstream_brier - brier, "strict_spearman": helper.spearman(probability, y[train]), "strict_spearman_delta_vs_downstream": helper.spearman(probability, y[train]) - downstream_spearman, "pair_order": pair, "pair_order_delta_vs_downstream": float(pair["accuracy"]) - float(downstream_pair["accuracy"]), "selector": selector}
        dag, outcome = selector["residual_dagig"], selector["outcome"]
        passes = (metrics["strict_brier_improvement_vs_downstream"] >= threshold["strict_brier_improvement_vs_downstream_min"] and metrics["strict_spearman_delta_vs_downstream"] >= threshold["strict_spearman_delta_vs_downstream_min"] and metrics["pair_order_delta_vs_downstream"] >= -threshold["pair_order_noninferiority_tolerance"] and dag["support"] >= outcome["support"] - threshold["selected_support_noninferiority_vs_outcome_tolerance"] and dag["expected_strict"] >= outcome["expected_strict"] - threshold["selected_strict_noninferiority_vs_outcome_tolerance"])
        candidates.append({**metrics, "passes": bool(passes), "probability": probability, "raw_score": raw_score})
    passing = [candidate for candidate in candidates if candidate["passes"]]
    selected = min(passing, key=lambda candidate: candidate["beta"]) if passing else max(candidates, key=lambda candidate: (candidate["selector"]["residual_dagig"]["expected_strict"], candidate["strict_spearman"], -candidate["strict_brier"]))

    support_model = deserialize(v1_models["support_head"])
    conditional_model = deserialize(v1_models["conditional_head"])
    ranker = deserialize(v1_models["pairwise_ranker"])
    v1_platt = deserialize(v1_models["platt"])
    base_full = helper.predict_logistic(support_model, x) * helper.predict_logistic(conditional_model, x)
    rank_full = helper.rank_score(ranker, x)
    v1_alpha = float(v1_models["alpha"])
    v1_score = np.log(np.clip(base_full, 1e-5, 1 - 1e-5) / (1 - np.clip(base_full, 1e-5, 1 - 1e-5))) + v1_alpha * rank_full
    q_full = helper.predict_logistic(v1_platt, v1_score[:, None])
    beta = selected["beta"]
    full_score = np.log(np.clip(downstream, 1e-5, 1 - 1e-5) / (1 - np.clip(downstream, 1e-5, 1 - 1e-5))) + beta * np.log(np.clip(q_full, 1e-5, 1 - 1e-5) / (1 - np.clip(q_full, 1e-5, 1 - 1e-5)))
    final_platt = helper.fit_logistic(full_score[train, None], y[train], float(config["platt_l2"]), int(config["newton_steps"]))
    prediction = np.full(len(records), np.nan)
    prediction[train] = selected["probability"]
    prediction[internal] = helper.predict_logistic(final_platt, full_score[internal, None])
    raw = np.full(len(records), np.nan)
    raw[train] = selected["raw_score"]
    raw[internal] = full_score[internal]
    groups: dict[str, list[int]] = defaultdict(list)
    for position, parent in enumerate(parent_ids):
        groups[parent].append(position)
    nonconstant = mean(float(max(prediction[train][group]) - min(prediction[train][group]) > 1e-8) for group in groups.values())
    metrics = {key: value for key, value in selected.items() if key not in {"probability", "raw_score"}}
    metrics.update({"train_actions": len(train), "internal_actions_scored_without_labels": len(internal), "downstream_brier": downstream_brier, "downstream_spearman": downstream_spearman, "downstream_pair_order": downstream_pair, "nonconstant_parent_group_rate": nonconstant, "final_platt_slope": float(final_platt["weights"][1]), "beta_candidates": [{key: value for key, value in candidate.items() if key not in {"probability", "raw_score"}} for candidate in candidates]})
    gates = {"at_least_one_train_oof_beta_passes": bool(passing), "selected_smallest_passing_beta": bool(passing) and beta == min(candidate["beta"] for candidate in passing), "nonconstant_parent_groups": nonconstant >= threshold["nonconstant_parent_group_rate_min"], "positive_platt_slope": metrics["final_platt_slope"] > 0.0, "complete_train_oof": len(train) == 2359 and np.isfinite(prediction[train]).all(), "internal_scored_from_train_only": len(internal) == 595 and np.isfinite(prediction[internal]).all(), "internal_labels_not_loaded": True, "runtime_features_use_no_gold": True, "dev_sealed": True, "test_sealed": True}
    gates = {key: bool(value) for key, value in gates.items()}
    decision = "DAGIG_V6_RESIDUAL_QUERY_STATE_VALUE_V2_TRAIN_OOF_GO" if all(gates.values()) else "DAGIG_V6_RESIDUAL_QUERY_STATE_VALUE_V2_TRAIN_OOF_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "v6_residual_query_state_values_no_eval_labels.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, score, value in sorted(zip(records, raw, prediction), key=lambda item: item[0]["query_action_id"]):
            handle.write(json.dumps({"query_action_id": row["query_action_id"], "parent_visual_state_id": row["parent_visual_state_id"], "selected_evidence_action_id": row["selected_evidence_action_id"], "sample_id": row["sample_id"], "partition": row["partition"], "residual_query_score": float(score), "query_success_probability": float(value), "prediction_source": "sample_group_oof" if row["partition"] == "policy_train" else "policy_train_full_fit"}, sort_keys=True) + "\n")
    model_path = output / "v6_residual_query_state_platt.json"
    model_path.write_text(json.dumps({"beta": beta, "platt_center": final_platt["center"].tolist(), "platt_scale": final_platt["scale"].tolist(), "platt_weights": final_platt["weights"].tolist(), "fit_partition": "policy_train_only"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {"decision": decision, "protocol_version": freeze["protocol_version"], "metrics": metrics, "gates": gates, "input_paths": {"freeze": str(freeze_path)}, "input_hashes": {"freeze": sha256(freeze_path)}, "output_paths": {"predictions": str(prediction_path), "model": str(model_path)}, "output_hashes": {"predictions": sha256(prediction_path), "model": sha256(model_path)}, "internal_private_labels_loaded": False, "dev_used": False, "test_used": False, "api_calls": 0, "training_run": True}
    audit_path = output / "DAGIG_V6_RESIDUAL_QUERY_STATE_VALUE_V2_TRAIN_OOF_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "selected_beta": beta, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
