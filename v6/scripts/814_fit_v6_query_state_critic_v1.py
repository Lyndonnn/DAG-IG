#!/usr/bin/env python3
"""Train-only grouped OOF fitting and selection for the query-state critic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_v6_query_critic_helper", path)
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


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positive = labels == 1
    positives = int(positive.sum())
    negatives = len(labels) - positives
    return 0.5 if not positives or not negatives else float((ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def serialize(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_QUERY_STATE_CRITIC_V1_FROZEN":
        raise ValueError("query-state critic protocol is not frozen")
    if freeze["input_hashes"]["fitter"] != sha256(Path(__file__).resolve()):
        raise ValueError("query-state fitter changed after freeze")
    for key, raw_path in freeze["input_paths"].items():
        if sha256(Path(raw_path)) != freeze["input_hashes"][key]:
            raise ValueError(f"query critic input changed: {key}")
    feature_path = Path(freeze["output_paths"]["features"])
    if sha256(feature_path) != freeze["output_hashes"]["features"]:
        raise ValueError("query-state features changed")
    helper_path = args.helper.resolve()
    if helper_path != Path(freeze["input_paths"]["helper"]) or sha256(helper_path) != freeze["input_hashes"]["helper"]:
        raise ValueError("query-state numeric helper differs from frozen dependency")
    helper = load_module(helper_path)

    records = read_jsonl(feature_path)
    x = np.asarray([row["features"] for row in records], dtype=np.float64)
    train = np.asarray([index for index, row in enumerate(records) if row["partition"] == "policy_train"])
    internal = np.asarray([index for index, row in enumerate(records) if row["partition"] == "internal_holdout"])
    shared = {row["evidence_action_id"]: row for row in read_jsonl(Path(freeze["input_paths"]["shared_answer_values"]))}
    support_map = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(Path(freeze["input_paths"]["private_support"]))
        if row["partition"] == "policy_train"
    }
    terminal = {
        row["answer_action_id"]: row
        for row in read_jsonl(Path(freeze["input_paths"]["terminal_private"]))
        if row["partition"] == "policy_train"
    }
    y = np.full(len(records), np.nan)
    support = np.full(len(records), np.nan)
    conditional = np.full(len(records), np.nan)
    downstream = np.full(len(records), np.nan)
    downstream_feature_index = freeze["feature_names"].index("selected_hybrid_evidence_value")
    for index in train:
        row = records[index]
        evidence_id = row["selected_evidence_action_id"]
        strategy = evidence_id.rsplit("::", 1)[-1]
        support[index] = float(support_map[row["query_action_id"]][strategy])
        value = shared[evidence_id]
        probabilities = np.asarray(value["answer_policy_probabilities"], dtype=np.float64)
        strict_labels = np.asarray([terminal[answer_id]["strict_proxy"] for answer_id in value["answer_action_ids"]], dtype=np.float64)
        correct_labels = np.asarray([terminal[answer_id]["answer_correct_proxy"] for answer_id in value["answer_action_ids"]], dtype=np.float64)
        y[index] = probabilities @ strict_labels
        conditional[index] = probabilities @ correct_labels
        if abs(y[index] - support[index] * conditional[index]) > 1e-9:
            raise ValueError(f"query target factorization mismatch: {row['query_action_id']}")
        downstream[index] = float(row["features"][downstream_feature_index])

    config = freeze["fit"]
    samples = [records[index]["sample_id"] for index in train]
    parent_ids = [row["parent_visual_state_id"] for row in records]
    base_repeats, rank_repeats, support_repeats = [], [], []
    for repeat in range(int(config["repeats"])):
        assignment = helper.folds_for_samples(samples, int(config["folds"]), f"{config['seed_prefix']}:{repeat}")
        base_probability = np.full(len(train), np.nan)
        rank_scores = np.full(len(train), np.nan)
        support_probability = np.full(len(train), np.nan)
        for fold in range(int(config["folds"])):
            fit = np.asarray([index for index in train if assignment[records[index]["sample_id"]] != fold])
            valid = np.asarray([index for index in train if assignment[records[index]["sample_id"]] == fold])
            support_model = helper.fit_logistic(x[fit], support[fit], float(config["support_l2"]), int(config["newton_steps"]))
            positive_fit = fit[support[fit] == 1.0]
            conditional_model = helper.fit_logistic(x[positive_fit], conditional[positive_fit], float(config["conditional_l2"]), int(config["newton_steps"]))
            ranker = helper.fit_pairwise(x, y, parent_ids, fit, float(config["pairwise_l2"]), int(config["newton_steps"]))
            positions = np.searchsorted(train, valid)
            p_support = helper.predict_logistic(support_model, x[valid])
            support_probability[positions] = p_support
            base_probability[positions] = p_support * helper.predict_logistic(conditional_model, x[valid])
            rank_scores[positions] = helper.rank_score(ranker, x[valid])
        if not np.isfinite(base_probability).all() or not np.isfinite(rank_scores).all():
            raise ValueError("incomplete query critic OOF")
        base_repeats.append(base_probability)
        rank_repeats.append(rank_scores)
        support_repeats.append(support_probability)
    base_oof = np.mean(np.stack(base_repeats), axis=0)
    rank_oof = np.mean(np.stack(rank_repeats), axis=0)
    support_oof = np.mean(np.stack(support_repeats), axis=0)
    base_logit = np.log(np.clip(base_oof, 1e-5, 1 - 1e-5) / (1 - np.clip(base_oof, 1e-5, 1 - 1e-5)))

    query_freeze = read_json(Path(freeze["input_paths"]["query_value_freeze"]))
    target_rows = read_jsonl(Path(query_freeze["output_paths"]["train_targets"]))
    index_by_query = {records[index]["query_action_id"]: index for index in train}
    baseline_pair = helper.pair_order(downstream[train], y[train], [records[index]["parent_visual_state_id"] for index in train])
    baseline_spearman = helper.spearman(downstream[train], y[train])
    baseline_brier = float(np.mean((downstream[train] - y[train]) ** 2))
    threshold = freeze["train_oof_gates"]
    candidates = []
    for alpha in config["alpha_grid"]:
        raw_score = base_logit + float(alpha) * rank_oof
        platt_repeats = []
        for repeat in range(int(config["repeats"])):
            assignment = helper.folds_for_samples(samples, int(config["folds"]), f"{config['seed_prefix']}:platt:{alpha}:{repeat}")
            probability = np.full(len(train), np.nan)
            for fold in range(int(config["folds"])):
                fit_positions = np.asarray([position for position in range(len(train)) if assignment[samples[position]] != fold])
                valid_positions = np.asarray([position for position in range(len(train)) if assignment[samples[position]] == fold])
                model = helper.fit_logistic(raw_score[fit_positions, None], y[train][fit_positions], float(config["platt_l2"]), int(config["newton_steps"]))
                probability[valid_positions] = helper.predict_logistic(model, raw_score[valid_positions, None])
            platt_repeats.append(probability)
        probability = np.mean(np.stack(platt_repeats), axis=0)
        pair = helper.pair_order(probability, y[train], [records[index]["parent_visual_state_id"] for index in train])
        groups = []
        for target in target_rows:
            indices = [index_by_query[diagnostic_id] for diagnostic_id in [row["query_action_id"] for row in records if row["parent_visual_state_id"] == target["parent_state_id"] and row["partition"] == "policy_train"]]
            indices = sorted(indices, key=lambda idx: records[idx]["query_action_id"])
            # Align to target action strategy order rather than lexical IDs.
            by_strategy = {records[index]["query_action_id"].rsplit("::", 1)[-1]: index for index in indices}
            indices = [by_strategy[action["strategy"]] for action in target["actions"]]
            posterior = normalize([float(probability[np.searchsorted(train, index)]) for index in indices])
            methods = {"no_credit": target["target_distributions"]["no_credit"], "local_ig_m": target["target_distributions"]["local_ig_m"], "outcome": target["target_distributions"]["outcome"], "query_critic_dagig": posterior}
            selected = {}
            for method, distribution in methods.items():
                choice = max(range(len(distribution)), key=lambda item: (float(distribution[item]), -item))
                index = indices[choice]
                selected[method] = {"support": float(support[index]), "strict": float(y[index])}
            groups.append(selected)
        selector = {method: {"support": mean(group[method]["support"] for group in groups), "expected_strict": mean(group[method]["strict"] for group in groups)} for method in ("no_credit", "local_ig_m", "outcome", "query_critic_dagig")}
        brier = float(np.mean((probability - y[train]) ** 2))
        metrics = {
            "alpha": float(alpha),
            "strict_brier": brier,
            "strict_brier_improvement_vs_downstream": baseline_brier - brier,
            "strict_spearman": helper.spearman(probability, y[train]),
            "strict_spearman_delta_vs_downstream": helper.spearman(probability, y[train]) - baseline_spearman,
            "pair_order": pair,
            "pair_order_delta_vs_downstream": float(pair["accuracy"]) - float(baseline_pair["accuracy"]),
            "selector": selector,
        }
        dag, outcome = selector["query_critic_dagig"], selector["outcome"]
        passes = (
            metrics["strict_brier_improvement_vs_downstream"] >= threshold["strict_brier_improvement_vs_downstream_min"]
            and metrics["strict_spearman_delta_vs_downstream"] >= threshold["strict_spearman_delta_vs_downstream_min"]
            and metrics["pair_order_delta_vs_downstream"] >= threshold["pair_order_delta_vs_downstream_min"]
            and dag["support"] >= outcome["support"] - threshold["selected_support_noninferiority_vs_outcome_tolerance"]
            and dag["expected_strict"] >= outcome["expected_strict"] - threshold["selected_strict_noninferiority_vs_outcome_tolerance"]
        )
        candidates.append({**metrics, "passes": bool(passes), "probability": probability, "raw_score": raw_score})
    passing = [candidate for candidate in candidates if candidate["passes"]]
    selected_candidate = min(passing, key=lambda candidate: candidate["alpha"]) if passing else max(candidates, key=lambda candidate: (candidate["pair_order"]["accuracy"], candidate["strict_spearman"], -candidate["strict_brier"]))

    support_model = helper.fit_logistic(x[train], support[train], float(config["support_l2"]), int(config["newton_steps"]))
    positive_train = train[support[train] == 1.0]
    conditional_model = helper.fit_logistic(x[positive_train], conditional[positive_train], float(config["conditional_l2"]), int(config["newton_steps"]))
    ranker = helper.fit_pairwise(x, y, parent_ids, train, float(config["pairwise_l2"]), int(config["newton_steps"]))
    base_full = helper.predict_logistic(support_model, x) * helper.predict_logistic(conditional_model, x)
    rank_full = helper.rank_score(ranker, x)
    alpha = selected_candidate["alpha"]
    full_score = np.log(np.clip(base_full, 1e-5, 1 - 1e-5) / (1 - np.clip(base_full, 1e-5, 1 - 1e-5))) + alpha * rank_full
    final_platt = helper.fit_logistic(full_score[train, None], y[train], float(config["platt_l2"]), int(config["newton_steps"]))
    prediction = np.full(len(records), np.nan)
    prediction[train] = selected_candidate["probability"]
    prediction[internal] = helper.predict_logistic(final_platt, full_score[internal, None])
    raw = np.full(len(records), np.nan)
    raw[train] = selected_candidate["raw_score"]
    raw[internal] = full_score[internal]

    parent_groups: dict[str, list[int]] = defaultdict(list)
    for position, index in enumerate(train):
        parent_groups[records[index]["parent_visual_state_id"]].append(position)
    nonconstant = mean(float(max(prediction[train][group]) - min(prediction[train][group]) > 1e-8) for group in parent_groups.values())
    metrics = {key: value for key, value in selected_candidate.items() if key not in {"probability", "raw_score"}}
    metrics.update({
        "train_actions": len(train),
        "internal_actions_scored_without_labels": len(internal),
        "support_head_oof_auc": auc(support_oof, support[train]),
        "downstream_brier": baseline_brier,
        "downstream_spearman": baseline_spearman,
        "downstream_pair_order": baseline_pair,
        "nonconstant_parent_group_rate": nonconstant,
        "final_platt_slope": float(final_platt["weights"][1]),
        "alpha_candidates": [{key: value for key, value in candidate.items() if key not in {"probability", "raw_score"}} for candidate in candidates],
    })
    gates = {
        "at_least_one_train_oof_alpha_passes": bool(passing),
        "selected_smallest_passing_alpha": bool(passing) and alpha == min(candidate["alpha"] for candidate in passing),
        "support_head_auc": metrics["support_head_oof_auc"] >= threshold["support_auc_min"],
        "nonconstant_parent_groups": nonconstant >= threshold["nonconstant_parent_group_rate_min"],
        "positive_platt_slope": metrics["final_platt_slope"] > 0.0,
        "complete_train_oof": len(train) == 2359 and np.isfinite(prediction[train]).all(),
        "internal_scored_from_train_only": len(internal) == 595 and np.isfinite(prediction[internal]).all(),
        "internal_labels_not_loaded": True,
        "runtime_features_use_no_gold": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {key: bool(value) for key, value in gates.items()}
    decision = "DAGIG_V6_QUERY_STATE_CRITIC_V1_TRAIN_OOF_GO" if all(gates.values()) else "DAGIG_V6_QUERY_STATE_CRITIC_V1_TRAIN_OOF_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "v6_query_state_values_no_eval_labels.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, score, value in sorted(zip(records, raw, prediction), key=lambda item: item[0]["query_action_id"]):
            handle.write(json.dumps({"query_action_id": row["query_action_id"], "parent_visual_state_id": row["parent_visual_state_id"], "selected_evidence_action_id": row["selected_evidence_action_id"], "sample_id": row["sample_id"], "partition": row["partition"], "query_state_rank_score": float(score), "query_success_probability": float(value), "prediction_source": "sample_group_oof" if row["partition"] == "policy_train" else "policy_train_full_fit"}, sort_keys=True) + "\n")
    model_path = output / "v6_query_state_critic_models.json"
    model_path.write_text(json.dumps({"alpha": alpha, "support_head": serialize(support_model), "conditional_head": serialize(conditional_model), "pairwise_ranker": serialize(ranker), "platt": serialize(final_platt), "fit_partition": "policy_train_only"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {"decision": decision, "protocol_version": freeze["protocol_version"], "metrics": metrics, "gates": gates, "input_paths": {"freeze": str(freeze_path)}, "input_hashes": {"freeze": sha256(freeze_path)}, "output_paths": {"predictions": str(prediction_path), "models": str(model_path)}, "output_hashes": {"predictions": sha256(prediction_path), "models": sha256(model_path)}, "internal_private_labels_loaded": False, "dev_used": False, "test_used": False, "api_calls": 0, "training_run": True}
    audit_path = output / "DAGIG_V6_QUERY_STATE_CRITIC_V1_TRAIN_OOF_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "selected_alpha": alpha, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
