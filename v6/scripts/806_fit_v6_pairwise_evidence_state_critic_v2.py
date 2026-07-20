#!/usr/bin/env python3
"""Cross-fit a cardinal pairwise evidence-state ranker and Platt value map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


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


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, steps: int) -> dict[str, np.ndarray]:
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    design = np.concatenate([np.ones((len(x), 1)), (x - center) / scale], axis=1)
    prevalence = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    weights[0] = math.log(prevalence / (1.0 - prevalence))
    for _ in range(steps):
        probability = sigmoid(design @ weights)
        gradient = design.T @ (probability - y) / len(y)
        gradient[1:] += l2 * weights[1:]
        curvature = probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / len(y)
        hessian[1:, 1:] += np.eye(design.shape[1] - 1) * l2
        hessian += np.eye(design.shape[1]) * 1e-7
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if float(np.max(np.abs(update))) < 1e-7:
            break
    return {"center": center, "scale": scale, "weights": weights}


def predict_logistic(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((len(x), 1)), (x - model["center"]) / model["scale"]], axis=1)
    return sigmoid(design @ model["weights"])


def fit_pairwise(x: np.ndarray, y: np.ndarray, query_ids: list[str], indices: np.ndarray, l2: float, steps: int) -> dict[str, Any]:
    center = x[indices].mean(axis=0)
    scale = x[indices].std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - center) / scale
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[query_ids[index]].append(int(index))
    differences, labels, cardinal_weights = [], [], []
    for group in groups.values():
        for position, left in enumerate(group):
            for right in group[position + 1 :]:
                delta = float(y[left] - y[right])
                if abs(delta) <= 1e-12:
                    continue
                oriented = (z[left] - z[right]) * (1.0 if delta > 0.0 else -1.0)
                differences.extend([oriented, -oriented])
                labels.extend([1.0, 0.0])
                cardinal_weights.extend([abs(delta), abs(delta)])
    design = np.asarray(differences, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    weights_by_pair = np.asarray(cardinal_weights, dtype=np.float64)
    weights_by_pair /= weights_by_pair.mean()
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(steps):
        probability = sigmoid(design @ weights)
        gradient = design.T @ (weights_by_pair * (probability - target)) / len(target) + l2 * weights
        curvature = weights_by_pair * probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / len(target) + np.eye(design.shape[1]) * (l2 + 1e-7)
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if float(np.max(np.abs(update))) < 1e-7:
            break
    return {"center": center, "scale": scale, "weights": weights, "training_pairs": len(target) // 2}


def rank_score(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return ((x - model["center"]) / model["scale"]) @ model["weights"]


def folds_for_samples(sample_ids: list[str], folds: int, seed: str) -> dict[str, int]:
    unique = sorted(set(sample_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    return {sample_id: index % folds for index, sample_id in enumerate(unique)}


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    x, y = rankdata(left), rankdata(right)
    return 0.0 if x.std() == 0.0 or y.std() == 0.0 else float(np.corrcoef(x, y)[0, 1])


def pair_order(scores: np.ndarray, labels: np.ndarray, query_ids: list[str]) -> dict[str, float | int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, query_id in enumerate(query_ids):
        groups[query_id].append(index)
    correct = total = 0.0
    for group in groups.values():
        for position, left in enumerate(group):
            for right in group[position + 1 :]:
                delta = labels[left] - labels[right]
                if abs(delta) <= 1e-12:
                    continue
                total += 1.0
                predicted = scores[left] - scores[right]
                correct += float(predicted * delta > 0.0) + 0.5 * float(abs(predicted) <= 1e-12)
    return {"pairs": int(total), "accuracy": correct / total if total else 0.0}


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def serialize_model(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_PAIRWISE_EVIDENCE_STATE_CRITIC_V2_FROZEN":
        raise ValueError("pairwise evidence-state v2 is not frozen")
    if freeze["input_hashes"]["fitter"] != sha256(Path(__file__).resolve()):
        raise ValueError("v2 fitter changed after freeze")
    for key, raw_path in freeze["input_paths"].items():
        if sha256(Path(raw_path)) != freeze["input_hashes"][key]:
            raise ValueError(f"v2 frozen input changed: {key}")
    feature_path = Path(freeze["feature_path"])
    if sha256(feature_path) != freeze["feature_hash"]:
        raise ValueError("v2 runtime features changed")
    for key, raw_path in freeze["label_and_control_paths"].items():
        if sha256(Path(raw_path)) != freeze["label_and_control_hashes"][key]:
            raise ValueError(f"v2 label/control source changed: {key}")

    records = read_jsonl(feature_path)
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
    x = np.asarray([row["features"] for row in records], dtype=np.float64)
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

    config = freeze["fit"]
    all_query_ids = [row["query_id"] for row in records]
    train_samples = [records[index]["sample_id"] for index in train]
    repeated_probability, repeated_score = [], []
    for repeat in range(int(config["repeats"])):
        assignment = folds_for_samples(train_samples, int(config["folds"]), f"{config['seed_prefix']}:{repeat}")
        probabilities = np.full(len(train), np.nan)
        scores = np.full(len(train), np.nan)
        for fold in range(int(config["folds"])):
            fit = np.asarray([index for index in train if assignment[records[index]["sample_id"]] != fold])
            valid = np.asarray([index for index in train if assignment[records[index]["sample_id"]] == fold])
            ranker = fit_pairwise(x, y, all_query_ids, fit, float(config["rank_l2"]), int(config["newton_steps"]))
            fit_score = rank_score(ranker, x[fit])
            valid_score = rank_score(ranker, x[valid])
            platt = fit_logistic(fit_score[:, None], y[fit], float(config["platt_l2"]), int(config["newton_steps"]))
            positions = np.searchsorted(train, valid)
            scores[positions] = valid_score
            probabilities[positions] = predict_logistic(platt, valid_score[:, None])
        if not np.isfinite(probabilities).all() or not np.isfinite(scores).all():
            raise ValueError("incomplete pairwise OOF predictions")
        repeated_probability.append(probabilities)
        repeated_score.append(scores)
    train_probability = np.mean(np.stack(repeated_probability), axis=0)
    train_score = np.mean(np.stack(repeated_score), axis=0)

    final_ranker = fit_pairwise(x, y, all_query_ids, train, float(config["rank_l2"]), int(config["newton_steps"]))
    final_train_score = rank_score(final_ranker, x[train])
    final_platt = fit_logistic(final_train_score[:, None], y[train], float(config["platt_l2"]), int(config["newton_steps"]))
    internal_score = rank_score(final_ranker, x[internal])
    internal_probability = predict_logistic(final_platt, internal_score[:, None])
    low, high = map(float, config["probability_clip"])
    train_probability = np.clip(train_probability, low, high)
    internal_probability = np.clip(internal_probability, low, high)
    prediction = np.full(len(records), np.nan)
    score = np.full(len(records), np.nan)
    prediction[train] = train_probability
    prediction[internal] = internal_probability
    score[train] = train_score
    score[internal] = internal_score

    old = np.asarray([float(shared[row["evidence_action_id"]]["shared_answer_value"]) for row in records])
    query_ids = [records[index]["query_id"] for index in train]
    pair = pair_order(train_probability, y[train], query_ids)
    old_pair = pair_order(old[train], y[train], query_ids)
    groups: dict[str, list[int]] = defaultdict(list)
    for position, query_id in enumerate(query_ids):
        groups[query_id].append(position)
    nonconstant = mean(float(max(train_probability[group]) - min(train_probability[group]) > 1e-8) for group in groups.values())

    categorical = read_jsonl(Path(freeze["label_and_control_paths"]["categorical_train"]))
    index_by_id = {records[index]["evidence_action_id"]: index for index in train}
    selected_rows = []
    for group in categorical:
        indices = [index_by_id[action_id] for action_id in group["action_ids"]]
        posterior = normalize([0.2 * max(float(prediction[index]), 1e-8) for index in indices])
        methods = {
            "no_credit": group["behavior_probabilities"],
            "local_ig_m": group["local_target_probabilities"],
            "outcome": group["outcome_target_probabilities"],
            "old_dagig": group["dagig_target_probabilities"],
            "pairwise_dagig": posterior,
        }
        selected = {}
        for method, probabilities in methods.items():
            chosen = max(range(5), key=lambda index: (float(probabilities[index]), -index))
            absolute = indices[chosen]
            selected[method] = {
                "strategy": records[absolute]["evidence_strategy"],
                "support": float(support[absolute]),
                "expected_strict": float(y[absolute]),
                "old_terminal_value": float(old[absolute]),
            }
        selected_rows.append(selected)
    selector = {}
    for method in ("no_credit", "local_ig_m", "outcome", "old_dagig", "pairwise_dagig"):
        rows = [group[method] for group in selected_rows]
        selector[method] = {
            "states": len(rows),
            "support": mean(row["support"] for row in rows),
            "expected_strict": mean(row["expected_strict"] for row in rows),
            "old_terminal_value": mean(row["old_terminal_value"] for row in rows),
            "strategy_distribution": dict(sorted(Counter(row["strategy"] for row in rows).items())),
        }

    brier = float(np.mean((train_probability - y[train]) ** 2))
    old_brier = float(np.mean((old[train] - y[train]) ** 2))
    metrics = {
        "train_actions": len(train),
        "internal_actions_scored_without_labels": len(internal),
        "pairwise_training_pairs_full_fit": int(final_ranker["training_pairs"]),
        "strict_oof_brier": brier,
        "old_value_strict_brier": old_brier,
        "strict_brier_improvement_vs_old": old_brier - brier,
        "strict_oof_spearman": spearman(train_probability, y[train]),
        "old_value_strict_spearman": spearman(old[train], y[train]),
        "strict_spearman_delta_vs_old": spearman(train_probability, y[train]) - spearman(old[train], y[train]),
        "pairwise_order": pair,
        "old_value_pairwise_order": old_pair,
        "pair_order_delta_vs_old": float(pair["accuracy"]) - float(old_pair["accuracy"]),
        "nonconstant_query_group_rate": nonconstant,
        "final_platt_slope": float(final_platt["weights"][1]),
        "selector_train_oof": selector,
    }
    threshold = freeze["train_oof_gates"]
    dag, outcome = selector["pairwise_dagig"], selector["outcome"]
    gates = {
        "complete_train_oof": len(train) == 11795 and np.isfinite(train_probability).all(),
        "internal_scored_from_train_fit_only": len(internal) == 2975 and np.isfinite(internal_probability).all(),
        "sample_grouped_repeated_cv": True,
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
    decision = "DAGIG_V6_PAIRWISE_EVIDENCE_STATE_CRITIC_V2_TRAIN_OOF_GO" if all(gates.values()) else "DAGIG_V6_PAIRWISE_EVIDENCE_STATE_CRITIC_V2_TRAIN_OOF_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "v6_pairwise_evidence_state_predictions_no_eval_labels.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, rank, value in sorted(zip(records, score, prediction), key=lambda item: item[0]["evidence_action_id"]):
            handle.write(json.dumps({
                "evidence_action_id": row["evidence_action_id"],
                "query_id": row["query_id"],
                "sample_id": row["sample_id"],
                "partition": row["partition"],
                "evidence_strategy": row["evidence_strategy"],
                "pairwise_rank_score": float(rank),
                "evidence_success_probability": float(value),
                "prediction_source": "repeated_sample_group_oof" if row["partition"] == "policy_train" else "policy_train_full_fit",
            }, sort_keys=True) + "\n")
    model_path = output / "v6_pairwise_evidence_state_models.json"
    model_path.write_text(json.dumps({
        "feature_names": freeze["runtime_feature_names"],
        "pairwise_ranker": serialize_model(final_ranker),
        "platt_calibrator": serialize_model(final_platt),
        "fit_partition": "policy_train_only",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "metrics": metrics,
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": sha256(freeze_path)},
        "output_paths": {"predictions": str(prediction_path), "models": str(model_path)},
        "output_hashes": {"predictions": sha256(prediction_path), "models": sha256(model_path)},
        "internal_private_labels_loaded": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": True,
    }
    audit_path = output / "DAGIG_V6_PAIRWISE_EVIDENCE_STATE_CRITIC_V2_TRAIN_OOF_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
