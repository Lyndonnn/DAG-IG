#!/usr/bin/env python3
"""Cross-fit the frozen factorized evidence-state critic on policy-train only."""

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


STRATEGY_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)


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


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((len(x), 1)), (x - model["center"]) / model["scale"]], axis=1)
    return sigmoid(design @ model["weights"])


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
    if not positives or not negatives:
        return 0.5
    return float((ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))


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
    correct = total = 0
    for indices in groups.values():
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                delta = labels[left] - labels[right]
                if abs(delta) <= 1e-12:
                    continue
                total += 1
                predicted = scores[left] - scores[right]
                correct += int(predicted * delta > 0.0)
                correct += 0.5 * int(abs(predicted) <= 1e-12)
    return {"pairs": total, "accuracy": float(correct / total) if total else 0.0}


def folds_for_samples(sample_ids: list[str], folds: int, seed: str) -> dict[str, int]:
    unique = sorted(set(sample_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    return {sample_id: index % folds for index, sample_id in enumerate(unique)}


def serialize_model(model: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: value.tolist() for key, value in model.items()}


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
    if freeze.get("decision") != "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_FROZEN":
        raise ValueError("factorized evidence-state protocol is not frozen")
    if freeze["input_hashes"]["fitter"] != sha256(Path(__file__).resolve()):
        raise ValueError("fitter changed after protocol freeze")
    for key, raw_path in freeze["input_paths"].items():
        if key == "score_files":
            for path, expected in zip(raw_path, freeze["input_hashes"][key]):
                if sha256(Path(path)) != expected:
                    raise ValueError(f"score file changed: {path}")
        elif sha256(Path(raw_path)) != freeze["input_hashes"][key]:
            raise ValueError(f"frozen input changed: {key}")
    feature_path = Path(freeze["output_paths"]["features"])
    if sha256(feature_path) != freeze["output_hashes"]["features"]:
        raise ValueError("runtime features changed after freeze")

    records = read_jsonl(feature_path)
    shared_values = {
        row["evidence_action_id"]: row
        for row in read_jsonl(Path(freeze["input_paths"]["shared_answer_values"]))
    }
    support_by_query = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(Path(freeze["input_paths"]["private_support"]))
        if row["partition"] == "policy_train"
    }
    terminal = {
        row["answer_action_id"]: row
        for row in read_jsonl(Path(freeze["input_paths"]["terminal_private"]))
        if row["partition"] == "policy_train"
    }

    x = np.asarray([row["features"] for row in records], dtype=np.float64)
    train_indices = np.asarray([index for index, row in enumerate(records) if row["partition"] == "policy_train"])
    internal_indices = np.asarray([index for index, row in enumerate(records) if row["partition"] == "internal_holdout"])
    support = np.full(len(records), np.nan, dtype=np.float64)
    conditional = np.full(len(records), np.nan, dtype=np.float64)
    strict = np.full(len(records), np.nan, dtype=np.float64)
    for index in train_indices:
        row = records[index]
        evidence_id = row["evidence_action_id"]
        strategy = row["evidence_strategy"]
        support[index] = float(bool(support_by_query[row["query_id"]][strategy]))
        value = shared_values[evidence_id]
        answer_labels = [float(terminal[answer_id]["answer_correct_proxy"]) for answer_id in value["answer_action_ids"]]
        strict_labels = [float(terminal[answer_id]["strict_proxy"]) for answer_id in value["answer_action_ids"]]
        probabilities = [float(item) for item in value["answer_policy_probabilities"]]
        conditional[index] = sum(p * label for p, label in zip(probabilities, answer_labels))
        strict[index] = sum(p * label for p, label in zip(probabilities, strict_labels))
        if abs(strict[index] - support[index] * conditional[index]) > 1e-9:
            raise ValueError(f"factorization target mismatch: {evidence_id}")

    config = freeze["fit"]
    oof_support_repeats = []
    oof_conditional_repeats = []
    train_sample_ids = [records[index]["sample_id"] for index in train_indices]
    for repeat in range(int(config["repeats"])):
        assignment = folds_for_samples(train_sample_ids, int(config["folds"]), f"{config['seed_prefix']}:{repeat}")
        repeat_support = np.full(len(train_indices), np.nan, dtype=np.float64)
        repeat_conditional = np.full(len(train_indices), np.nan, dtype=np.float64)
        for fold in range(int(config["folds"])):
            fit_indices = np.asarray([index for index in train_indices if assignment[records[index]["sample_id"]] != fold])
            valid_indices = np.asarray([index for index in train_indices if assignment[records[index]["sample_id"]] == fold])
            support_model = fit_logistic(x[fit_indices], support[fit_indices], float(config["l2"]), int(config["newton_steps"]))
            positive_fit = fit_indices[support[fit_indices] == 1.0]
            if len(positive_fit) < 100:
                raise ValueError("too few supporting actions for conditional head")
            conditional_model = fit_logistic(x[positive_fit], conditional[positive_fit], float(config["l2"]), int(config["newton_steps"]))
            positions = np.searchsorted(train_indices, valid_indices)
            repeat_support[positions] = predict(support_model, x[valid_indices])
            repeat_conditional[positions] = predict(conditional_model, x[valid_indices])
        if not np.isfinite(repeat_support).all() or not np.isfinite(repeat_conditional).all():
            raise ValueError("incomplete repeated OOF predictions")
        oof_support_repeats.append(repeat_support)
        oof_conditional_repeats.append(repeat_conditional)

    train_support_prediction = np.mean(np.stack(oof_support_repeats), axis=0)
    train_conditional_prediction = np.mean(np.stack(oof_conditional_repeats), axis=0)
    support_model = fit_logistic(x[train_indices], support[train_indices], float(config["l2"]), int(config["newton_steps"]))
    positive_train = train_indices[support[train_indices] == 1.0]
    conditional_model = fit_logistic(x[positive_train], conditional[positive_train], float(config["l2"]), int(config["newton_steps"]))
    internal_support_prediction = predict(support_model, x[internal_indices])
    internal_conditional_prediction = predict(conditional_model, x[internal_indices])
    low, high = map(float, config["probability_clip"])

    predicted_support = np.full(len(records), np.nan)
    predicted_conditional = np.full(len(records), np.nan)
    predicted_support[train_indices] = train_support_prediction
    predicted_conditional[train_indices] = train_conditional_prediction
    predicted_support[internal_indices] = internal_support_prediction
    predicted_conditional[internal_indices] = internal_conditional_prediction
    predicted_support = np.clip(predicted_support, low, high)
    predicted_conditional = np.clip(predicted_conditional, low, high)
    predicted_strict = predicted_support * predicted_conditional

    old_value = np.asarray([float(shared_values[row["evidence_action_id"]]["shared_answer_value"]) for row in records])
    train_prediction = predicted_strict[train_indices]
    train_labels = strict[train_indices]
    old_train = old_value[train_indices]
    query_ids = [records[index]["query_id"] for index in train_indices]
    factor_pair = pair_order(train_prediction, train_labels, query_ids)
    old_pair = pair_order(old_train, train_labels, query_ids)
    groups: dict[str, list[int]] = defaultdict(list)
    for position, query_id in enumerate(query_ids):
        groups[query_id].append(position)
    nonconstant_rate = mean(float(max(train_prediction[indices]) - min(train_prediction[indices]) > 1e-8) for indices in groups.values())

    categorical_train = read_jsonl(Path(freeze["input_paths"]["categorical_train"]))
    index_by_evidence = {records[index]["evidence_action_id"]: index for index in train_indices}
    selector_rows = []
    for group in categorical_train:
        action_ids = group["action_ids"]
        if any(action_id not in index_by_evidence for action_id in action_ids):
            raise ValueError(f"categorical train action absent: {group['parent_group_id']}")
        indices = [index_by_evidence[action_id] for action_id in action_ids]
        factor_values = [float(predicted_strict[index]) for index in indices]
        factor_posterior = normalize([0.2 * max(value, 1e-8) for value in factor_values])
        old_values = [float(old_value[index]) for index in indices]
        methods = {
            "no_credit": group["behavior_probabilities"],
            "local_ig_m": group["local_target_probabilities"],
            "outcome": group["outcome_target_probabilities"],
            "old_dagig": group["dagig_target_probabilities"],
            "factorized_dagig": factor_posterior,
        }
        selected = {}
        for method, posterior in methods.items():
            selected_index = max(range(5), key=lambda index: (float(posterior[index]), -index))
            absolute_index = indices[selected_index]
            selected[method] = {
                "evidence_action_id": action_ids[selected_index],
                "strategy": records[absolute_index]["evidence_strategy"],
                "support": float(support[absolute_index]),
                "expected_strict": float(strict[absolute_index]),
                "old_terminal_value": old_values[selected_index],
            }
        selector_rows.append(selected)

    selector_summary = {}
    for method in ("no_credit", "local_ig_m", "outcome", "old_dagig", "factorized_dagig"):
        chosen = [row[method] for row in selector_rows]
        selector_summary[method] = {
            "states": len(chosen),
            "support": mean(row["support"] for row in chosen),
            "expected_strict": mean(row["expected_strict"] for row in chosen),
            "old_terminal_value": mean(row["old_terminal_value"] for row in chosen),
            "strategy_distribution": dict(sorted(Counter(row["strategy"] for row in chosen).items())),
        }

    brier = float(np.mean((train_prediction - train_labels) ** 2))
    old_brier = float(np.mean((old_train - train_labels) ** 2))
    metrics = {
        "train_actions": len(train_indices),
        "internal_actions_scored_without_labels": len(internal_indices),
        "support_positive_actions": int(support[train_indices].sum()),
        "support_head_oof_auc": auc(train_support_prediction, support[train_indices]),
        "support_head_oof_brier": float(np.mean((train_support_prediction - support[train_indices]) ** 2)),
        "conditional_head_oof_brier_on_support": float(np.mean((train_conditional_prediction[support[train_indices] == 1.0] - conditional[train_indices][support[train_indices] == 1.0]) ** 2)),
        "factorized_strict_oof_brier": brier,
        "old_value_strict_brier": old_brier,
        "strict_brier_improvement_vs_old": old_brier - brier,
        "factorized_strict_spearman": spearman(train_prediction, train_labels),
        "old_value_strict_spearman": spearman(old_train, train_labels),
        "strict_spearman_delta_vs_old": spearman(train_prediction, train_labels) - spearman(old_train, train_labels),
        "factorized_pair_order": factor_pair,
        "old_value_pair_order": old_pair,
        "pair_order_delta_vs_old": float(factor_pair["accuracy"]) - float(old_pair["accuracy"]),
        "nonconstant_query_group_rate": nonconstant_rate,
        "selector_train_oof": selector_summary,
    }
    threshold = freeze["train_oof_gates"]
    factor = selector_summary["factorized_dagig"]
    outcome = selector_summary["outcome"]
    gates = {
        "complete_train_oof": len(train_indices) == 11795 and np.isfinite(train_prediction).all(),
        "internal_scored_from_train_fit_only": len(internal_indices) == 2975 and np.isfinite(predicted_strict[internal_indices]).all(),
        "sample_grouped_repeated_cv": True,
        "support_head_auc": metrics["support_head_oof_auc"] >= threshold["support_auc_min"],
        "strict_brier_improves_old": metrics["strict_brier_improvement_vs_old"] >= threshold["strict_brier_improvement_vs_old_min"],
        "strict_spearman_improves_old": metrics["strict_spearman_delta_vs_old"] >= threshold["strict_spearman_delta_vs_old_min"],
        "pair_order_improves_old": metrics["pair_order_delta_vs_old"] >= threshold["pair_order_delta_vs_old_min"],
        "selected_support_noninferior_outcome": factor["support"] >= outcome["support"] - threshold["selected_support_noninferiority_vs_outcome_tolerance"],
        "selected_strict_noninferior_outcome": factor["expected_strict"] >= outcome["expected_strict"] - threshold["selected_strict_noninferiority_vs_outcome_tolerance"],
        "query_groups_nonconstant": nonconstant_rate >= threshold["nonconstant_query_group_rate_min"],
        "runtime_predictions_use_no_gold": True,
        "internal_labels_not_loaded": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {key: bool(value) for key, value in gates.items()}
    decision = "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_TRAIN_OOF_GO" if all(gates.values()) else "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_TRAIN_OOF_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "v6_factorized_evidence_state_predictions_no_eval_labels.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, p_support, p_conditional, p_strict in sorted(
            zip(records, predicted_support, predicted_conditional, predicted_strict), key=lambda item: item[0]["evidence_action_id"]
        ):
            handle.write(json.dumps({
                "evidence_action_id": row["evidence_action_id"],
                "query_id": row["query_id"],
                "sample_id": row["sample_id"],
                "partition": row["partition"],
                "evidence_strategy": row["evidence_strategy"],
                "support_probability": float(p_support),
                "answer_correct_given_support_probability": float(p_conditional),
                "evidence_success_probability": float(p_strict),
                "prediction_source": "repeated_sample_group_oof" if row["partition"] == "policy_train" else "policy_train_full_fit",
            }, sort_keys=True) + "\n")
    model_path = output / "v6_factorized_evidence_state_models.json"
    model_path.write_text(json.dumps({
        "feature_names": freeze["feature_names"],
        "support_head": serialize_model(support_model),
        "conditional_answer_head": serialize_model(conditional_model),
        "fit_partition": "policy_train_only",
        "equivalence_logit_used": False,
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
        "policy_train_private_labels_used_for_fit": True,
        "internal_private_labels_loaded": False,
        "gold_or_qrels_in_runtime_features": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": True,
    }
    audit_path = output / "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_TRAIN_OOF_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
