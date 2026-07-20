"""Gold-free descendant aggregation and scalar query-selector calibration."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


FORBIDDEN_PUBLIC_FIELDS = {
    "answer_correct",
    "child_expected_strict",
    "child_success_probability",
    "equivalence_logit",
    "gold_answer",
    "ground_truth",
    "oracle",
    "positive_doc_ids",
    "qrels",
    "strict_proxy",
    "success_posterior_probability",
    "target_doc",
}


def sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid used by the frozen support verifier."""

    clipped = max(-20.0, min(20.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def reader_confidence(mean_logprob: float) -> float:
    """Map a frozen reader candidate mean log-probability into [0, 1]."""

    return math.exp(max(-10.0, min(0.0, float(mean_logprob))))


def _normalized_weights(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    weights = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError(f"invalid {key} values")
    return weights / float(weights.sum())


def aggregate_query_descendants(
    answer_actions: Iterable[dict[str, Any]],
    evidence_actions: Iterable[dict[str, Any]],
    terminal_scores: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate support/reader signals under frozen answer/evidence behavior.

    ``equivalence_logit`` may exist in private terminal-score rows because the
    terminal teacher also needs it. This function deliberately never reads it.
    """

    answers_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answer_actions:
        answers_by_evidence[str(row["evidence_action_id"])].append(row)
    evidence_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_actions:
        evidence_by_query[str(row["query_id"])].append(row)

    output: dict[str, dict[str, float]] = {}
    for query_id, evidence_rows in evidence_by_query.items():
        evidence_weights = _normalized_weights(evidence_rows, "behavior_weight")
        branch_support: list[float] = []
        branch_reader: list[float] = []
        branch_product: list[float] = []
        branch_minimum: list[float] = []
        unknown_mass = 0.0
        answer_count = 0
        for evidence_weight, evidence_row in zip(evidence_weights, evidence_rows):
            answer_rows = answers_by_evidence.get(str(evidence_row["evidence_action_id"]), [])
            if not answer_rows:
                raise ValueError("evidence action has no answer descendants")
            answer_weights = _normalized_weights(answer_rows, "behavior_weight")
            support_values: list[float] = []
            reader_values: list[float] = []
            product_values: list[float] = []
            minimum_values: list[float] = []
            for answer_weight, answer_row in zip(answer_weights, answer_rows):
                action_id = str(answer_row["answer_action_id"])
                if action_id not in terminal_scores:
                    raise ValueError(f"missing terminal score: {action_id}")
                score = terminal_scores[action_id]
                support = sigmoid(float(score["support_logit"]))
                reader = reader_confidence(float(score["reader_candidate_mean_logprob"]))
                support_values.append(support)
                reader_values.append(reader)
                product_values.append(support * reader)
                minimum_values.append(min(support, reader))
                if bool(score.get("is_unknown")):
                    unknown_mass += float(evidence_weight * answer_weight)
            branch_support.append(float(answer_weights @ np.asarray(support_values)))
            branch_reader.append(float(answer_weights @ np.asarray(reader_values)))
            branch_product.append(float(answer_weights @ np.asarray(product_values)))
            branch_minimum.append(float(answer_weights @ np.asarray(minimum_values)))
            answer_count += len(answer_rows)

        support_backup = float(evidence_weights @ np.asarray(branch_support))
        reader_backup = float(evidence_weights @ np.asarray(branch_reader))
        product_backup = float(evidence_weights @ np.asarray(branch_product))
        minimum_backup = float(evidence_weights @ np.asarray(branch_minimum))
        output[query_id] = {
            "descendant_support_backup": support_backup,
            "descendant_reader_backup": reader_backup,
            "descendant_product_backup": product_backup,
            "descendant_minimum_backup": minimum_backup,
            "descendant_product_branch_max": max(branch_product),
            "descendant_product_branch_std": float(np.std(branch_product)),
            "descendant_unknown_behavior_mass": float(unknown_mass),
            "descendant_answer_count": float(answer_count),
            "descendant_evidence_count": float(len(evidence_rows)),
        }
    return output


def selector_probabilities(
    feature_values: Iterable[float],
    behavior_probabilities: Iterable[float],
    beta: float,
    feature_mean: float,
    feature_std: float,
) -> list[float]:
    """Return behavior-anchored listwise selector probabilities."""

    feature = np.asarray(list(feature_values), dtype=np.float64)
    behavior = np.asarray(list(behavior_probabilities), dtype=np.float64)
    if feature.size != behavior.size or feature.size == 0:
        raise ValueError("feature/behavior dimensions differ")
    if np.any(behavior <= 0.0):
        raise ValueError("behavior probabilities must be positive")
    behavior /= float(behavior.sum())
    scale = max(float(feature_std), 1e-12)
    logits = np.log(behavior) + float(beta) * (feature - float(feature_mean)) / scale
    logits -= float(logits.max())
    mass = np.exp(logits)
    return (mass / float(mass.sum())).tolist()


def fit_scalar_listwise(
    groups: list[dict[str, Any]],
    *,
    feature_key: str,
    target_key: str,
    l2: float,
    max_iterations: int,
    tolerance: float,
) -> dict[str, float | int]:
    """Fit one behavior-anchored scalar using convex listwise cross-entropy."""

    if not groups:
        raise ValueError("no selector groups to fit")
    feature_values = [float(value) for group in groups for value in group[feature_key]]
    feature_mean = float(np.mean(feature_values))
    feature_std = float(np.std(feature_values))
    if feature_std <= 1e-12:
        raise ValueError("constant descendant selector feature")
    beta = 0.0
    iterations = 0
    for iteration in range(int(max_iterations)):
        gradient = float(l2) * beta
        hessian = float(l2)
        for group in groups:
            feature = (np.asarray(group[feature_key], dtype=np.float64) - feature_mean) / feature_std
            behavior = np.asarray(group["behavior_probabilities"], dtype=np.float64)
            behavior /= float(behavior.sum())
            target = np.asarray(group[target_key], dtype=np.float64)
            target /= float(target.sum())
            predicted = np.asarray(
                selector_probabilities(feature, behavior, beta, 0.0, 1.0),
                dtype=np.float64,
            )
            predicted_mean = float(predicted @ feature)
            gradient += float((predicted - target) @ feature)
            hessian += float(predicted @ ((feature - predicted_mean) ** 2))
        update = gradient / max(hessian, 1e-12)
        beta -= update
        iterations = iteration + 1
        if abs(update) <= float(tolerance):
            break
    loss = 0.5 * float(l2) * beta * beta
    for group in groups:
        predicted = selector_probabilities(
            group[feature_key],
            group["behavior_probabilities"],
            beta,
            feature_mean,
            feature_std,
        )
        target = np.asarray(group[target_key], dtype=np.float64)
        target /= float(target.sum())
        loss -= float(target @ np.log(np.maximum(np.asarray(predicted), 1e-12)))
    return {
        "beta": float(beta),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "iterations": iterations,
        "objective": float(loss / len(groups)),
    }


def linear_selector_probabilities(
    feature_rows: Iterable[Iterable[float]],
    behavior_probabilities: Iterable[float],
    weights: Iterable[float],
    feature_mean: Iterable[float],
    feature_std: Iterable[float],
) -> list[float]:
    """Return behavior-anchored probabilities from a small linear critic."""

    features = np.asarray(list(feature_rows), dtype=np.float64)
    behavior = np.asarray(list(behavior_probabilities), dtype=np.float64)
    parameters = np.asarray(list(weights), dtype=np.float64)
    center = np.asarray(list(feature_mean), dtype=np.float64)
    scale = np.maximum(np.asarray(list(feature_std), dtype=np.float64), 1e-12)
    if features.ndim != 2 or features.shape[0] != behavior.size:
        raise ValueError("feature/behavior dimensions differ")
    if features.shape[1] != parameters.size or center.size != parameters.size or scale.size != parameters.size:
        raise ValueError("feature/model dimensions differ")
    if np.any(behavior <= 0.0):
        raise ValueError("behavior probabilities must be positive")
    behavior /= float(behavior.sum())
    logits = np.log(behavior) + ((features - center) / scale) @ parameters
    logits -= float(logits.max())
    mass = np.exp(logits)
    return (mass / float(mass.sum())).tolist()


def fit_linear_listwise(
    groups: list[dict[str, Any]],
    *,
    feature_key: str,
    target_key: str,
    l2: float,
    max_iterations: int,
    tolerance: float,
) -> dict[str, Any]:
    """Fit a convex behavior-anchored linear listwise calibrator."""

    if not groups:
        raise ValueError("no selector groups to fit")
    all_features = np.asarray(
        [row for group in groups for row in group[feature_key]],
        dtype=np.float64,
    )
    if all_features.ndim != 2 or all_features.shape[1] == 0:
        raise ValueError("invalid multivariate selector features")
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    if np.any(feature_std <= 1e-12):
        raise ValueError("constant multivariate selector feature")
    weights = np.zeros(all_features.shape[1], dtype=np.float64)
    iterations = 0
    for iteration in range(int(max_iterations)):
        gradient = float(l2) * weights
        hessian = float(l2) * np.eye(weights.size, dtype=np.float64)
        for group in groups:
            features = (np.asarray(group[feature_key], dtype=np.float64) - feature_mean) / feature_std
            behavior = np.asarray(group["behavior_probabilities"], dtype=np.float64)
            behavior /= float(behavior.sum())
            target = np.asarray(group[target_key], dtype=np.float64)
            target /= float(target.sum())
            logits = np.log(behavior) + features @ weights
            logits -= float(logits.max())
            predicted = np.exp(logits)
            predicted /= float(predicted.sum())
            gradient += features.T @ (predicted - target)
            centered = features - predicted @ features
            hessian += centered.T @ (predicted[:, None] * centered)
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        iterations = iteration + 1
        if float(np.linalg.norm(update)) <= float(tolerance):
            break
    loss = 0.5 * float(l2) * float(weights @ weights)
    for group in groups:
        predicted = linear_selector_probabilities(
            group[feature_key],
            group["behavior_probabilities"],
            weights,
            feature_mean,
            feature_std,
        )
        target = np.asarray(group[target_key], dtype=np.float64)
        target /= float(target.sum())
        loss -= float(target @ np.log(np.maximum(np.asarray(predicted), 1e-12)))
    return {
        "weights": weights.tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "iterations": iterations,
        "objective": float(loss / len(groups)),
    }
