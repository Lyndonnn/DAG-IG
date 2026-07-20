"""Grouped cross-fitting for generated-answer supported-success values."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from typing import Any

import numpy as np

from .success_calibration import (
    calibration_metrics,
    fit_logistic,
    fit_platt,
    predict_logistic,
    predict_platt,
    probability_logit,
)


FEATURE_NAMES = [
    "generated_answer_equivalence_logit",
    "generated_answer_support_logit",
    "reader_gold_mean_logprob",
]


def sample_fold(sample_id: str, folds: int, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()[:16], 16) % folds


def pairwise_summary(sample_ids: list[str], labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        grouped[sample_id].append(index)
    wins = ties = losses = 0
    ranges: list[float] = []
    for indices in grouped.values():
        values = [float(probabilities[index]) for index in indices]
        ranges.append(max(values) - min(values))
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                if labels[left] == labels[right]:
                    continue
                signed = (probabilities[left] - probabilities[right]) * (labels[left] - labels[right])
                if signed > 0:
                    wins += 1
                elif signed < 0:
                    losses += 1
                else:
                    ties += 1
    total = wins + ties + losses
    return {
        "comparable_within_parent_pairs": total,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "accuracy_ties_half": (wins + 0.5 * ties) / total if total else None,
        "parent_value_range_mean": statistics.mean(ranges),
        "parent_value_range_median": statistics.median(ranges),
        "parents_value_range_gt_0_01_rate": sum(value > 0.01 for value in ranges) / len(ranges),
        "constant_probability_parents_at_1e_8": sum(value <= 1e-8 for value in ranges),
    }


def crossfit_generated_answer_value(
    *,
    sample_ids: list[str],
    equivalence_logits: np.ndarray,
    support_logits: np.ndarray,
    reader_means: np.ndarray,
    answer_labels: np.ndarray,
    strict_labels: np.ndarray,
    folds: int,
    fold_seed: int,
    reliability_bins: int,
) -> dict[str, Any]:
    fold_values = np.asarray([sample_fold(value, folds, fold_seed) for value in sample_ids], dtype=np.int64)
    sample_folds: dict[str, set[int]] = defaultdict(set)
    for sample_id, fold in zip(sample_ids, fold_values):
        sample_folds[sample_id].add(int(fold))
    if any(len(values) != 1 for values in sample_folds.values()) or set(fold_values.tolist()) != set(range(folds)):
        raise ValueError("sample-level fold assignment is invalid")
    n = len(sample_ids)
    equivalence_oof = np.full(n, np.nan)
    support_oof = np.full(n, np.nan)
    success_oof = np.full(n, np.nan)
    equivalence_base = np.full(n, np.nan)
    support_base = np.full(n, np.nan)
    success_base = np.full(n, np.nan)
    fold_models: list[dict[str, Any]] = []
    for outer in range(folds):
        train = np.where(fold_values != outer)[0]
        valid = np.where(fold_values == outer)[0]
        train_folds = fold_values[train]
        inner_equivalence = np.full(len(train), np.nan)
        inner_support = np.full(len(train), np.nan)
        for inner in sorted(set(int(value) for value in train_folds)):
            inner_valid = np.where(train_folds == inner)[0]
            inner_train = np.where(train_folds != inner)[0]
            equivalence_model = fit_platt(equivalence_logits[train[inner_train]], answer_labels[train[inner_train]])
            support_model = fit_platt(support_logits[train[inner_train]], strict_labels[train[inner_train]])
            inner_equivalence[inner_valid] = predict_platt(equivalence_model, equivalence_logits[train[inner_valid]])
            inner_support[inner_valid] = predict_platt(support_model, support_logits[train[inner_valid]])
        equivalence_model = fit_platt(equivalence_logits[train], answer_labels[train])
        support_model = fit_platt(support_logits[train], strict_labels[train])
        valid_equivalence = predict_platt(equivalence_model, equivalence_logits[valid])
        valid_support = predict_platt(support_model, support_logits[valid])
        equivalence_oof[valid] = valid_equivalence
        support_oof[valid] = valid_support
        strict_model = fit_logistic(
            np.column_stack([probability_logit(inner_equivalence), probability_logit(inner_support), reader_means[train]]),
            strict_labels[train],
            FEATURE_NAMES,
        )
        success_oof[valid] = predict_logistic(
            strict_model,
            np.column_stack([probability_logit(valid_equivalence), probability_logit(valid_support), reader_means[valid]]),
        )
        equivalence_base[valid] = float(np.mean(answer_labels[train]))
        support_base[valid] = float(np.mean(strict_labels[train]))
        success_base[valid] = float(np.mean(strict_labels[train]))
        train_samples = {sample_ids[index] for index in train}
        valid_samples = {sample_ids[index] for index in valid}
        if train_samples & valid_samples:
            raise ValueError("outer-fold sample leakage")
        fold_models.append({
            "fold": outer,
            "sample_overlap": 0,
            "equivalence_model": equivalence_model,
            "generated_support_model": support_model,
            "strict_success_model": strict_model,
        })
    if any(np.any(~np.isfinite(values)) for values in [equivalence_oof, support_oof, success_oof]):
        raise ValueError("generated-answer cross-fit is incomplete")
    full_equivalence_model = fit_platt(equivalence_logits, answer_labels)
    full_support_model = fit_platt(support_logits, strict_labels)
    full_equivalence = predict_platt(full_equivalence_model, equivalence_logits)
    full_support = predict_platt(full_support_model, support_logits)
    full_strict_model = fit_logistic(
        np.column_stack([probability_logit(full_equivalence), probability_logit(full_support), reader_means]),
        strict_labels,
        FEATURE_NAMES,
    )
    full_success = predict_logistic(
        full_strict_model,
        np.column_stack([probability_logit(full_equivalence), probability_logit(full_support), reader_means]),
    )
    return {
        "fold_values": fold_values,
        "equivalence_oof": equivalence_oof,
        "support_oof": support_oof,
        "success_oof": success_oof,
        "full_success": full_success,
        "equivalence_metrics": calibration_metrics(equivalence_oof, answer_labels, equivalence_base, reliability_bins),
        "support_metrics": calibration_metrics(support_oof, strict_labels, support_base, reliability_bins),
        "strict_metrics": calibration_metrics(success_oof, strict_labels, success_base, reliability_bins),
        "pairwise": pairwise_summary(sample_ids, strict_labels, success_oof),
        "parameters": {
            "equivalence_model_full_fit": full_equivalence_model,
            "generated_support_model_full_fit": full_support_model,
            "strict_success_model_full_fit": full_strict_model,
            "outer_fold_models": fold_models,
            "feature_contract": FEATURE_NAMES,
        },
    }
