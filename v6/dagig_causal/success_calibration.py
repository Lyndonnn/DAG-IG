"""Train-only grouped calibration utilities for frozen DAG success values."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def binary_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return float(
        -np.sum(
            labels * np.log(clipped)
            + (1.0 - labels) * np.log(1.0 - clipped)
        )
    )


def fit_platt(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    center = float(np.mean(values))
    scale = float(np.std(values))
    if not math.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    standardized = (values - center) / scale
    prevalence = float(np.clip(np.mean(labels), 1e-6, 1.0 - 1e-6))
    initial = np.asarray(
        [1.0, math.log(prevalence / (1.0 - prevalence))], dtype=np.float64
    )

    def objective(parameters: np.ndarray) -> float:
        probabilities = expit(parameters[0] * standardized + parameters[1])
        return binary_nll(probabilities, labels) + 1e-4 * float(parameters[0] ** 2)

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(0.0, None), (None, None)],
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Platt optimization failed: {result.message}")
    return {
        "type": "nonnegative_slope_platt",
        "center": center,
        "scale": scale,
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "optimizer": "scipy_L-BFGS-B",
    }


def predict_platt(model: dict[str, Any], values: np.ndarray) -> np.ndarray:
    standardized = (values - float(model["center"])) / float(model["scale"])
    return expit(float(model["slope"]) * standardized + float(model["intercept"]))


def probability_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def fit_logistic(
    features: np.ndarray, labels: np.ndarray, feature_names: list[str]
) -> dict[str, Any]:
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (features - center) / scale
    prevalence = float(np.clip(np.mean(labels), 1e-6, 1.0 - 1e-6))
    initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
    initial[-1] = math.log(prevalence / (1.0 - prevalence))

    def objective(parameters: np.ndarray) -> float:
        probabilities = expit(
            standardized @ parameters[:-1] + parameters[-1]
        )
        return binary_nll(probabilities, labels) + 1e-3 * float(
            np.sum(parameters[:-1] ** 2)
        )

    result = minimize(objective, initial, method="L-BFGS-B")
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"logistic optimization failed: {result.message}")
    return {
        "type": "standardized_logistic",
        "feature_names": feature_names,
        "center": [float(value) for value in center],
        "scale": [float(value) for value in scale],
        "weights": [float(value) for value in result.x[:-1]],
        "intercept": float(result.x[-1]),
        "optimizer": "scipy_L-BFGS-B",
    }


def predict_logistic(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    center = np.asarray(model["center"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    return expit(
        ((features - center) / scale) @ weights + float(model["intercept"])
    )


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    wins = 0.0
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return wins / (len(positives) * len(negatives))


def reliability(
    probabilities: np.ndarray, labels: np.ndarray, bins: int
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probabilities >= lower) & (
            probabilities < upper if index < bins - 1 else probabilities <= upper
        )
        count = int(np.sum(mask))
        if count:
            confidence = float(np.mean(probabilities[mask]))
            accuracy = float(np.mean(labels[mask]))
            ece += count / len(labels) * abs(confidence - accuracy)
        else:
            confidence = accuracy = None
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": confidence,
                "empirical_rate": accuracy,
            }
        )
    return ece, rows


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    bins: int,
) -> dict[str, Any]:
    ece, reliability_rows = reliability(probabilities, labels, bins)
    return {
        "n": int(len(labels)),
        "positives": int(np.sum(labels)),
        "base_rate": float(np.mean(labels)),
        "auc": auc(probabilities, labels),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "fold_base_rate_brier": float(
            np.mean((base_probabilities - labels) ** 2)
        ),
        "ece": float(ece),
        "probability_min_mean_max": [
            float(np.min(probabilities)),
            float(np.mean(probabilities)),
            float(np.max(probabilities)),
        ],
        "reliability_bins": reliability_rows,
    }
