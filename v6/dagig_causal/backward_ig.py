"""Exact backward values and information identities for a finite DAG policy.

The functions in this module operate on success probabilities.  A parent value
is the behavior-policy expectation of its child values.  The success
conditioned edge credit is therefore the pointwise information gain

    log P(success | child) - log P(success | parent).

For a complete path these edge credits telescope exactly.  The group-level
binary mutual information also accounts for the complementary failure event.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def normalized_mass(weights: Sequence[float]) -> list[float]:
    if len(weights) < 2:
        raise ValueError("An intervention group needs at least two actions")
    values = [float(value) for value in weights]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Behavior weights must be finite and positive")
    total = sum(values)
    return [value / total for value in values]


def validate_probability(value: float) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("Success probabilities must be finite and strictly inside (0, 1)")
    return probability


def behavior_backup(child_values: Sequence[float], weights: Sequence[float]) -> float:
    """Return P(success | parent) under the finite behavior-policy mass."""

    if len(child_values) != len(weights):
        raise ValueError("Child values and behavior weights do not align")
    probabilities = [validate_probability(value) for value in child_values]
    mass = normalized_mass(weights)
    return validate_probability(sum(weight * value for weight, value in zip(mass, probabilities)))


def success_edge_ig(child_value: float, parent_value: float) -> float:
    child = validate_probability(child_value)
    parent = validate_probability(parent_value)
    return math.log(child) - math.log(parent)


def failure_edge_ig(child_value: float, parent_value: float) -> float:
    child = validate_probability(child_value)
    parent = validate_probability(parent_value)
    return math.log1p(-child) - math.log1p(-parent)


def success_posterior(
    child_values: Sequence[float], weights: Sequence[float]
) -> list[float]:
    """Return P(action | success, parent) on the finite action support."""

    probabilities = [validate_probability(value) for value in child_values]
    mass = normalized_mass(weights)
    parent = behavior_backup(probabilities, mass)
    posterior = [weight * value / parent for weight, value in zip(mass, probabilities)]
    normalizer = sum(posterior)
    return [value / normalizer for value in posterior]


def failure_posterior(
    child_values: Sequence[float], weights: Sequence[float]
) -> list[float]:
    probabilities = [validate_probability(value) for value in child_values]
    mass = normalized_mass(weights)
    parent = behavior_backup(probabilities, mass)
    posterior = [
        weight * (1.0 - value) / (1.0 - parent)
        for weight, value in zip(mass, probabilities)
    ]
    normalizer = sum(posterior)
    return [value / normalizer for value in posterior]


def kl_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("KL distributions do not align")
    p = normalized_mass(left)
    q = normalized_mass(right)
    return sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q))


def binary_mutual_information(
    child_values: Sequence[float], weights: Sequence[float]
) -> float:
    """Return I(A; success | parent) in nats for one intervention group."""

    probabilities = [validate_probability(value) for value in child_values]
    mass = normalized_mass(weights)
    parent = behavior_backup(probabilities, mass)
    success_kl = kl_divergence(success_posterior(probabilities, mass), mass)
    failure_kl = kl_divergence(failure_posterior(probabilities, mass), mass)
    value = parent * success_kl + (1.0 - parent) * failure_kl
    if value < -1e-12:
        raise ArithmeticError("Binary mutual information became negative")
    return max(0.0, value)


def information_identity_error(
    child_values: Sequence[float], weights: Sequence[float]
) -> float:
    """Check E[success edge IG | success] = KL(pi(.|success) || pi)."""

    probabilities = [validate_probability(value) for value in child_values]
    mass = normalized_mass(weights)
    parent = behavior_backup(probabilities, mass)
    posterior = success_posterior(probabilities, mass)
    expected_ig = sum(
        probability * success_edge_ig(child, parent)
        for probability, child in zip(posterior, probabilities)
    )
    return abs(expected_ig - kl_divergence(posterior, mass))


def telescoping_error(
    root_value: float, leaf_value: float, edge_credits: Sequence[float]
) -> float:
    root = validate_probability(root_value)
    leaf = validate_probability(leaf_value)
    target = math.log(leaf) - math.log(root)
    return abs(sum(float(value) for value in edge_credits) - target)
