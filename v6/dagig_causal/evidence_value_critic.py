"""Leakage-safe utilities for the DAG-IG evidence action-value critic.

The critic observes only the causal parent state and the semantic content of one
executable evidence-set action.  Action IDs, strategy names, retrieval ranks,
gold answers, qrels, and verifier outputs are deliberately absent from the
encoder text.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable


FORBIDDEN_TRAINING_FIELDS = {
    "aliases",
    "answer_correct",
    "evidence_support",
    "evidence_support_proxy",
    "gold_answer",
    "ground_truth",
    "positive_doc_ids",
    "qrels",
    "strict_proxy",
    "strict_success",
    "support_label",
    "target_doc",
}


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_mass(values: Iterable[float]) -> list[float]:
    numbers = [float(value) for value in values]
    total = sum(numbers)
    if not numbers or not math.isfinite(total) or total <= 0:
        raise ValueError("cannot normalize empty or non-positive probability mass")
    result = [value / total for value in numbers]
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("DAG posterior must have finite positive support")
    return result


def success_posterior(behavior: Iterable[float], values: Iterable[float]) -> list[float]:
    behavior_values = [float(value) for value in behavior]
    success_values = [float(value) for value in values]
    if len(behavior_values) != len(success_values):
        raise ValueError("behavior and value vectors differ in length")
    return normalize_mass(pi * value for pi, value in zip(behavior_values, success_values))


def recover_action_values(
    behavior: Iterable[float], posterior: Iterable[float], parent_value: float
) -> list[float]:
    behavior_values = [float(value) for value in behavior]
    posterior_values = [float(value) for value in posterior]
    if len(behavior_values) != len(posterior_values):
        raise ValueError("behavior and posterior vectors differ in length")
    values = [q * float(parent_value) / pi for pi, q in zip(behavior_values, posterior_values)]
    if any(not 0 < value < 1 for value in values):
        raise ValueError("recovered terminal success value is outside (0, 1)")
    return values


def state_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Question: {clean(row['question'])}",
            f"Visual observation: {clean(row['visual_observation'])}",
            f"Search query: {clean(row['search_query'])}",
            f"Requested answer type: {clean(row.get('inferred_answer_type')) or 'other'}",
        ]
    )


def _semantic_document_text(doc: dict[str, Any], max_chars: int) -> str:
    text = "\n".join(
        [
            f"Title: {clean(doc.get('title'))}",
            f"Domain: {clean(doc.get('domain'))}",
            f"Evidence: {clean(doc.get('snippet'))}",
        ]
    )
    return text[:max_chars]


def action_text(row: dict[str, Any], max_chars_per_doc: int) -> str:
    """Return a permutation-invariant semantic representation of an action."""

    documents = [_semantic_document_text(doc, max_chars_per_doc) for doc in row["selected_docs"]]
    documents.sort(key=lambda value: (value.casefold(), value))
    return "\n\n--- evidence document ---\n\n".join(documents)


def deterministic_sample_folds(sample_ids: Iterable[str], folds: int, seed: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("at least two folds are required")
    unique = sorted(set(str(sample_id) for sample_id in sample_ids))
    ordered = sorted(
        unique,
        key=lambda sample_id: hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest(),
    )
    return {sample_id: index % folds for index, sample_id in enumerate(ordered)}


def local_observable_score(row: dict[str, Any]) -> float:
    values = [
        0.65 * float(doc["normalized_bge_score"])
        + 0.20 * float(doc["question_keyword_overlap"])
        + 0.15 * float(doc["answer_type_pattern_match"])
        for doc in row["selected_docs"]
    ]
    diversity = len({clean(doc.get("domain")).casefold() for doc in row["selected_docs"]}) / len(values)
    return sum(values) / len(values) + 0.10 * diversity
