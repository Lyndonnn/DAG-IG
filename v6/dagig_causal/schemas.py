"""Validation helpers for DAG-IG Causal v1 structured artifacts."""

from __future__ import annotations

from typing import Any


CREDITED_NODES = (
    "visual_action",
    "search_query",
    "evidence_action",
    "answer_hypothesis",
)

DESCENDANTS = {
    "visual_action": [
        "grounding_tool",
        "search_query",
        "retrieval",
        "evidence_action",
        "answer_hypothesis",
        "final_answer",
    ],
    "search_query": [
        "retrieval",
        "evidence_action",
        "answer_hypothesis",
        "final_answer",
    ],
    "evidence_action": ["answer_hypothesis", "final_answer"],
    "answer_hypothesis": ["final_answer"],
}

FORBIDDEN_POLICY_KEY_FRAGMENTS = (
    "gold",
    "oracle",
    "teacher",
    "positive_doc",
    "qrel",
    "target_doc",
    "support_label",
    "answer_correct",
    "strict_success",
)


def _walk_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_walk_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            keys.extend(_walk_keys(child, f"{prefix}[{index}]"))
    return keys


def forbidden_policy_keys(policy_input: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    for path in _walk_keys(policy_input):
        lowered = path.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_POLICY_KEY_FRAGMENTS):
            bad.append(path)
    return sorted(set(bad))


def validate_rollout_task(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("task_id", "sample_id", "split", "rollout_index", "paired_seed", "policy_input", "environment"):
        if key not in row:
            errors.append(f"missing:{key}")
    if row.get("split") not in {"train", "dev"}:
        errors.append("invalid_or_forbidden_split")
    if not isinstance(row.get("policy_input"), dict):
        errors.append("policy_input_not_object")
    else:
        errors.extend(f"forbidden_policy_key:{key}" for key in forbidden_policy_keys(row["policy_input"]))
    if row.get("credited_nodes") != list(CREDITED_NODES):
        errors.append("credited_nodes_mismatch")
    if row.get("protocol_version") != "dagig_causal_v1":
        errors.append("protocol_version_mismatch")
    return errors


def validate_counterfactual(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    node = row.get("intervened_node")
    if node not in CREDITED_NODES:
        errors.append("invalid_intervened_node")
    if not row.get("parent_state_id"):
        errors.append("missing_parent_state_id")
    if not row.get("actual_action_id"):
        errors.append("missing_actual_action_id")
    if not row.get("counterfactual_action_id"):
        errors.append("missing_counterfactual_action_id")
    if node in DESCENDANTS and row.get("required_descendant_replay") != DESCENDANTS[node]:
        errors.append("descendant_replay_mismatch")
    if row.get("paired_seed") is None:
        errors.append("missing_paired_seed")
    if row.get("counterfactual_action_seed") is None:
        errors.append("missing_counterfactual_action_seed")
    if row.get("baseline_role") not in {"main_policy_baseline", "diagnostic_control"}:
        errors.append("invalid_or_missing_baseline_role")
    return errors
