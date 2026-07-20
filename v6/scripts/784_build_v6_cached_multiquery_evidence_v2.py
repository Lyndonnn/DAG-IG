#!/usr/bin/env python3
"""Build clean five-action multi-query evidence-v2 posterior targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METHOD_KEYS = {
    "no_credit": "behavior_probabilities",
    "local_ig": "local_target_probabilities",
    "outcome": "outcome_target_probabilities",
    "dagig": "dagig_target_probabilities",
}
STRATEGY_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)
ACTION_LABELS = ("A", "B", "C", "D", "E")
ALLOWED_PUBLIC_KEYS = {"parent_state_id", "prompt", "actions", "target_distributions"}
FORBIDDEN_FIELD_TOKENS = (
    "gold",
    "qrel",
    "strict",
    "support_label",
    "evidence_hit",
    "answer_correct",
    "target_doc",
    "ground_truth",
    "success_probability",
    "terminal_value",
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


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} changed: expected {expected}, found {actual}: {path}")


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("cannot normalize target distribution")
    return [value / total for value in values]


def entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def tv(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def nested_field_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            names.add(str(key).lower())
            names.update(nested_field_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(nested_field_names(child))
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol_freeze.resolve()
    protocol = read_json(protocol_path)
    if protocol.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_FROZEN":
        raise ValueError("cached multi-query evidence v2 protocol is not frozen")
    for key, raw_path in protocol["input_paths"].items():
        assert_hash(Path(raw_path), protocol["input_hashes"][key], key)
    for key, raw_path in protocol["output_paths"].items():
        assert_hash(Path(raw_path), protocol["output_hashes"][key], key)

    state_path = Path(protocol["output_paths"]["state_ids"])
    states = read_jsonl(state_path)
    state_map = {row["parent_state_id"]: row for row in states}
    if len(state_map) != 1184:
        raise ValueError(f"expected 1184 frozen query states, found {len(state_map)}")

    input_paths = protocol["input_paths"]
    action_rows = read_jsonl(Path(input_paths["evidence_actions"]))
    action_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        if row["query_id"] in state_map:
            action_groups[row["query_id"]].append(row)
    values = {
        row["evidence_action_id"]: float(row["shared_answer_value"])
        for row in read_jsonl(Path(input_paths["shared_answer_values"]))
    }
    categorical_rows = (
        read_jsonl(Path(input_paths["categorical_train"]))
        + read_jsonl(Path(input_paths["categorical_internal"]))
    )
    categorical = {
        row["parent_group_id"]: row
        for row in categorical_rows
        if row["parent_group_id"] in state_map
    }
    if set(categorical) != set(state_map) or set(action_groups) != set(state_map):
        raise ValueError("frozen state, action, and target universes differ")

    output_rows: dict[str, list[dict[str, Any]]] = {"policy_train": [], "internal_holdout": []}
    max_normalization_error = 0.0
    max_dag_identity_error = 0.0
    max_action_mapping_error = 0
    method_entropies: dict[str, list[float]] = {method: [] for method in METHOD_KEYS}
    method_top_counts: dict[str, Counter[str]] = {method: Counter() for method in METHOD_KEYS}
    dag_outcome_tvs: list[float] = []
    dag_local_tvs: list[float] = []

    for state_id in sorted(state_map):
        state = state_map[state_id]
        source = categorical[state_id]
        rows = sorted(action_groups[state_id], key=lambda row: STRATEGY_ORDER.index(row["evidence_strategy"]))
        if tuple(row["evidence_strategy"] for row in rows) != STRATEGY_ORDER:
            raise ValueError(f"A-E strategy order changed for {state_id}")
        if source["categorical_action_labels"] != list(ACTION_LABELS):
            raise ValueError(f"categorical labels changed for {state_id}")
        if source["partition"] != state["partition"]:
            raise ValueError(f"partition mismatch for {state_id}")

        legal_actions: list[dict[str, Any]] = []
        for index, (label, strategy, action_row) in enumerate(zip(ACTION_LABELS, STRATEGY_ORDER, rows)):
            completion = json.loads(source["completions"][index])
            selected = source["selected_evidence_ids_by_action"][label]
            if completion != {"action": label}:
                raise ValueError(f"categorical completion changed for {state_id}::{label}")
            candidate_docs = sorted(action_row["candidate_docs"], key=lambda doc: int(doc["rank"]))
            id_map = {doc["doc_id"]: f"D{position}" for position, doc in enumerate(candidate_docs, 1)}
            expected_selected = [id_map[doc_id] for doc_id in action_row["selected_doc_ids"]]
            max_action_mapping_error += int(selected != expected_selected)
            legal_actions.append(
                {
                    "label": label,
                    "strategy": strategy,
                    "selected_evidence_ids": selected,
                    "completion": source["completions"][index],
                }
            )
        if len({tuple(action["selected_evidence_ids"]) for action in legal_actions}) != 5:
            raise ValueError(f"interventions are not unique for {state_id}")

        distributions = {
            method: [float(value) for value in source[key]]
            for method, key in METHOD_KEYS.items()
        }
        for method, distribution in distributions.items():
            max_normalization_error = max(max_normalization_error, abs(sum(distribution) - 1.0))
            if len(distribution) != 5 or any(value < 0.0 or not math.isfinite(value) for value in distribution):
                raise ValueError(f"invalid {method} target for {state_id}")
            method_entropies[method].append(entropy(distribution))
            top_index = max(range(5), key=distribution.__getitem__)
            method_top_counts[method][STRATEGY_ORDER[top_index]] += 1

        behavior = distributions["no_credit"]
        child_values = [values[row["evidence_action_id"]] for row in rows]
        exact_dag = normalize([probability * max(value, 1e-8) for probability, value in zip(behavior, child_values)])
        max_dag_identity_error = max(
            max_dag_identity_error,
            max(abs(left - right) for left, right in zip(exact_dag, distributions["dagig"])),
        )
        dag_outcome_tvs.append(tv(distributions["dagig"], distributions["outcome"]))
        dag_local_tvs.append(tv(distributions["dagig"], distributions["local_ig"]))

        public_row = {
            "parent_state_id": state_id,
            "prompt": source["prompt"],
            "actions": legal_actions,
            "target_distributions": distributions,
        }
        if set(public_row) != ALLOWED_PUBLIC_KEYS:
            raise ValueError(f"public schema changed for {state_id}")
        forbidden = sorted(
            field for field in nested_field_names(public_row)
            if any(token in field for token in FORBIDDEN_FIELD_TOKENS)
        )
        if forbidden:
            raise ValueError(f"public target row exposes evaluation fields: {state_id}: {forbidden}")
        output_rows[state["partition"]].append(public_row)

    gates = {
        "exact_946_train_states": len(output_rows["policy_train"]) == 946,
        "exact_238_internal_states": len(output_rows["internal_holdout"]) == 238,
        "five_unique_legal_actions_per_state": all(len(row["actions"]) == 5 and len({tuple(action["selected_evidence_ids"]) for action in row["actions"]}) == 5 for rows in output_rows.values() for row in rows),
        "exact_A_E_action_mapping": max_action_mapping_error == 0,
        "target_distributions_normalized": max_normalization_error <= 1e-10,
        "exact_dag_information_posterior_identity": max_dag_identity_error <= 1e-12,
        "public_schema_minimal": all(set(row) == ALLOWED_PUBLIC_KEYS for rows in output_rows.values() for row in rows),
        "public_files_have_no_evaluation_fields": True,
        "internal_unused_for_fit_or_tuning": True,
        "new_search_calls_zero": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    if not all(gates.values()):
        raise ValueError(f"cached multi-query evidence v2 build failed: {gates}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "v6_cached_multiquery_evidence_v2_targets_train.jsonl"
    internal_path = output_dir / "v6_cached_multiquery_evidence_v2_targets_internal_no_labels.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows["policy_train"]),
        encoding="utf-8",
    )
    internal_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows["internal_holdout"]),
        encoding="utf-8",
    )

    metrics = {
        "samples": len({row["parent_state_id"].split("::", 1)[0] for rows in output_rows.values() for row in rows}),
        "query_states": sum(len(rows) for rows in output_rows.values()),
        "policy_train_states": len(output_rows["policy_train"]),
        "internal_holdout_states": len(output_rows["internal_holdout"]),
        "evidence_actions": sum(len(row["actions"]) for rows in output_rows.values() for row in rows),
        "mean_target_entropy": {method: mean(values_) for method, values_ in method_entropies.items()},
        "top_action_strategy_distribution": {method: dict(sorted(counts.items())) for method, counts in method_top_counts.items()},
        "dagig_outcome_mean_tv": mean(dag_outcome_tvs),
        "dagig_local_mean_tv": mean(dag_local_tvs),
        "max_normalization_error": max_normalization_error,
        "max_dag_information_identity_error": max_dag_identity_error,
    }
    result = {
        "decision": "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_TARGETS_GO",
        "protocol_version": protocol["protocol_version"],
        "metrics": metrics,
        "gates": gates,
        "public_schema": {
            "top_level_fields": sorted(ALLOWED_PUBLIC_KEYS),
            "target_methods": list(METHOD_KEYS),
            "evaluation_only_fields_present": [],
        },
        "input_paths": {"protocol_freeze": str(protocol_path), **{key: value for key, value in input_paths.items() if key in {"evidence_actions", "shared_answer_values", "categorical_train", "categorical_internal"}}},
        "input_hashes": {"protocol_freeze": sha256(protocol_path), **{key: protocol["input_hashes"][key] for key in ("evidence_actions", "shared_answer_values", "categorical_train", "categorical_internal")}},
        "output_paths": {"train_targets": str(train_path), "internal_targets": str(internal_path)},
        "output_hashes": {"train_targets": sha256(train_path), "internal_targets": sha256(internal_path)},
        "gold_or_qrels_loaded": False,
        "private_evaluation_labels_loaded": False,
        "internal_holdout_used_for_training_or_tuning": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_TARGET_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "metrics": metrics, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
