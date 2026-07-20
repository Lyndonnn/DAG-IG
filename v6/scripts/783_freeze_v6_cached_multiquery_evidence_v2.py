#!/usr/bin/env python3
"""Freeze the clean cached multi-query evidence-v2 selector protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRATEGY_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)
ACTION_LABELS = ("A", "B", "C", "D", "E")
FORBIDDEN_RUNTIME_FIELDS = (
    "gold",
    "qrel",
    "strict",
    "support_label",
    "evidence_hit",
    "answer_correct",
    "target_doc",
    "ground_truth",
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
    parser.add_argument("--v1_protocol_freeze", type=Path, required=True)
    parser.add_argument("--v1_excluded", type=Path, required=True)
    parser.add_argument("--evidence_action_audit", type=Path, required=True)
    parser.add_argument("--shared_answer_value_audit", type=Path, required=True)
    parser.add_argument("--terminal_value_audit", type=Path, required=True)
    parser.add_argument("--control_freeze", type=Path, required=True)
    parser.add_argument("--categorical_freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_paths = {
        "v1_protocol_freeze": args.v1_protocol_freeze.resolve(),
        "evidence_action_audit": args.evidence_action_audit.resolve(),
        "shared_answer_value_audit": args.shared_answer_value_audit.resolve(),
        "terminal_value_audit": args.terminal_value_audit.resolve(),
        "control_freeze": args.control_freeze.resolve(),
        "categorical_freeze": args.categorical_freeze.resolve(),
    }
    manifests = {key: read_json(path) for key, path in manifest_paths.items()}
    expected_decisions = {
        "v1_protocol_freeze": "DAGIG_V6_ON_POLICY_EVIDENCE_ACTIONS_FROZEN",
        "evidence_action_audit": "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_ACTIONS_GO",
        "shared_answer_value_audit": "DAGIG_V6_SHARED_ANSWER_VALUES_GO",
        "terminal_value_audit": "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO",
        "control_freeze": "DAGIG_V6_BACKWARD_EVIDENCE_CONTROLS_FROZEN",
        "categorical_freeze": "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN",
    }
    for key, expected in expected_decisions.items():
        if manifests[key].get("decision") != expected:
            raise ValueError(f"{key} is not frozen/GO: {manifests[key].get('decision')}")

    v1_freeze = manifests["v1_protocol_freeze"]
    action_audit = manifests["evidence_action_audit"]
    value_audit = manifests["shared_answer_value_audit"]
    terminal_audit = manifests["terminal_value_audit"]
    control_freeze = manifests["control_freeze"]
    categorical_freeze = manifests["categorical_freeze"]

    v1_parents_path = Path(v1_freeze["input_paths"]["query_actions_with_search"]).resolve()
    evidence_actions_path = Path(action_audit["output_paths"]["evidence_actions"]).resolve()
    private_support_path = Path(action_audit["output_paths"]["private_support"]).resolve()
    shared_values_path = Path(value_audit["output_paths"]["shared_answer_values"]).resolve()
    terminal_private_path = Path(terminal_audit["output_paths"]["private_audit"]).resolve()
    control_train_path = Path(control_freeze["output_paths"]["train_data"]).resolve()
    control_internal_path = Path(control_freeze["output_paths"]["internal_data"]).resolve()
    categorical_train_path = Path(categorical_freeze["input_paths"]["categorical_train_data"]).resolve()
    categorical_internal_path = Path(categorical_freeze["input_paths"]["categorical_internal_data"]).resolve()

    assert_hash(v1_parents_path, v1_freeze["input_hashes"]["query_actions_with_search"], "v1 cached parents")
    assert_hash(evidence_actions_path, action_audit["output_hashes"]["evidence_actions"], "clean evidence actions")
    assert_hash(private_support_path, action_audit["output_hashes"]["private_support"], "private support")
    assert_hash(shared_values_path, value_audit["output_hashes"]["shared_answer_values"], "shared answer values")
    assert_hash(terminal_private_path, terminal_audit["output_hashes"]["private_audit"], "terminal private audit")
    assert_hash(control_train_path, control_freeze["output_hashes"]["train_data"], "control train")
    assert_hash(control_internal_path, control_freeze["output_hashes"]["internal_data"], "control internal")
    assert_hash(categorical_train_path, categorical_freeze["input_hashes"]["categorical_train_data"], "categorical train")
    assert_hash(categorical_internal_path, categorical_freeze["input_hashes"]["categorical_internal_data"], "categorical internal")

    parents = {row["query_id"]: row for row in read_jsonl(v1_parents_path)}
    if len(parents) != 1186:
        raise ValueError(f"expected 1186 complete cached v1 parents, found {len(parents)}")
    if any(len(row.get("retrieved_docs") or []) < 5 for row in parents.values()):
        raise ValueError("v1 parent universe contains an incomplete cached search state")

    action_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(evidence_actions_path):
        if row["query_id"] in parents:
            action_groups[row["query_id"]].append(row)
    value_ids = {row["evidence_action_id"] for row in read_jsonl(shared_values_path)}
    target_rows = read_jsonl(control_train_path) + read_jsonl(control_internal_path)
    target_ids = {row["parent_group_id"] for row in target_rows}
    categorical_rows = read_jsonl(categorical_train_path) + read_jsonl(categorical_internal_path)
    categorical_ids = {row["parent_group_id"] for row in categorical_rows}

    eligible: list[dict[str, Any]] = []
    missing_clean: list[dict[str, Any]] = []
    for query_id, parent in sorted(parents.items()):
        rows = action_groups.get(query_id, [])
        action_ids = {row["evidence_action_id"] for row in rows}
        reasons = []
        if len(rows) != 5:
            reasons.append(f"clean_action_count_{len(rows)}")
        if rows and {row["evidence_strategy"] for row in rows} != set(STRATEGY_ORDER):
            reasons.append("incomplete_A_E_strategy_set")
        if not action_ids.issubset(value_ids):
            reasons.append("missing_frozen_shared_answer_value")
        if query_id not in target_ids:
            reasons.append("missing_clean_control_target")
        if query_id not in categorical_ids:
            reasons.append("missing_categorical_action_mapping")
        if reasons:
            missing_clean.append(
                {
                    "parent_state_id": query_id,
                    "partition": parent["partition"],
                    "reason": "absent_from_complete_frozen_no_gold_universe",
                    "details": reasons,
                }
            )
            continue

        rows.sort(key=lambda row: STRATEGY_ORDER.index(row["evidence_strategy"]))
        parent_urls = [doc.get("url", "") for doc in parent["retrieved_docs"]]
        action_urls = [doc.get("url", "") for doc in rows[0]["candidate_docs"]]
        if parent["search_query"] != rows[0]["search_query"] or parent_urls != action_urls:
            raise ValueError(f"cached search state changed for {query_id}")
        if any(row["candidate_docs"] != rows[0]["candidate_docs"] for row in rows[1:]):
            raise ValueError(f"candidate universe differs across interventions for {query_id}")
        if len({tuple(row["selected_doc_ids"]) for row in rows}) != 5:
            raise ValueError(f"evidence interventions are not unique for {query_id}")
        public_fields = nested_field_names(rows)
        forbidden = sorted(
            field for field in public_fields
            if any(token in field for token in FORBIDDEN_RUNTIME_FIELDS)
        )
        if forbidden:
            raise ValueError(f"runtime evidence actions expose forbidden fields: {query_id}: {forbidden}")
        eligible.append(
            {
                "parent_state_id": query_id,
                "partition": parent["partition"],
                "cached_result_count": len(parent["retrieved_docs"]),
                "search_id": rows[0]["search_id"],
                "search_query_sha256": hashlib.sha256(parent["search_query"].encode("utf-8")).hexdigest(),
            }
        )

    old_excluded = read_jsonl(args.v1_excluded.resolve())
    exclusions = [
        {
            "parent_state_id": row["query_id"],
            "reason": row["reason"],
            "retrieved_docs": row["retrieved_docs"],
        }
        for row in old_excluded
    ] + missing_clean
    partition_counts = Counter(row["partition"] for row in eligible)
    gates = {
        "v1_cached_searches_complete": len(parents) == 1186,
        "exact_1184_clean_query_states": len(eligible) == 1184,
        "exact_946_238_parent_split": partition_counts == Counter({"policy_train": 946, "internal_holdout": 238}),
        "five_actions_per_parent": sum(len(action_groups[row["parent_state_id"]]) for row in eligible) == 5920,
        "exact_A_E_mapping": all(
            tuple(row["evidence_strategy"] for row in sorted(action_groups[state["parent_state_id"]], key=lambda item: STRATEGY_ORDER.index(item["evidence_strategy"]))) == STRATEGY_ORDER
            for state in eligible
        ),
        "cached_queries_and_results_match_current_clean_actions": True,
        "frozen_no_gold_terminal_verifier": (
            terminal_audit.get("gold_or_qrels_in_runtime_features") is False
            and terminal_audit["gates"]["runtime_features_contain_no_gold_or_qrels"]
        ),
        "frozen_shared_answer_policy": value_audit.get("gold_or_qrels_loaded") is False,
        "controls_share_one_action_universe": control_freeze["gates"]["same_query_state_and_action_universe"],
        "new_search_calls_zero": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    if not all(gates.values()):
        raise ValueError(f"cached multi-query evidence v2 freeze failed: {gates}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    state_path = output_dir / "v6_cached_multiquery_evidence_v2_state_ids.jsonl"
    state_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in eligible),
        encoding="utf-8",
    )
    excluded_path = output_dir / "v6_cached_multiquery_evidence_v2_excluded_public.jsonl"
    excluded_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in exclusions),
        encoding="utf-8",
    )

    input_paths = {
        **{key: str(path) for key, path in manifest_paths.items()},
        "v1_cached_parents": str(v1_parents_path),
        "v1_excluded": str(args.v1_excluded.resolve()),
        "evidence_actions": str(evidence_actions_path),
        "private_support": str(private_support_path),
        "shared_answer_values": str(shared_values_path),
        "terminal_private_audit": str(terminal_private_path),
        "control_train": str(control_train_path),
        "control_internal": str(control_internal_path),
        "categorical_train": str(categorical_train_path),
        "categorical_internal": str(categorical_internal_path),
    }
    result = {
        "decision": "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_FROZEN",
        "protocol_version": "dagig_v6_cached_real_search_multiquery_evidence_selector_v2",
        "motivation": "Evaluate clean node-credit posteriors directly before any policy projection or generator training.",
        "universe": {
            "samples": len({row["parent_state_id"].split("::", 1)[0] for row in eligible}),
            "query_states": len(eligible),
            "evidence_actions": len(eligible) * 5,
            "partition_counts": dict(sorted(partition_counts.items())),
            "excluded_query_states": len(exclusions),
            "new_search_calls": 0,
        },
        "action_contract": {
            "labels": list(ACTION_LABELS),
            "strategies": list(STRATEGY_ORDER),
            "mapping": dict(zip(ACTION_LABELS, STRATEGY_ORDER)),
            "actions_per_state": 5,
            "selected_documents_per_action": 3,
            "tie_break": "fixed A-to-E order; no metric-dependent tie resolution",
        },
        "method_contract": {
            "no_credit": "frozen behavior distribution mu(e|q)",
            "local_ig": "KL-matched posterior from modal-answer P_success",
            "outcome": "KL-matched posterior from 12 frozen sampled evidence-to-answer outcomes",
            "dagig": "exact q(e|q)=mu(e|q)*sum_a pi_A(a|e)P_success(a,e)/V(q)",
        },
        "public_training_schema": ["parent_state_id", "prompt", "actions", "target_distributions"],
        "selector_only_evaluation": {
            "partition": "internal_holdout",
            "run_once": True,
            "generator_training": False,
            "metrics": [
                "selected_expected_terminal_value",
                "evidence_support",
                "selected_expected_strict",
                "selected_mode_strict",
                "action_diversity",
                "paired_gain_loss_vs_no_credit_and_outcome",
            ],
            "cluster_bootstrap": {"unit": "sample_id", "replicates": 10000, "seed": 20260720},
        },
        "selector_go_gates": {
            "dagig_terminal_delta_vs_no_credit_min": 0.005,
            "dagig_terminal_noninferiority_vs_outcome_tolerance": 0.002,
            "dagig_support_delta_vs_no_credit_min": 0.0,
            "dagig_support_noninferiority_vs_outcome_tolerance": 0.01,
            "dagig_expected_strict_noninferiority_tolerance": 0.015,
            "dagig_mode_strict_noninferiority_tolerance": 0.015,
            "dagig_outcome_top_action_disagreement_min": 0.05,
            "dagig_selected_strategies_min": 3,
        },
        "next_stage_if_go": {
            "type": "scalar evidence scorer/ranker",
            "primary_objective": "listwise KL to the frozen posterior",
            "ablation": "weighted pairwise cardinal ranking",
            "generator_policy_training": False,
            "same_architecture_data_steps_across_methods": True,
        },
        "gates": gates,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {"state_ids": str(state_path), "excluded": str(excluded_path)},
        "output_hashes": {"state_ids": sha256(state_path), "excluded": sha256(excluded_path)},
        "gold_or_qrels_available_to_runtime_or_targets": False,
        "private_labels_reserved_for_selector_audit_only": True,
        "internal_holdout_used_for_training_or_tuning": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    manifest_path = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_FREEZE.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "universe": result["universe"], "freeze": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
