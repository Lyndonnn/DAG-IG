#!/usr/bin/env python3
"""Back up the frozen hybrid evidence value into full real-search query values."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


QUERY_ORDER = ("direct", "bridge", "entity_exact", "alternate_anchor", "source_targeted")
EVIDENCE_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)
METHODS = ("no_credit", "local_ig_m", "local_observable", "outcome", "dagig")


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_v6_hybrid_query_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid_value_freeze", type=Path, required=True)
    parser.add_argument("--hybrid_train_audit", type=Path, required=True)
    parser.add_argument("--hybrid_development_audit", type=Path, required=True)
    parser.add_argument("--query_search_audit", type=Path, required=True)
    parser.add_argument("--backup_audit", type=Path, required=True)
    parser.add_argument("--shared_answer_values", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--rollouts_per_group", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rollouts_per_group != 12:
        raise ValueError("Outcome control remains frozen to 12 downstream samples")
    paths = {key: value.resolve() for key, value in {
        "hybrid_value_freeze": args.hybrid_value_freeze,
        "hybrid_train_audit": args.hybrid_train_audit,
        "hybrid_development_audit": args.hybrid_development_audit,
        "query_search_audit": args.query_search_audit,
        "backup_audit": args.backup_audit,
        "shared_answer_values": args.shared_answer_values,
        "auditor": args.auditor,
        "helper": args.helper,
    }.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    helper = load_module(paths["helper"])
    value_freeze = read_json(paths["hybrid_value_freeze"])
    value_train = read_json(paths["hybrid_train_audit"])
    value_development = read_json(paths["hybrid_development_audit"])
    query_search = read_json(paths["query_search_audit"])
    backup = read_json(paths["backup_audit"])
    if value_development.get("decision") != "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_DEVELOPMENT_GO":
        raise ValueError("hybrid evidence value is not frozen/GO")
    if value_train.get("decision") != "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_TRAIN_OOF_GO":
        raise ValueError("hybrid evidence train audit is not GO")
    if query_search.get("decision") != "DAGIG_V6_IDENTIFYING_QUERY_SEARCH_GO":
        raise ValueError("real-search query universe is not GO")
    if backup.get("decision") != "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_GO":
        raise ValueError("full DAG action universe is not GO")
    for key, raw_path in value_train["output_paths"].items():
        if sha256(Path(raw_path)) != value_train["output_hashes"][key]:
            raise ValueError(f"hybrid value output changed: {key}")

    hybrid = {
        row["evidence_action_id"]: row
        for row in read_jsonl(Path(value_train["output_paths"]["predictions"]))
    }
    shared = {row["evidence_action_id"]: row for row in read_jsonl(paths["shared_answer_values"])}
    query_edges = {row["action_id"]: row for row in read_jsonl(Path(backup["output_paths"]["query_edges"]))}
    evidence_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(Path(backup["output_paths"]["evidence_edges"])):
        evidence_groups[row["parent_id"]].append(row)
    all_query_actions = {
        row["query_id"]: row
        for row in read_jsonl(Path(query_search["output_paths"]["actions_with_search"]))
    }
    query_actions = {query_id: all_query_actions[query_id] for query_id in query_edges}
    if set(query_actions) != set(evidence_groups) or len(query_actions) != 2954 or len(hybrid) != 14770:
        raise ValueError("hybrid query/evidence universe mismatch")

    evidence_choice = {}
    for query_id in sorted(query_actions):
        rows = sorted(evidence_groups[query_id], key=lambda row: EVIDENCE_ORDER.index(row["action_id"].rsplit("::", 1)[-1]))
        action_ids = [row["action_id"] for row in rows]
        values = [float(hybrid[action_id]["evidence_success_probability"]) for action_id in action_ids]
        posterior = helper.normalize([0.2 * max(value, 1e-8) for value in values])
        selected = max(range(5), key=lambda index: (posterior[index], -index))
        evidence_id = action_ids[selected]
        shared_value = shared[evidence_id]
        evidence_choice[query_id] = {
            "evidence_action_id": evidence_id,
            "evidence_strategy": evidence_id.rsplit("::", 1)[-1],
            "evidence_posterior": posterior,
            "hybrid_value": values[selected],
            "local_modal_value": float(shared_value["mode_child_success_probability"]),
            "answer_policy_probabilities": [float(item) for item in shared_value["answer_policy_probabilities"]],
            "answer_child_values": [float(item) for item in shared_value["child_success_probabilities"]],
        }

    grouped: dict[str, list[str]] = defaultdict(list)
    for query_id, edge in query_edges.items():
        grouped[edge["parent_id"]].append(query_id)
    groups = []
    for parent_id, query_ids in sorted(grouped.items()):
        query_ids = sorted(query_ids, key=lambda query_id: QUERY_ORDER.index(query_id.rsplit("::", 1)[-1]))
        if not 3 <= len(query_ids) <= 5:
            raise ValueError(f"invalid query intervention group: {parent_id}")
        behavior = helper.normalize([1.0 for _ in query_ids])
        values = [evidence_choice[query_id]["hybrid_value"] for query_id in query_ids]
        dagig = helper.normalize([probability * max(value, 1e-8) for probability, value in zip(behavior, values)])
        sampled: dict[str, list[float]] = {query_id: [] for query_id in query_ids}
        all_rewards = []
        for rollout in range(args.rollouts_per_group):
            rng = random.Random(f"hybrid-query-outcome:{args.seed}:{parent_id}:{rollout}")
            query_index = helper.weighted_index(behavior, rng)
            query_id = query_ids[query_index]
            choice = evidence_choice[query_id]
            answer_index = helper.weighted_index(choice["answer_policy_probabilities"], rng)
            reward = choice["answer_child_values"][answer_index]
            sampled[query_id].append(reward)
            all_rewards.append(reward)
        center = mean(all_rewards)
        std = math.sqrt(mean((value - center) ** 2 for value in all_rewards))
        groups.append({
            "parent_id": parent_id,
            "sample_id": query_ids[0].split("::", 1)[0],
            "partition": query_edges[query_ids[0]]["partition"],
            "query_ids": query_ids,
            "behavior": behavior,
            "hybrid_values": values,
            "local_values": [evidence_choice[query_id]["local_modal_value"] for query_id in query_ids],
            "local_observable_values": [helper.local_query_score(query_actions[query_id]) for query_id in query_ids],
            "outcome_values": [mean([(value - center) / std for value in sampled[query_id]]) if sampled[query_id] and std > 1e-12 else 0.0 for query_id in query_ids],
            "outcome_counts": [len(sampled[query_id]) for query_id in query_ids],
            "dagig": dagig,
        })

    train_groups = [group for group in groups if group["partition"] == "policy_train"]
    target_kl = mean(helper.kl(group["dagig"], group["behavior"]) for group in train_groups)
    local_beta = helper.find_scale(train_groups, "local_values", target_kl, log_values=True)
    local_observable_beta = helper.find_scale(train_groups, "local_observable_values", target_kl, log_values=True)
    outcome_eta = helper.find_scale(train_groups, "outcome_values", target_kl, log_values=False)
    public = {"policy_train": [], "internal_holdout": []}
    diagnostics = []
    selected_value = {method: [] for method in METHODS}
    identity_error = normalization_error = 0.0
    tv_outcome = []
    top_outcome = []
    for group in groups:
        local = helper.calibrated(group["behavior"], group["local_values"], local_beta, log_values=True)
        local_observable = helper.calibrated(group["behavior"], group["local_observable_values"], local_observable_beta, log_values=True)
        outcome = helper.calibrated(group["behavior"], group["outcome_values"], outcome_eta, log_values=False)
        distributions = {"no_credit": group["behavior"], "local_ig_m": local, "local_observable": local_observable, "outcome": outcome, "dagig": group["dagig"]}
        for distribution in distributions.values():
            normalization_error = max(normalization_error, abs(sum(distribution) - 1.0))
        exact = helper.normalize([probability * value for probability, value in zip(group["behavior"], group["hybrid_values"])])
        identity_error = max(identity_error, max(abs(a - b) for a, b in zip(exact, group["dagig"])))
        if group["partition"] == "policy_train":
            tv_outcome.append(helper.tv(group["dagig"], outcome))
            top_outcome.append(max(range(len(outcome)), key=outcome.__getitem__) == max(range(len(group["dagig"])), key=group["dagig"].__getitem__))
            for method, distribution in distributions.items():
                index = max(range(len(distribution)), key=lambda item: (distribution[item], -item))
                selected_value[method].append(group["hybrid_values"][index])
        source = [query_actions[query_id] for query_id in group["query_ids"]]
        row = {
            "parent_state_id": group["parent_id"],
            "prompt": helper.query_prompt(source[0]),
            "actions": [{"label": f"Q{index + 1}", "strategy": query_id.rsplit("::", 1)[-1], "completion": helper.query_completion(action)} for index, (query_id, action) in enumerate(zip(group["query_ids"], source))],
            "target_distributions": distributions,
        }
        public[group["partition"]].append(row)
        diagnostics.append({
            "parent_state_id": group["parent_id"],
            "partition": group["partition"],
            "query_action_ids": group["query_ids"],
            "selected_evidence_action_ids": [evidence_choice[query_id]["evidence_action_id"] for query_id in group["query_ids"]],
            "hybrid_query_values": group["hybrid_values"],
            "local_modal_descendant_values": group["local_values"],
            "local_observable_query_scores": group["local_observable_values"],
            "outcome_values": group["outcome_values"],
            "dagig_target_probabilities": group["dagig"],
        })

    metrics = {
        "samples": 198,
        "visual_parent_states": len(groups),
        "query_actions": sum(len(group["query_ids"]) for group in groups),
        "action_count_distribution": dict(sorted(Counter(len(group["query_ids"]) for group in groups).items())),
        "policy_train_groups": len(train_groups),
        "internal_groups": len(groups) - len(train_groups),
        "target_mean_kl_train": target_kl,
        "local_beta": local_beta,
        "local_observable_beta": local_observable_beta,
        "outcome_eta": outcome_eta,
        "dagig_outcome_mean_tv_train": mean(tv_outcome),
        "dagig_outcome_top_agreement_train": mean(top_outcome),
        "train_direct_selector_hybrid_value": {method: mean(values) for method, values in selected_value.items()},
        "max_identity_error": identity_error,
        "max_normalization_error": normalization_error,
    }
    gates = {
        "hybrid_evidence_value_frozen_and_go": True,
        "complete_594_visual_parents": len(groups) == 594,
        "complete_2954_query_actions": metrics["query_actions"] == 2954,
        "complete_474_120_split": metrics["policy_train_groups"] == 474 and metrics["internal_groups"] == 120,
        "uniform_query_intervention_prior": all(max(group["behavior"]) - min(group["behavior"]) <= 1e-12 for group in groups),
        "exact_query_dag_identity": identity_error <= 1e-12,
        "targets_normalized": normalization_error <= 1e-10,
        "dagig_train_improves_no_credit": metrics["train_direct_selector_hybrid_value"]["dagig"] >= metrics["train_direct_selector_hybrid_value"]["no_credit"] + 0.005,
        "dagig_train_noninferior_local": metrics["train_direct_selector_hybrid_value"]["dagig"] >= metrics["train_direct_selector_hybrid_value"]["local_ig_m"] - 0.002,
        "dagig_train_noninferior_outcome": metrics["train_direct_selector_hybrid_value"]["dagig"] >= metrics["train_direct_selector_hybrid_value"]["outcome"] - 0.002,
        "dagig_identifiable_from_outcome": metrics["dagig_outcome_mean_tv_train"] >= 0.01 and metrics["dagig_outcome_top_agreement_train"] <= 0.95,
        "runtime_targets_use_no_gold": True,
        "internal_unused_for_scale_or_tuning": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_HYBRID_QUERY_VALUES_V1_FROZEN" if all(gates.values()) else "DAGIG_V6_HYBRID_QUERY_VALUES_V1_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_path = output / "v6_hybrid_query_targets_train.jsonl"
    internal_path = output / "v6_hybrid_query_targets_internal_no_labels.jsonl"
    diagnostic_path = output / "v6_hybrid_query_diagnostics_no_labels.jsonl"
    for path, rows in ((train_path, public["policy_train"]), (internal_path, public["internal_holdout"]), (diagnostic_path, diagnostics)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    input_paths = {key: str(path) for key, path in paths.items()}
    input_paths.update({"hybrid_predictions": value_train["output_paths"]["predictions"], "query_actions": query_search["output_paths"]["actions_with_search"], "query_edges": backup["output_paths"]["query_edges"], "evidence_edges": backup["output_paths"]["evidence_edges"]})
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_hybrid_evidence_backed_full_query_value_v1",
        "downstream_contract": {"evidence_policy": "argmax hybrid DAG-IG evidence posterior", "answer_policy": "frozen shared answer policy", "query_value": "hybrid calibrated evidence-state P_success after hard evidence selection"},
        "metrics": metrics,
        "gates": gates,
        "development_gates": {"support_delta_vs_no_credit_min": 0.0, "support_noninferiority_vs_local_tolerance": 0.01, "support_noninferiority_vs_outcome_tolerance": 0.01, "strict_noninferiority_vs_no_credit_tolerance": 0.0, "strict_noninferiority_vs_local_tolerance": 0.015, "strict_noninferiority_vs_outcome_tolerance": 0.015, "top_action_disagreement_vs_outcome_min": 0.05, "selected_query_strategies_min": 4},
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {"train_targets": str(train_path), "internal_targets": str(internal_path), "diagnostics": str(diagnostic_path)},
        "output_hashes": {"train_targets": sha256(train_path), "internal_targets": sha256(internal_path), "diagnostics": sha256(diagnostic_path)},
        "auditor_path": str(paths["auditor"]),
        "auditor_hash": sha256(paths["auditor"]),
        "gold_or_qrels_loaded": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_HYBRID_QUERY_VALUE_V1_FREEZE.json"
    freeze_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "freeze": str(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
