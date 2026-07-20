#!/usr/bin/env python3
"""Build full multi-action query values under the frozen DAG-IG evidence selector.

The Local-IG control is deliberately query-local: it uses only observable search
result quality. DAG-IG alone propagates value through evidence and answer nodes.
"""

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


QUERY_STRATEGY_ORDER = ("direct", "bridge", "entity_exact", "alternate_anchor", "source_targeted")
EVIDENCE_STRATEGY_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)
METHODS = ("no_credit", "local_ig_m", "local_observable", "outcome", "dagig")
PUBLIC_KEYS = {"parent_state_id", "prompt", "actions", "target_distributions"}
FORBIDDEN_FIELD_TOKENS = (
    "gold",
    "qrel",
    "strict",
    "support_label",
    "evidence_hit",
    "answer_correct",
    "target_doc",
    "ground_truth",
    "terminal_value",
    "success_probability",
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
        raise ValueError("cannot normalize non-positive mass")
    result = [value / total for value in values]
    if any(value <= 0.0 or not math.isfinite(value) for value in result):
        raise ValueError("invalid normalized probability")
    return result


def softmax(logits: list[float]) -> list[float]:
    offset = max(logits)
    return normalize([math.exp(value - offset) for value in logits])


def kl(policy: list[float], behavior: list[float]) -> float:
    return sum(p * math.log(p / b) for p, b in zip(policy, behavior))


def tv(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def calibrated(behavior: list[float], values: list[float], scale: float, *, log_values: bool) -> list[float]:
    scores = [math.log(max(value, 1e-8)) if log_values else value for value in values]
    return softmax([math.log(probability) + scale * score for probability, score in zip(behavior, scores)])


def find_scale(groups: list[dict[str, Any]], key: str, target_kl: float, *, log_values: bool) -> float:
    def objective(scale: float) -> float:
        return mean(kl(calibrated(group["behavior"], group[key], scale, log_values=log_values), group["behavior"]) for group in groups)

    low, high = 0.0, 1.0
    while objective(high) < target_kl and high < 4096.0:
        high *= 2.0
    if objective(high) < target_kl:
        raise ValueError(f"cannot KL-match {key}")
    for _ in range(80):
        middle = (low + high) / 2.0
        if objective(middle) < target_kl:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def weighted_index(probabilities: list[float], rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative + 1e-12:
            return index
    return len(probabilities) - 1


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


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def local_query_score(row: dict[str, Any]) -> float:
    """Frozen gold-free quality of the executed query's observed search results."""
    docs = row.get("retrieved_docs") or []
    if not docs:
        return 0.0
    values = [
        0.65 * float(doc.get("normalized_bge_score", 0.0))
        + 0.20 * float(doc.get("question_keyword_overlap", 0.0))
        + 0.15 * float(doc.get("answer_type_pattern_match", 0.0))
        for doc in docs[:5]
    ]
    domains = {clean(doc.get("domain")).casefold() for doc in docs[:5] if clean(doc.get("domain"))}
    return sum(values) / len(values) + 0.10 * len(domains) / max(1, min(5, len(docs)))


def query_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are the structured query node of a multimodal web-search agent.",
            "Choose one legal query action using only the question and frozen visual observation.",
            "Do not answer the question and do not add information absent from the image/question.",
            f"Question: {row['question']}",
            f"Frozen image-only visual observation: {row['visual_observation']}",
            "Query action:",
        ]
    )


def query_completion(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "entity_quote": row.get("entity_quote") or "",
            "information_need": row.get("information_need") or "",
            "constraints": row.get("constraints") or [],
            "search_query": row["search_query"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_protocol_freeze", type=Path, required=True)
    parser.add_argument("--evidence_target_audit", type=Path, required=True)
    parser.add_argument("--evidence_selector_audit", type=Path, required=True)
    parser.add_argument("--backup_audit", type=Path, required=True)
    parser.add_argument("--query_search_audit", type=Path, required=True)
    parser.add_argument("--rollouts_per_group", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rollouts_per_group != 12:
        raise ValueError("query Outcome control is frozen to 12 downstream trajectories per visual state")

    manifest_paths = {
        "evidence_protocol_freeze": args.evidence_protocol_freeze.resolve(),
        "evidence_target_audit": args.evidence_target_audit.resolve(),
        "evidence_selector_audit": args.evidence_selector_audit.resolve(),
        "backup_audit": args.backup_audit.resolve(),
        "query_search_audit": args.query_search_audit.resolve(),
    }
    evidence_protocol = read_json(manifest_paths["evidence_protocol_freeze"])
    evidence_targets = read_json(manifest_paths["evidence_target_audit"])
    evidence_selector = read_json(manifest_paths["evidence_selector_audit"])
    backup = read_json(manifest_paths["backup_audit"])
    query_search = read_json(manifest_paths["query_search_audit"])
    if evidence_protocol.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_FROZEN":
        raise ValueError("evidence v2 protocol is not frozen")
    if evidence_targets.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_TARGETS_GO":
        raise ValueError("evidence v2 targets are not GO")
    if evidence_selector.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_SELECTOR_ONLY_GO":
        raise ValueError("direct evidence selector is not GO")
    if backup.get("decision") != "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_GO":
        raise ValueError("query behavior source is not frozen/GO")
    if query_search.get("decision") != "DAGIG_V6_IDENTIFYING_QUERY_SEARCH_GO":
        raise ValueError("full real-search query universe is not GO")
    assert_hash(manifest_paths["evidence_protocol_freeze"], evidence_targets["input_hashes"]["protocol_freeze"], "evidence protocol")
    for key, raw_path in evidence_targets["output_paths"].items():
        assert_hash(Path(raw_path), evidence_targets["output_hashes"][key], key)
    assert_hash(Path(evidence_selector["output_paths"]["private_rows"]), evidence_selector["output_hashes"]["private_rows"], "evidence selector private rows")

    shared_values_path = Path(evidence_protocol["input_paths"]["shared_answer_values"])
    assert_hash(shared_values_path, evidence_protocol["input_hashes"]["shared_answer_values"], "shared answer values")
    shared_values = {row["evidence_action_id"]: row for row in read_jsonl(shared_values_path)}
    query_edges_path = Path(backup["output_paths"]["query_edges"])
    assert_hash(query_edges_path, backup["output_hashes"]["query_edges"], "query behavior edges")
    query_edges = {row["action_id"]: row for row in read_jsonl(query_edges_path)}
    evidence_edges_path = Path(backup["output_paths"]["evidence_edges"])
    assert_hash(evidence_edges_path, backup["output_hashes"]["evidence_edges"], "evidence edges")
    evidence_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(evidence_edges_path):
        evidence_edges[row["parent_id"]].append(row)
    query_action_path = Path(query_search["output_paths"]["actions_with_search"])
    query_action_rows = read_jsonl(query_action_path)
    all_query_actions = {row["query_id"]: row for row in query_action_rows}
    query_actions = {query_id: all_query_actions[query_id] for query_id in query_edges}
    if set(query_actions) != set(query_edges) or set(evidence_edges) != set(query_edges):
        raise ValueError("full query, evidence, and real-search universes differ")
    exclusions = [
        {"query_action_id": query_id, "reason": "absent_from_frozen_complete_no_gold_full_dag"}
        for query_id in sorted(set(all_query_actions) - set(query_edges))
    ]

    evidence_choice: dict[str, dict[str, Any]] = {}
    for query_id in sorted(query_edges):
        rows = sorted(evidence_edges[query_id], key=lambda row: EVIDENCE_STRATEGY_ORDER.index(row["action_id"].rsplit("::", 1)[-1]))
        if len(rows) != 5:
            raise ValueError(f"expected five evidence interventions: {query_id}")
        action_ids = [row["action_id"] for row in rows]
        values = [float(shared_values[action_id]["shared_answer_value"]) for action_id in action_ids]
        posterior = normalize(values)
        evidence_index = max(range(5), key=lambda index: (posterior[index], -index))
        evidence_action_id = action_ids[evidence_index]
        evidence_strategy = evidence_action_id.rsplit("::", 1)[-1]
        value = shared_values[evidence_action_id]
        evidence_choice[query_id] = {
            "evidence_action_id": evidence_action_id,
            "evidence_strategy": evidence_strategy,
            "evidence_posterior": posterior,
            "hard_query_value": float(value["shared_answer_value"]),
            "local_modal_value": float(value["mode_child_success_probability"]),
            "answer_action_ids": value["answer_action_ids"],
            "answer_policy_probabilities": [float(item) for item in value["answer_policy_probabilities"]],
            "answer_child_values": [float(item) for item in value["child_success_probabilities"]],
        }

    grouped: dict[str, list[str]] = defaultdict(list)
    for query_id, edge in query_edges.items():
        grouped[edge["parent_id"]].append(query_id)
    complete_groups: dict[str, list[str]] = {}
    for parent_id, query_ids in sorted(grouped.items()):
        ordered = sorted(query_ids, key=lambda query_id: QUERY_STRATEGY_ORDER.index(query_id.split("::")[-1]))
        if not 3 <= len(ordered) <= 5:
            raise ValueError(f"full query parent has invalid action count: {parent_id}: {len(ordered)}")
        complete_groups[parent_id] = ordered

    groups: list[dict[str, Any]] = []
    constant_outcome = 0
    for parent_id, query_ids in sorted(complete_groups.items()):
        edge_rows = [query_edges[query_id] for query_id in query_ids]
        if any(row["parent_id"] != parent_id for row in edge_rows):
            raise ValueError(f"query edge parent changed: {parent_id}")
        behavior = normalize([1.0 for _ in edge_rows])
        hard_values = [evidence_choice[query_id]["hard_query_value"] for query_id in query_ids]
        dagig = normalize([probability * max(value, 1e-8) for probability, value in zip(behavior, hard_values)])
        observed: dict[str, list[float]] = {query_id: [] for query_id in query_ids}
        sampled: list[tuple[str, float]] = []
        for rollout in range(args.rollouts_per_group):
            rng = random.Random(f"full-query-outcome:{args.seed}:{parent_id}:{rollout}")
            query_index = weighted_index(behavior, rng)
            query_id = query_ids[query_index]
            choice = evidence_choice[query_id]
            answer_index = weighted_index(choice["answer_policy_probabilities"], rng)
            sampled.append((query_id, choice["answer_child_values"][answer_index]))
        rewards = [reward for _, reward in sampled]
        center = mean(rewards)
        reward_std = math.sqrt(mean((reward - center) ** 2 for reward in rewards))
        constant_outcome += int(reward_std <= 1e-12)
        for query_id, reward in sampled:
            observed[query_id].append((reward - center) / reward_std if reward_std > 1e-12 else 0.0)
        groups.append(
            {
                "parent_id": parent_id,
                "sample_id": query_ids[0].split("::", 1)[0],
                "partition": edge_rows[0]["partition"],
                "query_ids": query_ids,
                "behavior": behavior,
                "hard_values": hard_values,
                "local_m_values": [evidence_choice[query_id]["local_modal_value"] for query_id in query_ids],
                "local_observable_values": [local_query_score(query_actions[query_id]) for query_id in query_ids],
                "outcome_values": [mean(observed[query_id]) if observed[query_id] else 0.0 for query_id in query_ids],
                "outcome_counts": [len(observed[query_id]) for query_id in query_ids],
                "outcome_reward_std": reward_std,
                "dagig": dagig,
            }
        )

    train_groups = [group for group in groups if group["partition"] == "policy_train"]
    internal_groups = [group for group in groups if group["partition"] == "internal_holdout"]
    target_kl = mean(kl(group["dagig"], group["behavior"]) for group in train_groups)
    local_m_beta = find_scale(train_groups, "local_m_values", target_kl, log_values=True)
    local_observable_beta = find_scale(train_groups, "local_observable_values", target_kl, log_values=True)
    outcome_eta = find_scale(train_groups, "outcome_values", target_kl, log_values=False)

    public_rows: dict[str, list[dict[str, Any]]] = {"policy_train": [], "internal_holdout": []}
    diagnostics: list[dict[str, Any]] = []
    normalization_error = 0.0
    identity_error = 0.0
    local_m_tvs: list[float] = []
    local_observable_tvs: list[float] = []
    outcome_tvs: list[float] = []
    local_m_top: list[bool] = []
    local_observable_top: list[bool] = []
    outcome_top: list[bool] = []
    train_selected_terminal: dict[str, list[float]] = {method: [] for method in METHODS}
    for group in groups:
        local_m = calibrated(group["behavior"], group["local_m_values"], local_m_beta, log_values=True)
        local_observable = calibrated(
            group["behavior"], group["local_observable_values"], local_observable_beta, log_values=True
        )
        outcome = calibrated(group["behavior"], group["outcome_values"], outcome_eta, log_values=False)
        distributions = {
            "no_credit": group["behavior"],
            "local_ig_m": local_m,
            "local_observable": local_observable,
            "outcome": outcome,
            "dagig": group["dagig"],
        }
        for distribution in distributions.values():
            normalization_error = max(normalization_error, abs(sum(distribution) - 1.0))
        exact = normalize([probability * value for probability, value in zip(group["behavior"], group["hard_values"])])
        identity_error = max(identity_error, max(abs(a - b) for a, b in zip(exact, group["dagig"])))
        dag_top = max(range(len(group["query_ids"])), key=group["dagig"].__getitem__)
        if group["partition"] == "policy_train":
            local_m_tvs.append(tv(group["dagig"], local_m))
            local_observable_tvs.append(tv(group["dagig"], local_observable))
            outcome_tvs.append(tv(group["dagig"], outcome))
            local_m_top.append(dag_top == max(range(len(local_m)), key=local_m.__getitem__))
            local_observable_top.append(
                dag_top == max(range(len(local_observable)), key=local_observable.__getitem__)
            )
            outcome_top.append(dag_top == max(range(len(outcome)), key=outcome.__getitem__))
            for method, distribution in distributions.items():
                selected_index = max(range(len(distribution)), key=lambda index: (distribution[index], -index))
                train_selected_terminal[method].append(group["hard_values"][selected_index])

        source_rows = [query_actions[query_id] for query_id in group["query_ids"]]
        if len({row["question"] for row in source_rows}) != 1 or len({row["visual_observation"] for row in source_rows}) != 1:
            raise ValueError(f"query actions do not share one causal parent: {group['parent_id']}")
        actions = [
            {
                "label": f"Q{index + 1}",
                "strategy": query_id.split("::")[-1],
                "completion": query_completion(row),
            }
            for index, (query_id, row) in enumerate(zip(group["query_ids"], source_rows))
        ]
        if len({action["completion"] for action in actions}) != len(actions):
            raise ValueError(f"duplicate query completions: {group['parent_id']}")
        public = {
            "parent_state_id": group["parent_id"],
            "prompt": query_prompt(source_rows[0]),
            "actions": actions,
            "target_distributions": distributions,
        }
        if set(public) != PUBLIC_KEYS:
            raise ValueError("query public schema changed")
        forbidden = sorted(field for field in nested_field_names(public) if any(token in field for token in FORBIDDEN_FIELD_TOKENS))
        if forbidden:
            raise ValueError(f"query target exposes private fields: {group['parent_id']}: {forbidden}")
        public_rows[group["partition"]].append(public)
        diagnostics.append(
            {
                "parent_state_id": group["parent_id"],
                "partition": group["partition"],
                "query_action_ids": group["query_ids"],
                "behavior_probabilities": group["behavior"],
                "selected_evidence_action_ids": [evidence_choice[query_id]["evidence_action_id"] for query_id in group["query_ids"]],
                "hard_query_values": group["hard_values"],
                "local_modal_descendant_values": group["local_m_values"],
                "local_observable_query_scores": group["local_observable_values"],
                "outcome_values": group["outcome_values"],
                "outcome_counts": group["outcome_counts"],
                "dagig_target_probabilities": group["dagig"],
            }
        )

    metrics = {
        "samples": len({group["sample_id"] for group in groups}),
        "visual_parent_states": len(groups),
        "query_actions": sum(len(group["query_ids"]) for group in groups),
        "action_count_distribution": dict(sorted(Counter(len(group["query_ids"]) for group in groups).items())),
        "policy_train_groups": len(train_groups),
        "internal_holdout_groups": len(internal_groups),
        "policy_train_samples": len({group["sample_id"] for group in train_groups}),
        "internal_holdout_samples": len({group["sample_id"] for group in internal_groups}),
        "excluded_incomplete_groups": len(exclusions),
        "outcome_constant_group_rate": constant_outcome / len(groups),
        "target_mean_kl_from_behavior_train": target_kl,
        "local_m_beta": local_m_beta,
        "local_observable_beta": local_observable_beta,
        "outcome_eta": outcome_eta,
        "local_m_mean_kl_train": mean(kl(calibrated(group["behavior"], group["local_m_values"], local_m_beta, log_values=True), group["behavior"]) for group in train_groups),
        "local_observable_mean_kl_train": mean(kl(calibrated(group["behavior"], group["local_observable_values"], local_observable_beta, log_values=True), group["behavior"]) for group in train_groups),
        "outcome_mean_kl_train": mean(kl(calibrated(group["behavior"], group["outcome_values"], outcome_eta, log_values=False), group["behavior"]) for group in train_groups),
        "local_m_nonconstant_group_rate_train": mean(
            max(group["local_m_values"]) - min(group["local_m_values"]) > 1e-12 for group in train_groups
        ),
        "local_observable_nonconstant_group_rate_train": mean(
            max(group["local_observable_values"]) - min(group["local_observable_values"]) > 1e-12
            for group in train_groups
        ),
        "dagig_local_m_mean_tv_train": mean(local_m_tvs),
        "dagig_local_observable_mean_tv_train": mean(local_observable_tvs),
        "dagig_outcome_mean_tv_train": mean(outcome_tvs),
        "dagig_local_m_top_agreement_train": mean(local_m_top),
        "dagig_local_observable_top_agreement_train": mean(local_observable_top),
        "dagig_outcome_top_agreement_train": mean(outcome_top),
        "max_target_normalization_error": normalization_error,
        "max_query_dag_identity_error": identity_error,
        "train_direct_selector_terminal": {
            method: mean(values) for method, values in train_selected_terminal.items()
        },
    }
    gates = {
        "evidence_direct_selector_frozen_and_go": True,
        "exact_594_complete_visual_parents": len(groups) == 594,
        "exact_474_120_group_split": len(train_groups) == 474 and len(internal_groups) == 120,
        "sample_disjoint_158_40_split": metrics["policy_train_samples"] == 158 and metrics["internal_holdout_samples"] == 40,
        "exact_2954_query_actions": metrics["query_actions"] == 2954,
        "three_to_five_query_actions_per_parent": min(metrics["action_count_distribution"]) >= 3 and max(metrics["action_count_distribution"]) <= 5,
        "exact_twelve_uniform_exclusions": len(exclusions) == 12,
        "all_query_actions_have_at_least_five_real_search_results": all(
            len(query_actions[query_id].get("retrieved_docs") or []) >= 5
            for group in groups
            for query_id in group["query_ids"]
        ),
        "uniform_interventional_query_prior": all(
            max(group["behavior"]) - min(group["behavior"]) <= 1e-12 for group in groups
        ),
        "hard_evidence_argmax_policy_used": True,
        "hard_query_value_matches_deployed_policy": True,
        "exact_query_dag_identity": identity_error <= 1e-12,
        "targets_normalized": normalization_error <= 1e-10,
        "local_m_kl_matched_train_only": abs(metrics["local_m_mean_kl_train"] - target_kl) <= 1e-6,
        "local_observable_kl_matched_train_only": abs(metrics["local_observable_mean_kl_train"] - target_kl) <= 1e-6,
        "outcome_kl_matched_train_only": abs(metrics["outcome_mean_kl_train"] - target_kl) <= 1e-6,
        "outcome_groups_nonconstant": metrics["outcome_constant_group_rate"] <= 0.05,
        "local_m_signal_nonconstant_train": metrics["local_m_nonconstant_group_rate_train"] >= 0.90,
        "local_observable_signal_nonconstant_train": metrics["local_observable_nonconstant_group_rate_train"] >= 0.20,
        "dagig_local_m_identifiable_train": metrics["dagig_local_m_mean_tv_train"] >= 0.01 and metrics["dagig_local_m_top_agreement_train"] <= 0.90,
        "dagig_local_observable_identifiable_train": metrics["dagig_local_observable_mean_tv_train"] >= 0.05 and metrics["dagig_local_observable_top_agreement_train"] <= 0.90,
        "dagig_outcome_identifiable_train": metrics["dagig_outcome_mean_tv_train"] >= 0.01 and metrics["dagig_outcome_top_agreement_train"] <= 0.95,
        "dagig_train_terminal_improves_no_credit": metrics["train_direct_selector_terminal"]["dagig"] >= metrics["train_direct_selector_terminal"]["no_credit"] + 0.005,
        "dagig_train_terminal_noninferior_local_m": metrics["train_direct_selector_terminal"]["dagig"] >= metrics["train_direct_selector_terminal"]["local_ig_m"] - 0.002,
        "dagig_train_terminal_noninferior_local_observable": metrics["train_direct_selector_terminal"]["dagig"] >= metrics["train_direct_selector_terminal"]["local_observable"] - 0.002,
        "dagig_train_terminal_noninferior_outcome": metrics["train_direct_selector_terminal"]["dagig"] >= metrics["train_direct_selector_terminal"]["outcome"] - 0.002,
        "public_schema_has_no_private_evaluation_fields": True,
        "internal_unused_for_scale_or_tuning": True,
        "new_search_calls_zero": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_FULL_QUERY_VALUES_V2_FROZEN" if all(gates.values()) else "DAGIG_V6_FULL_QUERY_VALUES_V2_NO_GO"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "v6_full_query_targets_v2_train.jsonl"
    internal_path = output_dir / "v6_full_query_targets_v2_internal_no_labels.jsonl"
    diagnostics_path = output_dir / "v6_full_query_value_v2_diagnostics_no_gold.jsonl"
    excluded_path = output_dir / "v6_full_query_v2_excluded_public.jsonl"
    for path, rows in (
        (train_path, public_rows["policy_train"]),
        (internal_path, public_rows["internal_holdout"]),
        (diagnostics_path, diagnostics),
        (excluded_path, exclusions),
    ):
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    input_paths = {
        **{key: str(path) for key, path in manifest_paths.items()},
        "full_real_search_query_actions": str(query_action_path),
        "shared_answer_values": str(shared_values_path),
        "query_edges": str(query_edges_path),
        "evidence_edges": str(evidence_edges_path),
    }
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_full_real_search_uniform_intervention_query_backup_v2",
        "downstream_contract": {
            "evidence_policy": "frozen direct argmax of exact DAG-IG evidence posterior",
            "answer_policy": "frozen shared answer policy",
            "terminal_value": "frozen no-gold verifier P_success",
            "query_value": "V_Q(q)=V_A(argmax_e q_DAG(e|q))",
            "soft_query_value_used": False,
        },
        "method_contract": {
            "no_credit": "uniform proposal over every legal real-search query intervention",
            "local_ig_m": "same selected evidence intervention followed by the modal frozen answer child, KL-matched on train",
            "local_observable": "frozen query-local score from observed BGE/keyword/type/domain search-result features, KL-matched on train",
            "outcome": "12 sampled query-to-hard-evidence-to-frozen-answer outcomes, KL-matched on train",
            "dagig": "exact q(q|v,Y=1)=mu_Q(q|v)V_Q(q)/V_V(v), with uniform intervention proposal mu_Q",
        },
        "metrics": metrics,
        "gates": gates,
        "selector_go_gates": {
            "dagig_terminal_delta_vs_no_credit_min": 0.005,
            "dagig_terminal_noninferiority_vs_outcome_tolerance": 0.002,
            "dagig_support_delta_vs_no_credit_min": 0.0,
            "dagig_support_noninferiority_vs_outcome_tolerance": 0.01,
            "dagig_expected_strict_noninferiority_tolerance": 0.015,
            "dagig_mode_strict_noninferiority_tolerance": 0.015,
            "dagig_outcome_top_action_disagreement_min": 0.05,
            "dagig_selected_query_strategies_min": 4,
        },
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {
            "train_targets": str(train_path),
            "internal_targets": str(internal_path),
            "diagnostics": str(diagnostics_path),
            "excluded": str(excluded_path),
        },
        "output_hashes": {
            "train_targets": sha256(train_path),
            "internal_targets": sha256(internal_path),
            "diagnostics": sha256(diagnostics_path),
            "excluded": sha256(excluded_path),
        },
        "gold_or_qrels_loaded": False,
        "private_support_or_strict_loaded": False,
        "internal_holdout_used_for_scale_or_tuning": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_FULL_QUERY_VALUE_V2_FREEZE.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
