#!/usr/bin/env python3
"""Back up frozen evidence/answer values into matched query-node controls."""

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


METHODS = ("no_credit", "local_ig", "outcome", "dagig")
FORBIDDEN_KEYS = {
    "aliases",
    "answer_correct",
    "gold_answer",
    "ground_truth",
    "positive_doc_ids",
    "qrels",
    "strict",
    "support_label",
    "target_doc",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("cannot normalize non-positive probabilities")
    result = [value / total for value in values]
    if min(result) <= 0.0 or not all(math.isfinite(value) for value in result):
        raise ValueError("invalid normalized probabilities")
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


def find_scale(groups: list[dict[str, Any]], value_key: str, target_kl: float, *, log_values: bool) -> float:
    def objective(scale: float) -> float:
        return mean(
            kl(calibrated(group["behavior"], group[value_key], scale, log_values=log_values), group["behavior"])
            for group in groups
        )

    low, high = 0.0, 1.0
    while objective(high) < target_kl and high < 4096.0:
        high *= 2.0
    if objective(high) < target_kl:
        raise ValueError(f"cannot KL-match {value_key}")
    for _ in range(80):
        middle = (low + high) / 2.0
        if objective(middle) < target_kl:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def weighted_index(probabilities: list[float], rng: random.Random) -> int:
    draw, cumulative = rng.random(), 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative + 1e-12:
            return index
    return len(probabilities) - 1


def prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are the structured query node of a multimodal web-search agent.",
            "Return only compact valid JSON with exactly these fields:",
            '{"entity_quote":"...","information_need":"...","constraints":[],"search_query":"..."}',
            "Identify the visual entity and requested fact without guessing or including the answer.",
            f"Question: {row['question']}",
            f"Frozen image-only visual observation: {row['visual_observation']}",
            "Structured query action:",
        ]
    )


def completion(row: dict[str, Any]) -> str:
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


def collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(collect_keys(child) for child in value.values()), set())
    if isinstance(value, list):
        return set().union(*(collect_keys(child) for child in value), set())
    return set()


def load_score_audit(path: Path, method: str, partition: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    audit = read_json(path)
    if (
        audit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORES_READY"
        or audit.get("method") != method
        or audit.get("partition") != partition
    ):
        raise ValueError(f"invalid evidence score audit: {method} {partition}")
    score_path = Path(audit["output_paths"]["scores"])
    if sha256(score_path) != audit["output_hashes"]["scores"]:
        raise ValueError(f"evidence scores changed: {method} {partition}")
    return audit, {row["parent_group_id"]: row for row in read_jsonl(score_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_policy_freeze", type=Path, required=True)
    parser.add_argument("--rollouts_per_group", type=int, default=12)
    parser.add_argument("--max_query_tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=761943)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rollouts_per_group != 12:
        raise ValueError("matched query control budget is frozen to 12 downstream trajectories")

    evidence_freeze_path = args.evidence_policy_freeze.resolve()
    evidence_freeze = read_json(evidence_freeze_path)
    if evidence_freeze.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_FROZEN":
        raise ValueError("backward evidence policy is not frozen")
    for key, path in evidence_freeze["input_paths"].items():
        if sha256(Path(path)) != evidence_freeze["input_hashes"][key]:
            raise ValueError(f"evidence freeze input changed: {key}")

    training_path = Path(evidence_freeze["input_paths"]["training_freeze"])
    fit_path = Path(evidence_freeze["input_paths"]["train_fit"])
    internal_audit_path = Path(evidence_freeze["input_paths"]["internal_audit"])
    training, fit, internal_audit = map(read_json, (training_path, fit_path, internal_audit_path))
    control_path = Path(training["input_paths"]["control_freeze"])
    control = read_json(control_path)
    backup_path = Path(control["input_paths"]["backup_audit"])
    backup = read_json(backup_path)
    values_path = Path(control["input_paths"]["shared_answer_values"])
    evidence_action_audit_path = Path(control["input_paths"]["evidence_action_audit"])
    evidence_action_audit = read_json(evidence_action_audit_path)
    evidence_protocol_path = Path(evidence_action_audit["input_paths"]["freeze"])
    evidence_protocol = read_json(evidence_protocol_path)
    query_actions_path = Path(evidence_protocol["input_paths"]["query_actions_with_search"])
    query_edges_path = Path(backup["output_paths"]["query_edges"])

    for path, expected in (
        (control_path, training["input_hashes"]["control_freeze"]),
        (backup_path, control["input_hashes"]["backup_audit"]),
        (values_path, control["input_hashes"]["shared_answer_values"]),
        (evidence_action_audit_path, control["input_hashes"]["evidence_action_audit"]),
        (evidence_protocol_path, evidence_action_audit["input_hashes"]["freeze"]),
        (query_actions_path, evidence_protocol["input_hashes"]["query_actions_with_search"]),
        (query_edges_path, backup["output_hashes"]["query_edges"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"audited backward-query input changed: {path}")

    train_reference_path = Path(fit["input_paths"]["reference_score_audit"])
    train_dagig_path = Path(fit["input_paths"]["dagig_score_audit"])
    dagig_internal_method_path = Path(internal_audit["input_paths"]["dagig_method_audit"])
    dagig_internal_method = read_json(dagig_internal_method_path)
    internal_reference_path = Path(dagig_internal_method["input_paths"]["reference_score_audit"])
    internal_dagig_path = Path(dagig_internal_method["input_paths"]["method_score_audit"])
    train_reference_audit, train_reference = load_score_audit(train_reference_path, "reference", "policy_train")
    train_dagig_audit, train_dagig = load_score_audit(train_dagig_path, "dagig", "policy_train")
    internal_reference_audit, internal_reference = load_score_audit(internal_reference_path, "reference", "internal_holdout")
    internal_dagig_audit, internal_dagig = load_score_audit(internal_dagig_path, "dagig", "internal_holdout")

    evidence_groups: dict[str, dict[str, Any]] = {}
    for key in ("train_data", "internal_data"):
        path = Path(control["output_paths"][key])
        if sha256(path) != control["output_hashes"][key]:
            raise ValueError(f"evidence control data changed: {key}")
        evidence_groups.update({row["parent_group_id"]: row for row in read_jsonl(path)})
    reference_scores = {**train_reference, **internal_reference}
    dagig_scores = {**train_dagig, **internal_dagig}
    if set(evidence_groups) != set(reference_scores) or set(evidence_groups) != set(dagig_scores):
        raise ValueError("frozen evidence policy score universe is incomplete")

    values = {row["evidence_action_id"]: row for row in read_jsonl(values_path)}
    beta = float(training["training"]["beta"])
    evidence_policies: dict[str, list[float]] = {}
    query_values: dict[str, float] = {}
    local_query_values: dict[str, float] = {}
    for query_id, group in evidence_groups.items():
        action_ids = group["action_ids"]
        behavior = normalize([float(value) for value in group["behavior_probabilities"]])
        reference = reference_scores[query_id]
        current = dagig_scores[query_id]
        if reference["action_ids"] != action_ids or current["action_ids"] != action_ids:
            raise ValueError(f"evidence score action order changed: {query_id}")
        delta = [
            float(right) - float(left)
            for left, right in zip(reference["field_logprob_scores"], current["field_logprob_scores"])
        ]
        policy = softmax([math.log(probability) + beta * change for probability, change in zip(behavior, delta)])
        evidence_policies[query_id] = policy
        query_values[query_id] = sum(
            probability * float(values[action_id]["shared_answer_value"])
            for probability, action_id in zip(policy, action_ids)
        )
        modal = max(range(len(policy)), key=lambda index: (policy[index], action_ids[index]))
        local_query_values[query_id] = float(values[action_ids[modal]]["mode_child_success_probability"])

    query_actions = {row["query_id"]: row for row in read_jsonl(query_actions_path)}
    grouped_query_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(query_edges_path):
        grouped_query_edges[row["parent_id"]].append(row)
    edge_action_ids = {row["action_id"] for rows in grouped_query_edges.values() for row in rows}
    if edge_action_ids != set(query_actions) or edge_action_ids != set(evidence_groups):
        raise ValueError("query edges, query actions, and frozen evidence states differ")

    groups: list[dict[str, Any]] = []
    constant_outcome = 0
    for parent_id, rows in sorted(grouped_query_edges.items()):
        rows = sorted(rows, key=lambda row: row["action_id"])
        action_ids = [row["action_id"] for row in rows]
        behavior = normalize([float(row["behavior_probability"]) for row in rows])
        dag_values = [query_values[action_id] for action_id in action_ids]
        dagig = normalize([probability * value for probability, value in zip(behavior, dag_values)])
        observed: dict[str, list[float]] = {action_id: [] for action_id in action_ids}
        sampled_rewards: list[tuple[str, float]] = []
        for rollout in range(args.rollouts_per_group):
            rng = random.Random(f"backward-query-outcome:{args.seed}:{parent_id}:{rollout}")
            query_index = weighted_index(behavior, rng)
            query_id = action_ids[query_index]
            evidence_group = evidence_groups[query_id]
            evidence_index = weighted_index(evidence_policies[query_id], rng)
            evidence_id = evidence_group["action_ids"][evidence_index]
            value_row = values[evidence_id]
            answer_index = weighted_index(value_row["answer_policy_probabilities"], rng)
            reward = float(value_row["child_success_probabilities"][answer_index])
            sampled_rewards.append((query_id, reward))
        rewards = [reward for _, reward in sampled_rewards]
        center = mean(rewards)
        reward_std = math.sqrt(mean((reward - center) ** 2 for reward in rewards))
        constant_outcome += int(reward_std <= 1e-12)
        for query_id, reward in sampled_rewards:
            observed[query_id].append((reward - center) / reward_std if reward_std > 1e-12 else 0.0)
        groups.append(
            {
                "parent_id": parent_id,
                "sample_id": rows[0]["sample_id"],
                "partition": rows[0]["partition"],
                "action_ids": action_ids,
                "behavior": behavior,
                "dag_values": dag_values,
                "dagig": dagig,
                "local_values": [local_query_values[action_id] for action_id in action_ids],
                "outcome_values": [mean(observed[action_id]) if observed[action_id] else 0.0 for action_id in action_ids],
                "outcome_counts": [len(observed[action_id]) for action_id in action_ids],
                "outcome_reward_std": reward_std,
            }
        )

    train_groups = [group for group in groups if group["partition"] == "policy_train"]
    internal_groups = [group for group in groups if group["partition"] == "internal_holdout"]
    target_kl = mean(kl(group["dagig"], group["behavior"]) for group in train_groups)
    local_beta = find_scale(train_groups, "local_values", target_kl, log_values=True)
    outcome_eta = find_scale(train_groups, "outcome_values", target_kl, log_values=False)

    train_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    local_tvs: list[float] = []
    outcome_tvs: list[float] = []
    local_top: list[bool] = []
    outcome_top: list[bool] = []
    normalization_errors: list[float] = []
    duplicate_completion_groups = 0
    max_query_tokens = 0
    for group in groups:
        local = calibrated(group["behavior"], group["local_values"], local_beta, log_values=True)
        outcome = calibrated(group["behavior"], group["outcome_values"], outcome_eta, log_values=False)
        local_tvs.append(tv(group["dagig"], local))
        outcome_tvs.append(tv(group["dagig"], outcome))
        local_top.append(max(range(len(local)), key=local.__getitem__) == max(range(len(local)), key=group["dagig"].__getitem__))
        outcome_top.append(max(range(len(outcome)), key=outcome.__getitem__) == max(range(len(outcome)), key=group["dagig"].__getitem__))
        normalization_errors.extend(abs(sum(policy) - 1.0) for policy in (group["behavior"], local, outcome, group["dagig"]))
        source_rows = [query_actions[action_id] for action_id in group["action_ids"]]
        completions = [completion(row) for row in source_rows]
        duplicate_completion_groups += int(len(set(completions)) != len(completions))
        max_query_tokens = max(max_query_tokens, *(len(row["search_query"].split()) for row in source_rows))
        output_row = {
            "sample_id": group["sample_id"],
            "partition": group["partition"],
            "parent_group_id": group["parent_id"],
            "prompt": prompt(source_rows[0]),
            "completions": completions,
            "action_ids": group["action_ids"],
            "behavior_probabilities": group["behavior"],
            "local_target_probabilities": local,
            "outcome_target_probabilities": outcome,
            "dagig_target_probabilities": group["dagig"],
        }
        (train_rows if group["partition"] == "policy_train" else internal_rows).append(output_row)
        diagnostics.append(
            {
                "sample_id": group["sample_id"],
                "partition": group["partition"],
                "visual_parent_id": group["parent_id"],
                "query_action_ids": group["action_ids"],
                "behavior_probabilities": group["behavior"],
                "frozen_evidence_expected_values": group["dag_values"],
                "local_modal_descendant_values": group["local_values"],
                "outcome_values": group["outcome_values"],
                "outcome_counts": group["outcome_counts"],
                "outcome_reward_std": group["outcome_reward_std"],
                "dagig_target_probabilities": group["dagig"],
            }
        )

    metrics = {
        "samples": len({group["sample_id"] for group in groups}),
        "policy_train_samples": len({group["sample_id"] for group in train_groups}),
        "internal_holdout_samples": len({group["sample_id"] for group in internal_groups}),
        "visual_state_groups": len(groups),
        "policy_train_groups": len(train_groups),
        "internal_holdout_groups": len(internal_groups),
        "query_actions": sum(len(group["action_ids"]) for group in groups),
        "action_count_distribution": dict(sorted(Counter(len(group["action_ids"]) for group in groups).items())),
        "target_mean_kl_from_behavior_train": target_kl,
        "local_beta": local_beta,
        "outcome_eta": outcome_eta,
        "local_mean_kl_train": mean(kl(calibrated(group["behavior"], group["local_values"], local_beta, log_values=True), group["behavior"]) for group in train_groups),
        "outcome_mean_kl_train": mean(kl(calibrated(group["behavior"], group["outcome_values"], outcome_eta, log_values=False), group["behavior"]) for group in train_groups),
        "outcome_constant_group_rate": constant_outcome / len(groups),
        "dagig_local_mean_tv": mean(local_tvs),
        "dagig_outcome_mean_tv": mean(outcome_tvs),
        "dagig_local_top1_agreement": mean(local_top),
        "dagig_outcome_top1_agreement": mean(outcome_top),
        "max_target_normalization_error": max(normalization_errors),
        "duplicate_completion_groups": duplicate_completion_groups,
        "max_query_tokens": max_query_tokens,
        "rollouts_per_outcome_group": args.rollouts_per_group,
        "forbidden_public_keys": sorted(
            FORBIDDEN_KEYS.intersection(collect_keys([*train_rows, *internal_rows, *diagnostics]))
        ),
    }
    gates = {
        "evidence_policy_frozen": True,
        "complete_198_samples": metrics["samples"] == 198,
        "complete_158_40_sample_split": metrics["policy_train_samples"] == 158 and metrics["internal_holdout_samples"] == 40,
        "complete_594_visual_states": metrics["visual_state_groups"] == 594,
        "complete_474_120_group_split": metrics["policy_train_groups"] == 474 and metrics["internal_holdout_groups"] == 120,
        "complete_2954_query_actions": metrics["query_actions"] == 2954,
        "three_to_five_actions_per_state": min(metrics["action_count_distribution"]) >= 3 and max(metrics["action_count_distribution"]) <= 5,
        "no_duplicate_completions": metrics["duplicate_completion_groups"] == 0,
        "pre_frozen_queries_within_48_tokens": metrics["max_query_tokens"] <= args.max_query_tokens,
        "local_kl_matched_on_train": abs(metrics["local_mean_kl_train"] - target_kl) <= 1e-6,
        "outcome_kl_matched_on_train": abs(metrics["outcome_mean_kl_train"] - target_kl) <= 1e-6,
        "outcome_groups_nonconstant": metrics["outcome_constant_group_rate"] <= 0.05,
        "dagig_local_identifiable": metrics["dagig_local_mean_tv"] >= 0.01 and metrics["dagig_local_top1_agreement"] <= 0.95,
        "dagig_outcome_identifiable": metrics["dagig_outcome_mean_tv"] >= 0.01 and metrics["dagig_outcome_top1_agreement"] <= 0.95,
        "targets_normalized": metrics["max_target_normalization_error"] <= 1e-10,
        "same_query_action_universe": True,
        "frozen_evidence_and_answer_descendants": True,
        "runtime_controls_have_no_gold_or_qrels": not metrics["forbidden_public_keys"],
        "internal_unused_for_scale_or_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_BACKWARD_QUERY_CONTROLS_FROZEN" if all(gates.values()) else "DAGIG_V6_BACKWARD_QUERY_CONTROLS_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_path = output / "v6_backward_query_control_targets_train.jsonl"
    internal_path = output / "v6_backward_query_control_targets_internal_no_labels.jsonl"
    diagnostics_path = output / "v6_backward_query_control_diagnostics_no_gold.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(internal_path, internal_rows)
    write_jsonl(diagnostics_path, diagnostics)
    input_paths = {
        "evidence_policy_freeze": evidence_freeze_path,
        "evidence_training_freeze": training_path,
        "evidence_control_freeze": control_path,
        "backup_audit": backup_path,
        "query_edges": query_edges_path,
        "query_actions": query_actions_path,
        "shared_answer_values": values_path,
        "evidence_train_reference_score_audit": train_reference_path,
        "evidence_train_dagig_score_audit": train_dagig_path,
        "evidence_internal_reference_score_audit": internal_reference_path,
        "evidence_internal_dagig_score_audit": internal_dagig_path,
        "builder": Path(__file__).resolve(),
    }
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_fixed_evidence_answer_query_controls_v1",
        "method_contract": {
            "no_credit": "frozen behavior distribution over the same structured query actions",
            "local_ig": "modal frozen evidence action followed by the modal shared-answer child value",
            "outcome": "12 sampled query-to-frozen-evidence-to-frozen-answer trajectories",
            "dagig": "exact posterior pi_b(q|v) * sum_e pi_E(e|q) sum_a pi_A(a|e) P_success",
        },
        "target_keys": {
            "no_credit": "behavior_probabilities",
            "local_ig": "local_target_probabilities",
            "outcome": "outcome_target_probabilities",
            "dagig": "dagig_target_probabilities",
        },
        "metrics": metrics,
        "gates": gates,
        "seed": args.seed,
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "input_hashes": {key: sha256(path) for key, path in input_paths.items()},
        "output_paths": {
            "train_data": str(train_path),
            "internal_data": str(internal_path),
            "diagnostics": str(diagnostics_path),
        },
        "output_hashes": {
            "train_data": sha256(train_path),
            "internal_data": sha256(internal_path),
            "diagnostics": sha256(diagnostics_path),
        },
        "api_calls": 0,
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_BACKWARD_QUERY_CONTROL_FREEZE.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
