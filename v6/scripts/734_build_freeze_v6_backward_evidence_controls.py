#!/usr/bin/env python3
"""Build matched evidence controls after freezing the downstream answer policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dagig_causal.evidence_prompt import build_evidence_selection_prompt  # noqa: E402


METHODS = ("no_credit", "local_ig", "outcome", "dagig")
STRATEGY_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("cannot normalize evidence target")
    return [value / total for value in values]


def softmax(logits: list[float]) -> list[float]:
    offset = max(logits)
    return normalize([math.exp(value - offset) for value in logits])


def kl(policy: list[float], behavior: list[float]) -> float:
    return sum(p * math.log(p / b) for p, b in zip(policy, behavior) if p > 0.0)


def tv(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def calibrated(behavior: list[float], values: list[float], scale: float, *, log_values: bool) -> list[float]:
    scores = [math.log(max(value, 1e-8)) if log_values else value for value in values]
    return softmax([math.log(probability) + scale * score for probability, score in zip(behavior, scores)])


def find_scale(groups: list[dict[str, Any]], key: str, target_kl: float, *, log_values: bool) -> float:
    def objective(scale: float) -> float:
        return mean(
            kl(calibrated(group["behavior"], group[key], scale, log_values=log_values), group["behavior"])
            for group in groups
        )
    low, high = 0.0, 1.0
    while objective(high) < target_kl and high < 4096.0:
        high *= 2.0
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


def completion(row: dict[str, Any]) -> str:
    docs = sorted(row["candidate_docs"], key=lambda doc: int(doc["rank"]))
    mapping = {doc["doc_id"]: f"D{index}" for index, doc in enumerate(docs, 1)}
    return json.dumps(
        {"selected_evidence_ids": [mapping[doc_id] for doc_id in row["selected_doc_ids"]]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared_answer_value_audit", type=Path, required=True)
    parser.add_argument("--backup_audit", type=Path, required=True)
    parser.add_argument("--evidence_action_audit", type=Path, required=True)
    parser.add_argument("--rollouts_per_group", type=int, default=12)
    parser.add_argument("--seed", type=int, default=761943)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rollouts_per_group != 12:
        raise ValueError("paper control budget is frozen to 12 outcome trajectories per query state")

    paths = {
        "shared_answer_value_audit": args.shared_answer_value_audit.resolve(),
        "backup_audit": args.backup_audit.resolve(),
        "evidence_action_audit": args.evidence_action_audit.resolve(),
    }
    value_audit, backup, evidence_audit = [read_json(paths[key]) for key in paths]
    if value_audit.get("decision") != "DAGIG_V6_SHARED_ANSWER_VALUES_GO":
        raise ValueError("shared answer values are not GO")
    if backup.get("decision") != "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_GO":
        raise ValueError("terminal backup is not GO")
    if evidence_audit.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_ACTIONS_GO":
        raise ValueError("evidence actions are not GO")
    values_path = Path(value_audit["output_paths"]["shared_answer_values"])
    evidence_edges_path = Path(backup["output_paths"]["evidence_edges"])
    evidence_actions_path = Path(evidence_audit["output_paths"]["evidence_actions"])
    for path, expected in (
        (values_path, value_audit["output_hashes"]["shared_answer_values"]),
        (evidence_edges_path, backup["output_hashes"]["evidence_edges"]),
        (evidence_actions_path, evidence_audit["output_hashes"]["evidence_actions"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"audited input changed: {path}")

    values = {row["evidence_action_id"]: row for row in read_jsonl(values_path)}
    actions = {row["evidence_action_id"]: row for row in read_jsonl(evidence_actions_path)}
    grouped_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(evidence_edges_path):
        grouped_edges[row["parent_id"]].append(row)
    if set(values) != set(actions) or set(values) != {row["action_id"] for rows in grouped_edges.values() for row in rows}:
        raise ValueError("shared answer values and evidence action universes differ")
    if any(len(rows) != 5 for rows in grouped_edges.values()):
        raise ValueError("every query state must expose exactly five evidence actions")

    groups: list[dict[str, Any]] = []
    constant_outcome = 0
    for query_id, edge_rows in sorted(grouped_edges.items()):
        edge_rows.sort(key=lambda row: STRATEGY_ORDER.index(actions[row["action_id"]]["evidence_strategy"]))
        behavior = normalize([float(row["behavior_probability"]) for row in edge_rows])
        evidence_values = [float(values[row["action_id"]]["shared_answer_value"]) for row in edge_rows]
        dagig = normalize([probability * max(value, 1e-8) for probability, value in zip(behavior, evidence_values)])
        local_values = [float(values[row["action_id"]]["mode_child_success_probability"]) for row in edge_rows]
        observed: dict[str, list[float]] = {row["action_id"]: [] for row in edge_rows}
        sampled_rewards: list[tuple[str, float]] = []
        for rollout in range(args.rollouts_per_group):
            rng = random.Random(f"backward-evidence-outcome:{args.seed}:{query_id}:{rollout}")
            evidence_index = weighted_index(behavior, rng)
            evidence_id = edge_rows[evidence_index]["action_id"]
            value_row = values[evidence_id]
            answer_index = weighted_index(value_row["answer_policy_probabilities"], rng)
            reward = float(value_row["child_success_probabilities"][answer_index])
            sampled_rewards.append((evidence_id, reward))
        rewards = [reward for _, reward in sampled_rewards]
        center = mean(rewards)
        reward_std = math.sqrt(mean((reward - center) ** 2 for reward in rewards))
        constant_outcome += int(reward_std <= 1e-12)
        for evidence_id, reward in sampled_rewards:
            observed[evidence_id].append((reward - center) / reward_std if reward_std > 1e-12 else 0.0)
        outcome_values = [mean(observed[row["action_id"]]) if observed[row["action_id"]] else 0.0 for row in edge_rows]
        groups.append(
            {
                "query_id": query_id,
                "sample_id": edge_rows[0]["sample_id"],
                "partition": edge_rows[0]["partition"],
                "rows": edge_rows,
                "behavior": behavior,
                "evidence_values": evidence_values,
                "dagig": dagig,
                "local_values": local_values,
                "outcome_values": outcome_values,
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
    for group in groups:
        local = calibrated(group["behavior"], group["local_values"], local_beta, log_values=True)
        outcome = calibrated(group["behavior"], group["outcome_values"], outcome_eta, log_values=False)
        local_tvs.append(tv(group["dagig"], local))
        outcome_tvs.append(tv(group["dagig"], outcome))
        local_top.append(max(range(5), key=group["dagig"].__getitem__) == max(range(5), key=local.__getitem__))
        outcome_top.append(max(range(5), key=group["dagig"].__getitem__) == max(range(5), key=outcome.__getitem__))
        normalization_errors.extend(abs(sum(policy) - 1.0) for policy in (group["behavior"], local, outcome, group["dagig"]))
        action_rows = [actions[row["action_id"]] for row in group["rows"]]
        output_row = {
            "sample_id": group["sample_id"],
            "partition": group["partition"],
            "parent_group_id": group["query_id"],
            "prompt": build_evidence_selection_prompt(action_rows[0]),
            "completions": [completion(row) for row in action_rows],
            "action_ids": [row["action_id"] for row in group["rows"]],
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
                "query_id": group["query_id"],
                "action_ids": output_row["action_ids"],
                "behavior_probabilities": group["behavior"],
                "shared_answer_values": group["evidence_values"],
                "local_values": group["local_values"],
                "outcome_values": group["outcome_values"],
                "outcome_reward_std": group["outcome_reward_std"],
                "dagig_target_probabilities": group["dagig"],
            }
        )

    metrics = {
        "samples": len({group["sample_id"] for group in groups}),
        "policy_train_samples": len({group["sample_id"] for group in train_groups}),
        "internal_holdout_samples": len({group["sample_id"] for group in internal_groups}),
        "policy_train_groups": len(train_groups),
        "internal_holdout_groups": len(internal_groups),
        "evidence_actions": len(actions),
        "actions_per_group": 5,
        "rollouts_per_outcome_group": args.rollouts_per_group,
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
        "max_normalization_error": max(normalization_errors),
        "strategy_distribution": dict(sorted(Counter(row["evidence_strategy"] for row in actions.values()).items())),
    }
    gates = {
        "complete_198_samples": metrics["samples"] == 198,
        "complete_158_40_split": metrics["policy_train_samples"] == 158 and metrics["internal_holdout_samples"] == 40,
        "complete_2954_query_states": len(groups) == 2954,
        "complete_14770_evidence_actions": metrics["evidence_actions"] == 14770,
        "all_query_states_have_five_actions": len(groups) * 5 == metrics["evidence_actions"],
        "local_kl_matched": abs(metrics["local_mean_kl_train"] - target_kl) <= 1e-6,
        "outcome_kl_matched": abs(metrics["outcome_mean_kl_train"] - target_kl) <= 1e-6,
        "outcome_groups_nonconstant": metrics["outcome_constant_group_rate"] <= 0.05,
        "dagig_local_identifiable": metrics["dagig_local_mean_tv"] >= 0.01 and metrics["dagig_local_top1_agreement"] <= 0.95,
        "dagig_outcome_identifiable": metrics["dagig_outcome_mean_tv"] >= 0.01 and metrics["dagig_outcome_top1_agreement"] <= 0.95,
        "targets_normalized": metrics["max_normalization_error"] <= 1e-10,
        "shared_answer_policy_frozen_for_all_methods": True,
        "same_query_state_and_action_universe": True,
        "internal_holdout_unused_for_scale_or_training": True,
        "runtime_controls_have_no_gold_or_qrels": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_BACKWARD_EVIDENCE_CONTROLS_FROZEN" if all(gates.values()) else "DAGIG_V6_BACKWARD_EVIDENCE_CONTROLS_NO_GO"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "v6_backward_evidence_control_targets_train.jsonl"
    internal_path = output_dir / "v6_backward_evidence_control_targets_internal_no_labels.jsonl"
    diagnostics_path = output_dir / "v6_backward_evidence_control_diagnostics_no_gold.jsonl"
    for path, rows in ((train_path, train_rows), (internal_path, internal_rows), (diagnostics_path, diagnostics)):
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    input_paths = {**{key: str(path) for key, path in paths.items()}, "shared_answer_values": str(values_path), "evidence_edges": str(evidence_edges_path), "evidence_actions": str(evidence_actions_path)}
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_fixed_answer_all_query_states_evidence_v1",
        "method_contract": {
            "no_credit": "frozen evidence behavior distribution",
            "local_ig": "P_success of the modal answer under the shared answer policy",
            "outcome": "12 sampled evidence-to-answer trajectories under the shared answer policy",
            "dagig": "exact posterior pi_b(e|q) * sum_a pi_A(a|e) P_success(a,e) / V(q)",
        },
        "target_keys": {"no_credit": "behavior_probabilities", "local_ig": "local_target_probabilities", "outcome": "outcome_target_probabilities", "dagig": "dagig_target_probabilities"},
        "metrics": metrics,
        "gates": gates,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {"train_data": str(train_path), "internal_data": str(internal_path), "diagnostics": str(diagnostics_path)},
        "output_hashes": {"train_data": sha256(train_path), "internal_data": sha256(internal_path), "diagnostics": sha256(diagnostics_path)},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_CONTROL_FREEZE.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
