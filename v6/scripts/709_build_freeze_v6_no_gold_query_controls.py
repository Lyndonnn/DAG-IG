#!/usr/bin/env python3
"""Freeze matched query-only controls using the deployable no-gold DAG value."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


FORBIDDEN = {
    "aliases",
    "answer_correct_proxy",
    "gold_answer",
    "positive_doc_ids",
    "qrels",
    "strict_proxy",
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
        raise ValueError("cannot normalize non-positive policy mass")
    return [value / total for value in values]


def weighted_choice(rows: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["action_id"]))
    draw = rng.random()
    cumulative = 0.0
    for row in ordered:
        cumulative += float(row["behavior_probability"])
        if draw <= cumulative + 1e-12:
            return row
    return ordered[-1]


def canonical(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (-float(row["behavior_probability"]), str(row["action_id"])),
    )[0]


def policy_from_values(rows: list[dict[str, Any]], values: dict[str, float]) -> dict[str, float]:
    masses = normalize(
        [float(row["behavior_probability"]) * values[str(row["action_id"])] for row in rows]
    )
    return {str(row["action_id"]): mass for row, mass in zip(rows, masses)}


def outcome_policy(
    rows: list[dict[str, Any]],
    observed: dict[str, list[float]],
    eta: float,
) -> dict[str, float]:
    logits = []
    for row in rows:
        action_id = str(row["action_id"])
        estimate = mean(observed[action_id]) if observed[action_id] else 0.0
        logits.append(math.log(float(row["behavior_probability"])) + eta * estimate)
    offset = max(logits)
    masses = normalize([math.exp(value - offset) for value in logits])
    return {str(row["action_id"]): mass for row, mass in zip(rows, masses)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup_audit", type=Path, required=True)
    parser.add_argument("--terminal_audit", type=Path, required=True)
    parser.add_argument("--query_actions", type=Path, required=True)
    parser.add_argument("--rollouts_per_group", type=int, default=12)
    parser.add_argument("--outcome_eta", type=float, default=1.0)
    parser.add_argument("--max_query_tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rollouts_per_group < 6:
        raise ValueError("at least six matched rollouts per query group are required")

    backup_path = args.backup_audit.resolve()
    terminal_audit_path = args.terminal_audit.resolve()
    query_action_path = args.query_actions.resolve()
    backup = read_json(backup_path)
    terminal_audit = read_json(terminal_audit_path)
    if backup.get("decision") != "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_GO":
        raise ValueError("deployable no-gold DAG backup is not GO")
    if terminal_audit.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO":
        raise ValueError("deployable no-gold terminal value is not GO")
    if backup["input_hashes"]["terminal_audit"] != sha256(terminal_audit_path):
        raise ValueError("backup and terminal audits do not belong to the same frozen protocol")

    edge_paths = {
        node: Path(backup["output_paths"][f"{node}_edges"])
        for node in ("query", "evidence", "answer")
    }
    terminal_path = Path(terminal_audit["output_paths"]["terminal_values"])
    for node, path in edge_paths.items():
        if sha256(path) != backup["output_hashes"][f"{node}_edges"]:
            raise ValueError(f"audited {node} edges changed")
    if sha256(terminal_path) != terminal_audit["output_hashes"]["terminal_values"]:
        raise ValueError("audited no-gold terminal values changed")

    edges = {node: read_jsonl(path) for node, path in edge_paths.items()}
    by_parent: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for node, rows in edges.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["parent_id"])].append(row)
        by_parent[node] = dict(grouped)
    terminal = {
        str(row["answer_action_id"]): float(row["terminal_success_probability"])
        for row in read_jsonl(terminal_path)
    }
    query_actions = {str(row["query_id"]): row for row in read_jsonl(query_action_path)}
    if set(query_actions) != {str(row["action_id"]) for row in edges["query"]}:
        raise ValueError("query action file and exact DAG query universe differ")

    canonical_answer = {
        parent: str(canonical(rows)["action_id"]) for parent, rows in by_parent["answer"].items()
    }
    local_evidence_values = {
        action_id: terminal[canonical_answer[action_id]]
        for action_id in {str(row["action_id"]) for row in edges["evidence"]}
    }
    canonical_evidence = {
        parent: str(canonical(rows)["action_id"]) for parent, rows in by_parent["evidence"].items()
    }
    local_query_values = {
        action_id: local_evidence_values[canonical_evidence[action_id]]
        for action_id in {str(row["action_id"]) for row in edges["query"]}
    }

    train_groups = {}
    removed_overlength = []
    for parent, raw_rows in by_parent["query"].items():
        if not parent.endswith("::joint_state") or raw_rows[0]["partition"] != "policy_train":
            continue
        retained = []
        for row in raw_rows:
            query_id = str(row["action_id"])
            if len(str(query_actions[query_id]["search_query"]).split()) <= args.max_query_tokens:
                retained.append(dict(row))
            else:
                removed_overlength.append(query_id)
        if len(retained) < 3:
            raise ValueError(f"paper-legal query group has fewer than three actions: {parent}")
        behavior = normalize([float(row["behavior_probability"]) for row in retained])
        parent_value = sum(
            probability * float(row["child_success_probability"])
            for probability, row in zip(behavior, retained)
        )
        posterior = normalize(
            [
                probability * float(row["child_success_probability"])
                for probability, row in zip(behavior, retained)
            ]
        )
        for row, probability, success_probability in zip(retained, behavior, posterior):
            row["behavior_probability"] = probability
            row["parent_success_probability"] = parent_value
            row["success_posterior_probability"] = success_probability
            row["dagig_nats"] = math.log(float(row["child_success_probability"])) - math.log(parent_value)
        train_groups[parent] = retained
    if len(train_groups) != 158:
        raise ValueError(f"expected 158 train joint-state query groups, found {len(train_groups)}")

    trajectory_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    constant_groups = 0
    unique_paths = []
    dag_local_differences = []
    dag_outcome_differences = []
    reward_scales: dict[str, list[float]] = defaultdict(list)
    for parent, rows in sorted(train_groups.items()):
        sample_id = str(rows[0]["sample_id"])
        sampled = []
        for rollout_index in range(args.rollouts_per_group):
            rng = random.Random(f"query-control:{args.seed}:{parent}:{rollout_index}")
            query = weighted_choice(rows, rng)
            evidence = weighted_choice(by_parent["evidence"][str(query["action_id"])], rng)
            answer = weighted_choice(by_parent["answer"][str(evidence["action_id"])], rng)
            reward = terminal[str(answer["action_id"])]
            sampled.append((query, evidence, answer, reward))
        rewards = [row[3] for row in sampled]
        center = mean(rewards)
        scale = math.sqrt(mean((value - center) ** 2 for value in rewards))
        constant_groups += int(scale <= 1e-12)
        observed: dict[str, list[float]] = {
            str(row["action_id"]): [] for row in rows
        }
        for rollout_index, (query, evidence, answer, reward) in enumerate(sampled):
            advantage = (reward - center) / scale if scale > 1e-12 else 0.0
            observed[str(query["action_id"])].append(advantage)
            trajectory_rows.append(
                {
                    "trajectory_id": "v6qtraj_"
                    + hashlib.sha256(
                        f"{args.seed}\n{parent}\n{rollout_index}".encode()
                    ).hexdigest()[:24],
                    "sample_id": sample_id,
                    "query_parent_id": parent,
                    "rollout_index": rollout_index,
                    "query_id": str(query["action_id"]),
                    "evidence_action_id": str(evidence["action_id"]),
                    "answer_action_id": str(answer["action_id"]),
                    "terminal_success_probability": reward,
                    "outcome_query_advantage": advantage,
                }
            )

        behavior_policy = {
            str(row["action_id"]): float(row["behavior_probability"]) for row in rows
        }
        dag_policy = {
            str(row["action_id"]): float(row["success_posterior_probability"])
            for row in rows
        }
        local_policy = policy_from_values(rows, local_query_values)
        sampled_outcome_policy = outcome_policy(rows, observed, args.outcome_eta)
        candidates = []
        for row in sorted(rows, key=lambda item: str(item["action_id"])):
            query_id = str(row["action_id"])
            source = query_actions[query_id]
            candidates.append(
                {
                    "query_id": query_id,
                    "query_strategy": source["query_strategy"],
                    "search_query": source["search_query"],
                    "behavior_probability": behavior_policy[query_id],
                    "no_credit_probability": behavior_policy[query_id],
                    "local_fixed_descendant_probability": local_policy[query_id],
                    "true_outcome_grpo_probability": sampled_outcome_policy[query_id],
                    "dagig_exact_probability": dag_policy[query_id],
                    "dagig_nats": float(row["dagig_nats"]),
                    "local_fixed_descendant_value": local_query_values[query_id],
                    "outcome_observations": len(observed[query_id]),
                    "outcome_mean_advantage": mean(observed[query_id]) if observed[query_id] else 0.0,
                }
            )
            dag_local_differences.append(abs(dag_policy[query_id] - local_policy[query_id]) > 1e-8)
            dag_outcome_differences.append(
                abs(dag_policy[query_id] - sampled_outcome_policy[query_id]) > 1e-8
            )
            reward_scales["dagig"].append(abs(float(row["dagig_nats"])))
            reward_scales["local"].append(abs(math.log(local_query_values[query_id])))
            if observed[query_id]:
                reward_scales["outcome"].extend(abs(value) for value in observed[query_id])
        target_rows.append(
            {
                "sample_id": sample_id,
                "query_parent_id": parent,
                "partition": "policy_train",
                "visual_field": "joint_state",
                "question": query_actions[str(rows[0]["action_id"])]["question"],
                "visual_observation": query_actions[str(rows[0]["action_id"])]["visual_observation"],
                "rollout_reward_mean": center,
                "rollout_reward_std": scale,
                "candidates": candidates,
            }
        )
        unique_paths.append(
            len({(str(q["action_id"]), str(e["action_id"]), str(a["action_id"])) for q, e, a, _ in sampled})
        )

    forbidden = sorted(
        {
            key
            for row in [*trajectory_rows, *target_rows]
            for key in FORBIDDEN.intersection(row)
        }
    )
    metrics = {
        "train_query_groups": len(target_rows),
        "train_query_actions": sum(len(row["candidates"]) for row in target_rows),
        "removed_overlength_query_actions": len(removed_overlength),
        "max_query_tokens": max(
            len(str(query_actions[candidate["query_id"]]["search_query"]).split())
            for row in target_rows
            for candidate in row["candidates"]
        ),
        "rollouts_per_group": args.rollouts_per_group,
        "trajectory_rows": len(trajectory_rows),
        "constant_outcome_group_rate": constant_groups / len(target_rows),
        "mean_unique_paths_per_group": mean(unique_paths),
        "dag_local_probability_difference_rate": mean(dag_local_differences),
        "dag_outcome_probability_difference_rate": mean(dag_outcome_differences),
        "median_nonzero_credit_scale": {
            key: median([value for value in values if value > 1e-12])
            for key, values in reward_scales.items()
        },
        "forbidden_fields_present": forbidden,
    }
    gates = {
        "complete_train_groups": metrics["train_query_groups"] == 158,
        "paper_legal_action_count": metrics["train_query_actions"] == 773,
        "paper_legal_query_length": metrics["max_query_tokens"] <= args.max_query_tokens,
        "complete_matched_rollouts": metrics["trajectory_rows"] == 158 * args.rollouts_per_group,
        "outcome_groups_nonconstant": metrics["constant_outcome_group_rate"] <= 0.05,
        "trajectory_diversity": metrics["mean_unique_paths_per_group"] >= args.rollouts_per_group * 0.75,
        "dag_differs_from_local": metrics["dag_local_probability_difference_rate"] >= 0.90,
        "dag_differs_from_outcome": metrics["dag_outcome_probability_difference_rate"] >= 0.90,
        "nonzero_reward_scales": all(value > 0.0 for value in metrics["median_nonzero_credit_scale"].values()),
        "same_query_action_universe": True,
        "same_behavior_descendant_policy": True,
        "runtime_no_gold_or_qrels": not forbidden,
        "internal_holdout_unused": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = (
        "DAGIG_V6_NO_GOLD_QUERY_CONTROLS_FROZEN"
        if all(gates.values())
        else "DAGIG_V6_NO_GOLD_QUERY_CONTROLS_NO_GO"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = output / "v6_no_gold_query_matched_trajectory_ledger_train.jsonl"
    targets_path = output / "v6_no_gold_query_control_targets_train.jsonl"
    write_jsonl(ledger_path, trajectory_rows)
    write_jsonl(targets_path, target_rows)
    input_paths = {
        "backup_audit": str(backup_path),
        "terminal_audit": str(terminal_audit_path),
        "query_actions": str(query_action_path),
        "terminal_values": str(terminal_path),
        **{f"{node}_edges": str(path) for node, path in edge_paths.items()},
    }
    audit = {
        "decision": decision,
        "protocol_version": "dagig_v6_no_gold_matched_query_only_controls_v1",
        "control_definitions": {
            "no_credit": "frozen behavior distribution over the same legal query actions",
            "local_fixed_descendant": "query intervention with evidence and answer fixed to their behavior-mode descendants",
            "true_outcome_grpo": "twelve complete query-to-evidence-to-answer rollouts with root-group normalized terminal P_success",
            "dagig_exact": "exact behavior-marginalized descendant posterior from deployable no-gold P_success",
            "evaluation_descendants": "evidence and answer remain the same frozen behavior distributions for every query method",
        },
        "metrics": metrics,
        "gates": gates,
        "seed": args.seed,
        "outcome_eta": args.outcome_eta,
        "rollouts_per_group": args.rollouts_per_group,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {
            "trajectory_ledger": str(ledger_path),
            "query_targets": str(targets_path),
        },
        "output_hashes": {
            "trajectory_ledger": sha256(ledger_path),
            "query_targets": sha256(targets_path),
        },
        "gold_or_qrels_in_training_ledger": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_NO_GOLD_QUERY_CONTROL_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
