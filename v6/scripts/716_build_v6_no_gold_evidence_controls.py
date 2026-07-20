#!/usr/bin/env python3
"""Build matched evidence-node controls under the frozen DAG-IG query policy."""

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


METHODS = ("no_credit", "local_listwise", "outcome_listwise", "dagig_posterior")
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
    if not math.isfinite(total) or total <= 0:
        raise ValueError("cannot normalize evidence target")
    return [value / total for value in values]


def softmax(logits: list[float]) -> list[float]:
    offset = max(logits)
    return normalize([math.exp(value - offset) for value in logits])


def kl(policy: list[float], behavior: list[float]) -> float:
    return sum(p * math.log(p / b) for p, b in zip(policy, behavior) if p > 0)


def tv(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def completion(row: dict[str, Any]) -> str:
    docs = sorted(row["candidate_docs"], key=lambda doc: int(doc["rank"]))
    mapping = {doc["doc_id"]: f"D{index}" for index, doc in enumerate(docs, 1)}
    return json.dumps(
        {"selected_evidence_ids": [mapping[doc_id] for doc_id in row["selected_doc_ids"]]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def weighted_choice(rows: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    draw = rng.random()
    cumulative = 0.0
    for row in rows:
        cumulative += float(row["behavior_probability"])
        if draw <= cumulative + 1e-12:
            return row
    return rows[-1]


def calibrated_policy(rows: list[dict[str, Any]], values: list[float], scale: float, log_values: bool) -> list[float]:
    behavior = [float(row["behavior_probability"]) for row in rows]
    scores = [math.log(max(value, 1e-8)) if log_values else value for value in values]
    return softmax([math.log(probability) + scale * score for probability, score in zip(behavior, scores)])


def find_scale(groups: list[dict[str, Any]], value_key: str, target_kl: float, log_values: bool) -> float:
    def objective(scale: float) -> float:
        return mean(
            kl(
                calibrated_policy(group["rows"], group[value_key], scale, log_values),
                group["behavior"],
            )
            for group in groups
        )

    low, high = 0.0, 1.0
    while objective(high) < target_kl and high < 4096:
        high *= 2
    for _ in range(80):
        middle = (low + high) / 2
        if objective(middle) < target_kl:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_main_freeze", type=Path, required=True)
    parser.add_argument("--dagig_train_fit", type=Path, required=True)
    parser.add_argument("--dagig_internal_scores", type=Path, required=True)
    parser.add_argument("--evidence_actions", type=Path, required=True)
    parser.add_argument("--evidence_edges", type=Path, required=True)
    parser.add_argument("--answer_edges", type=Path, required=True)
    parser.add_argument("--rollouts_per_group", type=int, default=12)
    parser.add_argument("--seed", type=int, default=761943)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rollouts_per_group != 12:
        raise ValueError("paper control budget is frozen to 12 outcome rollouts per group")

    paths = {
        "query_main_freeze": args.query_main_freeze.resolve(),
        "dagig_train_fit": args.dagig_train_fit.resolve(),
        "dagig_internal_scores": args.dagig_internal_scores.resolve(),
        "evidence_actions": args.evidence_actions.resolve(),
        "evidence_edges": args.evidence_edges.resolve(),
        "answer_edges": args.answer_edges.resolve(),
    }
    query_freeze = read_json(paths["query_main_freeze"])
    if query_freeze.get("decision") != "DAGIG_V6_QUERY_NODE_MAIN_FROZEN":
        raise ValueError("query node is not frozen")
    fit_audit = read_json(paths["dagig_train_fit"])
    score_audit = read_json(paths["dagig_internal_scores"])
    if fit_audit.get("decision") != "DAGIG_V6_TRUE_CONTROL_QUERY_TRAIN_FIT_GO" or fit_audit.get("method") != "dagig_exact":
        raise ValueError("DAG-IG train query fit is not GO")
    if score_audit.get("decision") != "DAGIG_V6_QUERY_SELECTOR_SCORES_READY" or score_audit.get("method") != "dagig_exact":
        raise ValueError("DAG-IG internal candidate scores are not ready")
    fit_path = Path(fit_audit["output_paths"]["private_fit"])
    score_path = Path(score_audit["output_paths"]["scores"])
    if sha256(fit_path) != fit_audit["output_hashes"]["private_fit"] or sha256(score_path) != score_audit["output_hashes"]["scores"]:
        raise ValueError("frozen query selections changed")

    selected_queries: dict[str, str] = {}
    for row in read_jsonl(fit_path):
        selected_queries[row["sample_id"]] = row["action_ids"][max(range(len(row["posterior"])), key=row["posterior"].__getitem__)]
    for row in read_jsonl(score_path):
        if row["sample_id"] in selected_queries:
            raise ValueError("sample appears in train and internal query selections")
        selected_queries[row["sample_id"]] = row["selected_query_id"]
    if len(selected_queries) != 198:
        raise ValueError(f"expected 198 fixed query states, found {len(selected_queries)}")

    action_source = {row["evidence_action_id"]: row for row in read_jsonl(paths["evidence_actions"])}
    evidence_edges = read_jsonl(paths["evidence_edges"])
    answer_edges = read_jsonl(paths["answer_edges"])
    selected_query_ids = set(selected_queries.values())
    edge_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_edges:
        if row["parent_id"] in selected_query_ids:
            edge_groups[row["parent_id"]].append(row)
    answer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answer_edges:
        if row["parent_id"] in action_source:
            answer_groups[row["parent_id"]].append(row)
    if set(edge_groups) != selected_query_ids or any(len(rows) != 5 for rows in edge_groups.values()):
        raise ValueError("fixed-query evidence action matrix is incomplete")

    groups: list[dict[str, Any]] = []
    constant_outcome_groups = 0
    for query_id, rows in sorted(edge_groups.items()):
        rows = sorted(rows, key=lambda row: STRATEGY_ORDER.index(action_source[row["action_id"]]["evidence_strategy"]))
        behavior = normalize([float(row["behavior_probability"]) for row in rows])
        for row, probability in zip(rows, behavior):
            row["behavior_probability"] = probability
        dagig = normalize([float(row["success_posterior_probability"]) for row in rows])
        local_values = []
        for row in rows:
            children = sorted(
                answer_groups[row["action_id"]],
                key=lambda child: (-float(child["behavior_probability"]), str(child["action_id"])),
            )
            if not children:
                raise ValueError(f"missing answer descendants: {row['action_id']}")
            local_values.append(float(children[0]["child_success_probability"]))
        sampled: list[tuple[str, float]] = []
        for rollout in range(args.rollouts_per_group):
            rng = random.Random(f"evidence-outcome:{args.seed}:{query_id}:{rollout}")
            evidence = weighted_choice(rows, rng)
            answers = sorted(answer_groups[evidence["action_id"]], key=lambda row: str(row["action_id"]))
            answer = weighted_choice(answers, rng)
            sampled.append((str(evidence["action_id"]), float(answer["child_success_probability"])))
        rewards = [reward for _, reward in sampled]
        center = mean(rewards)
        scale = math.sqrt(mean((reward - center) ** 2 for reward in rewards))
        constant_outcome_groups += int(scale <= 1e-12)
        observed: dict[str, list[float]] = {str(row["action_id"]): [] for row in rows}
        for action_id, reward in sampled:
            observed[action_id].append((reward - center) / scale if scale > 1e-12 else 0.0)
        outcome_advantages = [mean(observed[str(row["action_id"])]) if observed[str(row["action_id"])] else 0.0 for row in rows]
        groups.append(
            {
                "query_id": query_id,
                "sample_id": rows[0]["sample_id"],
                "partition": rows[0]["partition"],
                "rows": rows,
                "behavior": behavior,
                "dagig": dagig,
                "local_values": local_values,
                "outcome_advantages": outcome_advantages,
                "outcome_reward_std": scale,
            }
        )

    train_groups = [group for group in groups if group["partition"] == "policy_train"]
    internal_groups = [group for group in groups if group["partition"] == "internal_holdout"]
    if len(train_groups) != 158 or len(internal_groups) != 40:
        raise ValueError("fixed-query partition matrix differs from 158/40")
    target_kl = mean(kl(group["dagig"], group["behavior"]) for group in train_groups)
    local_beta = find_scale(train_groups, "local_values", target_kl, True)
    outcome_eta = find_scale(train_groups, "outcome_advantages", target_kl, False)

    target_rows: list[dict[str, Any]] = []
    selected_actions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    normalization_errors: list[float] = []
    for group in groups:
        local = calibrated_policy(group["rows"], group["local_values"], local_beta, True)
        outcome = calibrated_policy(group["rows"], group["outcome_advantages"], outcome_eta, False)
        actions = [action_source[row["action_id"]] for row in group["rows"]]
        completions = [completion(row) for row in actions]
        for values in (group["behavior"], local, outcome, group["dagig"]):
            normalization_errors.append(abs(sum(values) - 1.0))
        diagnostics.append(
            {
                "sample_id": group["sample_id"],
                "query_id": group["query_id"],
                "partition": group["partition"],
                "action_ids": [row["action_id"] for row in group["rows"]],
                "behavior_probabilities": group["behavior"],
                "local_target_probabilities": local,
                "outcome_target_probabilities": outcome,
                "dagig_success_posterior": group["dagig"],
                "local_values": group["local_values"],
                "outcome_advantages": group["outcome_advantages"],
                "outcome_reward_std": group["outcome_reward_std"],
            }
        )
        selected_actions.extend(actions)
        if group["partition"] == "policy_train":
            target_rows.append(
                {
                    "sample_id": group["sample_id"],
                    "parent_group_id": group["query_id"],
                    "prompt": build_evidence_selection_prompt(actions[0]),
                    "completions": completions,
                    "action_ids": [row["action_id"] for row in group["rows"]],
                    "behavior_probabilities": group["behavior"],
                    "local_target_probabilities": local,
                    "outcome_target_probabilities": outcome,
                    "dagig_success_posterior": group["dagig"],
                }
            )

    train_dag_outcome_top1 = mean(
        max(range(5), key=group["dagig"].__getitem__)
        == max(range(5), key=lambda index: calibrated_policy(group["rows"], group["outcome_advantages"], outcome_eta, False)[index])
        for group in train_groups
    )
    train_dag_local_top1 = mean(
        max(range(5), key=group["dagig"].__getitem__)
        == max(range(5), key=lambda index: calibrated_policy(group["rows"], group["local_values"], local_beta, True)[index])
        for group in train_groups
    )
    metrics = {
        "samples": len(selected_queries),
        "policy_train_groups": len(train_groups),
        "internal_holdout_groups": len(internal_groups),
        "evidence_actions": len(selected_actions),
        "actions_per_group": 5,
        "rollouts_per_outcome_group": args.rollouts_per_group,
        "outcome_constant_group_rate_all": constant_outcome_groups / len(groups),
        "target_mean_kl_from_behavior_train": target_kl,
        "local_beta": local_beta,
        "outcome_eta": outcome_eta,
        "local_mean_kl_from_behavior_train": mean(
            kl(calibrated_policy(group["rows"], group["local_values"], local_beta, True), group["behavior"])
            for group in train_groups
        ),
        "outcome_mean_kl_from_behavior_train": mean(
            kl(calibrated_policy(group["rows"], group["outcome_advantages"], outcome_eta, False), group["behavior"])
            for group in train_groups
        ),
        "dagig_outcome_top1_agreement_train": train_dag_outcome_top1,
        "dagig_local_top1_agreement_train": train_dag_local_top1,
        "dagig_outcome_mean_tv_train": mean(
            tv(group["dagig"], calibrated_policy(group["rows"], group["outcome_advantages"], outcome_eta, False))
            for group in train_groups
        ),
        "dagig_local_mean_tv_train": mean(
            tv(group["dagig"], calibrated_policy(group["rows"], group["local_values"], local_beta, True))
            for group in train_groups
        ),
        "max_normalization_error": max(normalization_errors),
        "strategy_distribution": dict(Counter(action["evidence_strategy"] for action in selected_actions)),
    }
    gates = {
        "complete_fixed_query_samples": metrics["samples"] == 198,
        "complete_train_groups": metrics["policy_train_groups"] == 158,
        "complete_internal_groups": metrics["internal_holdout_groups"] == 40,
        "complete_five_action_matrix": metrics["evidence_actions"] == 990,
        "outcome_groups_nonconstant": metrics["outcome_constant_group_rate_all"] <= 0.05,
        "local_kl_matched": abs(metrics["local_mean_kl_from_behavior_train"] - target_kl) <= 1e-6,
        "outcome_kl_matched": abs(metrics["outcome_mean_kl_from_behavior_train"] - target_kl) <= 1e-6,
        "dagig_outcome_identifiable": train_dag_outcome_top1 <= 0.95 and metrics["dagig_outcome_mean_tv_train"] >= 0.01,
        "dagig_local_identifiable": train_dag_local_top1 <= 0.95 and metrics["dagig_local_mean_tv_train"] >= 0.01,
        "targets_normalized": metrics["max_normalization_error"] <= 1e-10,
        "query_policy_frozen": True,
        "same_query_and_action_universe": True,
        "no_gold_or_qrels_loaded": True,
        "internal_holdout_unused_for_calibration_or_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_NO_GOLD_EVIDENCE_CONTROLS_GO" if all(gates.values()) else "DAGIG_V6_NO_GOLD_EVIDENCE_CONTROLS_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    targets_path = output / "v6_no_gold_fixed_query_evidence_training_targets.jsonl"
    actions_path = output / "v6_no_gold_fixed_query_evidence_actions_no_labels.jsonl"
    diagnostics_path = output / "v6_no_gold_fixed_query_evidence_control_diagnostics_private.jsonl"
    write_jsonl(targets_path, target_rows)
    write_jsonl(actions_path, sorted(selected_actions, key=lambda row: (row["sample_id"], row["query_id"], row["evidence_strategy"])))
    write_jsonl(diagnostics_path, diagnostics)
    audit = {
        "decision": decision,
        "protocol_version": "dagig_v6_fixed_query_exact_no_gold_evidence_controls_v1",
        "metrics": metrics,
        "gates": gates,
        "method_contract": {
            "no_credit": "frozen evidence behavior distribution",
            "local_listwise": "terminal value with the answer descendant fixed to behavior mode",
            "outcome_listwise": "12 sampled evidence-to-answer trajectories with group-normalized outcome advantages",
            "dagig_posterior": "exact behavior-marginalized posterior over all answer descendants",
            "matched_update_budget": "Local and Outcome are calibrated on policy_train to DAG-IG mean KL from behavior",
        },
        "input_paths": {**{key: str(path) for key, path in paths.items()}, "dagig_train_fit_private": str(fit_path), "dagig_internal_score_rows": str(score_path)},
        "input_hashes": {**{key: sha256(path) for key, path in paths.items()}, "dagig_train_fit_private": sha256(fit_path), "dagig_internal_score_rows": sha256(score_path)},
        "output_paths": {
            "evidence_training_targets": str(targets_path),
            "evidence_actions": str(actions_path),
            "control_diagnostics_private": str(diagnostics_path),
        },
        "output_hashes": {
            "evidence_training_targets": sha256(targets_path),
            "evidence_actions": sha256(actions_path),
            "control_diagnostics_private": sha256(diagnostics_path),
        },
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_NO_GOLD_EVIDENCE_CONTROL_AUDIT.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
