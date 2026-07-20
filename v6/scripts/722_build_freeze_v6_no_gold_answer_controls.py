#!/usr/bin/env python3
"""Build clean matched answer-node controls from the compact v4 terminal value."""

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
from dagig_causal.answer_prompt import answer_completion, build_answer_policy_prompt  # noqa: E402


METHODS = ("no_credit", "local_ig", "outcome", "dagig")
TARGET_KEYS = {
    "no_credit": "no_credit_probability",
    "local_ig": "local_ig_probability",
    "outcome": "outcome_probability",
    "dagig": "dagig_probability",
}
FORBIDDEN = {
    "aliases", "answer_correct", "answer_correct_proxy", "equivalence_logit", "gold_answer",
    "positive_doc_ids", "qrels", "strict_proxy", "support_label", "target_doc",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(child.relative_to(root)).encode())
        digest.update(sha256(child).encode())
    return digest.hexdigest()


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("cannot normalize non-positive target")
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
        return mean(calculated_kl(group, key, scale, log_values) for group in groups)

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


def calculated_kl(group: dict[str, Any], key: str, scale: float, log_values: bool) -> float:
    return kl(calibrated(group["behavior"], group[key], scale, log_values=log_values), group["behavior"])


def weighted_choice(rows: list[dict[str, Any]], behavior: list[float], rng: random.Random) -> int:
    draw, cumulative = rng.random(), 0.0
    for index, probability in enumerate(behavior):
        cumulative += probability
        if draw <= cumulative + 1e-12:
            return index
    return len(rows) - 1


def local_value(score: dict[str, Any]) -> float:
    if bool(score["is_unknown"]):
        return 1e-8
    support = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(score["support_logit"])))))
    reader = math.exp(0.25 * max(-20.0, min(0.0, float(score["reader_candidate_mean_logprob"]))))
    return max(1e-8, support * reader)


def select_structural(groups: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    """Select groups using only IDs and structural provenance, never values or labels."""
    ordered = sorted(
        groups,
        key=lambda row: hashlib.sha256(f"answer-group:{seed}:{row['parent_id']}".encode()).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        if len(selected) < limit and row["parent_id"] not in seen:
            selected.append(row)
            seen.add(row["parent_id"])

    for field in sorted({row["visual_field"] for row in ordered}):
        add(next(row for row in ordered if row["visual_field"] == field))
    for strategy in sorted({row["evidence_strategy"] for row in ordered}):
        add(next(row for row in ordered if row["evidence_strategy"] == strategy))
    for row in ordered:
        add(row)
    if len(selected) != limit:
        raise ValueError(f"insufficient structurally eligible groups: {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup_audit", type=Path, required=True)
    parser.add_argument("--terminal_audit", type=Path, required=True)
    parser.add_argument("--answer_expansion_audit", type=Path, required=True)
    parser.add_argument("--evidence_action_audit", type=Path, required=True)
    parser.add_argument("--initializer_audit", type=Path, required=True)
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--groups_per_sample", type=int, default=12)
    parser.add_argument("--rollouts_per_group", type=int, default=12)
    parser.add_argument("--seed", type=int, default=761943)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.groups_per_sample != 12 or args.rollouts_per_group != 12:
        raise ValueError("paper protocol freezes 12 groups/sample and 12 outcome rollouts/group")

    paths = {
        "backup_audit": args.backup_audit.resolve(),
        "terminal_audit": args.terminal_audit.resolve(),
        "answer_expansion_audit": args.answer_expansion_audit.resolve(),
        "evidence_action_audit": args.evidence_action_audit.resolve(),
        "initializer_audit": args.initializer_audit.resolve(),
    }
    backup, terminal, expansion, evidence_audit, initializer_audit = [read_json(paths[key]) for key in paths]
    expected = (
        (backup, "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_GO"),
        (terminal, "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO"),
        (expansion, "DAGIG_V6_ANSWER_ACTION_EXPANSION_GO"),
        (evidence_audit, "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_ACTIONS_GO"),
        (initializer_audit, "DAGIG_V6_ANSWER_NEUTRAL_SFT_GO"),
    )
    if any(row.get("decision") != decision for row, decision in expected):
        raise ValueError("one or more frozen answer prerequisites are not GO")
    if backup["input_hashes"]["terminal_audit"] != sha256(paths["terminal_audit"]):
        raise ValueError("backup and terminal audits do not belong to the same v4 protocol")

    answer_edges_path = Path(backup["output_paths"]["answer_edges"])
    answer_actions_path = Path(expansion["output_paths"]["answer_actions"])
    evidence_actions_path = Path(evidence_audit["output_paths"]["evidence_actions"])
    for path, digest in (
        (answer_edges_path, backup["output_hashes"]["answer_edges"]),
        (answer_actions_path, expansion["output_hashes"]["answer_actions"]),
        (evidence_actions_path, evidence_audit["output_hashes"]["evidence_actions"]),
    ):
        if sha256(path) != digest:
            raise ValueError(f"audited input changed: {path}")

    source_freeze = read_json(Path(terminal["input_paths"]["freeze"]))
    score_manifests = [Path(path) for path in source_freeze["input_paths"]["score_manifests"]]
    if [sha256(path) for path in score_manifests] != source_freeze["input_hashes"]["score_manifests"]:
        raise ValueError("terminal score manifests changed")
    clean_scores: dict[str, dict[str, Any]] = {}
    for manifest_path in score_manifests:
        manifest = read_json(manifest_path)
        score_path = Path(manifest["output_paths"]["scores"])
        if sha256(score_path) != manifest["output_hashes"]["scores"]:
            raise ValueError("terminal score shard changed")
        for row in read_jsonl(score_path):
            # Deliberately whitelist observable fields; equivalence_logit is never read or copied.
            clean_scores[str(row["answer_action_id"])] = {
                "answer_action_id": str(row["answer_action_id"]),
                "support_logit": float(row["support_logit"]),
                "reader_candidate_mean_logprob": float(row["reader_candidate_mean_logprob"]),
                "is_unknown": bool(row["is_unknown"]),
            }

    actions = {str(row["answer_action_id"]): row for row in read_jsonl(answer_actions_path)}
    evidence = {str(row["evidence_action_id"]): row for row in read_jsonl(evidence_actions_path)}
    grouped_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(answer_edges_path):
        grouped_edges[str(row["parent_id"])].append(row)
    if set(actions) != clean_scores.keys() or set(actions) != {str(row["action_id"]) for rows in grouped_edges.values() for row in rows}:
        raise ValueError("answer action, edge, and clean local-feature universes differ")

    enriched_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for parent_id, rows in grouped_edges.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda row: str(row["action_id"]))
        parent = evidence[parent_id]
        behavior = normalize([float(row["behavior_probability"]) for row in rows])
        dagig = normalize([float(row["success_posterior_probability"]) for row in rows])
        locals_ = [local_value(clean_scores[str(row["action_id"])]) for row in rows]
        sampled: dict[str, list[float]] = {str(row["action_id"]): [] for row in rows}
        rewards = []
        sampled_ids = []
        for rollout in range(args.rollouts_per_group):
            rng = random.Random(f"answer-outcome:{args.seed}:{parent_id}:{rollout}")
            index = weighted_choice(rows, behavior, rng)
            reward = float(rows[index]["child_success_probability"])
            rewards.append(reward)
            sampled_ids.append(str(rows[index]["action_id"]))
        center = mean(rewards)
        std = math.sqrt(mean((reward - center) ** 2 for reward in rewards))
        for action_id, reward in zip(sampled_ids, rewards):
            sampled[action_id].append((reward - center) / std if std > 1e-12 else 0.0)
        outcome = [mean(sampled[str(row["action_id"])]) if sampled[str(row["action_id"])] else 0.0 for row in rows]
        enriched_by_sample[str(rows[0]["sample_id"])].append({
            "parent_id": parent_id,
            "sample_id": str(rows[0]["sample_id"]),
            "partition": str(rows[0]["partition"]),
            "visual_field": str(parent["visual_field"]),
            "evidence_strategy": str(parent["evidence_strategy"]),
            "rows": rows,
            "behavior": behavior,
            "dagig": dagig,
            "local_values": locals_,
            "outcome_values": outcome,
            "outcome_std": std,
        })
    if len(enriched_by_sample) != 198:
        raise ValueError(f"expected 198 samples with multi-action answer groups, got {len(enriched_by_sample)}")

    selected = [row for sample in sorted(enriched_by_sample) for row in select_structural(enriched_by_sample[sample], args.groups_per_sample, args.seed)]
    train_groups = [row for row in selected if row["partition"] == "policy_train"]
    internal_groups = [row for row in selected if row["partition"] == "internal_holdout"]
    target_kl = mean(kl(row["dagig"], row["behavior"]) for row in train_groups)
    local_beta = find_scale(train_groups, "local_values", target_kl, log_values=True)
    outcome_eta = find_scale(train_groups, "outcome_values", target_kl, log_values=False)

    train_rows, internal_rows, clean_feature_rows = [], [], []
    diagnostics = []
    normalization_errors, local_tvs, outcome_tvs = [], [], []
    local_top, outcome_top = [], []
    constant_outcome = 0
    for group in selected:
        local = calibrated(group["behavior"], group["local_values"], local_beta, log_values=True)
        outcome = calibrated(group["behavior"], group["outcome_values"], outcome_eta, log_values=False)
        constant_outcome += int(group["outcome_std"] <= 1e-12)
        local_tvs.append(tv(group["dagig"], local)); outcome_tvs.append(tv(group["dagig"], outcome))
        local_top.append(max(range(len(local)), key=local.__getitem__) == max(range(len(local)), key=group["dagig"].__getitem__))
        outcome_top.append(max(range(len(outcome)), key=outcome.__getitem__) == max(range(len(outcome)), key=group["dagig"].__getitem__))
        probabilities = (group["behavior"], local, outcome, group["dagig"])
        normalization_errors.extend(abs(sum(values) - 1.0) for values in probabilities)
        prompt = build_answer_policy_prompt(evidence[group["parent_id"]])
        destination = train_rows if group["partition"] == "policy_train" else internal_rows
        for index, edge in enumerate(group["rows"]):
            action_id = str(edge["action_id"])
            row = {
                "sample_id": group["sample_id"], "partition": group["partition"],
                "parent_group_id": group["parent_id"], "answer_action_id": action_id,
                "visual_field": group["visual_field"], "evidence_strategy": group["evidence_strategy"],
                "prompt": prompt, "completion": answer_completion(actions[action_id]["candidate_answer"]),
                "no_credit_probability": group["behavior"][index], "local_ig_probability": local[index],
                "outcome_probability": outcome[index], "dagig_probability": group["dagig"][index],
                "child_success_probability": float(edge["child_success_probability"]),
            }
            if FORBIDDEN.intersection(row):
                raise ValueError("forbidden evaluation field entered answer controls")
            destination.append(row)
            clean_feature_rows.append({"answer_action_id": action_id, **clean_scores[action_id]})
        diagnostics.append({
            "sample_id": group["sample_id"], "partition": group["partition"], "parent_group_id": group["parent_id"],
            "actions": len(group["rows"]), "outcome_reward_std": group["outcome_std"],
            "dagig_local_tv": local_tvs[-1], "dagig_outcome_tv": outcome_tvs[-1],
        })

    clean_feature_rows = list({row["answer_action_id"]: row for row in clean_feature_rows}.values())
    metrics = {
        "samples": len(enriched_by_sample), "policy_train_samples": len({row["sample_id"] for row in train_rows}),
        "internal_holdout_samples": len({row["sample_id"] for row in internal_rows}),
        "policy_train_groups": len(train_groups), "internal_holdout_groups": len(internal_groups),
        "policy_train_action_rows": len(train_rows), "internal_holdout_action_rows": len(internal_rows),
        "groups_per_sample": args.groups_per_sample, "rollouts_per_group": args.rollouts_per_group,
        "target_mean_kl_from_behavior_train": target_kl, "local_beta": local_beta, "outcome_eta": outcome_eta,
        "local_mean_kl_train": mean(calculated_kl(row, "local_values", local_beta, True) for row in train_groups),
        "outcome_mean_kl_train": mean(calculated_kl(row, "outcome_values", outcome_eta, False) for row in train_groups),
        "outcome_constant_group_rate": constant_outcome / len(selected),
        "dagig_local_mean_tv": mean(local_tvs), "dagig_outcome_mean_tv": mean(outcome_tvs),
        "dagig_local_top1_agreement": mean(local_top), "dagig_outcome_top1_agreement": mean(outcome_top),
        "max_normalization_error": max(normalization_errors),
        "structural_distribution": dict(
            Counter(f"{row['visual_field']}::{row['evidence_strategy']}" for row in selected)
        ),
    }
    gates = {
        "complete_198_samples": metrics["samples"] == 198,
        "complete_158_40_split": metrics["policy_train_samples"] == 158 and metrics["internal_holdout_samples"] == 40,
        "sample_balanced": len(train_groups) == 158 * 12 and len(internal_groups) == 40 * 12,
        "multi_action_groups": len(train_rows) > len(train_groups) * 2 and len(internal_rows) > len(internal_groups) * 2,
        "local_kl_matched": abs(metrics["local_mean_kl_train"] - target_kl) <= 1e-6,
        "outcome_kl_matched": abs(metrics["outcome_mean_kl_train"] - target_kl) <= 1e-6,
        "local_identifiable": metrics["dagig_local_mean_tv"] >= 0.005,
        "outcome_identifiable": metrics["dagig_outcome_mean_tv"] >= 0.005,
        "targets_normalized": metrics["max_normalization_error"] <= 1e-10,
        "selection_uses_only_structure": True, "runtime_controls_have_no_gold_or_qrels": True,
        "internal_holdout_unused_for_scale_or_training": True, "dev_sealed": True, "test_sealed": True,
    }
    decision = "DAGIG_V6_NO_GOLD_ANSWER_CONTROLS_FROZEN" if all(gates.values()) else "DAGIG_V6_NO_GOLD_ANSWER_CONTROLS_NO_GO"
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    train_path = output / "v6_no_gold_answer_control_targets_train.jsonl"
    internal_path = output / "v6_no_gold_answer_control_targets_internal_no_labels.jsonl"
    clean_path = output / "v6_no_gold_answer_observable_features_clean.jsonl"
    diagnostic_path = output / "v6_no_gold_answer_control_diagnostics.jsonl"
    write_jsonl(train_path, train_rows); write_jsonl(internal_path, internal_rows)
    write_jsonl(clean_path, sorted(clean_feature_rows, key=lambda row: row["answer_action_id"]))
    write_jsonl(diagnostic_path, diagnostics)
    trainer = Path(__file__).with_name("723_train_v6_no_gold_answer_policy.py").resolve()
    input_paths = {**{key: str(path) for key, path in paths.items()}, "answer_edges": str(answer_edges_path),
                   "answer_actions": str(answer_actions_path), "evidence_actions": str(evidence_actions_path),
                   "initializer_adapter": str(Path(initializer_audit["adapter"]).resolve()), "train_data": str(train_path),
                   "internal_data": str(internal_path), "clean_observable_features": str(clean_path)}
    freeze = {
        "decision": decision, "protocol_version": "dagig_v6_compact_v4_backward_answer_controls_v1",
        "methods": METHODS, "target_keys": TARGET_KEYS, "metrics": metrics, "gates": gates,
        "method_contract": {
            "no_credit": "frozen answer behavior distribution",
            "local_ig": "observable support-reader local confidence, KL matched to DAG-IG",
            "outcome": "12 behavior rollouts with verifier outcome advantages, KL matched to DAG-IG",
            "dagig": "exact posterior pi_b(a|e) P_success(a,e) / V(e)",
        },
        "base_model": str(args.base_model.resolve()), "base_model_tree_sha256": tree_hash(args.base_model.resolve()),
        "shared_initializer": str(Path(initializer_audit["adapter"]).resolve()),
        "shared_initializer_tree_sha256": tree_hash(Path(initializer_audit["adapter"]).resolve()),
        "parent_groups": len(train_groups), "input_paths": input_paths,
        "input_hashes": {key: tree_hash(Path(value)) if key == "initializer_adapter" else sha256(Path(value)) for key, value in input_paths.items()},
        "training": {"epochs": 2, "learning_rate": 2e-5, "group_batch_size": 1, "gradient_accumulation_groups": 8,
                     "max_input_tokens": 4096, "listwise_nll_weight": 0.05, "max_grad_norm": 1.0,
                     "seed": 761943, "logging_steps": 20},
        "runner_hashes": {"trainer": sha256(trainer)}, "gold_or_qrels_in_policy_data": False,
        "internal_holdout_used_for_training": False, "dev_used": False, "test_used": False, "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_NO_GOLD_ANSWER_CONTROL_FREEZE.json"
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "freeze": str(freeze_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
