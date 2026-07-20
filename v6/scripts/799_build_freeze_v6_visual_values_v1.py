#!/usr/bin/env python3
"""Build visual-node values under frozen DAG query/evidence and answer policies."""

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


VISUAL_ORDER = ("ocr_state", "caption_state", "joint_state")
METHODS = ("no_credit", "local_ig_m", "outcome", "dagig")
PUBLIC_KEYS = {"parent_state_id", "prompt", "actions", "target_distributions"}
FORBIDDEN = ("gold", "qrel", "strict", "answer_correct", "target_doc", "ground_truth", "terminal_value")


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
    if total <= 0.0:
        raise ValueError("cannot normalize non-positive mass")
    result = [value / total for value in values]
    if any(value <= 0.0 or not math.isfinite(value) for value in result):
        raise ValueError("invalid probability")
    return result


def softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    return normalize([math.exp(value - maximum) for value in logits])


def kl(policy: list[float], behavior: list[float]) -> float:
    return sum(p * math.log(p / b) for p, b in zip(policy, behavior))


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
    draw, cumulative = rng.random(), 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative + 1e-12:
            return index
    return len(probabilities) - 1


def field_names(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            result.add(str(key).lower())
            result.update(field_names(child))
    elif isinstance(value, list):
        for child in value:
            result.update(field_names(child))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_value_freeze", type=Path, required=True)
    parser.add_argument("--query_v1_freeze", type=Path, required=True)
    parser.add_argument("--query_development_audit", type=Path, required=True)
    parser.add_argument("--rollouts_per_group", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rollouts_per_group != 12:
        raise ValueError("visual Outcome control is frozen to 12 trajectories per root")

    query_freeze_path = args.query_value_freeze.resolve()
    query_v1_path = args.query_v1_freeze.resolve()
    development_path = args.query_development_audit.resolve()
    query_freeze, query_v1, development = map(read_json, (query_freeze_path, query_v1_path, development_path))
    if query_freeze.get("decision") != "DAGIG_V6_FULL_QUERY_VALUES_V2_FROZEN":
        raise ValueError("full query values v2 are not frozen")
    if query_v1.get("decision") != "DAGIG_V6_FULL_QUERY_VALUES_V1_FROZEN":
        raise ValueError("full query values v1 are not frozen")
    if development.get("decision") != "DAGIG_V6_FULL_QUERY_SELECTOR_DEVELOPMENT_GO":
        raise ValueError("full query selector development diagnostic is not GO")
    for freeze in (query_freeze, query_v1):
        for key, raw_path in freeze["input_paths"].items():
            assert_hash(Path(raw_path), freeze["input_hashes"][key], key)
        for key, raw_path in freeze["output_paths"].items():
            assert_hash(Path(raw_path), freeze["output_hashes"][key], key)

    v2_rows = read_jsonl(Path(query_freeze["output_paths"]["train_targets"])) + read_jsonl(
        Path(query_freeze["output_paths"]["internal_targets"])
    )
    v1_rows = read_jsonl(Path(query_v1["output_paths"]["train_targets"])) + read_jsonl(
        Path(query_v1["output_paths"]["internal_targets"])
    )
    v1_map = {row["parent_state_id"]: row for row in v1_rows}
    if {row["parent_state_id"] for row in v2_rows} != set(v1_map):
        raise ValueError("query v1/v2 parent universes differ")
    unchanged_error = 0.0
    for row in v2_rows:
        old = v1_map[row["parent_state_id"]]
        if row["actions"] != old["actions"]:
            raise ValueError("query actions changed when Local-IG-M was added")
        for method_v2, method_v1 in (("no_credit", "no_credit"), ("local_observable", "local_ig"), ("outcome", "outcome"), ("dagig", "dagig")):
            unchanged_error = max(
                unchanged_error,
                max(
                    abs(a - b)
                    for a, b in zip(
                        row["target_distributions"][method_v2], old["target_distributions"][method_v1]
                    )
                ),
            )

    diagnostics = {
        row["parent_state_id"]: row
        for row in read_jsonl(Path(query_freeze["output_paths"]["diagnostics"]))
    }
    query_actions_path = Path(query_freeze["input_paths"]["full_real_search_query_actions"])
    query_actions = {row["query_id"]: row for row in read_jsonl(query_actions_path)}
    shared_values_path = Path(query_freeze["input_paths"]["shared_answer_values"])
    shared_values = {row["evidence_action_id"]: row for row in read_jsonl(shared_values_path)}

    visual_children: dict[str, dict[str, Any]] = {}
    for row in v2_rows:
        parent_id = row["parent_state_id"]
        diagnostic = diagnostics[parent_id]
        posterior = row["target_distributions"]["dagig"]
        selected_index = max(range(len(posterior)), key=lambda index: (posterior[index], -index))
        query_id = diagnostic["query_action_ids"][selected_index]
        evidence_id = diagnostic["selected_evidence_action_ids"][selected_index]
        value = shared_values[evidence_id]
        source = query_actions[query_id]
        visual_children[parent_id] = {
            "partition": diagnostic["partition"],
            "sample_id": source["sample_id"],
            "question": source["question"],
            "visual_observation": source["visual_observation"],
            "selected_query_action_id": query_id,
            "selected_evidence_action_id": evidence_id,
            "hard_value": float(diagnostic["hard_query_values"][selected_index]),
            "local_m_value": float(diagnostic["local_modal_descendant_values"][selected_index]),
            "answer_policy_probabilities": [float(value_) for value_ in value["answer_policy_probabilities"]],
            "answer_child_values": [float(value_) for value_ in value["child_success_probabilities"]],
        }

    by_sample: dict[str, list[str]] = defaultdict(list)
    for parent_id, row in visual_children.items():
        by_sample[row["sample_id"]].append(parent_id)
    groups = []
    constant_outcome = 0
    for sample_id, parent_ids in sorted(by_sample.items()):
        parent_ids.sort(key=lambda parent_id: VISUAL_ORDER.index(parent_id.rsplit("::", 1)[-1]))
        if tuple(parent_id.rsplit("::", 1)[-1] for parent_id in parent_ids) != VISUAL_ORDER:
            raise ValueError(f"visual intervention set changed: {sample_id}")
        rows = [visual_children[parent_id] for parent_id in parent_ids]
        if len({row["question"] for row in rows}) != 1 or len({row["partition"] for row in rows}) != 1:
            raise ValueError(f"visual children do not share root state: {sample_id}")
        behavior = [1.0 / 3.0] * 3
        hard_values = [row["hard_value"] for row in rows]
        dagig = normalize([probability * value for probability, value in zip(behavior, hard_values)])
        sampled = []
        observed: dict[int, list[float]] = {index: [] for index in range(3)}
        for rollout in range(args.rollouts_per_group):
            rng = random.Random(f"visual-outcome:{args.seed}:{sample_id}:{rollout}")
            visual_index = weighted_index(behavior, rng)
            child = rows[visual_index]
            answer_index = weighted_index(child["answer_policy_probabilities"], rng)
            sampled.append((visual_index, child["answer_child_values"][answer_index]))
        rewards = [reward for _, reward in sampled]
        center = mean(rewards)
        reward_std = math.sqrt(mean((reward - center) ** 2 for reward in rewards))
        constant_outcome += int(reward_std <= 1e-12)
        for index, reward in sampled:
            observed[index].append((reward - center) / reward_std if reward_std > 1e-12 else 0.0)
        groups.append(
            {
                "sample_id": sample_id,
                "partition": rows[0]["partition"],
                "children": rows,
                "behavior": behavior,
                "hard_values": hard_values,
                "local_m_values": [row["local_m_value"] for row in rows],
                "outcome_values": [mean(observed[index]) if observed[index] else 0.0 for index in range(3)],
                "outcome_counts": [len(observed[index]) for index in range(3)],
                "dagig": dagig,
            }
        )

    train = [group for group in groups if group["partition"] == "policy_train"]
    internal = [group for group in groups if group["partition"] == "internal_holdout"]
    target_kl = mean(kl(group["dagig"], group["behavior"]) for group in train)
    local_beta = find_scale(train, "local_m_values", target_kl, log_values=True)
    outcome_eta = find_scale(train, "outcome_values", target_kl, log_values=False)

    public_rows = {"policy_train": [], "internal_holdout": []}
    diagnostics_rows = []
    normalization_error = 0.0
    identity_error = 0.0
    tv_local, tv_outcome, top_local, top_outcome = [], [], [], []
    train_selected: dict[str, list[float]] = {method: [] for method in METHODS}
    train_mix: dict[str, Counter[str]] = {method: Counter() for method in METHODS}
    for group in groups:
        local = calibrated(group["behavior"], group["local_m_values"], local_beta, log_values=True)
        outcome = calibrated(group["behavior"], group["outcome_values"], outcome_eta, log_values=False)
        distributions = {"no_credit": group["behavior"], "local_ig_m": local, "outcome": outcome, "dagig": group["dagig"]}
        for distribution in distributions.values():
            normalization_error = max(normalization_error, abs(sum(distribution) - 1.0))
        exact = normalize([p * v for p, v in zip(group["behavior"], group["hard_values"])])
        identity_error = max(identity_error, max(abs(a - b) for a, b in zip(exact, group["dagig"])))
        dag_top = max(range(3), key=group["dagig"].__getitem__)
        if group["partition"] == "policy_train":
            tv_local.append(tv(group["dagig"], local))
            tv_outcome.append(tv(group["dagig"], outcome))
            top_local.append(dag_top == max(range(3), key=local.__getitem__))
            top_outcome.append(dag_top == max(range(3), key=outcome.__getitem__))
            for method, distribution in distributions.items():
                index = max(range(3), key=lambda candidate: (distribution[candidate], -candidate))
                train_selected[method].append(group["hard_values"][index])
                train_mix[method][VISUAL_ORDER[index]] += 1
        actions = [
            {
                "label": f"V{index + 1}",
                "strategy": VISUAL_ORDER[index],
                "completion": json.dumps(
                    {"visual_observation": child["visual_observation"]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            for index, child in enumerate(group["children"])
        ]
        public = {
            "parent_state_id": group["sample_id"],
            "prompt": "Extract one image-only visual observation for the multimodal search agent. Do not answer the question or use outside knowledge.",
            "actions": actions,
            "target_distributions": distributions,
        }
        if set(public) != PUBLIC_KEYS:
            raise ValueError("visual public schema changed")
        leaked = sorted(name for name in field_names(public) if any(token in name for token in FORBIDDEN))
        if leaked:
            raise ValueError(f"private visual fields exposed: {group['sample_id']}: {leaked}")
        public_rows[group["partition"]].append(public)
        diagnostics_rows.append(
            {
                "parent_state_id": group["sample_id"],
                "partition": group["partition"],
                "visual_action_ids": [f"{group['sample_id']}::{strategy}" for strategy in VISUAL_ORDER],
                "selected_query_action_ids": [child["selected_query_action_id"] for child in group["children"]],
                "selected_evidence_action_ids": [child["selected_evidence_action_id"] for child in group["children"]],
                "hard_visual_values": group["hard_values"],
                "local_modal_descendant_values": group["local_m_values"],
                "outcome_values": group["outcome_values"],
                "outcome_counts": group["outcome_counts"],
                "dagig_target_probabilities": group["dagig"],
            }
        )

    metrics = {
        "samples": len(groups),
        "policy_train_samples": len(train),
        "internal_development_samples": len(internal),
        "visual_actions": 3 * len(groups),
        "target_mean_kl_from_behavior_train": target_kl,
        "local_beta": local_beta,
        "outcome_eta": outcome_eta,
        "local_mean_kl_train": mean(kl(calibrated(group["behavior"], group["local_m_values"], local_beta, log_values=True), group["behavior"]) for group in train),
        "outcome_mean_kl_train": mean(kl(calibrated(group["behavior"], group["outcome_values"], outcome_eta, log_values=False), group["behavior"]) for group in train),
        "outcome_constant_group_rate": constant_outcome / len(groups),
        "dagig_local_mean_tv_train": mean(tv_local),
        "dagig_outcome_mean_tv_train": mean(tv_outcome),
        "dagig_local_top_agreement_train": mean(top_local),
        "dagig_outcome_top_agreement_train": mean(top_outcome),
        "train_direct_selector_terminal": {method: mean(values) for method, values in train_selected.items()},
        "train_selected_visual_distribution": {method: dict(sorted(counts.items())) for method, counts in train_mix.items()},
        "max_query_v1_v2_unchanged_target_error": unchanged_error,
        "max_target_normalization_error": normalization_error,
        "max_visual_dag_identity_error": identity_error,
    }
    gates = {
        "query_v2_frozen": True,
        "query_v1_development_go_transfers_exactly": unchanged_error <= 1e-12,
        "complete_198_samples": len(groups) == 198,
        "complete_158_40_split": len(train) == 158 and len(internal) == 40,
        "exact_three_visual_interventions": all(len(group["children"]) == 3 for group in groups),
        "uniform_visual_intervention_prior": all(max(group["behavior"]) - min(group["behavior"]) <= 1e-12 for group in groups),
        "hard_query_and_evidence_policies_used": True,
        "exact_visual_dag_identity": identity_error <= 1e-12,
        "targets_normalized": normalization_error <= 1e-10,
        "local_kl_matched_train_only": abs(metrics["local_mean_kl_train"] - target_kl) <= 1e-6,
        "outcome_kl_matched_train_only": abs(metrics["outcome_mean_kl_train"] - target_kl) <= 1e-6,
        "outcome_groups_nonconstant": metrics["outcome_constant_group_rate"] <= 0.05,
        "dagig_local_identifiable_train": metrics["dagig_local_mean_tv_train"] >= 0.01 and metrics["dagig_local_top_agreement_train"] <= 0.95,
        "dagig_outcome_identifiable_train": metrics["dagig_outcome_mean_tv_train"] >= 0.01 and metrics["dagig_outcome_top_agreement_train"] <= 0.95,
        "dagig_train_terminal_improves_no_credit": metrics["train_direct_selector_terminal"]["dagig"] >= metrics["train_direct_selector_terminal"]["no_credit"] + 0.005,
        "dagig_train_terminal_noninferior_local": metrics["train_direct_selector_terminal"]["dagig"] >= metrics["train_direct_selector_terminal"]["local_ig_m"] - 0.002,
        "dagig_train_terminal_noninferior_outcome": metrics["train_direct_selector_terminal"]["dagig"] >= metrics["train_direct_selector_terminal"]["outcome"] - 0.002,
        "public_schema_has_no_private_fields": True,
        "private_support_or_strict_not_loaded": True,
        "development_labels_not_used": True,
        "new_search_calls_zero": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_VISUAL_VALUES_V1_FROZEN" if all(gates.values()) else "DAGIG_V6_VISUAL_VALUES_V1_NO_GO"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "train_targets": output_dir / "v6_visual_targets_v1_train.jsonl",
        "internal_targets": output_dir / "v6_visual_targets_v1_internal_no_labels.jsonl",
        "diagnostics": output_dir / "v6_visual_value_v1_diagnostics_no_gold.jsonl",
    }
    for path, rows in (
        (paths["train_targets"], public_rows["policy_train"]),
        (paths["internal_targets"], public_rows["internal_holdout"]),
        (paths["diagnostics"], diagnostics_rows),
    ):
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    input_paths = {
        "query_value_freeze": str(query_freeze_path),
        "query_v1_freeze": str(query_v1_path),
        "query_development_audit": str(development_path),
        "query_actions": str(query_actions_path),
        "shared_answer_values": str(shared_values_path),
    }
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_visual_under_full_query_v1",
        "downstream_contract": {
            "query_policy": "frozen direct argmax of full five-path DAG query posterior",
            "evidence_policy": "frozen direct argmax of exact DAG evidence posterior",
            "answer_policy": "frozen shared DAG answer policy",
            "visual_value": "V_V(v)=V_Q(argmax_q q_DAG(q|v,Y=1))",
        },
        "method_contract": {
            "no_credit": "uniform visual intervention proposal",
            "local_ig_m": "modal answer descendant value on the same frozen DAG query/evidence path",
            "outcome": "12 sampled visual-to-frozen-query/evidence-to-answer outcomes, train-KL matched",
            "dagig": "exact q(v|image,Y=1)=mu_V(v)V_V(v)/V_root",
        },
        "metrics": metrics,
        "gates": gates,
        "selector_go_gates": {
            "dagig_terminal_delta_vs_no_credit_min": 0.005,
            "dagig_terminal_noninferiority_tolerance": 0.002,
            "dagig_support_delta_vs_no_credit_min": 0.0,
            "dagig_support_noninferiority_tolerance": 0.01,
            "dagig_expected_strict_noninferiority_tolerance": 0.015,
            "dagig_mode_strict_noninferiority_tolerance": 0.015,
            "dagig_control_top_action_disagreement_min": 0.05,
            "dagig_selected_visual_strategies_min": 2,
        },
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {key: str(path) for key, path in paths.items()},
        "output_hashes": {key: sha256(path) for key, path in paths.items()},
        "internal_development_previously_consumed": True,
        "gold_or_qrels_loaded": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_VISUAL_VALUE_V1_FREEZE.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
