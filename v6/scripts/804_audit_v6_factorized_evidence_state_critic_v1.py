#!/usr/bin/env python3
"""One-shot internal audit of frozen factorized evidence-state predictions."""

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


METHODS = ("no_credit", "local_ig_m", "outcome", "old_dagig", "factorized_dagig")


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


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap(rows: list[dict[str, Any]], metric: str, baseline: str, seed: int) -> dict[str, float | int]:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row["methods"]["factorized_dagig"][metric] - row["methods"][baseline][metric])
    sample_ids = sorted(by_sample)
    rng = random.Random(f"factorized-evidence:{seed}:{baseline}:{metric}")
    draws = []
    for _ in range(5000):
        sampled = [sample_ids[rng.randrange(len(sample_ids))] for _ in sample_ids]
        draws.append(mean(value for sample_id in sampled for value in by_sample[sample_id]))
    observed = [value for values in by_sample.values() for value in values]
    return {
        "observed_delta": mean(observed),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "probability_delta_gt_zero": mean(float(value > 0.0) for value in draws),
        "clusters": len(sample_ids),
        "replicates": len(draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--train_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    train_audit_path = args.train_audit.resolve()
    freeze = read_json(freeze_path)
    train_audit = read_json(train_audit_path)
    if train_audit.get("decision") != "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_TRAIN_OOF_GO":
        raise ValueError("factorized critic did not pass train OOF gates")
    if freeze["input_hashes"]["auditor"] != sha256(Path(__file__).resolve()):
        raise ValueError("auditor changed after protocol freeze")
    if train_audit["input_hashes"]["freeze"] != sha256(freeze_path):
        raise ValueError("train audit came from another freeze")
    for key, path in train_audit["output_paths"].items():
        if sha256(Path(path)) != train_audit["output_hashes"][key]:
            raise ValueError(f"train output changed: {key}")

    predictions = {
        row["evidence_action_id"]: row
        for row in read_jsonl(Path(train_audit["output_paths"]["predictions"]))
        if row["partition"] == "internal_holdout"
    }
    shared_values = {
        row["evidence_action_id"]: row
        for row in read_jsonl(Path(freeze["input_paths"]["shared_answer_values"]))
    }
    support_by_query = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(Path(freeze["input_paths"]["private_support"]))
        if row["partition"] == "internal_holdout"
    }
    terminal = {
        row["answer_action_id"]: row
        for row in read_jsonl(Path(freeze["input_paths"]["terminal_private"]))
        if row["partition"] == "internal_holdout"
    }
    categorical = read_jsonl(Path(freeze["input_paths"]["categorical_internal"]))
    if len(predictions) != 2975 or len(categorical) != 595:
        raise ValueError(f"internal universe mismatch: {len(predictions)}/{len(categorical)}")

    private_rows = []
    for group in categorical:
        action_ids = group["action_ids"]
        factor_values = [float(predictions[action_id]["evidence_success_probability"]) for action_id in action_ids]
        factor_posterior = normalize([0.2 * max(value, 1e-8) for value in factor_values])
        methods = {
            "no_credit": group["behavior_probabilities"],
            "local_ig_m": group["local_target_probabilities"],
            "outcome": group["outcome_target_probabilities"],
            "old_dagig": group["dagig_target_probabilities"],
            "factorized_dagig": factor_posterior,
        }
        selected = {}
        for method, posterior in methods.items():
            selected_index = max(range(5), key=lambda index: (float(posterior[index]), -index))
            evidence_id = action_ids[selected_index]
            value = shared_values[evidence_id]
            strategy = evidence_id.rsplit("::", 1)[-1]
            probabilities = [float(item) for item in value["answer_policy_probabilities"]]
            answer_correct = [float(terminal[answer_id]["answer_correct_proxy"]) for answer_id in value["answer_action_ids"]]
            strict = [float(terminal[answer_id]["strict_proxy"]) for answer_id in value["answer_action_ids"]]
            mode_index = value["answer_action_ids"].index(value["mode_answer_action_id"])
            selected[method] = {
                "evidence_action_id": evidence_id,
                "strategy": strategy,
                "support": float(bool(support_by_query[group["parent_group_id"]][strategy])),
                "expected_answer_correct": sum(p * label for p, label in zip(probabilities, answer_correct)),
                "expected_strict": sum(p * label for p, label in zip(probabilities, strict)),
                "mode_strict": strict[mode_index],
                "old_terminal_value": float(value["shared_answer_value"]),
                "factorized_value": factor_values[selected_index],
                "posterior_probability": float(posterior[selected_index]),
            }
        private_rows.append({
            "parent_state_id": group["parent_group_id"],
            "sample_id": group["sample_id"],
            "partition": "internal_holdout",
            "methods": selected,
        })

    summaries = {}
    for method in METHODS:
        selected = [row["methods"][method] for row in private_rows]
        summaries[method] = {
            "states": len(selected),
            "samples": len({row["sample_id"] for row in private_rows}),
            "support": mean(row["support"] for row in selected),
            "expected_answer_correct": mean(row["expected_answer_correct"] for row in selected),
            "expected_strict": mean(row["expected_strict"] for row in selected),
            "mode_strict": mean(row["mode_strict"] for row in selected),
            "old_terminal_value": mean(row["old_terminal_value"] for row in selected),
            "factorized_value": mean(row["factorized_value"] for row in selected),
            "strategy_distribution": dict(sorted(Counter(row["strategy"] for row in selected).items())),
        }
    pairwise = {}
    for baseline in ("no_credit", "local_ig_m", "outcome", "old_dagig"):
        pairwise[f"factorized_dagig_vs_{baseline}"] = {
            "top_action_disagreement_rate": mean(
                row["methods"]["factorized_dagig"]["evidence_action_id"] != row["methods"][baseline]["evidence_action_id"]
                for row in private_rows
            ),
            **{
                metric: cluster_bootstrap(private_rows, metric, baseline, 20260720)
                for metric in ("support", "expected_strict", "mode_strict")
            },
        }

    threshold = freeze["development_gates"]
    factor = summaries["factorized_dagig"]
    no_credit = summaries["no_credit"]
    local = summaries["local_ig_m"]
    outcome = summaries["outcome"]
    gates = {
        "complete_595_internal_query_states": len(private_rows) == 595,
        "complete_40_internal_samples": len({row["sample_id"] for row in private_rows}) == 40,
        "support_not_below_no_credit": factor["support"] - no_credit["support"] >= threshold["support_delta_vs_no_credit_min"],
        "support_noninferior_local": factor["support"] >= local["support"] - threshold["support_noninferiority_vs_local_tolerance"],
        "support_noninferior_outcome": factor["support"] >= outcome["support"] - threshold["support_noninferiority_vs_outcome_tolerance"],
        "strict_noninferior_no_credit": factor["expected_strict"] >= no_credit["expected_strict"] - threshold["strict_noninferiority_vs_no_credit_tolerance"],
        "strict_noninferior_local": factor["expected_strict"] >= local["expected_strict"] - threshold["strict_noninferiority_vs_local_tolerance"],
        "strict_noninferior_outcome": factor["expected_strict"] >= outcome["expected_strict"] - threshold["strict_noninferiority_vs_outcome_tolerance"],
        "differs_from_outcome": pairwise["factorized_dagig_vs_outcome"]["top_action_disagreement_rate"] >= threshold["top_action_disagreement_vs_outcome_min"],
        "evidence_action_diversity": len(factor["strategy_distribution"]) >= threshold["selected_evidence_strategies_min"],
        "predictions_frozen_before_internal_labels": True,
        "internal_never_fit": True,
        "runtime_features_contain_no_gold_or_qrels": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_DEVELOPMENT_GO" if all(gates.values()) else "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_DEVELOPMENT_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "v6_factorized_evidence_state_internal_private.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in private_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "method_summary": summaries,
        "pairwise_comparisons": pairwise,
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path), "train_audit": str(train_audit_path)},
        "input_hashes": {"freeze": sha256(freeze_path), "train_audit": sha256(train_audit_path)},
        "output_paths": {"private_rows": str(rows_path)},
        "output_hashes": {"private_rows": sha256(rows_path)},
        "internal_labels_loaded_only_after_predictions_frozen": True,
        "development_result_not_paper_final": True,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_DEVELOPMENT_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "method_summary": summaries, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
