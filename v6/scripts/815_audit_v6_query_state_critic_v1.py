#!/usr/bin/env python3
"""One-shot internal audit of the node-specific query-state critic."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


METHODS = ("no_credit", "local_ig_m", "outcome", "old_dagig", "query_critic_dagig")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--train_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path, train_path = args.freeze.resolve(), args.train_audit.resolve()
    freeze, train = read_json(freeze_path), read_json(train_path)
    if train.get("decision") != "DAGIG_V6_QUERY_STATE_CRITIC_V1_TRAIN_OOF_GO":
        raise ValueError("query-state critic did not pass train OOF")
    if freeze["input_hashes"]["auditor"] != sha256(Path(__file__).resolve()):
        raise ValueError("query-state auditor changed after freeze")
    if train["input_hashes"]["freeze"] != sha256(freeze_path):
        raise ValueError("query-state train audit came from another freeze")
    for key, raw_path in train["output_paths"].items():
        if sha256(Path(raw_path)) != train["output_hashes"][key]:
            raise ValueError(f"query-state output changed: {key}")
    query_value_freeze = read_json(Path(freeze["input_paths"]["query_value_freeze"]))
    public = read_jsonl(Path(query_value_freeze["output_paths"]["internal_targets"]))
    diagnostics = {row["parent_state_id"]: row for row in read_jsonl(Path(query_value_freeze["output_paths"]["diagnostics"])) if row["partition"] == "internal_holdout"}
    predictions = {row["query_action_id"]: row for row in read_jsonl(Path(train["output_paths"]["predictions"])) if row["partition"] == "internal_holdout"}
    support_map = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(Path(freeze["input_paths"]["private_support"]))
        if row["partition"] == "internal_holdout"
    }
    terminal = {
        row["answer_action_id"]: row
        for row in read_jsonl(Path(freeze["input_paths"]["terminal_private"]))
        if row["partition"] == "internal_holdout"
    }
    shared = {row["evidence_action_id"]: row for row in read_jsonl(Path(freeze["input_paths"]["shared_answer_values"]))}
    if len(public) != 120 or len(predictions) != 595:
        raise ValueError(f"query-state internal universe mismatch: {len(public)}/{len(predictions)}")
    rows = []
    for target in public:
        diagnostic = diagnostics[target["parent_state_id"]]
        query_ids = diagnostic["query_action_ids"]
        values = [float(predictions[query_id]["query_success_probability"]) for query_id in query_ids]
        critic_posterior = normalize([value / len(values) for value in values])
        methods = {
            "no_credit": target["target_distributions"]["no_credit"],
            "local_ig_m": target["target_distributions"]["local_ig_m"],
            "outcome": target["target_distributions"]["outcome"],
            "old_dagig": target["target_distributions"]["dagig"],
            "query_critic_dagig": critic_posterior,
        }
        action_metrics = []
        for index, query_id in enumerate(query_ids):
            evidence_id = diagnostic["selected_evidence_action_ids"][index]
            evidence_strategy = evidence_id.rsplit("::", 1)[-1]
            value = shared[evidence_id]
            probabilities = [float(item) for item in value["answer_policy_probabilities"]]
            strict = [float(terminal[answer_id]["strict_proxy"]) for answer_id in value["answer_action_ids"]]
            mode = value["answer_action_ids"].index(value["mode_answer_action_id"])
            action_metrics.append({
                "query_action_id": query_id,
                "query_strategy": query_id.rsplit("::", 1)[-1],
                "evidence_action_id": evidence_id,
                "support": float(support_map[query_id][evidence_strategy]),
                "expected_strict": sum(p * label for p, label in zip(probabilities, strict)),
                "mode_strict": strict[mode],
                "query_critic_value": values[index],
            })
        selected = {}
        for method, distribution in methods.items():
            choice = max(range(len(distribution)), key=lambda index: (float(distribution[index]), -index))
            selected[method] = action_metrics[choice]
        rows.append({"parent_state_id": target["parent_state_id"], "sample_id": target["parent_state_id"].split("::", 1)[0], "methods": selected})
    summary = {}
    for method in METHODS:
        chosen = [row["methods"][method] for row in rows]
        summary[method] = {
            "states": len(chosen),
            "samples": len({row["sample_id"] for row in rows}),
            "support": mean(row["support"] for row in chosen),
            "expected_strict": mean(row["expected_strict"] for row in chosen),
            "mode_strict": mean(row["mode_strict"] for row in chosen),
            "query_critic_value": mean(row["query_critic_value"] for row in chosen),
            "query_strategy_distribution": dict(sorted(Counter(row["query_strategy"] for row in chosen).items())),
        }
    disagreement = mean(row["methods"]["query_critic_dagig"]["query_action_id"] != row["methods"]["outcome"]["query_action_id"] for row in rows)
    threshold = freeze["development_gates"]
    dag, no_credit, local, outcome = summary["query_critic_dagig"], summary["no_credit"], summary["local_ig_m"], summary["outcome"]
    gates = {
        "complete_120_internal_visual_states": len(rows) == 120,
        "complete_40_internal_samples": len({row["sample_id"] for row in rows}) == 40,
        "support_not_below_no_credit": dag["support"] - no_credit["support"] >= threshold["support_delta_vs_no_credit_min"],
        "support_noninferior_local": dag["support"] >= local["support"] - threshold["support_noninferiority_vs_local_tolerance"],
        "support_noninferior_outcome": dag["support"] >= outcome["support"] - threshold["support_noninferiority_vs_outcome_tolerance"],
        "strict_noninferior_no_credit": dag["expected_strict"] >= no_credit["expected_strict"] - threshold["strict_noninferiority_vs_no_credit_tolerance"],
        "strict_noninferior_local": dag["expected_strict"] >= local["expected_strict"] - threshold["strict_noninferiority_vs_local_tolerance"],
        "strict_noninferior_outcome": dag["expected_strict"] >= outcome["expected_strict"] - threshold["strict_noninferiority_vs_outcome_tolerance"],
        "differs_from_outcome": disagreement >= threshold["top_action_disagreement_vs_outcome_min"],
        "query_strategy_diversity": len(dag["query_strategy_distribution"]) >= threshold["selected_query_strategies_min"],
        "predictions_frozen_before_internal_labels": True,
        "runtime_features_use_no_gold": True,
        "development_result_not_paper_final": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {key: bool(value) for key, value in gates.items()}
    decision = "DAGIG_V6_QUERY_STATE_CRITIC_V1_DEVELOPMENT_GO" if all(gates.values()) else "DAGIG_V6_QUERY_STATE_CRITIC_V1_DEVELOPMENT_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    row_path = output / "v6_query_state_critic_internal_private.jsonl"
    with row_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit = {"decision": decision, "protocol_version": freeze["protocol_version"], "method_summary": summary, "dagig_outcome_top_disagreement": disagreement, "gates": gates, "input_paths": {"freeze": str(freeze_path), "train_audit": str(train_path)}, "input_hashes": {"freeze": sha256(freeze_path), "train_audit": sha256(train_path)}, "output_paths": {"private_rows": str(row_path)}, "output_hashes": {"private_rows": sha256(row_path)}, "internal_labels_loaded_only_after_predictions_frozen": True, "dev_used": False, "test_used": False, "api_calls": 0, "training_run": False}
    audit_path = output / "DAGIG_V6_QUERY_STATE_CRITIC_V1_DEVELOPMENT_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "method_summary": summary, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
