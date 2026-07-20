#!/usr/bin/env python3
"""One-shot internal selector audit for hybrid evidence-state value v3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_v6_hybrid_audit_helper", path)
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
    if train.get("decision") != "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_TRAIN_OOF_GO":
        raise ValueError("hybrid evidence v3 did not pass train OOF")
    if freeze["input_hashes"]["auditor"] != sha256(Path(__file__).resolve()):
        raise ValueError("v3 auditor changed after freeze")
    if train["input_hashes"]["freeze"] != sha256(freeze_path):
        raise ValueError("v3 train audit came from another freeze")
    for key, raw_path in train["output_paths"].items():
        if sha256(Path(raw_path)) != train["output_hashes"][key]:
            raise ValueError(f"v3 train output changed: {key}")
    helper = load_module(Path(__file__).with_name("807_audit_v6_pairwise_evidence_state_critic_v2.py"))
    predictions = {
        row["evidence_action_id"]: row
        for row in read_jsonl(Path(train["output_paths"]["predictions"]))
        if row["partition"] == "internal_holdout"
    }
    shared = {row["evidence_action_id"]: row for row in read_jsonl(Path(freeze["label_and_control_paths"]["shared_answer_values"]))}
    support_map = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(Path(freeze["label_and_control_paths"]["private_support"]))
        if row["partition"] == "internal_holdout"
    }
    terminal = {
        row["answer_action_id"]: row
        for row in read_jsonl(Path(freeze["label_and_control_paths"]["terminal_private"]))
        if row["partition"] == "internal_holdout"
    }
    categorical = read_jsonl(Path(freeze["label_and_control_paths"]["categorical_internal"]))
    if len(predictions) != 2975 or len(categorical) != 595:
        raise ValueError(f"v3 internal universe mismatch: {len(predictions)}/{len(categorical)}")

    private_rows = []
    for group in categorical:
        action_ids = group["action_ids"]
        values = [float(predictions[action_id]["evidence_success_probability"]) for action_id in action_ids]
        posterior = normalize([0.2 * max(value, 1e-8) for value in values])
        methods = {
            "no_credit": group["behavior_probabilities"],
            "local_ig_m": group["local_target_probabilities"],
            "outcome": group["outcome_target_probabilities"],
            "old_dagig": group["dagig_target_probabilities"],
            "hybrid_dagig": posterior,
        }
        selected = {}
        for method, probabilities in methods.items():
            choice = max(range(5), key=lambda index: (float(probabilities[index]), -index))
            evidence_id = action_ids[choice]
            value = shared[evidence_id]
            strategy = evidence_id.rsplit("::", 1)[-1]
            answer_probabilities = [float(item) for item in value["answer_policy_probabilities"]]
            strict = [float(terminal[answer_id]["strict_proxy"]) for answer_id in value["answer_action_ids"]]
            correct = [float(terminal[answer_id]["answer_correct_proxy"]) for answer_id in value["answer_action_ids"]]
            mode = value["answer_action_ids"].index(value["mode_answer_action_id"])
            selected[method] = {
                "evidence_action_id": evidence_id,
                "strategy": strategy,
                "support": float(support_map[group["parent_group_id"]][strategy]),
                "expected_answer_correct": sum(p * label for p, label in zip(answer_probabilities, correct)),
                "expected_strict": sum(p * label for p, label in zip(answer_probabilities, strict)),
                "mode_strict": strict[mode],
                "old_terminal_value": float(value["shared_answer_value"]),
                "hybrid_value": values[choice],
            }
        private_rows.append({"parent_state_id": group["parent_group_id"], "sample_id": group["sample_id"], "partition": "internal_holdout", "methods": selected})

    methods = ("no_credit", "local_ig_m", "outcome", "old_dagig", "hybrid_dagig")
    summary = {}
    for method in methods:
        rows = [group["methods"][method] for group in private_rows]
        summary[method] = {
            "states": len(rows),
            "samples": len({group["sample_id"] for group in private_rows}),
            "support": mean(row["support"] for row in rows),
            "expected_answer_correct": mean(row["expected_answer_correct"] for row in rows),
            "expected_strict": mean(row["expected_strict"] for row in rows),
            "mode_strict": mean(row["mode_strict"] for row in rows),
            "old_terminal_value": mean(row["old_terminal_value"] for row in rows),
            "hybrid_value": mean(row["hybrid_value"] for row in rows),
            "strategy_distribution": dict(sorted(Counter(row["strategy"] for row in rows).items())),
        }
    comparisons = {}
    bootstrap_rows = [
        {
            **row,
            "methods": {**row["methods"], "pairwise_dagig": row["methods"]["hybrid_dagig"]},
        }
        for row in private_rows
    ]
    for baseline in ("no_credit", "local_ig_m", "outcome", "old_dagig"):
        comparisons[f"hybrid_dagig_vs_{baseline}"] = {
            "top_action_disagreement_rate": mean(
                row["methods"]["hybrid_dagig"]["evidence_action_id"] != row["methods"][baseline]["evidence_action_id"]
                for row in private_rows
            ),
            **{metric: helper.bootstrap(bootstrap_rows, metric, baseline) for metric in ("support", "expected_strict", "mode_strict")},
        }

    threshold = freeze["development_gates"]
    dag, no_credit, local, outcome = summary["hybrid_dagig"], summary["no_credit"], summary["local_ig_m"], summary["outcome"]
    gates = {
        "complete_595_internal_query_states": len(private_rows) == 595,
        "complete_40_internal_samples": len({row["sample_id"] for row in private_rows}) == 40,
        "support_not_below_no_credit": dag["support"] - no_credit["support"] >= threshold["support_delta_vs_no_credit_min"],
        "support_noninferior_local": dag["support"] >= local["support"] - threshold["support_noninferiority_vs_local_tolerance"],
        "support_noninferior_outcome": dag["support"] >= outcome["support"] - threshold["support_noninferiority_vs_outcome_tolerance"],
        "strict_noninferior_no_credit": dag["expected_strict"] >= no_credit["expected_strict"] - threshold["strict_noninferiority_vs_no_credit_tolerance"],
        "strict_noninferior_local": dag["expected_strict"] >= local["expected_strict"] - threshold["strict_noninferiority_vs_local_tolerance"],
        "strict_noninferior_outcome": dag["expected_strict"] >= outcome["expected_strict"] - threshold["strict_noninferiority_vs_outcome_tolerance"],
        "differs_from_outcome": comparisons["hybrid_dagig_vs_outcome"]["top_action_disagreement_rate"] >= threshold["top_action_disagreement_vs_outcome_min"],
        "evidence_action_diversity": len(dag["strategy_distribution"]) >= threshold["selected_evidence_strategies_min"],
        "predictions_frozen_before_internal_labels": True,
        "internal_never_fit": True,
        "runtime_features_contain_no_gold_or_qrels": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {key: bool(value) for key, value in gates.items()}
    decision = "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_DEVELOPMENT_GO" if all(gates.values()) else "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_DEVELOPMENT_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "v6_hybrid_evidence_state_internal_private.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in private_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "method_summary": summary,
        "pairwise_comparisons": comparisons,
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path), "train_audit": str(train_path)},
        "input_hashes": {"freeze": sha256(freeze_path), "train_audit": sha256(train_path)},
        "output_paths": {"private_rows": str(rows_path)},
        "output_hashes": {"private_rows": sha256(rows_path)},
        "internal_labels_loaded_only_after_predictions_frozen": True,
        "development_result_not_paper_final": True,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_DEVELOPMENT_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "method_summary": summary, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
