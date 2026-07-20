#!/usr/bin/env python3
"""Freeze a memory-safe deterministic categorical evidence protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_deterministic_freeze", type=Path, required=True)
    parser.add_argument("--preflight_oom_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source_deterministic_freeze.resolve()
    preflight_path = args.preflight_oom_audit.resolve()
    source = read_json(source_path)
    preflight = read_json(preflight_path)
    if source.get("protocol_version") != "dagig_v6_backward_evidence_explicit_categorical_deterministic_v2":
        raise ValueError("source is not deterministic categorical v2")
    if preflight.get("decision") != "DAGIG_V6_CATEGORICAL_DETERMINISTIC_V2_PREFLIGHT_OOM_NO_GO":
        raise ValueError("v2 preflight OOM was not established")
    if preflight["input_hashes"]["freeze"] != sha256(source_path):
        raise ValueError("preflight audit belongs to another deterministic freeze")
    if preflight.get("internal_holdout_used") or preflight.get("dev_used") or preflight.get("test_used"):
        raise ValueError("held-out data was opened during preflight diagnosis")

    scripts = Path(__file__).resolve().parent
    runner_paths = {
        "freezer": Path(__file__).resolve(),
        "trainer": scripts / "770_train_v6_backward_evidence_policy_categorical_deterministic_microbatch.py",
        "scorer": scripts / "771_score_v6_backward_evidence_policy_categorical_deterministic_microbatch.py",
        "train_fit_auditor": scripts / "772_audit_v6_backward_evidence_categorical_deterministic_microbatch_train_fit.py",
        "fixed_point_auditor": scripts / "773_audit_v6_categorical_microbatch_no_credit_fixed_point.py",
    }
    if not all(path.is_file() for path in runner_paths.values()):
        raise FileNotFoundError("deterministic categorical runner set is incomplete")
    input_paths = {
        "source_deterministic_freeze": str(source_path),
        "preflight_oom_audit": str(preflight_path),
        "control_freeze": source["input_paths"]["control_freeze"],
        "categorical_train_data": source["input_paths"]["categorical_train_data"],
        "categorical_internal_data": source["input_paths"]["categorical_internal_data"],
        "sft_adapter_model": source["input_paths"]["sft_adapter_model"],
    }
    for key, path in input_paths.items():
        if key in {"source_deterministic_freeze", "preflight_oom_audit"}:
            continue
        source_key = key
        if sha256(Path(path)) != source["input_hashes"][source_key]:
            raise ValueError(f"deterministic v2 input changed: {key}")

    training = dict(source["training"])
    training["group_batch_size"] = 1
    training["gradient_accumulation_batches"] = 8
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "decision": "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN",
        "protocol_version": "dagig_v6_backward_evidence_explicit_categorical_deterministic_microbatch_v3",
        "repair_reason": "v2 preflight OOM; halve simultaneous groups while preserving effective batch and objective",
        "base_model": source["base_model"],
        "base_model_tree_sha256": source["base_model_tree_sha256"],
        "shared_sft_adapter": source["shared_sft_adapter"],
        "shared_sft_adapter_tree_sha256": source["shared_sft_adapter_tree_sha256"],
        "groups": source["groups"],
        "internal_groups": source["internal_groups"],
        "actions_per_group": 5,
        "categorical_action_labels": source["categorical_action_labels"],
        "target_keys": source["target_keys"],
        "training": training,
        "metrics": source["metrics"],
        "objective": {
            **source["objective"],
            "deterministic_current_and_reference_logits": True,
            "dropout_active_during_optimization": False,
        },
        "matched_controls": {
            **source["matched_controls"],
            "same_deterministic_forward_function": True,
            "same_effective_groups_per_optimizer_step": training["group_batch_size"] * training["gradient_accumulation_batches"] == 8,
        },
        "frozen_train_fit_gates": source["frozen_train_fit_gates"],
        "required_preflight_gate": {
            "name": "no_credit_fixed_point",
            "groups": 32,
            "max_mean_tv": 1e-5,
            "max_mean_abs_policy_shift": 1e-5,
            "max_grad_norm": 1e-4,
            "must_pass_before_any_full_method_run": True,
        },
        "pre_registered_escalation": source["pre_registered_escalation"],
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "runner_paths": {key: str(path) for key, path in runner_paths.items()},
        "runner_hashes": {key: sha256(path) for key, path in runner_paths.items()},
        "gold_or_qrels_available_to_trainer": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    manifest = output / "DAGIG_V6_BACKWARD_EVIDENCE_CATEGORICAL_DETERMINISTIC_MICROBATCH_FREEZE.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
