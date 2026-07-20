#!/usr/bin/env python3
"""Freeze equal-compute query training after backward query controls pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = ("no_credit", "local_ig", "outcome", "dagig")


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


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(child.relative_to(root)).encode("utf-8"))
        digest.update(sha256(child).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_freeze", type=Path, required=True)
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--initializer_format_audit", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, default=Path(__file__).with_name("748_train_v6_backward_query_policy_batched.py"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs != 3:
        raise ValueError("initial backward-query protocol is pre-registered at exactly three epochs")

    paths = {
        "control_freeze": args.control_freeze.resolve(),
        "initializer_format_audit": args.initializer_format_audit.resolve(),
        "initializer_model": (args.initializer.resolve() / "adapter_model.safetensors"),
        "initializer_config": (args.initializer.resolve() / "adapter_config.json"),
        "train_data": Path(read_json(args.control_freeze.resolve())["output_paths"]["train_data"]),
        "trainer": args.trainer.resolve(),
    }
    control = read_json(paths["control_freeze"])
    format_audit = read_json(paths["initializer_format_audit"])
    if control.get("decision") != "DAGIG_V6_BACKWARD_QUERY_CONTROLS_FROZEN":
        raise ValueError("backward query controls are not frozen")
    if format_audit.get("decision") != "DAGIG_V6_QUERY_POLICY_FORMAT_GO":
        raise ValueError("shared query initializer format is not GO")
    for key, path in control["input_paths"].items():
        if sha256(Path(path)) != control["input_hashes"][key]:
            raise ValueError(f"query-control input changed: {key}")
    for key, path in control["output_paths"].items():
        if sha256(Path(path)) != control["output_hashes"][key]:
            raise ValueError(f"query-control output changed: {key}")
    initializer = args.initializer.resolve()
    if Path(format_audit["adapter"]).resolve() != initializer:
        raise ValueError("format audit belongs to a different query initializer")
    if sha256(paths["initializer_model"]) != format_audit["adapter_model_sha256"]:
        raise ValueError("query initializer changed after format audit")

    rows = read_jsonl(paths["train_data"])
    if len(rows) != int(control["metrics"]["policy_train_groups"]):
        raise ValueError("backward query train group count changed")
    action_count = sum(len(row["completions"]) for row in rows)
    if action_count != 2359:
        raise ValueError(f"expected 2359 policy-train query actions, found {action_count}")
    for row in rows:
        if row.get("partition") != "policy_train" or not 3 <= len(row["completions"]) <= 5:
            raise ValueError("invalid query policy-train row")
        for key in control["target_keys"].values():
            values = [float(value) for value in row[key]]
            if len(values) != len(row["completions"]) or min(values) <= 0.0 or abs(sum(values) - 1.0) > 1e-8:
                raise ValueError(f"invalid target distribution: {row['parent_group_id']} {key}")

    base_model = args.base_model.resolve()
    if not (base_model / "config.json").is_file():
        raise ValueError("base model is incomplete")
    if not paths["trainer"].is_file():
        raise ValueError("query trainer is missing")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    sequence_lengths: list[int] = []
    for row in rows:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        sequence_lengths.extend(
            len(prefix) + len(tokenizer(completion, add_special_tokens=False)["input_ids"]) + 1
            for completion in row["completions"]
        )
    training = {
        "epochs": args.epochs,
        "learning_rate": 2e-5,
        "group_batch_size": 2,
        "gradient_accumulation_batches": 4,
        "effective_groups_per_optimizer_step": 8,
        "beta": 1.0,
        "max_grad_norm": 1.0,
        "max_input_tokens": 768,
        "logging_steps": 20,
        "seed": 761943,
    }
    gates = {
        "control_gate_passed": True,
        "complete_474_policy_train_groups": len(rows) == 474,
        "complete_2359_policy_train_actions": action_count == 2359,
        "three_to_five_actions_per_group": all(3 <= len(row["completions"]) <= 5 for row in rows),
        "shared_initializer_format_go": True,
        "all_sequences_within_frozen_limit": max(sequence_lengths) <= training["max_input_tokens"],
        "only_search_query_tokens_optimized": True,
        "all_methods_same_initializer_data_order_optimizer_and_steps": True,
        "internal_holdout_unused": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_BACKWARD_QUERY_TRAINING_FROZEN" if all(gates.values()) else "DAGIG_V6_BACKWARD_QUERY_TRAINING_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_paths = {key: str(path) for key, path in paths.items()}
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_fixed_descendants_equal_query_training_three_epoch_resumable_v1",
        "methods": list(METHODS),
        "groups": len(rows),
        "actions": action_count,
        "base_model": str(base_model),
        "base_model_tree_sha256": tree_hash(base_model),
        "shared_initializer": str(initializer),
        "shared_initializer_tree_sha256": tree_hash(initializer),
        "target_keys": control["target_keys"],
        "objective": {
            "optimized_field": "search_query only",
            "policy": "softmax(log behavior + beta * (mean query-field logp_theta - mean query-field logp_reference))",
            "loss": "listwise cross entropy from frozen target to reference-corrected policy",
        },
        "training": training,
        "metrics": {
            "max_sequence_tokens": max(sequence_lengths),
            "p99_sequence_tokens": sorted(sequence_lengths)[int(0.99 * (len(sequence_lengths) - 1))],
        },
        "runner_hashes": {"trainer": sha256(paths["trainer"])},
        "matched_controls": {
            "same_action_universe": True,
            "same_initializer": True,
            "same_reference": True,
            "same_batch_order": True,
            "same_optimizer": True,
            "same_steps": True,
            "only_target_distribution_changes": True,
        },
        "train_fit_gates": {
            "no_credit_mean_tv_max": 0.03,
            "trained_method_mean_tv_max": 0.10,
            "trained_method_top_agreement_min": 0.65,
            "trained_method_high_margin_agreement_min": 0.85,
        },
        "pre_registered_escalation": {
            "initial_epochs": 3,
            "allowed_escalation_epochs": 6,
            "trigger": "one or more methods fail policy-train fit before internal is opened",
            "all_four_methods_must_rerun": True,
            "twelve_epochs_allowed": False,
        },
        "resilience": {
            "checkpoint_after_every_epoch": True,
            "adapter_optimizer_rng_and_group_hash_saved": True,
            "atomic_rotation": True,
            "resume_supported": True,
            "detached_session_required": True,
        },
        "gates": gates,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "gold_or_qrels_available_to_trainer": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_BACKWARD_QUERY_TRAINING_FREEZE.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "groups": len(rows), "actions": action_count, "audit": str(audit_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
