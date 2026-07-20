#!/usr/bin/env python3
"""Freeze equal-budget evidence-node training after downstream answer freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
    parser.add_argument("--shared_sft_adapter", type=Path, required=True)
    parser.add_argument("--sft_holdout_audit", type=Path, required=True)
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_groups", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs != 3:
        raise ValueError("v1 freezes three epochs; escalation may only follow a train-fit-only NO-GO")
    if args.learning_rate != 2e-5 or args.gradient_accumulation_groups != 8:
        raise ValueError("v1 freezes lr=2e-5 and gradient accumulation=8")

    control_path = args.control_freeze.resolve()
    control = read_json(control_path)
    if control.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_CONTROLS_FROZEN":
        raise ValueError("backward evidence controls are not frozen")
    train_path = Path(control["output_paths"]["train_data"])
    if sha256(train_path) != control["output_hashes"]["train_data"]:
        raise ValueError("evidence training targets changed")
    rows = read_jsonl(train_path)
    if len(rows) != int(control["metrics"]["policy_train_groups"]) or any(len(row["completions"]) != 5 for row in rows):
        raise ValueError("evidence training group universe is incomplete")
    shared = args.shared_sft_adapter.resolve()
    sft_audit_path = args.sft_holdout_audit.resolve()
    sft_audit = read_json(sft_audit_path)
    if sft_audit.get("decision") != "DAGIG_V6_EVIDENCE_NEUTRAL_SFT_GO":
        raise ValueError("shared evidence format SFT is not GO")
    if sha256(shared / "adapter_model.safetensors") != sft_audit["adapter_model_sha256"]:
        raise ValueError("shared evidence initializer changed")
    base_model = args.base_model.resolve()
    if Path(read_json(shared / "adapter_config.json")["base_model_name_or_path"]).resolve() != base_model:
        raise ValueError("initializer and base model differ")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    maximum = 0
    for row in rows:
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True)
        for completion in row["completions"]:
            maximum = max(maximum, len(prefix) + len(tokenizer(completion, add_special_tokens=False)["input_ids"]) + 1)
    if maximum > args.max_input_tokens:
        raise ValueError(f"evidence sequence length {maximum} exceeds {args.max_input_tokens}")
    trainer = Path(__file__).with_name("736_train_v6_backward_evidence_policy.py").resolve()
    input_paths = {
        "control_freeze": str(control_path),
        "train_data": str(train_path.resolve()),
        "sft_holdout_audit": str(sft_audit_path),
        "sft_adapter_config": str((shared / "adapter_config.json").resolve()),
        "sft_adapter_model": str((shared / "adapter_model.safetensors").resolve()),
    }
    freeze = {
        "decision": "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN",
        "protocol_version": "dagig_v6_backward_fixed_answer_equal_evidence_training_v1",
        "methods": ["no_credit", "local_ig", "outcome", "dagig"],
        "target_keys": control["target_keys"],
        "groups": len(rows),
        "actions_per_group": 5,
        "base_model": str(base_model),
        "base_model_tree_sha256": tree_hash(base_model),
        "shared_sft_adapter": str(shared),
        "shared_sft_adapter_tree_sha256": tree_hash(shared),
        "objective": {
            "loss": "listwise cross entropy from frozen target to reference-corrected policy",
            "policy": "softmax(log behavior + beta * (field_logp_theta - field_logp_reference))",
            "optimized_field": "selected_evidence_ids only",
            "beta": 1.0,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_groups": args.gradient_accumulation_groups,
            "group_batch_size": 1,
            "max_input_tokens": args.max_input_tokens,
            "max_grad_norm": 1.0,
            "logging_steps": 50,
            "seed": 761943,
            "beta": 1.0,
        },
        "train_fit_only_escalation_rule": (
            "If and only if a method fails the frozen policy-train fit gate, create a new version "
            "with six epochs without opening internal/dev/test. Do not jump to 12 epochs."
        ),
        "matched_controls": {
            "same_groups": True,
            "same_actions": True,
            "same_initializer": True,
            "same_reference": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_batch_order": True,
            "only_target_distribution_changes": True,
            "same_frozen_answer_policy": True,
        },
        "metrics": {"max_sequence_tokens": maximum},
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "runner_hashes": {"trainer": sha256(trainer)},
        "gold_or_qrels_available_to_trainer": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FREEZE.json"
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
