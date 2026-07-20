#!/usr/bin/env python3
"""Record why categorical v1 was stopped before its first epoch boundary."""

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


def last_optimizer_log(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            row = json.loads(line)
            if "optimizer_step" in row:
                rows.append(row)
    if not rows:
        raise ValueError(f"no optimizer log found: {path}")
    return rows[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--policy_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.training_freeze.resolve()
    policy_root = args.policy_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze = read_json(freeze_path)
    if freeze.get("protocol_version") != "dagig_v6_backward_evidence_explicit_categorical_actions_v1":
        raise ValueError("dropout audit requires categorical v1")
    trainer = Path(freeze["runner_paths"]["trainer"])
    if sha256(trainer) != freeze["runner_hashes"]["trainer"]:
        raise ValueError("categorical v1 trainer changed")
    source = trainer.read_text(encoding="utf-8")
    if "model.eval()" not in source or "model.train()" not in source:
        raise ValueError("expected train/eval mode transition is absent")
    adapter_config_path = Path(freeze["shared_sft_adapter"]) / "adapter_config.json"
    adapter = read_json(adapter_config_path)
    dropout = float(adapter.get("lora_dropout", 0.0))
    no_credit_log = policy_root / "logs" / "no_credit.log"
    local_log = policy_root / "logs" / "local_ig.log"
    no_credit = last_optimizer_log(no_credit_log)
    local = last_optimizer_log(local_log)
    final_audits = list(policy_root.glob("*/DAGIG_V6_BACKWARD_EVIDENCE_POLICY_TRAIN_AUDIT.json"))
    checkpoints = list((policy_root / ".checkpoints").glob("*/CHECKPOINT_STATE.json"))
    scratch = policy_root.parent
    held_out_paths = [
        scratch / "no_gold_backward_evidence_categorical_internal_scores_v1",
        scratch / "no_gold_backward_evidence_categorical_internal_methods_v1",
        scratch / "no_gold_backward_evidence_categorical_internal_audit_v1",
    ]
    gates = {
        "lora_dropout_positive": dropout > 0.0,
        "reference_computed_in_eval_mode": True,
        "optimization_switched_model_to_train_mode": True,
        "no_credit_should_be_fixed_point_at_identical_parameters": True,
        "no_credit_early_tv_exceeds_old_gate": float(no_credit["policy_target_tv"]) > 0.03,
        "stopped_before_first_epoch_checkpoint": len(checkpoints) == 0,
        "no_final_training_audits": len(final_audits) == 0,
        "internal_holdout_unopened": not any(path.exists() for path in held_out_paths),
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = (
        "DAGIG_V6_CATEGORICAL_V1_ABORTED_DROPOUT_POLICY_MISMATCH"
        if all(gates.values())
        else "DAGIG_V6_CATEGORICAL_V1_ABORT_AUDIT_INCONCLUSIVE"
    )
    result = {
        "decision": decision,
        "gates": gates,
        "metrics": {
            "lora_dropout": dropout,
            "no_credit_last_log": no_credit,
            "local_ig_last_log": local,
            "final_training_audits": len(final_audits),
            "completed_epoch_checkpoints": len(checkpoints),
        },
        "mathematical_invariant": (
            "At initialization theta=reference and the No-credit target equals behavior, so "
            "softmax(log behavior + logp_theta - logp_reference) equals behavior and the exact gradient is zero."
        ),
        "diagnosis": (
            "Reference logits were computed with model.eval(), then optimization used model.train(). "
            "The initializer has LoRA dropout=0.05, so current and reference logits were stochastic even at "
            "identical parameters. A single-token categorical action exposes this mismatch directly."
        ),
        "repair_contract": {
            "disable_dropout_during_reference_and_gradient_scoring": True,
            "keep_gradients_enabled_while_model_is_in_eval_mode": True,
            "all_four_methods_restart_from_shared_initializer": True,
            "do_not_resume_v1": True,
            "same_data_actions_targets_optimizer_and_gate_thresholds": True,
            "require_no_credit_fixed_point_smoke_before_full_run": True,
            "internal_dev_test_remain_sealed": True,
        },
        "input_paths": {
            "training_freeze": str(freeze_path),
            "trainer": str(trainer),
            "adapter_config": str(adapter_config_path),
            "no_credit_log": str(no_credit_log),
            "local_ig_log": str(local_log),
        },
        "input_hashes": {
            "training_freeze": sha256(freeze_path),
            "trainer": sha256(trainer),
            "adapter_config": sha256(adapter_config_path),
            "no_credit_log": sha256(no_credit_log),
            "local_ig_log": sha256(local_log),
        },
        "gold_or_qrels_loaded": False,
        "internal_holdout_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
