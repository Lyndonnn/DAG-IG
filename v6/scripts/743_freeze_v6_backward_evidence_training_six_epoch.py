#!/usr/bin/env python3
"""Freeze the pre-registered six-epoch evidence train-fit escalation."""

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
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--train_fit_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_freeze.resolve()
    audit_path = args.train_fit_audit.resolve()
    source = read_json(source_path)
    audit = read_json(audit_path)
    if source.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN":
        raise ValueError("source evidence training protocol is not frozen")
    if source.get("training", {}).get("epochs") != 3:
        raise ValueError("six-epoch escalation must start from the frozen three-epoch protocol")
    if "six epochs" not in source.get("train_fit_only_escalation_rule", ""):
        raise ValueError("source protocol did not pre-register the six-epoch escalation")
    if audit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAIN_FIT_NO_GO":
        raise ValueError("six-epoch escalation requires a policy-train-only NO-GO")
    if audit.get("internal_holdout_used") or audit.get("dev_used") or audit.get("test_used"):
        raise ValueError("holdout data was opened before the train-fit escalation")
    if Path(audit["input_paths"]["training_freeze"]).resolve() != source_path:
        raise ValueError("train-fit audit does not belong to the source protocol")
    for key, path in source["input_paths"].items():
        if sha256(Path(path)) != source["input_hashes"][key]:
            raise ValueError(f"source frozen input changed: {key}")

    failed_methods = [
        method
        for method in ("local_ig", "outcome", "dagig")
        if audit["metrics"][method]["mean_policy_target_tv"] > 0.10
        or audit["metrics"][method]["top_action_agreement"] < 0.65
        or (
            audit["metrics"][method]["margin_ge_0p05_top_action_agreement"] is not None
            and audit["metrics"][method]["margin_ge_0p05_top_action_agreement"] < 0.85
        )
    ]
    if not failed_methods:
        raise ValueError("no trained method failed the frozen train-fit gates")

    freeze = dict(source)
    freeze["protocol_version"] = "dagig_v6_backward_fixed_answer_equal_evidence_training_batched_six_epoch_v3"
    freeze["training"] = dict(source["training"])
    freeze["training"]["epochs"] = 6
    freeze["six_epoch_train_fit_escalation"] = {
        "trigger": "pre-registered policy-train-only fit failure",
        "triggered_by_methods": failed_methods,
        "source_epochs": 3,
        "escalated_epochs": 6,
        "all_four_methods_rerun": True,
        "reason_all_four_methods_rerun": "preserve equal optimizer steps and compute across matched controls",
        "hyperparameters_changed_other_than_epochs": False,
        "internal_holdout_opened": False,
        "dev_opened": False,
        "test_opened": False,
        "twelve_epoch_run_allowed": False,
    }
    freeze["input_paths"] = {
        "source_v2_freeze": str(source_path),
        "source_v2_train_fit_audit": str(audit_path),
        **source["input_paths"],
    }
    freeze["input_hashes"] = {key: sha256(Path(path)) for key, path in freeze["input_paths"].items()}
    trainer = Path(__file__).with_name("742_train_v6_backward_evidence_policy_batched.py").resolve()
    freeze["runner_hashes"] = {"trainer": sha256(trainer)}
    freeze["matched_controls"] = {
        **source["matched_controls"],
        "same_steps": True,
        "same_epochs": True,
        "all_methods_restarted_from_shared_initializer": True,
    }
    freeze["training_run"] = False

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FREEZE.json"
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
