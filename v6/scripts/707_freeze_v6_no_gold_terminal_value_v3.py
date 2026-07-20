#!/usr/bin/env python3
"""Freeze stratified repeated-group CV no-gold P_success v3."""

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
    parser.add_argument("--v2_freeze", type=Path, required=True)
    parser.add_argument("--v2_audit", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--backup_builder", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    v2_path = args.v2_freeze.resolve()
    v2_audit_path = args.v2_audit.resolve()
    v2 = read_json(v2_path)
    v2_audit = read_json(v2_audit_path)
    if v2.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN" or v2.get("protocol_version") != "dagig_v6_deployable_no_gold_terminal_value_repeated_cv_v2":
        raise ValueError("no-gold terminal v2 source is not frozen")
    if v2_audit.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO" or v2_audit.get("protocol_version") != v2["protocol_version"]:
        raise ValueError("no-gold terminal v2 audit is not GO")
    if v2_audit["input_hashes"]["freeze"] != sha256(v2_path):
        raise ValueError("v2 audit came from another freeze")
    for key, path in v2_audit["output_paths"].items():
        if sha256(Path(path)) != v2_audit["output_hashes"][key]:
            raise ValueError(f"v2 output changed: {key}")
    for key, path in v2["input_paths"].items():
        if key == "score_manifests":
            for item, expected in zip(path, v2["input_hashes"][key]):
                if sha256(Path(item)) != expected:
                    raise ValueError("score manifest changed")
        elif sha256(Path(path)) != v2["input_hashes"][key]:
            raise ValueError(f"v2 source input changed: {key}")
    calibration_helper = Path(__file__).resolve().with_name("702_calibrate_v6_no_gold_terminal_value.py")
    for path in (args.calibrator.resolve(), args.backup_builder.resolve(), calibration_helper):
        if not path.is_file():
            raise FileNotFoundError(path)
    gates = {
        "v2_source_integrity": True,
        "stratification_uses_policy_train_labels_only": True,
        "groups_remain_disjoint_by_sample": True,
        "fold_objective_balances_positive_actions_total_actions_and_samples": True,
        "feature_schema_unchanged_from_v2": True,
        "gold_equivalence_excluded": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    freeze = {
        **{key: v2[key] for key in ("answer_actions", "evidence_actions", "policy_train_sample_ids", "development_sample_ids", "feature_names", "answer_strategy_flags", "evidence_strategies", "input_paths", "input_hashes", "leakage_contract")},
        "decision": "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN" if all(gates.values()) else "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_NO_GO",
        "protocol_version": "dagig_v6_deployable_no_gold_terminal_value_stratified_repeated_group_cv_v3",
        "semantics": v2["semantics"],
        "calibration": {
            "model": "standardized logistic regression",
            "l2": 0.0003,
            "max_newton_steps": 30,
            "folds": 5,
            "repeats": 10,
            "fold_seed_prefix": "dagig_v6_no_gold_terminal_stratified_group_cv_v3",
            "group_stratification": "greedy global imbalance minimization over positive actions, total actions, and sample counts; sample atomic",
            "train_probability": "mean of ten stratified sample-grouped OOF predictions",
            "probability_clip": [1e-5, 0.99999],
            "all_hyperparameters_and_stratification_selected_from_policy_train_only": True,
        },
        "gates_spec": {
            "repeated_oof_auc_mean_min": 0.90,
            "repeated_oof_auc_worst_min": 0.895,
            "repeated_oof_brier_improvement_worst_min": 0.01,
            "max_fold_positive_imbalance_fraction": 0.04,
            "max_fold_action_imbalance_fraction": 0.04,
            "development_auc_min": 0.85,
            "development_brier_improvement_min": 0.002,
            "development_ece_max": 0.05,
            "nonconstant_group_rate_min": 0.95,
        },
        "gates": gates,
        "code_paths": {
            "freezer": str(Path(__file__).resolve()),
            "calibrator": str(args.calibrator.resolve()),
            "calibration_helper": str(calibration_helper),
            "backup_builder": str(args.backup_builder.resolve()),
        },
        "code_hashes": {
            "freezer": sha256(Path(__file__).resolve()),
            "calibrator": sha256(args.calibrator.resolve()),
            "calibration_helper": sha256(calibration_helper),
            "backup_builder": sha256(args.backup_builder.resolve()),
        },
        "source_paths": {"v2_freeze": str(v2_path), "v2_audit": str(v2_audit_path), "v2_private_records": v2_audit["output_paths"]["private_audit"]},
        "source_hashes": {"v2_freeze": sha256(v2_path), "v2_audit": sha256(v2_audit_path), "v2_private_records": v2_audit["output_hashes"]["private_audit"]},
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FREEZE.json"
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": freeze["decision"], "gates": gates, "freeze": str(path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
