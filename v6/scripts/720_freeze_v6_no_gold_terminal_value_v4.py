#!/usr/bin/env python3
"""Freeze the compact, shortcut-free no-gold terminal value v4."""

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
    parser.add_argument("--v3_freeze", type=Path, required=True)
    parser.add_argument("--v3_audit", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--backup_builder", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    v3_path = args.v3_freeze.resolve()
    audit_path = args.v3_audit.resolve()
    v3 = read_json(v3_path)
    audit = read_json(audit_path)
    if v3.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN":
        raise ValueError("no-gold terminal v3 source is not frozen")
    if v3.get("protocol_version") != "dagig_v6_deployable_no_gold_terminal_value_stratified_repeated_group_cv_v3":
        raise ValueError("unexpected no-gold terminal v3 protocol")
    if audit.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO":
        raise ValueError("no-gold terminal v3 source did not pass")
    if audit["input_hashes"]["freeze"] != sha256(v3_path):
        raise ValueError("v3 audit came from another freeze")
    for key, path in audit["output_paths"].items():
        if sha256(Path(path)) != audit["output_hashes"][key]:
            raise ValueError(f"v3 output changed: {key}")
    calibrator = args.calibrator.resolve()
    backup_builder = args.backup_builder.resolve()
    helper = Path(__file__).with_name("702_calibrate_v6_no_gold_terminal_value.py").resolve()
    fold_helper = Path(__file__).with_name("708_calibrate_v6_no_gold_terminal_value_v3.py").resolve()
    for path in (calibrator, backup_builder, helper, fold_helper):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_records = Path(audit["output_paths"]["private_audit"])
    gates = {
        "v3_source_integrity": True,
        "runtime_features_are_compact_and_observable": True,
        "strategy_identity_features_removed": True,
        "interaction_and_minimum_shortcuts_removed": True,
        "fit_uses_policy_train_labels_only": True,
        "groups_remain_disjoint_by_sample": True,
        "gold_equivalence_excluded": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    freeze = {
        **{
            key: v3[key]
            for key in (
                "answer_actions",
                "evidence_actions",
                "policy_train_sample_ids",
                "development_sample_ids",
                "answer_strategy_flags",
                "evidence_strategies",
                "input_paths",
                "input_hashes",
                "leakage_contract",
            )
        },
        "decision": "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN" if all(gates.values()) else "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_NO_GO",
        "protocol_version": "dagig_v6_compact_monotone_no_gold_terminal_value_v4",
        "semantics": "P(strict success | question, evidence, candidate answer) from frozen no-gold support and reader scores",
        "feature_names": ["support_logit", "reader_candidate_mean_logprob"],
        "calibration": {
            "model": "standardized logistic regression",
            "l2": 0.0003,
            "max_newton_steps": 30,
            "folds": 5,
            "repeats": 10,
            "fold_seed_prefix": "dagig_v6_no_gold_terminal_compact_v4",
            "train_probability": "mean of ten stratified sample-grouped OOF predictions",
            "probability_clip": [1e-5, 0.99999],
            "monotonicity_contract": "both final standardized feature coefficients must be strictly positive",
            "all_hyperparameters_selected_before v4_calibration": True,
        },
        "gates_spec": {
            "repeated_oof_auc_mean_min": 0.895,
            "repeated_oof_auc_worst_min": 0.89,
            "repeated_oof_brier_improvement_worst_min": 0.01,
            "development_auc_min": 0.90,
            "development_brier_improvement_min": 0.008,
            "development_ece_max": 0.05,
            "development_evidence_value_strict_spearman_min": 0.35,
            "nonconstant_group_rate_min": 0.95,
        },
        "gates": gates,
        "source_paths": {
            "v3_freeze": str(v3_path),
            "v3_audit": str(audit_path),
            "v3_private_records": str(source_records),
        },
        "source_hashes": {
            "v3_freeze": sha256(v3_path),
            "v3_audit": sha256(audit_path),
            "v3_private_records": sha256(source_records),
        },
        "code_paths": {
            "freezer": str(Path(__file__).resolve()),
            "calibrator": str(calibrator),
            "calibration_helper": str(helper),
            "fold_helper": str(fold_helper),
            "backup_builder": str(backup_builder),
        },
        "code_hashes": {
            "freezer": sha256(Path(__file__).resolve()),
            "calibrator": sha256(calibrator),
            "calibration_helper": sha256(helper),
            "fold_helper": sha256(fold_helper),
            "backup_builder": sha256(backup_builder),
        },
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FREEZE.json"
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": freeze["decision"], "freeze": str(path), "gates": gates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
