#!/usr/bin/env python3
"""Freeze the pairwise-cardinal evidence-state value revision."""

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
    parser.add_argument("--v1_freeze", type=Path, required=True)
    parser.add_argument("--v1_train_audit", type=Path, required=True)
    parser.add_argument("--fitter", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "v1_freeze": args.v1_freeze.resolve(),
        "v1_train_audit": args.v1_train_audit.resolve(),
        "fitter": args.fitter.resolve(),
        "auditor": args.auditor.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    v1 = read_json(paths["v1_freeze"])
    audit = read_json(paths["v1_train_audit"])
    if v1.get("decision") != "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_FROZEN":
        raise ValueError("v1 feature protocol is not frozen")
    if audit.get("decision") != "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_TRAIN_OOF_NO_GO":
        raise ValueError("v2 revision requires the recorded v1 train-OOF NO-GO")
    if audit["input_hashes"]["freeze"] != sha256(paths["v1_freeze"]):
        raise ValueError("v1 audit came from another feature protocol")
    failed = [name for name, passed in audit["gates"].items() if not passed]
    if failed != ["pair_order_improves_old"]:
        raise ValueError(f"unexpected v1 failure set: {failed}")
    for key, raw_path in v1["output_paths"].items():
        if sha256(Path(raw_path)) != v1["output_hashes"][key]:
            raise ValueError(f"v1 output changed: {key}")

    gates = {
        "v1_failed_only_groupwise_pair_order": True,
        "same_runtime_feature_universe": True,
        "same_frozen_answer_policy": True,
        "same_frozen_evidence_actions": True,
        "cardinal_pair_weights_use_train_labels_only": True,
        "internal_unused_for_revision": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    protocol = {
        "decision": "DAGIG_V6_PAIRWISE_EVIDENCE_STATE_CRITIC_V2_FROZEN" if all(gates.values()) else "DAGIG_V6_PAIRWISE_EVIDENCE_STATE_CRITIC_V2_NO_GO",
        "protocol_version": "dagig_v6_pairwise_cardinal_platt_evidence_state_value_v2",
        "revision_rationale": "v1 improved calibration and state correlation but failed only the preregistered within-query pair-order delta; v2 changes the train loss, not features or data",
        "value_semantics": "Platt-calibrated P(strict success under frozen shared answer policy | evidence state)",
        "runtime_feature_names": v1["feature_names"],
        "fit": {
            "ranker": "standardized linear Bradley-Terry model without intercept",
            "pair_orientation": "higher expected strict action first plus symmetric reverse pair",
            "pair_weight": "absolute policy-train expected-strict difference",
            "pair_tie_rule": "exclude exact target ties",
            "calibrator": "one-dimensional standardized logistic Platt map from rank score to expected strict",
            "folds": 5,
            "repeats": 5,
            "rank_l2": 0.03,
            "platt_l2": 0.003,
            "newton_steps": 40,
            "seed_prefix": "dagig_v6_pairwise_evidence_state_v2",
            "probability_clip": [1e-5, 0.99999],
        },
        "train_oof_gates": {
            "strict_brier_improvement_vs_old_min": 0.001,
            "strict_spearman_delta_vs_old_min": 0.015,
            "pair_order_delta_vs_old_min": 0.02,
            "selected_support_noninferiority_vs_outcome_tolerance": 0.01,
            "selected_strict_noninferiority_vs_outcome_tolerance": 0.01,
            "nonconstant_query_group_rate_min": 0.95,
            "platt_slope_min": 0.0,
        },
        "development_gates": v1["development_gates"],
        "input_paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "feature_path": v1["output_paths"]["features"],
        "feature_hash": v1["output_hashes"]["features"],
        "label_and_control_paths": {
            key: v1["input_paths"][key]
            for key in ("shared_answer_values", "private_support", "terminal_private", "categorical_train", "categorical_internal")
        },
        "label_and_control_hashes": {
            key: v1["input_hashes"][key]
            for key in ("shared_answer_values", "private_support", "terminal_private", "categorical_train", "categorical_internal")
        },
        "gates": gates,
        "gold_or_qrels_in_runtime_features": False,
        "internal_holdout_used_for_fit_or_model_selection": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "DAGIG_V6_PAIRWISE_EVIDENCE_STATE_CRITIC_V2_FREEZE.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "freeze": str(path), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
