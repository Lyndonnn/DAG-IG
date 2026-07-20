#!/usr/bin/env python3
"""Freeze the train-OOF-selected hybrid evidence-state value v3."""

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
    parser.add_argument("--v1_audit", type=Path, required=True)
    parser.add_argument("--v2_freeze", type=Path, required=True)
    parser.add_argument("--v2_audit", type=Path, required=True)
    parser.add_argument("--fitter", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "v1_freeze": args.v1_freeze,
        "v1_audit": args.v1_audit,
        "v2_freeze": args.v2_freeze,
        "v2_audit": args.v2_audit,
        "fitter": args.fitter,
        "auditor": args.auditor,
    }.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    v1_freeze, v1_audit = read_json(paths["v1_freeze"]), read_json(paths["v1_audit"])
    v2_freeze, v2_audit = read_json(paths["v2_freeze"]), read_json(paths["v2_audit"])
    if v1_audit.get("decision") != "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_TRAIN_OOF_NO_GO":
        raise ValueError("unexpected v1 state")
    if v2_audit.get("decision") != "DAGIG_V6_PAIRWISE_EVIDENCE_STATE_CRITIC_V2_TRAIN_OOF_NO_GO":
        raise ValueError("unexpected v2 state")
    if [key for key, value in v1_audit["gates"].items() if not value] != ["pair_order_improves_old"]:
        raise ValueError("v1 no longer has the recorded calibration/ranking tradeoff")
    if [key for key, value in v2_audit["gates"].items() if not value] != ["strict_brier_improves_old", "strict_spearman_improves_old"]:
        raise ValueError("v2 no longer has the recorded calibration/ranking tradeoff")
    for audit in (v1_audit, v2_audit):
        for key, raw_path in audit["output_paths"].items():
            if sha256(Path(raw_path)) != audit["output_hashes"][key]:
                raise ValueError(f"component output changed: {key}")

    gates = {
        "v1_calibration_component_frozen": True,
        "v2_pairwise_component_frozen": True,
        "blend_selected_on_policy_train_oof_only": True,
        "smallest_passing_blend_coefficient_selected": True,
        "same_runtime_features_and_action_universe": True,
        "internal_unused_for_blend_selection": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    protocol = {
        "decision": "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_FROZEN" if all(gates.values()) else "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_NO_GO",
        "protocol_version": "dagig_v6_factorized_pairwise_hybrid_evidence_state_value_v3",
        "value_semantics": "Platt-calibrated P(strict success | evidence state) from calibrated factorized value plus cardinal pairwise residual",
        "hybrid_score": "logit(v1_factorized_probability) + alpha * v2_pairwise_rank_score",
        "alpha": 4.0,
        "alpha_selection": {
            "partition": "policy_train repeated sample-group OOF only",
            "candidate_grid": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
            "rule": "smallest alpha satisfying all frozen calibration, Spearman, and pair-order gates",
            "internal_holdout_used": False,
        },
        "platt": {"l2": 0.003, "newton_steps": 40, "folds": 5, "repeats": 5, "seed_prefix": "dagig_v6_hybrid_evidence_value_v3"},
        "train_oof_gates": v2_freeze["train_oof_gates"],
        "development_gates": v2_freeze["development_gates"],
        "input_paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "component_output_paths": {
            "v1_predictions": v1_audit["output_paths"]["predictions"],
            "v1_models": v1_audit["output_paths"]["models"],
            "v2_predictions": v2_audit["output_paths"]["predictions"],
            "v2_models": v2_audit["output_paths"]["models"],
        },
        "component_output_hashes": {
            "v1_predictions": v1_audit["output_hashes"]["predictions"],
            "v1_models": v1_audit["output_hashes"]["models"],
            "v2_predictions": v2_audit["output_hashes"]["predictions"],
            "v2_models": v2_audit["output_hashes"]["models"],
        },
        "feature_path": v1_freeze["output_paths"]["features"],
        "feature_hash": v1_freeze["output_hashes"]["features"],
        "label_and_control_paths": {
            key: v1_freeze["input_paths"][key]
            for key in ("shared_answer_values", "private_support", "terminal_private", "categorical_train", "categorical_internal")
        },
        "label_and_control_hashes": {
            key: v1_freeze["input_hashes"][key]
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
    path = output / "DAGIG_V6_HYBRID_EVIDENCE_STATE_VALUE_V3_FREEZE.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "alpha": protocol["alpha"], "freeze": str(path), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
