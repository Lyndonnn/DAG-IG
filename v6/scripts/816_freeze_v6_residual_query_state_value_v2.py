#!/usr/bin/env python3
"""Freeze conservative residual query-state value selection on train OOF."""

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
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--fitter", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {"v1_freeze": args.v1_freeze, "v1_audit": args.v1_audit, "helper": args.helper, "fitter": args.fitter, "auditor": args.auditor}.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    v1_freeze, v1_audit = read_json(paths["v1_freeze"]), read_json(paths["v1_audit"])
    if v1_audit.get("decision") != "DAGIG_V6_QUERY_STATE_CRITIC_V1_TRAIN_OOF_NO_GO":
        raise ValueError("residual v2 requires recorded query critic v1 train NO-GO")
    failed = [key for key, value in v1_audit["gates"].items() if not value]
    if failed != ["at_least_one_train_oof_alpha_passes", "selected_smallest_passing_alpha"]:
        raise ValueError(f"unexpected query critic v1 failure: {failed}")
    for key, raw_path in v1_audit["output_paths"].items():
        if sha256(Path(raw_path)) != v1_audit["output_hashes"][key]:
            raise ValueError(f"query critic v1 output changed: {key}")
    gates = {
        "v1_failed_only_composite_selection_rule": True,
        "downstream_value_remains_anchor": True,
        "query_critic_is_residual_not_replacement": True,
        "beta_selected_on_train_oof_only": True,
        "internal_unused_for_revision": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    protocol = {
        "decision": "DAGIG_V6_RESIDUAL_QUERY_STATE_VALUE_V2_FROZEN" if all(gates.values()) else "DAGIG_V6_RESIDUAL_QUERY_STATE_VALUE_V2_NO_GO",
        "protocol_version": "dagig_v6_conservative_residual_query_state_value_v2",
        "semantics": "P(strict success | query state) as a calibrated residual correction around frozen downstream evidence value",
        "residual_score": "logit(downstream_hybrid_evidence_value) + beta*logit(query_critic_v1_probability)",
        "fit": {"beta_grid": [0.5, 0.75, 1.0, 1.5, 2.0], "folds": 5, "repeats": 5, "platt_l2": 0.003, "newton_steps": 40, "seed_prefix": "dagig_v6_residual_query_state_v2"},
        "selection_rule": "smallest beta passing train-OOF calibration, rank noninferiority, and selector gates",
        "train_oof_gates": {"strict_brier_improvement_vs_downstream_min": 0.001, "strict_spearman_delta_vs_downstream_min": 0.0, "pair_order_noninferiority_tolerance": 0.005, "selected_support_noninferiority_vs_outcome_tolerance": 0.01, "selected_strict_noninferiority_vs_outcome_tolerance": 0.01, "nonconstant_parent_group_rate_min": 0.95},
        "development_gates": v1_freeze["development_gates"],
        "input_paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "component_output_paths": v1_audit["output_paths"],
        "component_output_hashes": v1_audit["output_hashes"],
        "feature_path": v1_freeze["output_paths"]["features"],
        "feature_hash": v1_freeze["output_hashes"]["features"],
        "label_and_control_paths": {key: v1_freeze["input_paths"][key] for key in ("private_support", "terminal_private", "shared_answer_values", "query_value_freeze")},
        "label_and_control_hashes": {key: v1_freeze["input_hashes"][key] for key in ("private_support", "terminal_private", "shared_answer_values", "query_value_freeze")},
        "gates": gates,
        "gold_or_qrels_in_runtime_features": False,
        "internal_used_for_fit_or_selection": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "DAGIG_V6_RESIDUAL_QUERY_STATE_VALUE_V2_FREEZE.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "freeze": str(path), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
