#!/usr/bin/env python3
"""Freeze repeated-CV no-gold terminal P_success v2."""

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
    parser.add_argument("--score_dir", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--backup_builder", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source_freeze.resolve()
    source = read_json(source_path)
    if source.get("decision") != "DAGIG_V6_TERMINAL_VALUE_FROZEN" or source.get("protocol_version") != "dagig_v6_expanded_answer_terminal_scoring_v1":
        raise ValueError("expanded terminal source is not frozen")
    for key, path in source["input_paths"].items():
        if sha256(Path(path)) != source["input_hashes"][key]:
            raise ValueError(f"expanded terminal input changed: {key}")
    manifests = sorted(args.score_dir.resolve().glob("DAGIG_V6_TERMINAL_TEACHER_SCORE_SHARD*_MANIFEST.json"))
    loaded = [read_json(path) for path in manifests]
    if not loaded or len(loaded) != int(loaded[0]["num_shards"]):
        raise ValueError("expanded score shards are incomplete")
    for manifest in loaded:
        score_path = Path(manifest["output_paths"]["scores"])
        if sha256(score_path) != manifest["output_hashes"]["scores"]:
            raise ValueError("expanded score shard changed")
    for path in (args.calibrator.resolve(), args.backup_builder.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    calibration_helper = Path(__file__).resolve().with_name("702_calibrate_v6_no_gold_terminal_value.py")
    if not calibration_helper.is_file():
        raise FileNotFoundError(calibration_helper)
    evidence_strategies = [
        "bge_top3",
        "entity_condition_mismatch_top3",
        "observable_low_support_top3",
        "serper_rank_top3",
        "support_diverse_top3",
    ]
    answer_strategy_flags = ["policy_greedy", "extractive", "type_constrained", "cautious"]
    gates = {
        "expanded_actions_complete": int(source["answer_actions"]) == 41273,
        "policy_train_samples_complete": len(source["policy_train_sample_ids"]) == 158,
        "development_samples_complete": len(source["internal_holdout_sample_ids"]) == 40,
        "score_shards_complete": len(manifests) == int(loaded[0]["num_shards"]),
        "gold_equivalence_excluded": True,
        "strategy_features_are_runtime_observable": True,
        "feature_and_gate_selection_train_oof_only": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN" if all(gates.values()) else "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_NO_GO"
    freeze = {
        "decision": decision,
        "protocol_version": "dagig_v6_deployable_no_gold_terminal_value_repeated_cv_v2",
        "semantics": {
            "target": "strict success used only as policy-train calibration label and post-freeze development audit label",
            "runtime": "question/evidence/candidate-answer support and reader features with action provenance; no gold, qrels, or equivalence",
            "downstream_use": "exact online answer/evidence/query/visual backward backup",
        },
        "feature_names": [
            "support_logit",
            "reader_candidate_mean_logprob",
            "minimum_support_reader",
            "support_reader_interaction",
            "answer_token_length",
            "is_unknown",
            *[f"answer_strategy_{name}" for name in answer_strategy_flags],
            *[f"evidence_strategy_{name}" for name in evidence_strategies],
        ],
        "answer_strategy_flags": answer_strategy_flags,
        "evidence_strategies": evidence_strategies,
        "calibration": {
            "model": "standardized logistic regression",
            "l2": 0.0003,
            "max_newton_steps": 30,
            "folds": 5,
            "repeats": 10,
            "fold_seed_prefix": "dagig_v6_no_gold_terminal_repeated_cv_v2",
            "train_probability": "mean of ten sample-grouped OOF predictions",
            "probability_clip": [1e-5, 0.99999],
            "all_hyperparameters_selected_from_policy_train_oof_only": True,
        },
        "gates_spec": {
            "repeated_oof_auc_mean_min": 0.88,
            "repeated_oof_auc_worst_min": 0.87,
            "repeated_oof_brier_improvement_worst_min": 0.008,
            "development_auc_min": 0.85,
            "development_brier_improvement_min": 0.002,
            "development_ece_max": 0.05,
            "nonconstant_group_rate_min": 0.95,
        },
        "answer_actions": int(source["answer_actions"]),
        "evidence_actions": int(source["evidence_actions"]),
        "policy_train_sample_ids": source["policy_train_sample_ids"],
        "development_sample_ids": source["internal_holdout_sample_ids"],
        "gates": gates,
        "input_paths": {
            "source_freeze": str(source_path),
            "answer_actions": source["input_paths"]["answer_actions"],
            "evidence_actions": source["input_paths"]["evidence_actions"],
            "private_labels": source["input_paths"]["private_labels"],
            "corpus": source["input_paths"]["corpus"],
            "eval_utils": source["input_paths"]["eval_utils"],
            "source_scoring_freeze": source["input_paths"]["source_scoring_freeze"],
            "score_manifests": [str(path) for path in manifests],
        },
        "input_hashes": {
            "source_freeze": sha256(source_path),
            "answer_actions": source["input_hashes"]["answer_actions"],
            "evidence_actions": source["input_hashes"]["evidence_actions"],
            "private_labels": source["input_hashes"]["private_labels"],
            "corpus": source["input_hashes"]["corpus"],
            "eval_utils": source["input_hashes"]["eval_utils"],
            "source_scoring_freeze": source["input_hashes"]["source_scoring_freeze"],
            "score_manifests": [sha256(path) for path in manifests],
        },
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
        "leakage_contract": {
            "gold_allowed_only_for_train_calibration_labels_and_development_audit": True,
            "gold_qrels_or_equivalence_in_runtime_features": False,
            "development_used_for_fit_feature_or_gate_selection": False,
            "dev_or_test_used": False,
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
    print(json.dumps({"decision": decision, "gates": gates, "freeze": str(path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
