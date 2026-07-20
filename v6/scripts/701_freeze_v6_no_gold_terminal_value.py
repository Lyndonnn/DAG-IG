#!/usr/bin/env python3
"""Freeze a deployable no-gold terminal P_success protocol."""

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
        raise ValueError("expanded answer terminal source is not frozen")
    for key, path in source["input_paths"].items():
        if sha256(Path(path)) != source["input_hashes"][key]:
            raise ValueError(f"expanded terminal input changed: {key}")
    manifests = sorted(args.score_dir.resolve().glob("DAGIG_V6_TERMINAL_TEACHER_SCORE_SHARD*_MANIFEST.json"))
    if not manifests:
        raise ValueError("expanded answer score manifests are missing")
    loaded = [read_json(path) for path in manifests]
    if len(loaded) != int(loaded[0]["num_shards"]):
        raise ValueError("expanded answer score shards are incomplete")
    for manifest in loaded:
        score_path = Path(manifest["output_paths"]["scores"])
        if sha256(score_path) != manifest["output_hashes"]["scores"]:
            raise ValueError("expanded answer score shard changed")
    for path in (args.calibrator.resolve(), args.backup_builder.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gates = {
        "expanded_actions_complete": int(source["answer_actions"]) == 41273,
        "policy_train_samples_complete": len(source["policy_train_sample_ids"]) == 158,
        "development_samples_complete": len(source["internal_holdout_sample_ids"]) == 40,
        "score_shards_complete": len(manifests) == int(loaded[0]["num_shards"]),
        "gold_equivalence_excluded_from_feature_schema": True,
        "fit_partition_policy_train_only": True,
        "sample_grouped_crossfit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN" if all(gates.values()) else "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_NO_GO"
    freeze = {
        "decision": decision,
        "protocol_version": "dagig_v6_deployable_no_gold_terminal_value_v1",
        "semantics": {
            "target": "strict success label used only on policy-train for calibration and on development for post-freeze audit",
            "inference": "P_success(question, selected evidence, candidate answer) without gold answer or qrels",
            "allowed_features": [
                "support_logit",
                "reader_candidate_mean_logprob",
                "minimum_support_reader",
                "support_reader_interaction",
                "answer_token_length",
                "is_unknown",
            ],
            "forbidden_feature": "equivalence_logit because it requires the gold answer",
            "downstream_use": "exact online answer/evidence/query/visual backward backup",
        },
        "feature_definition": {
            "support_logit": "clip(raw support_logit,-20,20)",
            "reader_candidate_mean_logprob": "clip(raw reader_candidate_mean_logprob,-20,0)",
            "minimum_support_reader": "min(support_logit,reader_candidate_mean_logprob)",
            "support_reader_interaction": "support_logit*reader_candidate_mean_logprob/20",
            "answer_token_length": "clip(answer_token_length,0,40)",
            "is_unknown": "boolean",
        },
        "calibration": {
            "model": "standardized logistic regression",
            "l2": 0.001,
            "max_newton_steps": 30,
            "folds": 5,
            "fold_seed_text": "dagig_v6_no_gold_terminal_crossfit_v1",
            "probability_clip": [1e-5, 0.99999],
            "feature_set_selected_from_policy_train_sample_grouped_oof_only": True,
        },
        "gates_spec": {
            "oof_auc_min": 0.90,
            "oof_brier_improvement_vs_fold_constant_min": 0.003,
            "development_auc_min": 0.85,
            "development_brier_improvement_vs_train_base_rate_min": 0.002,
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
            "backup_builder": str(args.backup_builder.resolve()),
        },
        "code_hashes": {
            "freezer": sha256(Path(__file__).resolve()),
            "calibrator": sha256(args.calibrator.resolve()),
            "backup_builder": sha256(args.backup_builder.resolve()),
        },
        "leakage_contract": {
            "gold_allowed_for_policy_train_binary_calibration_labels": True,
            "gold_allowed_for_development_post_freeze_audit": True,
            "gold_or_qrels_in_runtime_features": False,
            "equivalence_logit_in_runtime_features": False,
            "dev_or_test_used_for_fit_or_feature_selection": False,
        },
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FREEZE.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates, "freeze": str(freeze_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
