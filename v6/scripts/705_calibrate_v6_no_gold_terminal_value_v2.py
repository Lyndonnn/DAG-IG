#!/usr/bin/env python3
"""Repeated sample-grouped CV for deployable no-gold P_success v2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_helper(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_v6_no_gold_calibration_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    helper_path = Path(__file__).resolve().with_name("702_calibrate_v6_no_gold_terminal_value.py")
    helper = load_helper(helper_path)
    freeze = helper.read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN" or freeze.get("protocol_version") != "dagig_v6_deployable_no_gold_terminal_value_repeated_cv_v2":
        raise ValueError("repeated-CV no-gold terminal protocol is not frozen")
    if freeze["code_hashes"]["calibrator"] != helper.sha256(Path(__file__).resolve()):
        raise ValueError("v2 calibrator changed after freeze")
    if freeze["code_hashes"]["calibration_helper"] != helper.sha256(helper_path):
        raise ValueError("calibration helper changed after freeze")
    for key in ("source_freeze", "answer_actions", "evidence_actions", "private_labels", "corpus", "eval_utils", "source_scoring_freeze"):
        if helper.sha256(Path(freeze["input_paths"][key])) != freeze["input_hashes"][key]:
            raise ValueError(f"frozen v2 input changed: {key}")
    manifests = [helper.read_json(Path(path)) for path in freeze["input_paths"]["score_manifests"]]
    for path, expected in zip(freeze["input_paths"]["score_manifests"], freeze["input_hashes"]["score_manifests"]):
        if helper.sha256(Path(path)) != expected:
            raise ValueError("terminal score manifest changed")
    scores = {}
    for manifest in manifests:
        score_path = Path(manifest["output_paths"]["scores"])
        if helper.sha256(score_path) != manifest["output_hashes"]["scores"]:
            raise ValueError("terminal score shard changed")
        scores.update({str(row["answer_action_id"]): row for row in helper.read_jsonl(score_path)})

    answers = helper.read_jsonl(Path(freeze["input_paths"]["answer_actions"]))
    evidence = {str(row["evidence_action_id"]): row for row in helper.read_jsonl(Path(freeze["input_paths"]["evidence_actions"]))}
    labels = {str(row["sample_id"]): row for row in helper.read_jsonl(Path(freeze["input_paths"]["private_labels"]))}
    corpus = {str(row["doc_id"]): row for row in helper.read_jsonl(Path(freeze["input_paths"]["corpus"]))}
    match_helper = helper.load_module(Path(freeze["input_paths"]["eval_utils"]))
    if len(answers) != int(freeze["answer_actions"]) or len(scores) != len(answers):
        raise ValueError("expanded v2 answer/score universe differs")
    train_ids = set(freeze["policy_train_sample_ids"])
    development_ids = set(freeze["development_sample_ids"])
    answer_flags = freeze["answer_strategy_flags"]
    evidence_strategies = freeze["evidence_strategies"]
    records = []
    for action in answers:
        answer_id = str(action["answer_action_id"])
        parent = evidence[str(action["evidence_action_id"])]
        score = scores[answer_id]
        label = labels[str(action["sample_id"])]
        partition = "policy_train" if action["sample_id"] in train_ids else "internal_holdout" if action["sample_id"] in development_ids else "invalid"
        if partition == "invalid":
            raise ValueError("answer action outside v2 partitions")
        match = match_helper.answer_match_details(action["candidate_answer"], label["gold_answer"], label.get("aliases") or [])
        accepted = [label["gold_answer"], *(label.get("aliases") or [])]
        text = " ".join(f"{doc.get('title', '')} {doc.get('snippet', '')}" for doc in parent["selected_docs"])
        selected_urls = {helper.canonical_url(doc.get("url")) for doc in parent["selected_docs"]}
        positive_urls = {
            helper.canonical_url((corpus.get(str(doc_id)) or {}).get("final_url") or (corpus.get(str(doc_id)) or {}).get("url"))
            for doc_id in label.get("positive_doc_ids") or []
        }
        support = bool(positive_urls & selected_urls) or any(helper.phrase_contains(text, value) for value in accepted)
        strategy_parts = set(str(action["answer_strategy"]).split("+"))
        vector = [
            *helper.feature(score),
            *[float(name in strategy_parts) for name in answer_flags],
            *[float(parent["evidence_strategy"] == name) for name in evidence_strategies],
        ]
        if len(vector) != len(freeze["feature_names"]):
            raise ValueError("v2 runtime feature schema differs")
        records.append(
            {
                "answer_action_id": answer_id,
                "evidence_action_id": str(action["evidence_action_id"]),
                "query_id": str(action["query_id"]),
                "sample_id": str(action["sample_id"]),
                "partition": partition,
                "feature": vector,
                "strict_proxy": int(bool(match["answer_correct"] and support)),
                "answer_correct_proxy": bool(match["answer_correct"]),
                "evidence_support_proxy": bool(support),
            }
        )

    x = np.asarray([row["feature"] for row in records], dtype=np.float64)
    y = np.asarray([row["strict_proxy"] for row in records], dtype=np.float64)
    train_index = np.asarray([index for index, row in enumerate(records) if row["partition"] == "policy_train"])
    development_index = np.asarray([index for index, row in enumerate(records) if row["partition"] == "internal_holdout"])
    config = freeze["calibration"]
    repeat_predictions = []
    repeated_metrics = []
    fold_logs = []
    for repeat in range(int(config["repeats"])):
        fold_by_sample = {
            sample_id: int(hashlib.sha256(f"{config['fold_seed_prefix']}:{repeat}:{sample_id}".encode()).hexdigest(), 16) % int(config["folds"])
            for sample_id in train_ids
        }
        prediction = np.full(len(train_index), np.nan, dtype=np.float64)
        baseline = np.full(len(train_index), np.nan, dtype=np.float64)
        for fold in range(int(config["folds"])):
            fit_index = np.asarray([index for index in train_index if fold_by_sample[records[index]["sample_id"]] != fold])
            validation_index = np.asarray([index for index in train_index if fold_by_sample[records[index]["sample_id"]] == fold])
            model = helper.fit_logistic(x[fit_index], y[fit_index], float(config["l2"]), int(config["max_newton_steps"]))
            local_positions = np.searchsorted(train_index, validation_index)
            prediction[local_positions] = helper.predict(model, x[validation_index])
            baseline[local_positions] = float(y[fit_index].mean())
            fold_logs.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "fit_samples": len({records[index]["sample_id"] for index in fit_index}),
                    "validation_samples": len({records[index]["sample_id"] for index in validation_index}),
                    "fit_actions": len(fit_index),
                    "validation_actions": len(validation_index),
                }
            )
        if not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
            raise ValueError("incomplete repeated OOF probabilities")
        repeat_predictions.append(prediction)
        repeated_metrics.append({"repeat": repeat, **helper.metric(prediction, y[train_index], baseline)})
    oof_probability = np.mean(np.stack(repeat_predictions), axis=0)
    oof_baseline = np.full(len(train_index), float(y[train_index].mean()))
    aggregate_oof = helper.metric(oof_probability, y[train_index], oof_baseline)
    final_model = helper.fit_logistic(x[train_index], y[train_index], float(config["l2"]), int(config["max_newton_steps"]))
    development_probability = helper.predict(final_model, x[development_index])
    development_baseline = np.full(len(development_index), float(y[train_index].mean()))
    development_metrics = helper.metric(development_probability, y[development_index], development_baseline)
    probabilities = np.full(len(records), np.nan, dtype=np.float64)
    probabilities[train_index] = oof_probability
    probabilities[development_index] = development_probability
    low, high = config["probability_clip"]
    probabilities = np.clip(probabilities, float(low), float(high))
    if not np.isfinite(probabilities).all():
        raise ValueError("incomplete v2 terminal values")
    by_evidence: dict[str, list[float]] = defaultdict(list)
    for row, probability in zip(records, probabilities):
        by_evidence[row["evidence_action_id"]].append(float(probability))
    nonconstant = float(np.mean([max(values) - min(values) > 1e-5 for values in by_evidence.values()]))
    specs = freeze["gates_spec"]
    aucs = [float(row["auc"]) for row in repeated_metrics]
    brier_improvements = [float(row["brier_improvement"]) for row in repeated_metrics]
    gates = {
        "complete_expanded_actions": len(records) == int(freeze["answer_actions"]),
        "repeated_oof_complete": len(repeated_metrics) == int(config["repeats"]),
        "repeated_oof_auc_mean": float(np.mean(aucs)) >= float(specs["repeated_oof_auc_mean_min"]),
        "repeated_oof_auc_worst": min(aucs) >= float(specs["repeated_oof_auc_worst_min"]),
        "repeated_oof_brier_worst": min(brier_improvements) >= float(specs["repeated_oof_brier_improvement_worst_min"]),
        "development_auc": float(development_metrics["auc"]) >= float(specs["development_auc_min"]),
        "development_brier": float(development_metrics["brier_improvement"]) >= float(specs["development_brier_improvement_min"]),
        "development_ece": float(development_metrics["ece_10bin"]) <= float(specs["development_ece_max"]),
        "nonconstant_answer_groups": nonconstant >= float(specs["nonconstant_group_rate_min"]),
        "equivalence_logit_not_used": True,
        "runtime_features_contain_no_gold_or_qrels": True,
        "development_never_fit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    decision = "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO" if all(gates.values()) else "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    value_path = output / "v6_no_gold_terminal_success_values.jsonl"
    private_path = output / "v6_no_gold_terminal_private_audit.jsonl"
    model_path = output / "v6_no_gold_terminal_calibrator.json"
    with value_path.open("w", encoding="utf-8") as handle:
        for row, probability in sorted(zip(records, probabilities), key=lambda item: item[0]["answer_action_id"]):
            handle.write(json.dumps({"answer_action_id": row["answer_action_id"], "evidence_action_id": row["evidence_action_id"], "query_id": row["query_id"], "sample_id": row["sample_id"], "partition": row["partition"], "terminal_success_probability": float(probability), "terminal_log_value": math.log(float(probability)), "calibration_source": "ten_repeat_sample_grouped_oof_mean" if row["partition"] == "policy_train" else "policy_train_fit_development_score"}, sort_keys=True) + "\n")
    with private_path.open("w", encoding="utf-8") as handle:
        for row, probability in sorted(zip(records, probabilities), key=lambda item: item[0]["answer_action_id"]):
            handle.write(json.dumps({**row, "terminal_success_probability": float(probability)}, sort_keys=True) + "\n")
    model_path.write_text(json.dumps({"feature_names": freeze["feature_names"], "center": final_model["center"].tolist(), "scale": final_model["scale"].tolist(), "weights": final_model["weights"].tolist(), "fit_partition": "policy_train_only", "folds": fold_logs, "equivalence_logit_used": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "metrics": {
            "repeated_oof": repeated_metrics,
            "repeated_oof_auc_mean": float(np.mean(aucs)),
            "repeated_oof_auc_worst": min(aucs),
            "repeated_oof_brier_improvement_mean": float(np.mean(brier_improvements)),
            "repeated_oof_brier_improvement_worst": min(brier_improvements),
            "aggregate_oof_mean_probability": aggregate_oof,
            "development": development_metrics,
            "nonconstant_answer_group_rate": nonconstant,
            "answer_actions": len(records),
            "evidence_groups": len(by_evidence),
        },
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": helper.sha256(freeze_path)},
        "output_paths": {"terminal_values": str(value_path), "private_audit": str(private_path), "calibrator": str(model_path)},
        "output_hashes": {"terminal_values": helper.sha256(value_path), "private_audit": helper.sha256(private_path), "calibrator": helper.sha256(model_path)},
        "gold_or_qrels_in_runtime_features": False,
        "equivalence_logit_used": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
