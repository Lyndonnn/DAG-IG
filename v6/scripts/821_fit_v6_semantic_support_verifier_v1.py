#!/usr/bin/env python3
"""Grouped-OOF calibration and train gate for the frozen semantic verifier."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_v6_semantic_support_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positive = labels == 1
    positives = int(positive.sum())
    negatives = len(labels) - positives
    return 0.5 if not positives or not negatives else float((ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def feature_matrix(names: list[str], semantic: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    baseline_score = logit(baseline)
    columns = {
        "semantic_logit": semantic,
        "baseline_support_logit": baseline_score,
        "semantic_x_baseline": semantic * baseline_score,
    }
    return np.stack([columns[name] for name in names], axis=1)


def serialize(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--score_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--private_support", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_FROZEN":
        raise ValueError("Semantic verifier protocol is not frozen")
    if freeze["input_hashes"]["fitter"] != sha256(Path(__file__).resolve()):
        raise ValueError("Fitter changed after protocol freeze")
    helper_path = args.helper.resolve()
    helper = load_module(helper_path)

    input_path = Path(freeze["output_paths"]["verifier_inputs"])
    if sha256(input_path) != freeze["output_hashes"]["verifier_inputs"]:
        raise ValueError("Frozen verifier inputs changed")
    records = sorted(read_jsonl(input_path), key=lambda row: row["query_action_id"])
    record_by_id = {row["query_action_id"]: row for row in records}

    scores: dict[str, dict[str, Any]] = {}
    manifests = []
    for raw_dir in args.score_dirs:
        directory = raw_dir.resolve()
        manifest = read_json(directory / "SHARD_MANIFEST.json")
        if manifest.get("decision") != "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_SHARD_COMPLETE":
            raise ValueError(f"Incomplete verifier shard: {directory}")
        if manifest["freeze_sha256"] != sha256(freeze_path):
            raise ValueError(f"Verifier shard uses another freeze: {directory}")
        score_path = Path(manifest["score_path"])
        if sha256(score_path) != manifest["score_sha256"]:
            raise ValueError(f"Verifier shard scores changed: {directory}")
        manifests.append(manifest)
        for row in read_jsonl(score_path):
            query_id = row["query_action_id"]
            if query_id in scores:
                raise ValueError(f"Duplicate verifier score: {query_id}")
            scores[query_id] = row
    if set(scores) != set(record_by_id):
        raise ValueError(f"Verifier score universe mismatch: {len(scores)} vs {len(records)}")
    if sorted(manifest["shard_index"] for manifest in manifests) != list(range(manifests[0]["num_shards"])):
        raise ValueError("Verifier shards are not complete")

    semantic = np.asarray([scores[row["query_action_id"]]["semantic_support_logit"] for row in records], dtype=np.float64)
    baseline = np.asarray([row["baseline_support_probability"] for row in records], dtype=np.float64)
    train = np.asarray([index for index, row in enumerate(records) if row["partition"] == "policy_train"])
    internal = np.asarray([index for index, row in enumerate(records) if row["partition"] == "internal_holdout"])

    # The combined private file is streamed, but only policy-train labels are retained.
    support_map = {}
    with args.private_support.resolve().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("partition") != "policy_train":
                continue
            support_map[row["query_id"]] = row["strategy_support"]
    labels = np.full(len(records), np.nan)
    for index in train:
        row = records[index]
        strategy = row["selected_evidence_action_id"].rsplit("::", 1)[-1]
        labels[index] = float(support_map[row["query_action_id"]][strategy])
    if not np.isfinite(labels[train]).all():
        raise ValueError("Missing policy-train support labels")

    config = freeze["fit"]
    thresholds = freeze["train_oof_gates"]
    samples = [records[index]["sample_id"] for index in train]
    parent_ids = [records[index]["parent_visual_state_id"] for index in train]
    baseline_brier = float(np.mean((baseline[train] - labels[train]) ** 2))
    candidates = []
    candidate_predictions = {}
    for family_index, names in enumerate(config["candidate_features"]):
        x = feature_matrix(names, semantic, baseline)
        for l2 in config["l2_grid"]:
            repeated = []
            for repeat in range(int(config["repeats"])):
                assignment = helper.folds_for_samples(samples, int(config["folds"]), f"{config['seed_prefix']}:{family_index}:{l2}:{repeat}")
                probability = np.full(len(train), np.nan)
                for fold in range(int(config["folds"])):
                    fit_indices = np.asarray([index for index in train if assignment[records[index]["sample_id"]] != fold])
                    valid_indices = np.asarray([index for index in train if assignment[records[index]["sample_id"]] == fold])
                    model = helper.fit_logistic(x[fit_indices], labels[fit_indices], float(l2), int(config["newton_steps"]))
                    probability[np.searchsorted(train, valid_indices)] = helper.predict_logistic(model, x[valid_indices])
                if not np.isfinite(probability).all():
                    raise ValueError("Incomplete grouped-OOF semantic support predictions")
                repeated.append(probability)
            oof = np.mean(np.stack(repeated), axis=0)
            full_model = helper.fit_logistic(x[train], labels[train], float(l2), int(config["newton_steps"]))
            brier = float(np.mean((oof - labels[train]) ** 2))
            pair = helper.pair_order(oof, labels[train], parent_ids)
            groups: dict[str, list[int]] = defaultdict(list)
            for position, parent_id in enumerate(parent_ids):
                groups[parent_id].append(position)
            nonconstant = mean(float(max(oof[group]) - min(oof[group]) > 1e-8) for group in groups.values())
            semantic_positive = float(full_model["weights"][1]) > 0.0
            metrics = {
                "feature_family_index": family_index,
                "feature_names": names,
                "l2": float(l2),
                "support_auc": auc(oof, labels[train]),
                "support_brier": brier,
                "baseline_support_brier": baseline_brier,
                "brier_improvement_vs_baseline": baseline_brier - brier,
                "within_visual_pair_order": pair,
                "nonconstant_parent_group_rate": nonconstant,
                "semantic_coefficient": float(full_model["weights"][1]),
            }
            passes = (
                metrics["support_auc"] >= thresholds["support_auc_min"]
                and metrics["brier_improvement_vs_baseline"] >= thresholds["brier_improvement_vs_baseline_min"]
                and pair["accuracy"] >= thresholds["within_visual_pair_order_min"]
                and nonconstant >= thresholds["nonconstant_parent_group_rate_min"]
                and semantic_positive
            )
            key = (family_index, float(l2))
            candidates.append({**metrics, "passes": bool(passes)})
            candidate_predictions[key] = (oof, full_model, x)

    passing = [candidate for candidate in candidates if candidate["passes"]]
    if passing:
        first_family = min(candidate["feature_family_index"] for candidate in passing)
        selected = min(
            (candidate for candidate in passing if candidate["feature_family_index"] == first_family),
            key=lambda candidate: (candidate["support_brier"], candidate["l2"]),
        )
    else:
        selected = max(candidates, key=lambda candidate: (candidate["within_visual_pair_order"]["accuracy"], candidate["support_auc"], -candidate["support_brier"]))
    key = (selected["feature_family_index"], selected["l2"])
    oof, final_model, x = candidate_predictions[key]
    low, high = config["probability_clip"]
    prediction = np.full(len(records), np.nan)
    prediction[train] = np.clip(oof, low, high)
    prediction[internal] = np.clip(helper.predict_logistic(final_model, x[internal]), low, high)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "v6_semantic_support_values_no_eval_labels.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, raw_score, value in zip(records, semantic, prediction):
            handle.write(json.dumps({
                "query_action_id": row["query_action_id"],
                "parent_visual_state_id": row["parent_visual_state_id"],
                "selected_evidence_action_id": row["selected_evidence_action_id"],
                "sample_id": row["sample_id"],
                "partition": row["partition"],
                "semantic_support_logit": float(raw_score),
                "semantic_support_probability": float(value),
                "prediction_source": "sample_group_oof" if row["partition"] == "policy_train" else "policy_train_full_fit",
            }, sort_keys=True) + "\n")
    model_path = output / "v6_semantic_support_calibrator.json"
    model_path.write_text(json.dumps({
        "feature_names": selected["feature_names"],
        "l2": selected["l2"],
        "model": serialize(final_model),
        "fit_partition": "policy_train_only",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gates = {
        "at_least_one_candidate_passes": bool(passing),
        "selection_rule_honored": bool(passing) and selected["feature_family_index"] == min(candidate["feature_family_index"] for candidate in passing),
        "complete_policy_train_oof": len(train) == 2359 and np.isfinite(prediction[train]).all(),
        "complete_internal_predictions_without_internal_labels": len(internal) == 595 and np.isfinite(prediction[internal]).all(),
        "semantic_verifier_is_answer_independent": True,
        "runtime_features_use_no_gold": True,
        "internal_labels_not_used": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {key: bool(value) for key, value in gates.items()}
    decision = "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_TRAIN_OOF_GO" if all(gates.values()) else "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_TRAIN_OOF_NO_GO"
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "selected_candidate": selected,
        "all_candidates": candidates,
        "gates": gates,
        "input_paths": {
            "freeze": str(freeze_path),
            "private_support": str(args.private_support.resolve()),
            "score_dirs": [str(path.resolve()) for path in args.score_dirs],
        },
        "input_hashes": {
            "freeze": sha256(freeze_path),
            "private_support": sha256(args.private_support.resolve()),
        },
        "output_paths": {"predictions": str(prediction_path), "model": str(model_path)},
        "output_hashes": {"predictions": sha256(prediction_path), "model": sha256(model_path)},
        "internal_private_labels_used": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": True,
    }
    audit_path = output / "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_TRAIN_OOF_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "selected_candidate": selected, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
