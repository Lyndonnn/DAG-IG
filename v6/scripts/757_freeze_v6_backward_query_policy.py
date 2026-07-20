#!/usr/bin/env python3
"""Persist the backward DAG-IG query policy after train and fresh-search GO."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(child for child in root.rglob("*") if child.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def require(payload: dict[str, Any], decision: str, label: str) -> None:
    if payload.get("decision") != decision:
        raise ValueError(f"{label} is not {decision}: {payload.get('decision')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--train_fit", type=Path, required=True)
    parser.add_argument("--fresh_search_freeze", type=Path, required=True)
    parser.add_argument("--fresh_internal_audit", type=Path, required=True)
    parser.add_argument("--dagig_train_audit", type=Path, required=True)
    parser.add_argument("--dagig_train_scores", type=Path, required=True)
    parser.add_argument("--dagig_internal_scores", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "training_freeze": args.training_freeze.resolve(),
        "train_fit": args.train_fit.resolve(),
        "fresh_search_freeze": args.fresh_search_freeze.resolve(),
        "fresh_internal_audit": args.fresh_internal_audit.resolve(),
        "dagig_train_audit": args.dagig_train_audit.resolve(),
        "dagig_train_scores": args.dagig_train_scores.resolve(),
        "dagig_internal_scores": args.dagig_internal_scores.resolve(),
        "freeze_runner": Path(__file__).resolve(),
    }
    payloads = {key: read_json(path) for key, path in paths.items() if key != "freeze_runner"}
    training = payloads["training_freeze"]
    fit = payloads["train_fit"]
    search = payloads["fresh_search_freeze"]
    internal = payloads["fresh_internal_audit"]
    train_audit = payloads["dagig_train_audit"]
    train_scores = payloads["dagig_train_scores"]
    internal_scores = payloads["dagig_internal_scores"]
    require(training, "DAGIG_V6_BACKWARD_QUERY_TRAINING_FROZEN", "query training protocol")
    require(fit, "DAGIG_V6_BACKWARD_QUERY_TRAIN_FIT_GO", "query train-fit audit")
    require(search, "DAGIG_V6_SERPER_SEARCH_PLAN_FROZEN", "fresh-search plan")
    require(internal, "DAGIG_V6_BACKWARD_QUERY_INTERNAL_GO", "fresh query internal audit")
    require(train_audit, "DAGIG_V6_BACKWARD_QUERY_POLICY_READY", "DAG-IG query train audit")
    require(train_scores, "DAGIG_V6_BACKWARD_QUERY_POLICY_SCORES_READY", "DAG-IG query train scores")
    require(internal_scores, "DAGIG_V6_BACKWARD_QUERY_POLICY_SCORES_READY", "DAG-IG query internal scores")
    if search.get("protocol_version") != "dagig_v6_backward_query_all_visual_states_internal_fresh_search_v1":
        raise ValueError("fresh-search protocol mismatch")
    if search.get("reserved_final_eval_key_used"):
        raise ValueError("reserved final evaluation key was used during internal selection")
    if train_audit.get("method") != "dagig":
        raise ValueError("main query train audit is not DAG-IG")
    if train_scores.get("method") != "dagig" or train_scores.get("partition") != "policy_train":
        raise ValueError("DAG-IG query train score identity mismatch")
    if internal_scores.get("method") != "dagig" or internal_scores.get("partition") != "internal_holdout":
        raise ValueError("DAG-IG query internal score identity mismatch")

    training_hash = sha256(paths["training_freeze"])
    if train_audit["input_hashes"].get("freeze") != training_hash:
        raise ValueError("DAG-IG query adapter belongs to another training protocol")
    if fit["input_hashes"].get("training_freeze") != training_hash:
        raise ValueError("query train-fit belongs to another training protocol")
    if search["input_hashes"].get("training_freeze") != training_hash:
        raise ValueError("fresh-search selection belongs to another query protocol")
    if search["input_hashes"].get("dagig_score_audit") != sha256(paths["dagig_internal_scores"]):
        raise ValueError("fresh-search DAG-IG selections belong to another internal score audit")
    internal_action_freeze_path = Path(internal["input_paths"]["fresh_action_freeze"])
    if sha256(internal_action_freeze_path) != internal["input_hashes"]["fresh_action_freeze"]:
        raise ValueError("fresh internal action freeze changed after private audit")
    internal_action_freeze = read_json(internal_action_freeze_path)
    if internal_action_freeze["input_hashes"].get("search_freeze") != sha256(paths["fresh_search_freeze"]):
        raise ValueError("fresh internal audit descends from another search plan")
    if internal_action_freeze["input_hashes"].get("evidence_policy_freeze") is None:
        raise ValueError("fresh internal audit is missing frozen evidence-policy lineage")
    if train_scores["input_hashes"].get("train_audit") != sha256(paths["dagig_train_audit"]):
        raise ValueError("DAG-IG train scores belong to another adapter")
    if internal_scores["input_hashes"].get("train_audit") != sha256(paths["dagig_train_audit"]):
        raise ValueError("DAG-IG internal scores belong to another adapter")

    source_adapter = Path(train_audit["output_paths"]["adapter"])
    if sha256(source_adapter / "adapter_model.safetensors") != train_audit["output_hashes"]["adapter_model"]:
        raise ValueError("DAG-IG query source adapter changed")
    output = args.output_dir.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite query-policy freeze: {output}")
    temporary.mkdir(parents=True)
    frozen_adapter_tmp = temporary / "adapter"
    shutil.copytree(source_adapter, frozen_adapter_tmp)
    if tree_sha256(frozen_adapter_tmp) != tree_sha256(source_adapter):
        raise ValueError("persisted query adapter differs from source")

    final_adapter = output / "adapter"
    gates = {
        "query_training_protocol_frozen": True,
        "query_train_fit_go": True,
        "fresh_search_internal_go": True,
        "dagig_adapter_hash_valid": True,
        "dagig_train_and_internal_scores_valid": True,
        "frozen_evidence_and_answer_descendants_used": bool(internal.get("gates", {}).get("same_frozen_evidence_and_answer_descendants")),
        "reserved_final_eval_key_unused": not bool(internal.get("reserved_final_eval_key_used")),
        "internal_holdout_unused_for_training_or_tuning": not bool(internal.get("internal_holdout_used_for_training_or_tuning")),
        "dev_sealed": not bool(internal.get("dev_used")),
        "test_sealed": not bool(internal.get("test_used")),
    }
    decision = "DAGIG_V6_BACKWARD_QUERY_POLICY_FROZEN" if all(gates.values()) else "DAGIG_V6_BACKWARD_QUERY_POLICY_NO_GO"
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_frozen_evidence_answer_query_policy_v1",
        "method": "dagig",
        "method_contract": (
            "pi_Q(q|v) projected from the exact posterior proportional to pi_b(q|v) * "
            "sum_e pi_E(e|q) sum_a pi_A(a|e) P_success(a,e)"
        ),
        "runtime_interface": {
            "candidate_universe": "three-to-five pre-frozen structured query actions per visual state",
            "optimized_field": "search_query",
            "selection": "argmax reference-corrected listwise policy",
            "search": "execute only the selected query",
        },
        "gates": gates,
        "train_fit_metrics": fit.get("metrics"),
        "fresh_internal_metrics": internal.get("metrics"),
        "input_paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "source_adapter_tree_sha256": tree_sha256(source_adapter),
        "output_paths": {"adapter": str(final_adapter)},
        "output_hashes": {
            "adapter_tree": tree_sha256(frozen_adapter_tmp),
            "adapter_model": sha256(frozen_adapter_tmp / "adapter_model.safetensors"),
        },
        "gold_or_qrels_in_runtime_policy": False,
        "internal_holdout_used_for_training_or_tuning": False,
        "reserved_final_eval_key_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    manifest_tmp = temporary / "DAGIG_V6_BACKWARD_QUERY_POLICY_FREEZE.json"
    manifest_tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.rename(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
