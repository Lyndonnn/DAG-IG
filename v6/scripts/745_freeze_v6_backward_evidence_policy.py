#!/usr/bin/env python3
"""Persist the DAG-IG evidence policy only after every frozen evidence gate passes."""

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


def require_decision(payload: dict[str, Any], expected: str, label: str) -> None:
    if payload.get("decision") != expected:
        raise ValueError(f"{label} is not {expected}: {payload.get('decision')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--train_fit", type=Path, required=True)
    parser.add_argument("--internal_audit", type=Path, required=True)
    parser.add_argument("--dagig_train_audit", type=Path, required=True)
    parser.add_argument("--dagig_train_scores", type=Path, required=True)
    parser.add_argument("--dagig_internal_scores", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "training_freeze": args.training_freeze.resolve(),
        "train_fit": args.train_fit.resolve(),
        "internal_audit": args.internal_audit.resolve(),
        "dagig_train_audit": args.dagig_train_audit.resolve(),
        "dagig_train_scores": args.dagig_train_scores.resolve(),
        "dagig_internal_scores": args.dagig_internal_scores.resolve(),
        "freeze_runner": Path(__file__).resolve(),
    }
    payloads = {key: read_json(path) for key, path in paths.items() if key != "freeze_runner"}
    training = payloads["training_freeze"]
    fit = payloads["train_fit"]
    internal = payloads["internal_audit"]
    train_audit = payloads["dagig_train_audit"]
    train_scores = payloads["dagig_train_scores"]
    internal_scores = payloads["dagig_internal_scores"]

    require_decision(training, "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN", "training protocol")
    require_decision(fit, "DAGIG_V6_BACKWARD_EVIDENCE_TRAIN_FIT_GO", "train-fit audit")
    require_decision(internal, "DAGIG_V6_BACKWARD_EVIDENCE_INTERNAL_GO", "internal audit")
    require_decision(train_audit, "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_READY", "DAG-IG train audit")
    require_decision(train_scores, "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORES_READY", "DAG-IG train scores")
    require_decision(internal_scores, "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORES_READY", "DAG-IG internal scores")
    if train_audit.get("method") != "dagig":
        raise ValueError("main train audit is not DAG-IG")
    if train_scores.get("method") != "dagig" or train_scores.get("partition") != "policy_train":
        raise ValueError("main policy-train score identity mismatch")
    if internal_scores.get("method") != "dagig" or internal_scores.get("partition") != "internal_holdout":
        raise ValueError("main internal score identity mismatch")

    training_hash = sha256(paths["training_freeze"])
    if train_audit["input_hashes"].get("freeze") != training_hash:
        raise ValueError("DAG-IG adapter was not trained under this frozen protocol")
    if fit["input_hashes"].get("training_freeze") != training_hash:
        raise ValueError("train-fit audit belongs to another training protocol")
    if internal["input_hashes"].get("train_fit") != sha256(paths["train_fit"]):
        raise ValueError("internal audit belongs to another train-fit audit")
    for label, audit in (("train", train_scores), ("internal", internal_scores)):
        if audit["input_hashes"].get("train_audit") != sha256(paths["dagig_train_audit"]):
            raise ValueError(f"DAG-IG {label} scores belong to another adapter")

    source_adapter = Path(train_audit["output_paths"]["adapter"]).resolve()
    source_model = source_adapter / "adapter_model.safetensors"
    if sha256(source_model) != train_audit["output_hashes"]["adapter_model"]:
        raise ValueError("DAG-IG source adapter changed")

    output = args.output_dir.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite evidence-policy freeze: {output}")
    temporary.mkdir(parents=True)
    frozen_adapter_tmp = temporary / "adapter"
    shutil.copytree(source_adapter, frozen_adapter_tmp)
    if tree_sha256(frozen_adapter_tmp) != tree_sha256(source_adapter):
        raise ValueError("persisted evidence adapter differs from source")

    final_adapter = output / "adapter"
    gates = {
        "training_protocol_frozen": True,
        "train_fit_go": True,
        "internal_holdout_go": True,
        "dagig_adapter_hash_valid": True,
        "dagig_train_and_internal_scores_valid": True,
        "shared_answer_policy_was_fixed": bool(
            internal.get("gates", {}).get("same_shared_answer_policy")
        ),
        "internal_holdout_unused_for_training": not bool(internal.get("internal_holdout_used_for_training")),
        "dev_sealed": not bool(internal.get("dev_used")),
        "test_sealed": not bool(internal.get("test_used")),
    }
    decision = (
        "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_FROZEN"
        if all(gates.values())
        else "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_NO_GO"
    )
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_frozen_downstream_evidence_policy_v1",
        "method": "dagig",
        "method_contract": (
            "pi_E(e|q) projected from the exact posterior proportional to "
            "pi_b(e|q) * sum_a pi_A(a|e) P_success(a,e)"
        ),
        "gates": gates,
        "train_fit_metrics": fit.get("metrics"),
        "internal_metrics": internal.get("metrics"),
        "input_paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "source_adapter_tree_sha256": tree_sha256(source_adapter),
        "output_paths": {"adapter": str(final_adapter)},
        "output_hashes": {
            "adapter_tree": tree_sha256(frozen_adapter_tmp),
            "adapter_model": sha256(frozen_adapter_tmp / "adapter_model.safetensors"),
        },
        "gold_or_qrels_in_runtime_policy": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    manifest_tmp = temporary / "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_FREEZE.json"
    manifest_tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.rename(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
