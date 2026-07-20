#!/usr/bin/env python3
"""Persist and freeze one shared downstream answer policy for DAG-IG v6.

All evidence/query/visual control methods must use this exact reader so that an
upstream comparison changes only the node credit being tested.  The original
internal NO-GO and post-hoc normalization sensitivity are both retained.
"""

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
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_freeze", type=Path, required=True)
    parser.add_argument("--train_fit", type=Path, required=True)
    parser.add_argument("--primary_internal_audit", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amended_sensitivity", type=Path, required=True)
    parser.add_argument("--source_train_audit", type=Path, required=True)
    parser.add_argument("--source_adapter", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "answer_freeze": args.answer_freeze.resolve(),
        "train_fit": args.train_fit.resolve(),
        "primary_internal_audit": args.primary_internal_audit.resolve(),
        "amendment": args.amendment.resolve(),
        "amended_sensitivity": args.amended_sensitivity.resolve(),
        "source_train_audit": args.source_train_audit.resolve(),
    }
    documents = {key: read_json(path) for key, path in paths.items()}
    source_adapter = args.source_adapter.resolve()
    train_audit = documents["source_train_audit"]
    gates = {
        "answer_controls_frozen": documents["answer_freeze"].get("decision") == "DAGIG_V6_NO_GOLD_ANSWER_CONTROLS_FROZEN",
        "all_methods_train_fit_go": documents["train_fit"].get("decision") == "DAGIG_V6_NO_GOLD_ANSWER_TRAIN_FIT_GO",
        "dagig_policy_training_ready": train_audit.get("decision") == "DAGIG_V6_NO_GOLD_ANSWER_POLICY_READY" and train_audit.get("method") == "dagig",
        "primary_internal_no_go_preserved": documents["primary_internal_audit"].get("decision") == "DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_NO_GO",
        "normalization_amendment_frozen": documents["amendment"].get("decision") == "DAGIG_V6_ANSWER_NORMALIZATION_AMENDMENT_FROZEN",
        "amended_sensitivity_go": documents["amended_sensitivity"].get("decision") == "DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_AMENDED_SENSITIVITY_GO",
        "amended_sensitivity_not_mislabeled_pristine": documents["amended_sensitivity"].get("eligible_as_pristine_holdout_claim") is False,
        "source_adapter_exists": (source_adapter / "adapter_model.safetensors").is_file(),
        "source_adapter_hash_matches_training_audit": (
            (source_adapter / "adapter_model.safetensors").is_file()
            and sha256(source_adapter / "adapter_model.safetensors") == train_audit["output_hashes"]["adapter_model"]
        ),
        "no_dev_used": not any(document.get("dev_used", False) for document in documents.values()),
        "no_test_used": not any(document.get("test_used", False) for document in documents.values()),
    }
    if not all(gates.values()):
        raise ValueError(f"shared answer freeze prerequisites failed: {gates}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    persistent_adapter = output_dir / "adapter"
    shutil.copytree(source_adapter, persistent_adapter)
    training_log_source = Path(train_audit["output_paths"]["training_log"])
    persistent_training_log = output_dir / "training_log.jsonl"
    shutil.copy2(training_log_source, persistent_training_log)
    source_train_audit_copy = output_dir / "SOURCE_DAGIG_ANSWER_POLICY_TRAIN_AUDIT.json"
    shutil.copy2(paths["source_train_audit"], source_train_audit_copy)

    gates["persistent_adapter_hash_matches_source"] = tree_sha256(persistent_adapter) == tree_sha256(source_adapter)
    gates["persistent_training_log_hash_matches_source"] = sha256(persistent_training_log) == sha256(training_log_source)
    decision = "DAGIG_V6_SHARED_ANSWER_POLICY_FROZEN" if all(gates.values()) else "DAGIG_V6_SHARED_ANSWER_POLICY_FREEZE_FAILED"
    input_paths = {key: str(path) for key, path in paths.items()}
    input_paths["source_adapter_model"] = str(source_adapter / "adapter_model.safetensors")
    input_paths["source_training_log"] = str(training_log_source)
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_induction_shared_answer_v1",
        "selected_method": "dagig",
        "selection_contract": (
            "Use the DAG-IG answer policy as one shared fixed reader for every upstream "
            "evidence/query/visual method. The answer-node primary internal NO-GO is retained; "
            "the amended result is disclosed as post-hoc sensitivity, and final validity is "
            "deferred to sealed dev/test."
        ),
        "downstream_control_contract": (
            "No upstream method may update, replace, or conditionally switch this answer policy."
        ),
        "gates": gates,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {
            "adapter": str(persistent_adapter),
            "training_log": str(persistent_training_log),
            "source_train_audit_copy": str(source_train_audit_copy),
        },
        "output_hashes": {
            "adapter_tree": tree_sha256(persistent_adapter),
            "adapter_model": sha256(persistent_adapter / "adapter_model.safetensors"),
            "training_log": sha256(persistent_training_log),
            "source_train_audit_copy": sha256(source_train_audit_copy),
        },
        "primary_internal_claim": documents["primary_internal_audit"]["decision"],
        "amended_sensitivity_claim": documents["amended_sensitivity"]["decision"],
        "internal_holdout_previously_observed": True,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    manifest = output_dir / "DAGIG_V6_SHARED_ANSWER_POLICY_FREEZE.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
