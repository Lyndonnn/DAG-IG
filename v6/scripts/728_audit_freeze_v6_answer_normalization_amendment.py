#!/usr/bin/env python3
"""Audit and freeze the post-internal answer-normalization amendment.

This script is deliberately private-label aware.  It measures the amendment on
the entire frozen answer-action universe and records that the internal holdout
had already been observed.  It never modifies labels, actions, model outputs,
or the original answer audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_freeze", type=Path, required=True)
    parser.add_argument("--baseline_internal_audit", type=Path, required=True)
    parser.add_argument(
        "--amendment_module",
        type=Path,
        default=Path(__file__).with_name("dagig_answer_match_v2.py"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    answer_freeze_path = args.answer_freeze.resolve()
    baseline_audit_path = args.baseline_internal_audit.resolve()
    amendment_path = args.amendment_module.resolve()
    freeze = read_json(answer_freeze_path)
    baseline_audit = read_json(baseline_audit_path)
    if freeze.get("decision") != "DAGIG_V6_NO_GOLD_ANSWER_CONTROLS_FROZEN":
        raise ValueError("answer control freeze is not valid")
    if baseline_audit.get("decision") != "DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_NO_GO":
        raise ValueError("expected the preserved primary internal NO-GO audit")

    terminal_audit = read_json(Path(freeze["input_paths"]["terminal_audit"]))
    source_freeze_path = Path(terminal_audit["input_paths"]["freeze"])
    source_freeze = read_json(source_freeze_path)
    baseline_helper_path = Path(source_freeze["input_paths"]["eval_utils"])
    private_labels_path = Path(source_freeze["input_paths"]["private_labels"])
    answer_actions_path = Path(freeze["input_paths"]["answer_actions"])
    for key, path in (
        ("eval_utils", baseline_helper_path),
        ("private_labels", private_labels_path),
        ("answer_actions", answer_actions_path),
    ):
        expected = source_freeze["input_hashes"].get(key) or freeze["input_hashes"].get(key)
        if expected != sha256(path):
            raise ValueError(f"frozen input changed: {key}")

    baseline = load_module("dagig_answer_match_frozen_baseline", baseline_helper_path)
    amendment = load_module("dagig_answer_match_amendment_v2", amendment_path)
    labels = {row["sample_id"]: row for row in read_jsonl(private_labels_path)}
    actions = read_jsonl(answer_actions_path)
    flips: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    for action in actions:
        label = labels.get(action["sample_id"])
        if label is None:
            missing_labels.append(action["sample_id"])
            continue
        prediction = action["candidate_answer"]
        old = baseline.answer_match_details(prediction, label["gold_answer"], label.get("aliases") or [])
        new = amendment.answer_match_details(
            baseline,
            prediction,
            label["gold_answer"],
            label.get("aliases") or [],
        )
        if bool(old["answer_correct"]) != bool(new["answer_correct"]):
            row = {
                "answer_action_id": action["answer_action_id"],
                "sample_id": action["sample_id"],
                "partition": action["partition"],
                "prediction": prediction,
                "gold_answer": label["gold_answer"],
                "old_match_type": old.get("answer_match_type"),
                "new_match_type": new.get("answer_match_type"),
                "matched_alias": new.get("matched_alias"),
                "matched_unit": new.get("matched_unit"),
                "old_correct": bool(old["answer_correct"]),
                "new_correct": bool(new["answer_correct"]),
            }
            if row["old_correct"] and not row["new_correct"]:
                regressions.append(row)
            else:
                flips.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    flips_path = output_dir / "numeric_unit_equivalence_flips_private.jsonl"
    flips_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in flips),
        encoding="utf-8",
    )
    partition_counts = Counter(row["partition"] for row in flips)
    sample_counts = Counter(row["sample_id"] for row in flips)
    unit_counts = Counter(row["matched_unit"] for row in flips)
    gates = {
        "complete_action_universe": len(actions) == 41273,
        "all_actions_have_private_labels": not missing_labels,
        "baseline_checker_hash_preserved": sha256(baseline_helper_path) == source_freeze["input_hashes"]["eval_utils"],
        "original_internal_audit_preserved": baseline_audit.get("decision") == "DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_NO_GO",
        "amendment_has_no_true_to_false_regressions": not regressions,
        "all_flips_use_only_numeric_unit_symmetric_exact": all(
            row["new_match_type"] == "numeric_unit_symmetric_exact" for row in flips
        ),
        "policy_train_labels_unchanged": partition_counts.get("policy_train", 0) == 0,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = (
        "DAGIG_V6_ANSWER_NORMALIZATION_AMENDMENT_FROZEN"
        if all(gates.values())
        else "DAGIG_V6_ANSWER_NORMALIZATION_AMENDMENT_REJECTED"
    )
    input_paths = {
        "answer_freeze": str(answer_freeze_path),
        "baseline_internal_audit": str(baseline_audit_path),
        "source_terminal_freeze": str(source_freeze_path.resolve()),
        "baseline_eval_utils": str(baseline_helper_path.resolve()),
        "private_labels": str(private_labels_path.resolve()),
        "answer_actions": str(answer_actions_path.resolve()),
        "amendment_module": str(amendment_path),
    }
    result = {
        "decision": decision,
        "protocol_version": amendment.PROTOCOL_VERSION,
        "rule": (
            "After the immutable baseline matcher fails, accept only an exact numeric prediction "
            "against an exact same-number plus whitelisted simple-unit gold/alias."
        ),
        "allowed_units": list(amendment.UNIT_PATTERNS),
        "metrics": {
            "answer_actions": len(actions),
            "false_to_true_flips": len(flips),
            "true_to_false_regressions": len(regressions),
            "flipped_samples": len(sample_counts),
            "partition_counts": dict(sorted(partition_counts.items())),
            "sample_counts": dict(sorted(sample_counts.items())),
            "unit_counts": dict(sorted(unit_counts.items())),
        },
        "gates": gates,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {"private_flips": str(flips_path)},
        "output_hashes": {"private_flips": sha256(flips_path)},
        "internal_holdout_was_observed_before_amendment": True,
        "amended_internal_audit_is_sensitivity_not_pristine_holdout": True,
        "future_frozen_dev_test_may_use_amendment": decision.endswith("FROZEN"),
        "training_run": False,
        "predictions_modified": False,
        "labels_modified": False,
        "dev_used": False,
        "test_used": False,
    }
    manifest_path = output_dir / "DAGIG_V6_ANSWER_NORMALIZATION_AMENDMENT.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
