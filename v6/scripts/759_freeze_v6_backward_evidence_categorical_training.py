#!/usr/bin/env python3
"""Freeze a matched five-way categorical repair after the v5 train-fit NO-GO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LABELS = list("ABCDE")
METHODS = ("no_credit", "local_ig", "outcome", "dagig")


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


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(child for child in root.rglob("*") if child.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def transform_prompt(source: str, evidence_sets: list[list[str]]) -> str:
    marker = "Question:"
    start = source.find(marker)
    if start < 0:
        raise ValueError("source evidence prompt has no Question field")
    body = source[start:].rstrip()
    suffix = "Evidence selection:"
    if body.endswith(suffix):
        body = body[: -len(suffix)].rstrip()
    actions = "\n".join(
        f"[{label}] selected_evidence_ids={compact(ids)}" for label, ids in zip(LABELS, evidence_sets)
    )
    return (
        "You are the evidence-selection node of a multimodal web-search agent.\n\n"
        "Exactly five legal evidence-set actions are listed below. Select one action using the question, "
        "frozen visual observation, executed query, and retrieved document content.\n\n"
        "Return only compact valid JSON with exactly one field, for example {\"action\":\"A\"}. "
        "The action must be one of A, B, C, D, or E. Do not answer the question and do not add reasoning.\n\n"
        f"{body}\n\nLegal candidate evidence-set actions:\n{actions}\n\nEvidence action:"
    )


def transform_row(row: dict[str, Any]) -> dict[str, Any]:
    if len(row.get("action_ids", [])) != 5 or len(row.get("completions", [])) != 5:
        raise ValueError("every source group must contain exactly five actions")
    evidence_sets: list[list[str]] = []
    for completion in row["completions"]:
        parsed = json.loads(completion)
        if set(parsed) != {"selected_evidence_ids"}:
            raise ValueError("source completion schema changed")
        ids = parsed["selected_evidence_ids"]
        if not isinstance(ids, list) or len(ids) != 3 or len(set(ids)) != 3 or not all(isinstance(x, str) for x in ids):
            raise ValueError("source completion is not a three-document evidence action")
        evidence_sets.append(ids)
    transformed = {key: value for key, value in row.items() if key not in {"prompt", "completions"}}
    transformed.update(
        {
            "prompt": transform_prompt(row["prompt"], evidence_sets),
            "completions": [compact({"action": label}) for label in LABELS],
            "categorical_action_labels": LABELS,
            "selected_evidence_ids_by_action": {
                label: ids for label, ids in zip(LABELS, evidence_sets)
            },
            "source_completions": row["completions"],
        }
    )
    return transformed


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_training_freeze", type=Path, required=True)
    parser.add_argument("--source_train_fit", type=Path, required=True)
    parser.add_argument("--representation_audit", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs != 3:
        raise ValueError("categorical v1 is pre-registered for exactly three epochs")

    source_freeze_path = args.source_training_freeze.resolve()
    source_fit_path = args.source_train_fit.resolve()
    representation_path = args.representation_audit.resolve()
    source = read_json(source_freeze_path)
    fit = read_json(source_fit_path)
    representation = read_json(representation_path)
    if source.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN":
        raise ValueError("source training protocol is not frozen")
    if fit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAIN_FIT_NO_GO":
        raise ValueError("categorical repair requires the source train-fit NO-GO")
    if representation.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_CATEGORICAL_REPAIR_JUSTIFIED":
        raise ValueError("categorical repair was not justified by the train-only audit")
    if any(bool(fit.get(key)) for key in ("internal_holdout_used", "dev_used", "test_used")):
        raise ValueError("a held-out partition was opened before freezing the repair")
    if sha256(source_freeze_path) != representation["input_hashes"]["training_freeze"]:
        raise ValueError("representation audit belongs to another source training protocol")
    if sha256(source_fit_path) != representation["input_hashes"]["train_fit"]:
        raise ValueError("representation audit belongs to another source train-fit audit")

    control_path = Path(source["input_paths"]["control_freeze"])
    control = read_json(control_path)
    train_source = Path(control["output_paths"]["train_data"])
    internal_source = Path(control["output_paths"]["internal_data"])
    if sha256(train_source) != control["output_hashes"]["train_data"]:
        raise ValueError("source policy-train data changed")
    if sha256(internal_source) != control["output_hashes"]["internal_data"]:
        raise ValueError("source internal data changed")
    train_rows = [transform_row(row) for row in read_jsonl(train_source)]
    internal_rows = [transform_row(row) for row in read_jsonl(internal_source)]
    if len(train_rows) != 2359 or len(internal_rows) != 595:
        raise ValueError("categorical group universe differs from the frozen five-action universe")

    scripts = Path(__file__).resolve().parent
    runner_paths = {
        "freezer": Path(__file__).resolve(),
        "trainer": scripts / "760_train_v6_backward_evidence_policy_categorical.py",
        "scorer": scripts / "761_score_v6_backward_evidence_policy_categorical.py",
        "train_fit_auditor": scripts / "762_audit_v6_backward_evidence_categorical_train_fit.py",
    }
    if not all(path.is_file() for path in runner_paths.values()):
        raise FileNotFoundError("categorical runner set is incomplete")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_path = output / "v6_backward_evidence_categorical_targets_train.jsonl"
    internal_path = output / "v6_backward_evidence_categorical_targets_internal_no_labels.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(internal_path, internal_rows)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source["base_model"], local_files_only=True)
    if any(len(tokenizer(label, add_special_tokens=False).input_ids) != 1 for label in LABELS):
        raise ValueError("A-E are not single-token actions under the frozen tokenizer")
    max_sequence_tokens = 0
    for row in train_rows + internal_rows:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True
        )
        for completion in row["completions"]:
            length = len(prefix) + len(tokenizer(completion, add_special_tokens=False).input_ids) + 1
            max_sequence_tokens = max(max_sequence_tokens, length)
    if max_sequence_tokens > int(source["training"]["max_input_tokens"]):
        raise ValueError("categorical prompt exceeds the frozen context limit")

    shared_adapter = Path(source["shared_sft_adapter"])
    if tree_hash(shared_adapter) != source["shared_sft_adapter_tree_sha256"]:
        raise ValueError("shared evidence initializer changed")
    training = dict(source["training"])
    training["epochs"] = 3
    training["seed"] = 762943
    input_paths = {
        "source_training_freeze": str(source_freeze_path),
        "source_train_fit": str(source_fit_path),
        "representation_audit": str(representation_path),
        "control_freeze": str(control_path),
        "source_train_data": str(train_source),
        "source_internal_data": str(internal_source),
        "categorical_train_data": str(train_path),
        "categorical_internal_data": str(internal_path),
        "sft_adapter_model": str(shared_adapter / "adapter_model.safetensors"),
    }
    result = {
        "decision": "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN",
        "protocol_version": "dagig_v6_backward_evidence_explicit_categorical_actions_v1",
        "repair_reason": "frozen v5 train-fit NO-GO caused by a coupled implicit multi-token action representation",
        "base_model": source["base_model"],
        "base_model_tree_sha256": source["base_model_tree_sha256"],
        "shared_sft_adapter": str(shared_adapter),
        "shared_sft_adapter_tree_sha256": source["shared_sft_adapter_tree_sha256"],
        "groups": len(train_rows),
        "internal_groups": len(internal_rows),
        "actions_per_group": 5,
        "categorical_action_labels": LABELS,
        "target_keys": source["target_keys"],
        "training": training,
        "metrics": {"max_sequence_tokens": max_sequence_tokens},
        "objective": {
            **source["objective"],
            "optimized_field": "one A-E action token",
            "action_mapping": "A-E deterministically map to the original five selected_evidence_ids triples",
        },
        "matched_controls": {
            **source["matched_controls"],
            "same_original_five_action_sets": True,
            "same_target_distributions_as_v5": True,
            "all_methods_restart_from_shared_initializer": True,
            "only_method_specific_target_distribution_changes": True,
        },
        "frozen_train_fit_gates": {
            "no_credit_mean_tv_at_most": 0.03,
            "trained_method_mean_tv_at_most": 0.10,
            "trained_method_top_agreement_at_least": 0.65,
            "trained_method_high_margin_agreement_at_least": 0.85,
            "high_margin_threshold": 0.05,
        },
        "pre_registered_escalation": {
            "trigger": "categorical three-epoch policy-train fit NO-GO",
            "allowed_once": True,
            "next_epochs": 6,
            "all_four_methods_must_rerun": True,
            "no_other_hyperparameter_change": True,
            "internal_dev_test_remain_sealed": True,
            "more_than_six_epochs_allowed": False,
        },
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "runner_paths": {key: str(path) for key, path in runner_paths.items()},
        "runner_hashes": {key: sha256(path) for key, path in runner_paths.items()},
        "gold_or_qrels_available_to_trainer": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    manifest = output / "DAGIG_V6_BACKWARD_EVIDENCE_CATEGORICAL_TRAINING_FREEZE.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
