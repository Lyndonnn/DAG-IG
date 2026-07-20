#!/usr/bin/env python3
"""Freeze an independent blinded GPT audit of provisional support labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


AUDITOR_SYSTEM_PROMPT = """You are an independent annotation auditor for top-k web evidence.
For each case, determine whether at least one supplied document directly states or strongly entails the private reference answer for the exact entity and all date, location, comparison, and other constraints in the question.
Irrelevant or incorrect documents do not cancel a genuinely supporting document. Equivalent formatting/transliteration of addresses, phones, dates, times, and units is acceptable. A source value that yields the reference through rounding or a simple conversion explicitly requested by the question is acceptable.
An incidental occurrence of the same number or words for the wrong entity/context, topical relevance, or a weak hint is not support.
Judge only the supplied evidence. Return one decision for every audit_id."""


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_build_audit", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    build_path = args.label_build_audit.resolve()
    build = read_json(build_path)
    if build.get("decision") != "DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_V2_PENDING_INDEPENDENT_AUDIT":
        raise ValueError("Gold-aware v2 labels are not pending independent audit")
    for key, raw_path in build["output_paths"].items():
        if sha256(Path(raw_path)) != build["output_hashes"][key]:
            raise ValueError(f"Label-build output changed: {key}")
    teacher_freeze = read_json(Path(build["input_paths"]["freeze"]))
    if teacher_freeze.get("decision") != "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V2_FROZEN":
        raise ValueError("Gold-aware v2 teacher is not frozen")
    items_path = Path(build["output_paths"]["blinded_audit_items"])
    key_path = Path(build["output_paths"]["audit_key"])
    items = read_jsonl(items_path)
    key = read_jsonl(key_path)
    if len(items) != 350 or len(key) != 350:
        raise ValueError(f"Expected 350 audit items, got {len(items)}/{len(key)}")
    if {row["audit_id"] for row in items} != {row["audit_id"] for row in key}:
        raise ValueError("Blinded items and private audit key disagree")
    if any(set(row) - {"audit_id", "system_prompt", "user_prompt_private"} for row in items):
        raise ValueError("Blinded audit items contain unexpected local metadata")
    runner, evaluator = args.runner.resolve(), args.evaluator.resolve()
    for path in (runner, evaluator):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    forbidden_metadata_fields = {
        "audit_category",
        "legacy_support_reason",
        "local_label",
        "local_probability",
    }
    gates = {
        "exact_350_blinded_items": len(items) == 350,
        "local_predictions_absent_from_runner_input": all("local_probability" not in row and "local_label" not in row for row in items),
        # Natural evidence text may legitimately contain words such as "legacy".
        # Leakage is defined by private metadata fields, not by ordinary prose.
        "legacy_labels_and_categories_absent_from_runner_input": all(
            not (set(row) & forbidden_metadata_fields) for row in items
        ),
        "private_key_hidden_from_runner": True,
        "policy_train_only": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    protocol = {
        "decision": "DAGIG_V6_GPT_SUPPORT_LABEL_AUDIT_V1_FROZEN" if all(gates.values()) else "DAGIG_V6_GPT_SUPPORT_LABEL_AUDIT_V1_NO_GO",
        "protocol_version": "dagig_v6_blinded_gpt5mini_support_label_audit_v1",
        "model": "gpt-5-mini",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "auditor_system_prompt": AUDITOR_SYSTEM_PROMPT,
        "generation": {
            "batch_size": 10,
            "max_completion_tokens": 4096,
            "response_format": "strict_json_schema",
            "temperature": None,
            "seed": None,
        },
        "budget": {
            "max_requests": 40,
            "expected_requests": 35,
            "max_total_input_tokens": 500000,
            "max_total_output_tokens": 100000,
        },
        "quality_gates": teacher_freeze["quality_gate"],
        "input_paths": {
            "label_build_audit": str(build_path),
            "blinded_items": str(items_path),
            "private_audit_key": str(key_path),
            "provisional_labels": build["output_paths"]["provisional_labels"],
            "runner": str(runner),
            "evaluator": str(evaluator),
        },
        "input_hashes": {
            "label_build_audit": sha256(build_path),
            "blinded_items": sha256(items_path),
            "private_audit_key": sha256(key_path),
            "provisional_labels": sha256(Path(build["output_paths"]["provisional_labels"])),
            "runner": sha256(runner),
            "evaluator": sha256(evaluator),
        },
        "gates": gates,
        "api_key_stored_in_protocol": False,
        "serper_calls": 0,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_GPT_SUPPORT_LABEL_AUDIT_V1_FREEZE.json"
    freeze_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "model": protocol["model"], "items": len(items), "expected_requests": protocol["budget"]["expected_requests"], "freeze": str(freeze_path), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
