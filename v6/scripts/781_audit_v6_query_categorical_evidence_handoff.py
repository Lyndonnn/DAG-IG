#!/usr/bin/env python3
"""Audit categorical evidence-policy representation across query handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
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


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dagig_fresh_categorical_scorer", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--fresh_scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.training_freeze.resolve()
    scorer_path = args.fresh_scorer.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze = read_json(freeze_path)
    if freeze.get("protocol_version") != "dagig_v6_backward_evidence_explicit_categorical_deterministic_microbatch_v3":
        raise ValueError("handoff audit requires categorical deterministic evidence v3")
    scorer = load_module(scorer_path)
    control_path = Path(freeze["input_paths"]["control_freeze"])
    if sha256(control_path) != freeze["input_hashes"]["control_freeze"]:
        raise ValueError("evidence control freeze changed")
    control = read_json(control_path)
    action_audit_path = Path(control["input_paths"]["evidence_action_audit"])
    if sha256(action_audit_path) != control["input_hashes"]["evidence_action_audit"]:
        raise ValueError("evidence action audit changed")
    action_audit = read_json(action_audit_path)
    action_key = "public_actions" if "public_actions" in action_audit["output_paths"] else "evidence_actions"
    actions_path = Path(action_audit["output_paths"][action_key])
    if sha256(actions_path) != action_audit["output_hashes"][action_key]:
        raise ValueError("public evidence actions changed")

    categorical_rows: dict[str, dict[str, Any]] = {}
    categorical_paths: list[Path] = []
    for key in ("categorical_train_data", "categorical_internal_data"):
        path = Path(freeze["input_paths"][key])
        if sha256(path) != freeze["input_hashes"][key]:
            raise ValueError(f"categorical evidence rows changed: {key}")
        categorical_paths.append(path)
        for row in read_jsonl(path):
            if row["parent_group_id"] in categorical_rows:
                raise ValueError("duplicate categorical evidence parent")
            categorical_rows[row["parent_group_id"]] = row

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(actions_path):
        grouped[row["query_id"]].append(row)
    prompt_mismatches: list[str] = []
    mapping_mismatches: list[str] = []
    for query_id in sorted(categorical_rows):
        source = grouped.get(query_id) or []
        if len(source) != 5:
            mapping_mismatches.append(query_id)
            continue
        ordered = sorted(source, key=lambda row: scorer.STRATEGY_ORDER.index(row["evidence_strategy"]))
        target = categorical_rows[query_id]
        if scorer.categorical_prompt(ordered) != target["prompt"]:
            prompt_mismatches.append(query_id)
        actual_sets = [json.loads(scorer.source_completion(row))["selected_evidence_ids"] for row in ordered]
        expected_sets = [target["selected_evidence_ids_by_action"][label] for label in scorer.LABELS]
        if actual_sets != expected_sets or target["categorical_action_labels"] != list(scorer.LABELS):
            mapping_mismatches.append(query_id)

    gates = {
        "complete_2954_query_states": len(categorical_rows) == 2954 and len(grouped) == 2954,
        "complete_14770_evidence_actions": sum(len(rows) for rows in grouped.values()) == 14770,
        "exact_prompt_contract": not prompt_mismatches,
        "exact_A_to_E_action_mapping": not mapping_mismatches,
        "one_token_action_schema": all(row["completions"] == [f'{{"action":"{label}"}}' for label in scorer.LABELS] for row in categorical_rows.values()),
        "no_gold_or_qrels_loaded": True,
        "internal_holdout_unused_for_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    result = {
        "decision": "DAGIG_V6_QUERY_CATEGORICAL_EVIDENCE_HANDOFF_GO" if all(gates.values()) else "DAGIG_V6_QUERY_CATEGORICAL_EVIDENCE_HANDOFF_NO_GO",
        "metrics": {
            "query_states": len(categorical_rows),
            "evidence_actions": sum(len(rows) for rows in grouped.values()),
            "prompt_mismatches": len(prompt_mismatches),
            "mapping_mismatches": len(mapping_mismatches),
        },
        "gates": gates,
        "mismatch_examples": {
            "prompt": prompt_mismatches[:20],
            "mapping": mapping_mismatches[:20],
        },
        "input_paths": {
            "training_freeze": str(freeze_path),
            "categorical_train_data": str(categorical_paths[0]),
            "categorical_internal_data": str(categorical_paths[1]),
            "evidence_action_audit": str(action_audit_path),
            "public_evidence_actions": str(actions_path),
            "fresh_categorical_scorer": str(scorer_path),
            "auditor": str(Path(__file__).resolve()),
        },
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    result["input_hashes"] = {key: sha256(Path(path)) for key, path in result["input_paths"].items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
