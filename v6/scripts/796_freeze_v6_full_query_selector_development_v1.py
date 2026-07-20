#!/usr/bin/env python3
"""Freeze the full-query development audit after the narrow protocol was consumed."""

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


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} changed: expected {expected}, found {actual}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_value_freeze", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    query_freeze_path = args.query_value_freeze.resolve()
    auditor_path = args.auditor.resolve()
    query_freeze = read_json(query_freeze_path)
    if query_freeze.get("decision") != "DAGIG_V6_FULL_QUERY_VALUES_V1_FROZEN":
        raise ValueError("full query values v1 are not frozen")
    for key, raw_path in query_freeze["input_paths"].items():
        assert_hash(Path(raw_path), query_freeze["input_hashes"][key], key)
    for key, raw_path in query_freeze["output_paths"].items():
        assert_hash(Path(raw_path), query_freeze["output_hashes"][key], key)
    if not auditor_path.is_file():
        raise FileNotFoundError(auditor_path)

    evidence_protocol_path = Path(query_freeze["input_paths"]["evidence_protocol_freeze"])
    evidence_protocol = read_json(evidence_protocol_path)
    if evidence_protocol.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_FROZEN":
        raise ValueError("evidence protocol is not frozen")
    private_paths = {
        "private_support": Path(evidence_protocol["input_paths"]["private_support"]),
        "terminal_private": Path(evidence_protocol["input_paths"]["terminal_private_audit"]),
        "shared_answer_values": Path(query_freeze["input_paths"]["shared_answer_values"]),
    }
    for key, path in private_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{key}: {path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "decision": "DAGIG_V6_FULL_QUERY_SELECTOR_DEVELOPMENT_PROTOCOL_FROZEN",
        "protocol_version": "dagig_v6_full_query_selector_development_v1",
        "selection_rule": "direct argmax of each frozen full-query posterior; fixed Q1-to-Q5 tie break",
        "evaluation_unit": "120 visual parent states clustered by 40 development sample ids",
        "metrics": ["expected_terminal_value", "support", "expected_strict", "mode_strict"],
        "controls": ["no_credit", "local_ig", "outcome"],
        "go_gates": {
            **query_freeze["selector_go_gates"],
            "dagig_terminal_noninferiority_vs_local_tolerance": 0.002,
            "dagig_support_noninferiority_vs_local_tolerance": 0.01,
            "dagig_expected_strict_noninferiority_vs_local_tolerance": 0.015,
            "dagig_mode_strict_noninferiority_vs_local_tolerance": 0.015,
            "dagig_local_top_action_disagreement_min": 0.05,
        },
        "cluster_bootstrap": {"clusters": "sample_id", "replicates": 10000, "seed": 20260720},
        "input_paths": {
            "query_value_freeze": str(query_freeze_path),
            "internal_targets": query_freeze["output_paths"]["internal_targets"],
            "diagnostics": query_freeze["output_paths"]["diagnostics"],
            "private_support": str(private_paths["private_support"]),
            "terminal_private": str(private_paths["terminal_private"]),
            "shared_answer_values": str(private_paths["shared_answer_values"]),
        },
        "input_hashes": {
            "query_value_freeze": sha256(query_freeze_path),
            "internal_targets": query_freeze["output_hashes"]["internal_targets"],
            "diagnostics": query_freeze["output_hashes"]["diagnostics"],
            "private_support": sha256(private_paths["private_support"]),
            "terminal_private": sha256(private_paths["terminal_private"]),
            "shared_answer_values": sha256(private_paths["shared_answer_values"]),
        },
        "runner_path": str(auditor_path),
        "runner_hash": sha256(auditor_path),
        "internal_holdout_previously_consumed_by_narrow_query_protocol": True,
        "current_full_query_targets_fit_to_private_internal_labels": False,
        "development_diagnostic_only": True,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    output_path = output_dir / "DAGIG_V6_FULL_QUERY_SELECTOR_DEVELOPMENT_PROTOCOL_FREEZE.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "freeze": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
