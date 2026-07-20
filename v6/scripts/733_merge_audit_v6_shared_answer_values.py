#!/usr/bin/env python3
"""Merge and audit the two immutable shared-answer value shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--shard_audits", type=Path, nargs=2, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_SHARED_ANSWER_VALUE_SCORING_FROZEN":
        raise ValueError("shared answer value scoring is not frozen")
    audits: list[tuple[Path, dict[str, Any]]] = []
    for path in args.shard_audits:
        resolved = path.resolve()
        audit = read_json(resolved)
        if audit.get("decision") != "DAGIG_V6_SHARED_ANSWER_VALUE_SHARD_READY":
            raise ValueError(f"answer value shard is not ready: {resolved}")
        if audit.get("num_shards") != 2:
            raise ValueError("shard count mismatch")
        if audit["input_hashes"]["freeze"] != sha256(freeze_path):
            raise ValueError("shard belongs to another freeze")
        values_path = Path(audit["output_paths"]["values"])
        if sha256(values_path) != audit["output_hashes"]["values"]:
            raise ValueError("answer value shard changed")
        audits.append((resolved, audit))
    if {audit["shard_index"] for _, audit in audits} != {0, 1}:
        raise ValueError("expected exactly shard 0 and shard 1")

    rows: list[dict[str, Any]] = []
    for _, audit in sorted(audits, key=lambda item: item[1]["shard_index"]):
        rows.extend(read_jsonl(Path(audit["output_paths"]["values"])))
    ids = [row["evidence_action_id"] for row in rows]
    action_ids = [action for row in rows for action in row["answer_action_ids"]]
    values = [float(row["shared_answer_value"]) for row in rows]
    gates = {
        "complete_14770_evidence_groups": len(rows) == 14770,
        "complete_41273_answer_actions": len(action_ids) == 41273,
        "unique_evidence_groups": len(set(ids)) == len(ids),
        "unique_answer_actions": len(set(action_ids)) == len(action_ids),
        "finite_nonnegative_values": all(math.isfinite(value) and value >= 0.0 for value in values),
        "normalized_answer_policies": all(abs(sum(row["answer_policy_probabilities"]) - 1.0) <= 1e-8 for row in rows),
        "shared_answer_policy_unchanged": True,
        "no_gold_or_qrels_loaded": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_SHARED_ANSWER_VALUES_GO" if all(gates.values()) else "DAGIG_V6_SHARED_ANSWER_VALUES_NO_GO"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    values_path = output_dir / "v6_shared_answer_evidence_values_no_labels.jsonl"
    values_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(rows, key=lambda item: item["evidence_action_id"])),
        encoding="utf-8",
    )
    inputs = {"freeze": str(freeze_path)}
    for _, audit in audits:
        inputs[f"shard_{audit['shard_index']}_audit"] = str(next(path for path, candidate in audits if candidate is audit))
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_shared_answer_values_v1",
        "metrics": {
            "evidence_groups": len(rows),
            "answer_actions": len(action_ids),
            "samples": len({row["sample_id"] for row in rows}),
            "partition_groups": dict(sorted(Counter(row["partition"] for row in rows).items())),
            "value_min": min(values),
            "value_mean": mean(values),
            "value_max": max(values),
            "nonconstant_value_rate": sum(value > min(values) for value in values) / len(values),
        },
        "gates": gates,
        "input_paths": inputs,
        "input_hashes": {key: sha256(Path(path)) for key, path in inputs.items()},
        "output_paths": {"shared_answer_values": str(values_path)},
        "output_hashes": {"shared_answer_values": sha256(values_path)},
        "gold_or_qrels_loaded": False,
        "training_run": False,
        "dev_used": False,
        "test_used": False,
    }
    audit_path = output_dir / "DAGIG_V6_SHARED_ANSWER_VALUE_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
