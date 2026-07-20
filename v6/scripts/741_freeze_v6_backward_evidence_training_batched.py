#!/usr/bin/env python3
"""Freeze a throughput-only batched implementation of evidence training v1."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source_freeze.resolve()
    source = read_json(source_path)
    if source.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN":
        raise ValueError("source evidence training is not frozen")
    for key, path in source["input_paths"].items():
        if sha256(Path(path)) != source["input_hashes"][key]:
            raise ValueError(f"source frozen input changed: {key}")
    trainer = Path(__file__).with_name("742_train_v6_backward_evidence_policy_batched.py").resolve()
    freeze = dict(source)
    freeze["protocol_version"] = "dagig_v6_backward_fixed_answer_equal_evidence_training_batched_v2"
    freeze["throughput_amendment"] = {
        "reason": "v1 used only about 24GB of each 80GB A800 and required approximately six hours for four methods",
        "semantic_change": False,
        "group_batch_size": 2,
        "gradient_accumulation_batches": 4,
        "effective_groups_per_optimizer_step": 8,
        "v1_effective_groups_per_optimizer_step": 8,
        "loss_equivalence": "mean(two group losses)/4 equals sum(eight group losses)/8 before each optimizer step",
        "v1_full_runs_completed": False,
        "v1_internal_opened": False,
        "dev_opened": False,
        "test_opened": False,
    }
    freeze["training"] = dict(source["training"])
    freeze["training"].pop("gradient_accumulation_groups", None)
    freeze["training"]["group_batch_size"] = 2
    freeze["training"]["gradient_accumulation_batches"] = 4
    freeze["training"]["effective_groups_per_optimizer_step"] = 8
    freeze["input_paths"] = {"source_v1_freeze": str(source_path), **source["input_paths"]}
    freeze["input_hashes"] = {key: sha256(Path(path)) for key, path in freeze["input_paths"].items()}
    freeze["runner_hashes"] = {"trainer": sha256(trainer)}
    freeze["matched_controls"] = {**source["matched_controls"], "same_group_batch_size": True, "same_effective_batch_size": True}
    freeze["training_run"] = False
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FREEZE.json"
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
