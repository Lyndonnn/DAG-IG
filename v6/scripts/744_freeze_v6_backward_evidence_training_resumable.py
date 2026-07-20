#!/usr/bin/env python3
"""Freeze a crash-resilient implementation of the six-epoch evidence run."""

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
    expected = "dagig_v6_backward_fixed_answer_equal_evidence_training_batched_six_epoch_v3"
    if source.get("protocol_version") != expected or source.get("training", {}).get("epochs") != 6:
        raise ValueError("source is not the frozen six-epoch evidence protocol")
    if source.get("internal_holdout_used_for_training") or source.get("dev_used") or source.get("test_used"):
        raise ValueError("holdout data was opened before resilience freezing")
    for key, path in source["input_paths"].items():
        if sha256(Path(path)) != source["input_hashes"][key]:
            raise ValueError(f"source frozen input changed: {key}")

    trainer = Path(__file__).with_name("742_train_v6_backward_evidence_policy_batched.py").resolve()
    freeze = dict(source)
    freeze["protocol_version"] = (
        "dagig_v6_backward_fixed_answer_equal_evidence_training_batched_six_epoch_resumable_v5"
    )
    freeze["resilience_amendment"] = {
        "semantic_change": False,
        "checkpoint_boundary": "after every completed epoch",
        "checkpoint_contents": [
            "adapter",
            "adam_optimizer_state",
            "completed_epoch",
            "optimizer_step",
            "training_log",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state",
        ],
        "atomic_rotation": True,
        "automatic_resume_supported": True,
        "checkpoint_bound_to_group_universe_hash": True,
        "intentional_stop_resume_smoke_required_before_full_run": True,
        "screen_or_equivalent_required": True,
        "loss_batch_order_hyperparameters_changed": False,
        "internal_holdout_opened": False,
        "dev_opened": False,
        "test_opened": False,
    }
    freeze["input_paths"] = {"source_v3_freeze": str(source_path), **source["input_paths"]}
    freeze["input_hashes"] = {key: sha256(Path(path)) for key, path in freeze["input_paths"].items()}
    freeze["runner_hashes"] = {"trainer": sha256(trainer)}
    freeze["training_run"] = False

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FREEZE.json"
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
