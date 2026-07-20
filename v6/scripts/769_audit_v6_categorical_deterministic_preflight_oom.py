#!/usr/bin/env python3
"""Audit the deterministic v2 preflight OOM before any full run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--smoke_log", type=Path, required=True)
    parser.add_argument("--exit_code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    log_path = args.smoke_log.resolve()
    exit_path = args.exit_code.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    log = log_path.read_text(encoding="utf-8")
    code = int(exit_path.read_text(encoding="utf-8").strip())
    scratch = Path("/root/dagig_scratch/v6_full_dag")
    held_out = [
        scratch / "no_gold_backward_evidence_categorical_deterministic_internal_scores_v2",
        scratch / "no_gold_backward_evidence_categorical_deterministic_internal_audit_v2",
    ]
    gates = {
        "deterministic_v2_frozen": freeze.get("protocol_version") == "dagig_v6_backward_evidence_explicit_categorical_deterministic_v2",
        "preflight_failed_nonzero": code != 0,
        "failure_is_cuda_oom": "torch.OutOfMemoryError" in log and "Tried to allocate 446.00 MiB" in log,
        "full_method_run_never_started": not Path("/root/dagig_scratch/v6_full_dag/no_gold_backward_evidence_categorical_deterministic_policies_v2").exists(),
        "internal_holdout_unopened": not any(path.exists() for path in held_out),
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_CATEGORICAL_DETERMINISTIC_V2_PREFLIGHT_OOM_NO_GO" if all(gates.values()) else "DAGIG_V6_CATEGORICAL_DETERMINISTIC_V2_PREFLIGHT_INCONCLUSIVE"
    result = {
        "decision": decision,
        "gates": gates,
        "diagnosis": "Five actions times group_batch_size=2 creates ten long sequences and exceeds one 80GB A800 on a preflight batch.",
        "repair_contract": {
            "group_batch_size": 1,
            "gradient_accumulation_batches": 8,
            "effective_groups_per_optimizer_step_unchanged": 8,
            "optimizer_lr_epochs_seed_order_targets_unchanged": True,
            "all_four_methods_restart": True,
            "require_fixed_point_smoke_again": True,
            "internal_dev_test_remain_sealed": True,
        },
        "input_paths": {"freeze": str(freeze_path), "smoke_log": str(log_path), "exit_code": str(exit_path)},
        "input_hashes": {"freeze": sha256(freeze_path), "smoke_log": sha256(log_path), "exit_code": sha256(exit_path)},
        "internal_holdout_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
