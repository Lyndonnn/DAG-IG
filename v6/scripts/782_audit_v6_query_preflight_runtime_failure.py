#!/usr/bin/env python3
"""Record a terminal no-holdout query preflight runtime failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--exit_code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    log_path = args.log.resolve()
    output = args.output.resolve()
    if args.exit_code == 0:
        raise ValueError("runtime-failure audit requires a nonzero exit code")
    if output.exists():
        raise FileExistsError(output)
    freeze = read_json(freeze_path)
    if freeze.get("protocol_version") != "dagig_v6_backward_fixed_descendants_equal_query_training_deterministic_v2":
        raise ValueError("runtime-failure audit requires deterministic query v2")
    if sha256(Path(__file__).resolve()) != freeze["runner_hashes"]["preflight_failure_auditor"]:
        raise ValueError("query preflight failure auditor differs from frozen runner")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failure_type = "cuda_out_of_memory" if re.search(r"out of memory|CUDA.*OOM", text, re.IGNORECASE) else "runtime_failure"
    result = {
        "decision": "DAGIG_V6_QUERY_DETERMINISTIC_PREFLIGHT_RUNTIME_NO_GO",
        "failure_type": failure_type,
        "exit_code": args.exit_code,
        "input_paths": {
            "freeze": str(freeze_path),
            "preflight_log": str(log_path),
            "auditor": str(Path(__file__).resolve()),
        },
        "internal_holdout_used": False,
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
