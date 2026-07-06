#!/usr/bin/env python3
"""Lightweight release gate for the DAG-IG core repository."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "dagig_grpo" / "02_train_grpo.py"
AUDIT = ROOT / "results" / "reports" / "CRITICAL_PAPER_AUDIT_20260706.md"


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if res.returncode:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(res.returncode)


def check_no_top_level_7b_import() -> None:
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts.dagig_7b_extension"):
            raise AssertionError(f"Top-level 7B import remains: {node.module}")
    print("No top-level 7B extension imports in trainer.")


def main() -> None:
    if not AUDIT.exists():
        raise FileNotFoundError(AUDIT)
    check_no_top_level_7b_import()
    run([sys.executable, "scripts/dagig_grpo/02_train_grpo.py", "--help"])
    print("Trainer help path works.")
    run([sys.executable, "scripts/verify_paper_main_results.py"])
    print("Metric consistency verifier works.")
    print("Release gate passed. Corrected KL/checker-v4/fixed-reader/corpus audits are included in verify_paper_main_results.py.")


if __name__ == "__main__":
    main()
