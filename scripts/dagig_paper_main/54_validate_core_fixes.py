#!/usr/bin/env python3
"""Validate critical paper-main implementation fixes.

This is a fast, deterministic gate for the July 2026 audit fixes:
- answer checker v4 false-positive guards;
- non-negative k3 KL penalty with a real policy-gradient path;
- no top-level dependency on the 7B reward extension modules.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts/dagig_grpo/02_train_grpo.py"
UTILS_SCRIPT = ROOT / "scripts/dagig_grpo/grpo_utils.py"
REPORT = ROOT / "outputs/dagig_paper_main_v1/reports/core_fix_validation.json"
REPORT_MD = ROOT / "outputs/dagig_paper_main_v1/reports/CORE_FIX_VALIDATION.md"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_no_top_level_7b_import(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "dagig_7b_extension" in alias.name:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "dagig_7b_extension" in module:
                offenders.append(module)
    if offenders:
        raise AssertionError(f"top-level 7B imports remain: {offenders}")
    return offenders


def validate_checker(utils_module) -> list[dict[str, object]]:
    cases = [
        {
            "name": "am_pm_range_not_bare_time",
            "prediction": "11:30 AM - 10:00 PM",
            "gold": "10 am",
            "expected": False,
        },
        {
            "name": "am_pm_exact_suffix_required",
            "prediction": "5:00 AM",
            "gold": "5 pm",
            "expected": False,
        },
        {
            "name": "substring_word_boundary",
            "prediction": "Romeo cafe",
            "gold": "Rome",
            "expected": False,
        },
        {
            "name": "numeric_range_not_single_year",
            "prediction": "from 2019 to 2023",
            "gold": "2019",
            "expected": False,
        },
        {
            "name": "numeric_contained_still_valid",
            "prediction": "The farm has 3 proprietary varieties of plums.",
            "gold": "3",
            "expected": True,
        },
        {
            "name": "phone_compact_match",
            "prediction": "65-6535-6455",
            "gold": "(65) 6535 6455",
            "expected": True,
        },
    ]
    results: list[dict[str, object]] = []
    for case in cases:
        detail = utils_module.answer_match_details(case["prediction"], case["gold"])
        observed = bool(detail.get("answer_correct"))
        if observed != case["expected"]:
            raise AssertionError(f"checker case failed: {case['name']} -> {detail}")
        results.append({**case, "detail": detail})
    return results


def validate_k3_kl(train_module) -> dict[str, float | bool]:
    same_policy = torch.tensor([-4.0], requires_grad=True)
    same_ref = torch.tensor([-4.0])
    token_count = torch.tensor([2.0])
    same_kl, same_signed = train_module.k3_kl_per_token(same_policy, same_ref, token_count)
    if float(same_kl.item()) != 0.0:
        raise AssertionError(f"k3 KL should be zero when policy==ref, got {same_kl.item()}")
    if float(same_signed.item()) != 0.0:
        raise AssertionError(f"signed log-ratio should be zero when policy==ref, got {same_signed.item()}")

    policy = torch.tensor([-8.0], requires_grad=True)
    ref = torch.tensor([-10.0])
    kl, signed = train_module.k3_kl_per_token(policy, ref, torch.tensor([2.0]))
    if float(kl.item()) < 0.0:
        raise AssertionError(f"k3 KL must be non-negative, got {kl.item()}")
    loss = 0.1 * kl.mean()
    loss.backward()
    grad = float(policy.grad.item())
    if abs(grad) < 1e-8:
        raise AssertionError("k3 KL produced zero policy gradient")
    old_constant_grad = 0.1 / 2.0
    if abs(grad - old_constant_grad) < 1e-6:
        raise AssertionError("k3 KL gradient matches old constant log-ratio penalty")
    bf16_policy = torch.tensor([-8.0], dtype=torch.bfloat16)
    bf16_ref = torch.tensor([-8.015625], dtype=torch.bfloat16)
    bf16_kl, _ = train_module.k3_kl_per_token(bf16_policy, bf16_ref, torch.tensor([35.0], dtype=torch.bfloat16))
    if float(bf16_kl.item()) < 0.0:
        raise AssertionError(f"k3 KL must stay non-negative under bf16 inputs, got {bf16_kl.item()}")
    return {
        "same_kl": float(same_kl.item()),
        "same_signed_log_ratio": float(same_signed.item()),
        "positive_case_kl": float(kl.item()),
        "positive_case_signed_log_ratio": float(signed.item()),
        "positive_case_grad": grad,
        "old_constant_grad_reference": old_constant_grad,
        "bf16_near_zero_kl": float(bf16_kl.item()),
    }


def main() -> None:
    no_top_level_7b = assert_no_top_level_7b_import(TRAIN_SCRIPT)
    utils_module = load_module(UTILS_SCRIPT, "dagig_grpo_utils_validation")
    train_module = load_module(TRAIN_SCRIPT, "dagig_grpo_train_validation")
    checker_results = validate_checker(utils_module)
    k3_result = validate_k3_kl(train_module)
    summary = {
        "passed": True,
        "train_script": str(TRAIN_SCRIPT),
        "utils_script": str(UTILS_SCRIPT),
        "no_top_level_7b_imports": no_top_level_7b == [],
        "checker_cases": checker_results,
        "k3_kl": k3_result,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Core Fix Validation\n\n",
        "- status: `passed`\n",
        "- answer checker false-positive guard cases: `passed`\n",
        "- k3 KL non-negativity/gradient check: `passed`\n",
        "- top-level 7B extension imports: `none`\n",
        f"- machine-readable report: `{REPORT}`\n",
    ]
    REPORT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
