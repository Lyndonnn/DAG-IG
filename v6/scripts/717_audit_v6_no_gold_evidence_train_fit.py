#!/usr/bin/env python3
"""Audit matched evidence policies against all policy-train targets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch


METHODS = ("local_listwise", "outcome_listwise", "dagig_posterior")


def load_core() -> Any:
    path = Path(__file__).with_name("613_run_v6_listwise_evidence_selector_eval.py")
    spec = importlib.util.spec_from_file_location("dagig_v6_no_gold_evidence_fit_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mapping(values: list[str], allowed: tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        method, separator, path = value.partition("=")
        if not separator or method not in allowed or method in result:
            raise ValueError(f"invalid method mapping: {value}")
        result[method] = Path(path).resolve()
    if set(result) != set(allowed):
        raise ValueError("method mappings must cover all trained evidence controls")
    return result


def summarize(policies: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, Any]:
    tvs, agreements, margins, kls = [], [], [], []
    for policy, target in zip(policies, targets):
        tvs.append(float(0.5 * np.abs(policy - target).sum()))
        agreements.append(int(np.argmax(policy) == np.argmax(target)))
        ordered = np.sort(target)
        margins.append(float(ordered[-1] - ordered[-2]))
        kls.append(float(np.sum(target * (np.log(target.clip(1e-12)) - np.log(policy.clip(1e-12))))))
    high = [index for index, margin in enumerate(margins) if margin >= 0.05]
    return {
        "groups": len(policies),
        "mean_policy_target_tv": mean(tvs),
        "mean_target_policy_kl": mean(kls),
        "top_action_agreement": mean(agreements),
        "margin_ge_0.05_groups": len(high),
        "margin_ge_0.05_top_action_agreement": mean(agreements[index] for index in high) if high else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--train_audit", action="append", required=True, help="method=/path/to/train_audit.json")
    parser.add_argument("--adapter", action="append", required=True, help="method=/path/to/adapter")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.training_freeze.resolve()
    freeze = core.read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_LISTWISE_EVIDENCE_NODE_GDPO_FROZEN":
        raise ValueError("evidence training is not frozen")
    target_path = Path(freeze["input_paths"]["training_targets"])
    if sha256(target_path) != freeze["input_hashes"]["training_targets"]:
        raise ValueError("evidence targets changed")
    train_audits = parse_mapping(args.train_audit, METHODS)
    adapters = parse_mapping(args.adapter, METHODS)
    input_paths: dict[str, str] = {"training_freeze": str(freeze_path), "training_targets": str(target_path)}
    for method in METHODS:
        audit = core.read_json(train_audits[method])
        if audit.get("decision") != "DAGIG_V6_LISTWISE_EVIDENCE_NODE_READY" or audit.get("method") != method:
            raise ValueError(f"evidence training is incomplete: {method}")
        model_path = adapters[method] / "adapter_model.safetensors"
        if sha256(model_path) != audit["output_hashes"]["adapter_model"]:
            raise ValueError(f"trained evidence adapter changed: {method}")
        input_paths[f"{method}_train_audit"] = str(train_audits[method])
        input_paths[f"{method}_adapter_model"] = str(model_path)
    if not torch.cuda.is_available():
        raise RuntimeError("one GPU is required")

    rows = core.read_jsonl(target_path)
    groups = [
        {"query_id": row["parent_group_id"], "prompt": row["prompt"], "completions": row["completions"]}
        for row in rows
    ]
    if len(groups) != 158 or any(len(group["completions"]) != 5 for group in groups):
        raise ValueError("evidence train target matrix is incomplete")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(freeze["base_model"], local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    reference = core.score_adapter(
        freeze["base_model"],
        freeze["shared_sft_adapter"],
        tokenizer,
        groups,
        int(freeze["training"]["max_input_tokens"]),
    )
    target_keys = freeze["target_keys"]
    beta = float(freeze["training"]["beta"])
    metrics: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        current = core.score_adapter(
            freeze["base_model"],
            str(adapters[method]),
            tokenizer,
            groups,
            int(freeze["training"]["max_input_tokens"]),
        )
        policies, targets = [], []
        for row in rows:
            group_id = row["parent_group_id"]
            behavior = np.asarray(row["behavior_probabilities"], dtype=np.float64)
            delta = np.asarray(current[group_id], dtype=np.float64) - np.asarray(reference[group_id], dtype=np.float64)
            logits = np.log(behavior) + beta * delta
            policy = np.exp(logits - logits.max())
            policy /= policy.sum()
            policies.append(policy)
            targets.append(np.asarray(row[target_keys[method]], dtype=np.float64))
        metrics[method] = summarize(policies, targets)
    gates = {
        "complete_methods": set(metrics) == set(METHODS),
        "complete_groups": all(metrics[method]["groups"] == 158 for method in METHODS),
        "mean_tv": all(metrics[method]["mean_policy_target_tv"] <= 0.15 for method in METHODS),
        "top_action_agreement": all(metrics[method]["top_action_agreement"] >= 0.55 for method in METHODS),
        "high_margin_agreement": all(
            metrics[method]["margin_ge_0.05_top_action_agreement"] is None
            or metrics[method]["margin_ge_0.05_top_action_agreement"] >= 0.75
            for method in METHODS
        ),
        "policy_train_groups_only": True,
        "no_gold_or_qrels_loaded": True,
        "internal_holdout_unused": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_NO_GOLD_EVIDENCE_TRAIN_FIT_GO" if all(gates.values()) else "DAGIG_V6_NO_GOLD_EVIDENCE_TRAIN_FIT_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "decision": decision,
        "metrics": metrics,
        "gates": gates,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    path = output / "DAGIG_V6_NO_GOLD_EVIDENCE_TRAIN_FIT_AUDIT.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
