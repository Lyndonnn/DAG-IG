#!/usr/bin/env python3
"""Freeze matched scalar ranker training after selector-only GO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dagig_causal.evidence_value_critic import action_text, state_text  # noqa: E402


METHODS = ("no_credit", "local_ig", "outcome", "dagig")
STRATEGY_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    parser.add_argument("--protocol_freeze", type=Path, required=True)
    parser.add_argument("--target_audit", type=Path, required=True)
    parser.add_argument("--selector_audit", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, default=Path("/root/dagig_models/bge-reranker-v2-m3"))
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "protocol_freeze": args.protocol_freeze.resolve(),
        "target_audit": args.target_audit.resolve(),
        "selector_audit": args.selector_audit.resolve(),
    }
    protocol, target_audit, selector_audit = [read_json(paths[key]) for key in paths]
    if protocol.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_FROZEN":
        raise ValueError("cached multi-query evidence v2 protocol is not frozen")
    if target_audit.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_TARGETS_GO":
        raise ValueError("cached multi-query targets are not GO")
    if selector_audit.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_SELECTOR_ONLY_GO":
        raise ValueError("selector-only gate did not pass")
    assert_hash(paths["protocol_freeze"], target_audit["input_hashes"]["protocol_freeze"], "protocol freeze")
    for key, raw_path in target_audit["output_paths"].items():
        assert_hash(Path(raw_path), target_audit["output_hashes"][key], key)
    assert_hash(Path(selector_audit["output_paths"]["private_rows"]), selector_audit["output_hashes"]["private_rows"], "selector private rows")

    train_path = Path(target_audit["output_paths"]["train_targets"])
    internal_path = Path(target_audit["output_paths"]["internal_targets"])
    train_targets = read_jsonl(train_path)
    internal_targets = read_jsonl(internal_path)
    if len(train_targets) != 946 or len(internal_targets) != 238:
        raise ValueError("frozen 946/238 ranker split changed")
    train_ids = {row["parent_state_id"] for row in train_targets}
    internal_ids = {row["parent_state_id"] for row in internal_targets}
    if train_ids & internal_ids:
        raise ValueError("ranker train/internal state overlap")
    train_samples = {state_id.split("::", 1)[0] for state_id in train_ids}
    internal_samples = {state_id.split("::", 1)[0] for state_id in internal_ids}
    if train_samples & internal_samples or len(train_samples) != 158 or len(internal_samples) != 40:
        raise ValueError("ranker split is not sample-disjoint 158/40")

    actions_path = Path(protocol["input_paths"]["evidence_actions"])
    assert_hash(actions_path, protocol["input_hashes"]["evidence_actions"], "evidence actions")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(actions_path):
        if row["query_id"] in train_ids:
            grouped[row["query_id"]].append(row)
    if set(grouped) != train_ids:
        raise ValueError("ranker train action universe is incomplete")
    for state_id, rows in grouped.items():
        rows.sort(key=lambda row: STRATEGY_ORDER.index(row["evidence_strategy"]))
        if len(rows) != 5 or tuple(row["evidence_strategy"] for row in rows) != STRATEGY_ORDER:
            raise ValueError(f"incomplete A-E group: {state_id}")

    model_path = args.encoder.resolve()
    required_model_files = (
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for relative in required_model_files:
        if not (model_path / relative).is_file():
            raise FileNotFoundError(model_path / relative)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    max_tokens = 768
    max_chars_per_doc = 700
    observed_lengths: list[int] = []
    truncated = 0
    for state_id in sorted(grouped):
        for row in grouped[state_id]:
            encoded = tokenizer(state_text(row), action_text(row, max_chars_per_doc), add_special_tokens=True)
            observed_lengths.append(len(encoded["input_ids"]))
            truncated += int(len(encoded["input_ids"]) > max_tokens)

    code_paths = {
        "freezer": Path(__file__).resolve(),
        "trainer": args.trainer.resolve(),
        "scorer": args.scorer.resolve(),
        "auditor": args.auditor.resolve(),
        "input_contract": (ROOT / "dagig_causal" / "evidence_value_critic.py").resolve(),
    }
    for path in code_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    model_hashes = {relative: sha256(model_path / relative) for relative in required_model_files}
    gates = {
        "selector_only_go": True,
        "exact_946_238_state_split": len(train_targets) == 946 and len(internal_targets) == 238,
        "sample_disjoint_158_40_split": not (train_samples & internal_samples),
        "five_semantic_actions_per_state": sum(len(rows) for rows in grouped.values()) == 946 * 5,
        "strategy_rank_score_url_id_hidden_from_encoder": True,
        "no_gold_qrel_support_strict_in_encoder_input": True,
        "same_initializer_optimizer_schedule_all_methods": True,
        "only_target_posterior_changes": True,
        "internal_not_used_for_hyperparameters_or_early_stopping": True,
        "no_variant_sweep": True,
        "generator_training_disabled": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    if not all(gates.values()):
        raise ValueError(f"ranker protocol freeze failed: {gates}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "decision": "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_FROZEN",
        "protocol_version": "dagig_v6_cached_multiquery_state_action_scalar_ranker_v2",
        "methods": list(METHODS),
        "architecture": {
            "encoder": "BGE reranker cross-encoder with one scalar sequence-classification head",
            "input": "causal parent state plus one permutation-invariant semantic evidence-set action",
            "output": "one unconstrained scalar score per action",
            "group_posterior": "softmax over the five scalar scores",
            "raw_image_used": False,
            "strategy_or_action_id_encoded": False,
        },
        "objective": {
            "primary": "listwise KL(target posterior || softmax(scalar scores))",
            "implementation": "target cross-entropy; target entropy is a constant",
            "pairwise_cardinal_ablation_deferred": True,
            "reason": "one preregistered primary objective avoids internal-holdout method selection",
        },
        "data": {
            "policy_train_samples": len(train_samples),
            "policy_train_states": len(train_targets),
            "policy_train_actions": len(train_targets) * 5,
            "internal_samples": len(internal_samples),
            "internal_states": len(internal_targets),
            "internal_actions": len(internal_targets) * 5,
            "sample_disjoint": True,
        },
        "training": {
            "epochs": 3,
            "group_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "effective_groups_per_step": 8,
            "learning_rate": 5e-5,
            "weight_decay": 1e-4,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["query", "value"],
            "max_tokens": max_tokens,
            "max_chars_per_doc": max_chars_per_doc,
            "max_grad_norm": 1.0,
            "logging_steps": 25,
            "seed": 20260720,
            "early_stopping": False,
            "hyperparameter_sweep": False,
            "checkpoint_each_epoch": True,
        },
        "input_length_audit_train_only": {
            "max_untruncated_tokens": max(observed_lengths),
            "mean_untruncated_tokens": sum(observed_lengths) / len(observed_lengths),
            "truncated_action_pairs": truncated,
            "truncation": "only_second; causal parent state is never truncated",
        },
        "internal_go_gates": {
            "dagig_mean_target_tv_max": 0.15,
            "dagig_top_action_agreement_min": 0.60,
            "dagig_high_margin_top_agreement_min": 0.70,
            "high_margin_threshold": 0.05,
            "dagig_terminal_delta_vs_no_credit_ranker_min": 0.005,
            "dagig_terminal_noninferiority_vs_outcome_ranker_tolerance": 0.002,
            "dagig_support_noninferiority_tolerance": 0.01,
            "dagig_expected_strict_noninferiority_tolerance": 0.015,
            "dagig_mode_strict_noninferiority_tolerance": 0.015,
            "dagig_selected_strategies_min": 3,
        },
        "encoder_model": str(model_path),
        "encoder_model_hashes": model_hashes,
        "input_paths": {
            **{key: str(path) for key, path in paths.items()},
            "train_targets": str(train_path),
            "internal_targets": str(internal_path),
            "evidence_actions": str(actions_path),
        },
        "input_hashes": {
            **{key: sha256(path) for key, path in paths.items()},
            "train_targets": sha256(train_path),
            "internal_targets": sha256(internal_path),
            "evidence_actions": sha256(actions_path),
        },
        "code_paths": {key: str(path) for key, path in code_paths.items()},
        "code_hashes": {key: sha256(path) for key, path in code_paths.items()},
        "method_output_root": str(output_dir.parent / "cached_multiquery_ranker_v2_models"),
        "gates": gates,
        "internal_holdout_used_for_training_or_tuning": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    freeze_path = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_FREEZE.json"
    freeze_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "data": result["data"], "input_length_audit_train_only": result["input_length_audit_train_only"], "freeze": str(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
