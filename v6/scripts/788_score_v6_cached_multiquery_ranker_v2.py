#!/usr/bin/env python3
"""Score the frozen internal split with one trained scalar evidence ranker."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


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


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_FROZEN":
        raise ValueError("cached multi-query ranker v2 is not frozen")
    if freeze["code_hashes"]["scorer"] != sha256(Path(__file__).resolve()):
        raise ValueError("ranker scorer changed after freeze")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ranker scorer requires exactly one visible CUDA GPU")

    model_dir = args.model_dir.resolve()
    training_audit_path = model_dir / "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_TRAINING_COMPLETE.json"
    training_audit = read_json(training_audit_path)
    if training_audit.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_METHOD_READY" or training_audit.get("method") != args.method:
        raise ValueError("ranker training is incomplete or method-mismatched")
    adapter = Path(training_audit["adapter"])
    if tree_sha256(adapter) != training_audit["adapter_tree_sha256"]:
        raise ValueError("trained ranker adapter changed")

    internal_path = Path(freeze["input_paths"]["internal_targets"])
    if sha256(internal_path) != freeze["input_hashes"]["internal_targets"]:
        raise ValueError("internal target file changed")
    target_rows = read_jsonl(internal_path)
    target_by_state = {row["parent_state_id"]: row for row in target_rows}
    if len(target_by_state) != 238:
        raise ValueError("ranker internal target universe is incomplete")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions_path = Path(freeze["input_paths"]["evidence_actions"])
    if sha256(actions_path) != freeze["input_hashes"]["evidence_actions"]:
        raise ValueError("evidence action source changed")
    for row in read_jsonl(actions_path):
        if row["query_id"] in target_by_state:
            grouped[row["query_id"]].append(row)
    if set(grouped) != set(target_by_state):
        raise ValueError("internal scorer action and target universes differ")
    for state_id, rows in grouped.items():
        rows.sort(key=lambda row: STRATEGY_ORDER.index(row["evidence_strategy"]))
        if len(rows) != 5 or tuple(row["evidence_strategy"] for row in rows) != STRATEGY_ORDER:
            raise ValueError(f"invalid internal five-action group: {state_id}")

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(freeze["encoder_model"])
    for relative, expected in freeze["encoder_model_hashes"].items():
        if sha256(model_path / relative) != expected:
            raise ValueError(f"encoder changed: {relative}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).to("cuda")
    model.eval()
    config = freeze["training"]
    group_ids = sorted(grouped)
    predictions: list[dict[str, Any]] = []
    observed_tokens: list[int] = []
    group_batch_size = 8
    with torch.inference_mode():
        for start in range(0, len(group_ids), group_batch_size):
            batch_ids = group_ids[start : start + group_batch_size]
            rows = [row for state_id in batch_ids for row in grouped[state_id]]
            encoded = tokenizer(
                [state_text(row) for row in rows],
                [action_text(row, int(config["max_chars_per_doc"])) for row in rows],
                padding=True,
                truncation="only_second",
                max_length=int(config["max_tokens"]),
                return_tensors="pt",
            ).to("cuda")
            observed_tokens.extend(int(value) for value in encoded["attention_mask"].sum(dim=1).tolist())
            logits = model(**encoded).logits.squeeze(-1).float().reshape(len(batch_ids), 5)
            posterior = torch.softmax(logits, dim=-1).cpu().tolist()
            for state_id, state_logits, state_posterior in zip(batch_ids, logits.cpu().tolist(), posterior):
                target = [float(value) for value in target_by_state[state_id]["target_distributions"][args.method]]
                top_index = max(range(5), key=state_posterior.__getitem__)
                predictions.append(
                    {
                        "parent_state_id": state_id,
                        "method": args.method,
                        "action_strategies": list(STRATEGY_ORDER),
                        "scalar_scores": state_logits,
                        "predicted_posterior": state_posterior,
                        "target_posterior": target,
                        "selected_action_index": top_index,
                        "selected_strategy": STRATEGY_ORDER[top_index],
                    }
                )
            if start == 0 or start + group_batch_size >= len(group_ids) or (start // group_batch_size + 1) % 10 == 0:
                print(json.dumps({"method": args.method, "scored_states": min(start + group_batch_size, len(group_ids)), "total_states": len(group_ids)}), flush=True)

    if len(predictions) != 238 or {row["parent_state_id"] for row in predictions} != set(group_ids):
        raise ValueError("internal scorer predictions are incomplete")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / f"v6_cached_multiquery_ranker_v2_{args.method}_internal_predictions_no_eval_labels.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    audit = {
        "decision": "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_INTERNAL_PREDICTIONS_READY",
        "method": args.method,
        "states": len(predictions),
        "actions": len(predictions) * 5,
        "max_observed_tokens": max(observed_tokens),
        "mean_observed_tokens": sum(observed_tokens) / len(observed_tokens),
        "input_hashes": {
            "freeze": sha256(freeze_path),
            "training_audit": sha256(training_audit_path),
            "internal_targets": sha256(internal_path),
        },
        "output_paths": {"predictions": str(predictions_path)},
        "output_hashes": {"predictions": sha256(predictions_path)},
        "private_labels_loaded": False,
        "internal_used_for_training_tuning_or_early_stopping": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_SCORE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": audit["decision"], "method": args.method, "audit": str(audit_path)}, indent=2))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
