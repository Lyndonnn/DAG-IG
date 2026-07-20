#!/usr/bin/env python3
"""Audit final ranker target fit on policy-train without private labels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
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
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("train-fit auditor requires exactly one visible GPU")

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_FROZEN":
        raise ValueError("ranker protocol is not frozen")
    train_path = Path(freeze["input_paths"]["train_targets"])
    if sha256(train_path) != freeze["input_hashes"]["train_targets"]:
        raise ValueError("policy-train target file changed")
    targets = read_jsonl(train_path)
    target_by_state = {row["parent_state_id"]: row for row in targets}
    if len(target_by_state) != 946:
        raise ValueError("policy-train target universe is incomplete")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions_path = Path(freeze["input_paths"]["evidence_actions"])
    for row in read_jsonl(actions_path):
        if row["query_id"] in target_by_state:
            grouped[row["query_id"]].append(row)
    for state_id, rows in grouped.items():
        rows.sort(key=lambda row: STRATEGY_ORDER.index(row["evidence_strategy"]))
        if len(rows) != 5:
            raise ValueError(f"incomplete action group: {state_id}")

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(freeze["encoder_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = freeze["training"]
    group_ids = sorted(grouped)
    summaries: dict[str, dict[str, Any]] = {}
    model_root = args.model_root.resolve()
    for method in METHODS:
        training_audit = read_json(model_root / method / "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_TRAINING_COMPLETE.json")
        adapter = Path(training_audit["adapter"])
        if tree_sha256(adapter) != training_audit["adapter_tree_sha256"]:
            raise ValueError(f"adapter changed for {method}")
        base = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False).to("cuda")
        model.eval()
        tvs: list[float] = []
        kls: list[float] = []
        top: list[float] = []
        high_margin: list[float] = []
        high_margin_count = 0
        batch_groups = 8
        with torch.inference_mode():
            for start in range(0, len(group_ids), batch_groups):
                batch_ids = group_ids[start : start + batch_groups]
                rows = [row for state_id in batch_ids for row in grouped[state_id]]
                encoded = tokenizer(
                    [state_text(row) for row in rows],
                    [action_text(row, int(config["max_chars_per_doc"])) for row in rows],
                    padding=True,
                    truncation="only_second",
                    max_length=int(config["max_tokens"]),
                    return_tensors="pt",
                ).to("cuda")
                logits = model(**encoded).logits.squeeze(-1).float().reshape(len(batch_ids), 5)
                posterior = torch.softmax(logits, dim=-1).cpu().tolist()
                for state_id, predicted in zip(batch_ids, posterior):
                    target = [float(value) for value in target_by_state[state_id]["target_distributions"][method]]
                    tvs.append(0.5 * sum(abs(a - b) for a, b in zip(target, predicted)))
                    kls.append(sum(q * math.log(q / max(p, 1e-12)) for q, p in zip(target, predicted) if q > 0.0))
                    target_top = max(range(5), key=target.__getitem__)
                    predicted_top = max(range(5), key=predicted.__getitem__)
                    top.append(float(target_top == predicted_top))
                    ordered_target = sorted(target, reverse=True)
                    if ordered_target[0] - ordered_target[1] >= float(freeze["internal_go_gates"]["high_margin_threshold"]):
                        high_margin_count += 1
                        high_margin.append(float(target_top == predicted_top))
        summaries[method] = {
            "states": len(tvs),
            "mean_target_tv": mean(tvs),
            "mean_target_forward_kl": mean(kls),
            "top_action_agreement": mean(top),
            "high_margin_states": high_margin_count,
            "high_margin_top_action_agreement": mean(high_margin) if high_margin else None,
        }
        print(json.dumps({"method": method, "train_fit": summaries[method]}), flush=True)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    trainer_source = Path(freeze["code_paths"]["trainer"]).read_text(encoding="utf-8")
    seed_position = trainer_source.find("torch.manual_seed(seed)")
    init_position = trainer_source.find("get_peft_model(base, lora)")
    deterministic_initializer = seed_position >= 0 and init_position >= 0 and seed_position < init_position
    result = {
        "decision": "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_TRAIN_FIT_AUDITED",
        "train_fit": summaries,
        "protocol_conformance": {
            "deterministic_seed_set_before_lora_initialization": deterministic_initializer,
            "same_initializer_claim_implemented": deterministic_initializer,
            "internal_loaded": False,
            "private_labels_loaded": False,
        },
        "diagnosis": (
            "ranker protocol breach: LoRA initialization was not deterministically shared across methods"
            if not deterministic_initializer
            else "initializer implementation matches freeze"
        ),
        "input_hashes": {"freeze": sha256(freeze_path), "train_targets": sha256(train_path)},
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    audit_path = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_TRAIN_FIT_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(audit_path), **result}, indent=2))


if __name__ == "__main__":
    main()
