#!/usr/bin/env python3
"""Score one frozen evidence adapter on train or sealed internal action groups."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


METHODS = ("reference", "no_credit", "local_ig", "outcome", "dagig")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field_token_mask(tokenizer: Any, completion: str) -> tuple[list[int], list[int]]:
    parsed = json.loads(completion)
    marker = json.dumps("selected_evidence_ids") + ":"
    start = completion.find(marker)
    value_start = start + len(marker)
    serialized = json.dumps(parsed["selected_evidence_ids"], ensure_ascii=False, separators=(",", ":"))
    value_end = value_start + len(serialized)
    if start < 0 or completion[value_start:value_end] != serialized:
        raise ValueError("selected_evidence_ids field span mismatch")
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    mask = [int(end > value_start and begin < value_end) for begin, end in encoded["offset_mapping"]]
    if not any(mask):
        raise ValueError("empty evidence field mask")
    return list(encoded["input_ids"]), mask


def score_group(model: Any, tokenizer: Any, row: dict[str, Any], max_tokens: int) -> list[float]:
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True)
    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    for completion in row["completions"]:
        tokens, field_mask = field_token_mask(tokenizer, completion)
        sequence = prefix + tokens + [tokenizer.eos_token_id]
        mask = [0] * len(prefix) + field_mask + [0]
        if len(sequence) > max_tokens:
            raise ValueError("evidence scoring sequence too long")
        sequences.append(sequence)
        masks.append(mask)
    width = max(map(len, sequences))
    input_ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    field_masks = torch.zeros((len(sequences), width), dtype=torch.float32)
    for index, (sequence, mask) in enumerate(zip(sequences, masks)):
        input_ids[index, : len(sequence)] = torch.tensor(sequence)
        attention[index, : len(sequence)] = 1
        field_masks[index, : len(sequence)] = torch.tensor(mask)
    with torch.inference_mode():
        logits = model(input_ids=input_ids.cuda(), attention_mask=attention.cuda(), use_cache=False).logits[:, :-1].float()
    labels = input_ids[:, 1:].cuda()
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = field_masks[:, 1:].cuda()
    return ((token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)).cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train_audit", type=Path)
    parser.add_argument("--partition", choices=("policy_train", "internal_holdout"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN":
        raise ValueError("backward evidence training is not frozen")
    adapter = args.adapter.resolve()
    inputs = {"freeze": str(freeze_path), "adapter_model": str(adapter / "adapter_model.safetensors")}
    if args.method == "reference":
        if adapter != Path(freeze["shared_sft_adapter"]).resolve():
            raise ValueError("reference adapter path differs from frozen initializer")
        if sha256(adapter / "adapter_model.safetensors") != freeze["input_hashes"]["sft_adapter_model"]:
            raise ValueError("reference adapter changed")
    else:
        if args.train_audit is None:
            raise ValueError("trained methods require --train_audit")
        audit_path = args.train_audit.resolve()
        audit = read_json(audit_path)
        if audit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_READY" or audit.get("method") != args.method:
            raise ValueError("evidence policy training audit mismatch")
        if sha256(adapter / "adapter_model.safetensors") != audit["output_hashes"]["adapter_model"]:
            raise ValueError("trained evidence adapter changed")
        inputs["train_audit"] = str(audit_path)
    data_key = "train_data" if args.partition == "policy_train" else "internal_data"
    control = read_json(Path(freeze["input_paths"]["control_freeze"]))
    data_path = Path(control["output_paths"][data_key])
    if sha256(data_path) != control["output_hashes"][data_key]:
        raise ValueError("evidence action data changed")
    rows = read_jsonl(data_path)

    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(freeze["base_model"], local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        freeze["base_model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    model = PeftModel.from_pretrained(base, adapter).cuda().eval()
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        scores = score_group(model, tokenizer, row, int(freeze["training"]["max_input_tokens"]))
        output_rows.append(
            {
                "method": args.method,
                "partition": args.partition,
                "sample_id": row["sample_id"],
                "parent_group_id": row["parent_group_id"],
                "action_ids": row["action_ids"],
                "field_logprob_scores": scores,
            }
        )
        if index % 500 == 0:
            print(json.dumps({"method": args.method, "partition": args.partition, "scored": index, "total": len(rows)}), flush=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    scores_path = output_dir / f"v6_backward_evidence_{args.method}_{args.partition}_scores_no_labels.jsonl"
    scores_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    gates = {
        "complete_partition": len(output_rows) == len(rows),
        "finite_scores": all(math.isfinite(value) for row in output_rows for value in row["field_logprob_scores"]),
        "complete_five_actions": all(len(row["field_logprob_scores"]) == 5 for row in output_rows),
        "no_gold_or_qrels_loaded": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    result = {
        "decision": "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORES_READY" if all(gates.values()) else "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORES_FAILED",
        "method": args.method,
        "partition": args.partition,
        "metrics": {"groups": len(output_rows), "actions": 5 * len(output_rows)},
        "gates": gates,
        "input_paths": inputs,
        "input_hashes": {key: sha256(Path(path)) for key, path in inputs.items()},
        "output_paths": {"scores": str(scores_path)},
        "output_hashes": {"scores": sha256(scores_path)},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORE_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
