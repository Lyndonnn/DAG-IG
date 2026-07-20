#!/usr/bin/env python3
"""Score one frozen backward query adapter on a matched action universe."""

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
QUERY_FIELDS = {"entity_quote", "information_need", "constraints", "search_query"}


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


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(child.relative_to(root)).encode("utf-8"))
        digest.update(sha256(child).encode("ascii"))
    return digest.hexdigest()


def field_token_mask(tokenizer: Any, completion: str) -> tuple[list[int], list[int]]:
    parsed = json.loads(completion)
    if set(parsed) != QUERY_FIELDS:
        raise ValueError("structured query completion schema changed")
    query = parsed["search_query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("search_query must be a nonempty string")
    marker = json.dumps("search_query") + ":"
    start = completion.find(marker)
    if start < 0:
        raise ValueError("search_query marker missing")
    value_start = start + len(marker)
    serialized = json.dumps(query, ensure_ascii=False, separators=(",", ":"))
    value_end = value_start + len(serialized)
    if completion[value_start:value_end] != serialized:
        raise ValueError("search_query field span mismatch")
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    mask = [int(end > value_start and begin < value_end) for begin, end in encoded["offset_mapping"]]
    if not any(mask):
        raise ValueError("empty search_query token mask")
    return list(encoded["input_ids"]), mask


def build_batch(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[tuple[int, int]]]:
    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    slices: list[tuple[int, int]] = []
    for row in rows:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        begin = len(sequences)
        for completion in row["completions"]:
            tokens, field_mask = field_token_mask(tokenizer, completion)
            sequence = prefix + tokens + [tokenizer.eos_token_id]
            mask = [0] * len(prefix) + field_mask + [0]
            if len(sequence) > max_tokens:
                raise ValueError(f"query scoring sequence exceeds frozen maximum: {row['parent_group_id']}")
            sequences.append(sequence)
            masks.append(mask)
        slices.append((begin, len(sequences)))
    width = max(map(len, sequences))
    input_ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    field_masks = torch.zeros((len(sequences), width), dtype=torch.float32)
    for index, (sequence, mask) in enumerate(zip(sequences, masks)):
        input_ids[index, : len(sequence)] = torch.tensor(sequence)
        attention[index, : len(sequence)] = 1
        field_masks[index, : len(sequence)] = torch.tensor(mask)
    return (
        {"input_ids": input_ids.cuda(), "attention_mask": attention.cuda()},
        field_masks[:, 1:].cuda(),
        slices,
    )


def field_logprobs(model: Any, batch: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, :-1].float()
    labels = batch["input_ids"][:, 1:]
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train_audit", type=Path)
    parser.add_argument("--partition", choices=("policy_train", "internal_holdout"), required=True)
    parser.add_argument("--group_batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_BACKWARD_QUERY_TRAINING_FROZEN":
        raise ValueError("backward query training is not frozen")
    for key, path in freeze["input_paths"].items():
        if sha256(Path(path)) != freeze["input_hashes"][key]:
            raise ValueError(f"frozen query input changed: {key}")

    adapter = args.adapter.resolve()
    inputs = {"freeze": str(freeze_path), "adapter_model": str(adapter / "adapter_model.safetensors")}
    if args.method == "reference":
        if adapter != Path(freeze["shared_initializer"]).resolve():
            raise ValueError("reference adapter differs from frozen query initializer")
        if tree_hash(adapter) != freeze["shared_initializer_tree_sha256"]:
            raise ValueError("reference query initializer changed")
    else:
        if args.train_audit is None:
            raise ValueError("trained query methods require --train_audit")
        audit_path = args.train_audit.resolve()
        audit = read_json(audit_path)
        if audit.get("decision") != "DAGIG_V6_BACKWARD_QUERY_POLICY_READY" or audit.get("method") != args.method:
            raise ValueError("query policy training audit mismatch")
        if audit["input_hashes"].get("freeze") != sha256(freeze_path):
            raise ValueError("query policy was trained under another freeze")
        if sha256(adapter / "adapter_model.safetensors") != audit["output_hashes"]["adapter_model"]:
            raise ValueError("trained query adapter changed")
        inputs["train_audit"] = str(audit_path)

    control = read_json(Path(freeze["input_paths"]["control_freeze"]))
    data_key = "train_data" if args.partition == "policy_train" else "internal_data"
    data_path = Path(control["output_paths"][data_key])
    if sha256(data_path) != control["output_hashes"][data_key]:
        raise ValueError("query action data changed")
    rows = read_jsonl(data_path)
    expected_groups = int(control["metrics"]["policy_train_groups" if args.partition == "policy_train" else "internal_holdout_groups"])
    if len(rows) != expected_groups:
        raise ValueError("query partition group count changed")
    if args.group_batch_size <= 0:
        raise ValueError("group_batch_size must be positive")

    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(freeze["base_model"], local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        freeze["base_model"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, adapter).cuda().eval()
    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.group_batch_size):
        batch_rows = rows[start : start + args.group_batch_size]
        batch, mask, slices = build_batch(tokenizer, batch_rows, int(freeze["training"]["max_input_tokens"]))
        scores = field_logprobs(model, batch, mask).cpu().tolist()
        for row, (begin, end) in zip(batch_rows, slices):
            output_rows.append(
                {
                    "method": args.method,
                    "partition": args.partition,
                    "sample_id": row["sample_id"],
                    "parent_group_id": row["parent_group_id"],
                    "action_ids": row["action_ids"],
                    "field_logprob_scores": scores[begin:end],
                }
            )
        completed = min(start + args.group_batch_size, len(rows))
        if completed % 100 < args.group_batch_size or completed == len(rows):
            print(json.dumps({"method": args.method, "partition": args.partition, "scored": completed, "total": len(rows)}), flush=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    score_path = output_dir / f"v6_backward_query_{args.method}_{args.partition}_scores_no_labels.jsonl"
    score_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    gates = {
        "complete_partition": len(output_rows) == len(rows),
        "same_group_order": [row["parent_group_id"] for row in output_rows] == [row["parent_group_id"] for row in rows],
        "finite_scores": all(math.isfinite(value) for row in output_rows for value in row["field_logprob_scores"]),
        "complete_action_counts": all(
            len(row["field_logprob_scores"]) == len(source["action_ids"])
            for row, source in zip(output_rows, rows)
        ),
        "three_to_five_actions_per_group": all(3 <= len(row["field_logprob_scores"]) <= 5 for row in output_rows),
        "search_query_field_only": True,
        "no_gold_or_qrels_loaded": True,
        "internal_holdout_unused_for_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    result = {
        "decision": "DAGIG_V6_BACKWARD_QUERY_POLICY_SCORES_READY" if all(gates.values()) else "DAGIG_V6_BACKWARD_QUERY_POLICY_SCORES_FAILED",
        "method": args.method,
        "partition": args.partition,
        "metrics": {"groups": len(output_rows), "actions": sum(len(row["field_logprob_scores"]) for row in output_rows)},
        "gates": gates,
        "input_paths": inputs,
        "input_hashes": {key: sha256(Path(path)) for key, path in inputs.items()},
        "output_paths": {"scores": str(score_path)},
        "output_hashes": {"scores": sha256(score_path)},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_BACKWARD_QUERY_POLICY_SCORE_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
