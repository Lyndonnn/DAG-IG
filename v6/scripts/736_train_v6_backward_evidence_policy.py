#!/usr/bin/env python3
"""Train one equal-budget backward evidence policy from frozen targets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch


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
    marker = json.dumps("selected_evidence_ids") + ":"
    start = completion.find(marker)
    if start < 0:
        raise ValueError("selected_evidence_ids marker missing")
    value_start = start + len(marker)
    serialized = json.dumps(parsed["selected_evidence_ids"], ensure_ascii=False, separators=(",", ":"))
    value_end = value_start + len(serialized)
    if completion[value_start:value_end] != serialized:
        raise ValueError("field span mismatch")
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    mask = [int(end > value_start and begin < value_end) for begin, end in encoded["offset_mapping"]]
    if not any(mask):
        raise ValueError("empty optimized field token mask")
    return list(encoded["input_ids"]), mask


def build_group(tokenizer: Any, row: dict[str, Any], max_tokens: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True)
    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    for completion in row["completions"]:
        tokens, field_mask = field_token_mask(tokenizer, completion)
        sequence = prefix + tokens + [tokenizer.eos_token_id]
        mask = [0] * len(prefix) + field_mask + [0]
        if len(sequence) > max_tokens:
            raise ValueError(f"group sequence exceeds frozen maximum: {row['parent_group_id']}")
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
    return {"input_ids": input_ids.cuda(), "attention_mask": attention.cuda()}, field_masks[:, 1:].cuda()


def field_logprobs(model: Any, batch: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    logits = model(**batch, use_cache=False).logits[:, :-1].float()
    labels = batch["input_ids"][:, 1:]
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--method", choices=("no_credit", "local_ig", "outcome", "dagig"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_groups", type=int, default=0)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN":
        raise ValueError("backward evidence training is not frozen")
    if sha256(Path(__file__).resolve()) != freeze["runner_hashes"]["trainer"]:
        raise ValueError("trainer differs from frozen runner")
    for key, path in freeze["input_paths"].items():
        if sha256(Path(path)) != freeze["input_hashes"][key]:
            raise ValueError(f"frozen input changed: {key}")
    if tree_hash(Path(freeze["shared_sft_adapter"])) != freeze["shared_sft_adapter_tree_sha256"]:
        raise ValueError("shared evidence initializer changed")
    rows = read_jsonl(Path(freeze["input_paths"]["train_data"]))
    rows = rows[: args.max_groups] if args.max_groups else rows
    if not rows or any(len(row["completions"]) != 5 for row in rows):
        raise ValueError("invalid evidence training group universe")
    target_key = freeze["target_keys"][args.method]
    config = freeze["training"]

    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(freeze["base_model"], local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        freeze["base_model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    model = PeftModel.from_pretrained(base, freeze["shared_sft_adapter"], is_trainable=True).cuda()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    reference: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            batch, mask = build_group(tokenizer, row, int(config["max_input_tokens"]))
            reference[row["parent_group_id"]] = field_logprobs(model, batch, mask).cpu()
            if index % 500 == 0:
                print(json.dumps({"method": args.method, "reference_groups": index, "total": len(rows)}), flush=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]), weight_decay=0.0)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logs: list[dict[str, Any]] = []
    optimizer_steps = 0
    accumulated = 0
    for epoch in range(int(config["epochs"])):
        ordered = list(rows)
        random.Random(seed + epoch).shuffle(ordered)
        for index, row in enumerate(ordered):
            batch, mask = build_group(tokenizer, row, int(config["max_input_tokens"]))
            current = field_logprobs(model, batch, mask)
            old = reference[row["parent_group_id"]].cuda()
            behavior = torch.tensor(row["behavior_probabilities"], dtype=torch.float32, device="cuda")
            target = torch.tensor(row[target_key], dtype=torch.float32, device="cuda")
            logits = torch.log(behavior) + float(config["beta"]) * (current - old)
            log_policy = torch.log_softmax(logits, dim=0)
            loss = -(target * log_policy).sum()
            (loss / int(config["gradient_accumulation_groups"])).backward()
            accumulated += 1
            final_group = index + 1 == len(ordered)
            if accumulated == int(config["gradient_accumulation_groups"]) or final_group:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(config["max_grad_norm"])).item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0
                policy = torch.softmax(logits.detach(), dim=0)
                log = {
                    "epoch": epoch,
                    "optimizer_step": optimizer_steps,
                    "loss": float(loss.detach()),
                    "policy_target_tv": float(0.5 * torch.abs(policy - target).sum()),
                    "mean_abs_policy_shift": float(torch.abs(current.detach() - old).mean()),
                    "grad_norm": grad_norm,
                }
                logs.append(log)
                if optimizer_steps % int(config["logging_steps"]) == 0:
                    print(json.dumps({"method": args.method, **log}), flush=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter = output_dir / "adapter"
    adapter.mkdir()
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    log_path = output_dir / "training_log.jsonl"
    log_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in logs), encoding="utf-8")
    result = {
        "decision": "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_READY" if not args.max_groups else "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SMOKE_READY",
        "method": args.method,
        "target_key": target_key,
        "groups": int(freeze["groups"]),
        "trained_groups": len(rows),
        "epochs": int(config["epochs"]),
        "optimizer_steps": optimizer_steps,
        "mean_final_policy_target_tv": mean(row["policy_target_tv"] for row in logs[-20:]),
        "mean_final_policy_shift": mean(row["mean_abs_policy_shift"] for row in logs[-20:]),
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": sha256(freeze_path)},
        "output_paths": {"adapter": str(adapter), "training_log": str(log_path)},
        "output_hashes": {"adapter_model": sha256(adapter / "adapter_model.safetensors"), "training_log": sha256(log_path)},
        "same_group_universe_for_all_methods": True,
        "gold_or_qrels_loaded": False,
        "internal_holdout_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": True,
    }
    audit_path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_TRAIN_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    del model, base, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
