#!/usr/bin/env python3
"""Train one matched clean v4 answer-node control by listwise field distillation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch


def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def read_jsonl(path: Path) -> list[dict[str, Any]]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(child.relative_to(root)).encode()); digest.update(sha256(child).encode())
    return digest.hexdigest()


def field_tokens(tokenizer: Any, completion: str) -> tuple[list[int], list[int]]:
    parsed = json.loads(completion)
    if set(parsed) != {"final_answer"} or not isinstance(parsed["final_answer"], str): raise ValueError("answer schema changed")
    marker = json.dumps("final_answer") + ":"; start = completion.find(marker)
    if start < 0: raise ValueError("final_answer marker missing")
    value_start = start + len(marker)
    serialized = json.dumps(parsed["final_answer"], ensure_ascii=False, separators=(",", ":")); value_end = value_start + len(serialized)
    if completion[value_start:value_end] != serialized: raise ValueError("final_answer span mismatch")
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    mask = [int(end > value_start and begin < value_end) for begin, end in encoded["offset_mapping"]]
    if not any(mask): raise ValueError("empty final_answer token mask")
    return list(encoded["input_ids"]), mask


def build_group(tokenizer: Any, rows: list[dict[str, Any]], key: str, max_tokens: int):
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": rows[0]["prompt"]}], tokenize=True, add_generation_prompt=True)
    sequences, masks, targets = [], [], []
    for row in rows:
        if row["prompt"] != rows[0]["prompt"]: raise ValueError("group prompt mismatch")
        ids, field_mask = field_tokens(tokenizer, row["completion"])
        sequence = prefix + ids + [tokenizer.eos_token_id]; mask = [0] * len(prefix) + field_mask + [0]
        if len(sequence) > max_tokens: raise ValueError(f"answer sequence exceeds frozen max: {len(sequence)}")
        sequences.append(sequence); masks.append(mask); targets.append(float(row[key]))
    if abs(sum(targets) - 1.0) > 1e-8 or min(targets) <= 0.0: raise ValueError("invalid answer target")
    width = max(map(len, sequences)); input_ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids); field_masks = torch.zeros((len(sequences), width), dtype=torch.float32)
    for index, (sequence, mask) in enumerate(zip(sequences, masks)):
        input_ids[index, :len(sequence)] = torch.tensor(sequence); attention[index, :len(sequence)] = 1
        field_masks[index, :len(sequence)] = torch.tensor(mask)
    return {"input_ids": input_ids.cuda(), "attention_mask": attention.cuda()}, field_masks[:, 1:].cuda(), torch.tensor(targets, dtype=torch.float32, device="cuda")


def field_logprobs(model: Any, batch: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    logits = model(**batch, use_cache=False).logits[:, :-1].float(); labels = batch["input_ids"][:, 1:]
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--method", choices=("no_credit", "local_ig", "outcome", "dagig"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True); parser.add_argument("--max_groups", type=int, default=0)
    args = parser.parse_args(); freeze_path = args.freeze.resolve(); freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_NO_GOLD_ANSWER_CONTROLS_FROZEN": raise ValueError("answer controls are not frozen")
    if freeze["runner_hashes"]["trainer"] != sha256(Path(__file__).resolve()): raise ValueError("trainer changed after freeze")
    for key, path in freeze["input_paths"].items():
        actual = tree_hash(Path(path)) if key == "initializer_adapter" else sha256(Path(path))
        if actual != freeze["input_hashes"][key]: raise ValueError(f"frozen input changed: {key}")
    if tree_hash(Path(freeze["base_model"])) != freeze["base_model_tree_sha256"] or not torch.cuda.is_available(): raise RuntimeError("base model changed or GPU unavailable")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(Path(freeze["input_paths"]["train_data"])): grouped[str(row["parent_group_id"])].append(row)
    groups = [sorted(rows, key=lambda row: row["answer_action_id"]) for _, rows in sorted(grouped.items())]
    if len(groups) != int(freeze["parent_groups"]): raise ValueError("answer group universe changed")
    if args.max_groups: groups = groups[:args.max_groups]
    key = freeze["target_keys"][args.method]; config = freeze["training"]
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(freeze["base_model"], local_files_only=True); tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(freeze["base_model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(base, freeze["shared_initializer"], is_trainable=True).cuda(); model.config.use_cache = False
    model.enable_input_require_grads(); model.gradient_checkpointing_enable(); parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]), weight_decay=0.0)
    seed = int(config["seed"]); torch.manual_seed(seed); random.seed(seed); optimizer.zero_grad(set_to_none=True)
    logs = []; accumulated = optimizer_steps = 0
    for epoch in range(int(config["epochs"])):
        ordered = list(groups); random.Random(seed + epoch).shuffle(ordered)
        for index, rows in enumerate(ordered):
            batch, mask, targets = build_group(tokenizer, rows, key, int(config["max_input_tokens"])); action_logp = field_logprobs(model, batch, mask)
            log_policy = torch.log_softmax(action_logp, dim=0); ce = -(targets * log_policy).sum(); nll = -(targets * action_logp).sum()
            loss = ce + float(config["listwise_nll_weight"]) * nll; (loss / int(config["gradient_accumulation_groups"])).backward(); accumulated += 1
            if accumulated == int(config["gradient_accumulation_groups"]) or index + 1 == len(ordered):
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(config["max_grad_norm"])).item()); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1; accumulated = 0
                logs.append({"epoch": epoch, "optimizer_step": optimizer_steps, "loss": float(loss.detach()), "posterior_tv": float(0.5 * torch.abs(log_policy.exp() - targets).sum().detach()), "grad_norm": grad_norm})
                if optimizer_steps % int(config["logging_steps"]) == 0: print(json.dumps(logs[-1]), flush=True)
    finite = bool(logs) and all(math.isfinite(float(row[key])) for row in logs for key in ("loss", "posterior_tv", "grad_norm"))
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False); adapter = output / "adapter"; adapter.mkdir()
    model.save_pretrained(adapter, safe_serialization=True); tokenizer.save_pretrained(adapter)
    log_path = output / "training_log.jsonl"; log_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in logs), encoding="utf-8")
    audit = {"decision": "DAGIG_V6_NO_GOLD_ANSWER_POLICY_SMOKE_READY" if finite and args.max_groups else "DAGIG_V6_NO_GOLD_ANSWER_POLICY_READY" if finite else "DAGIG_V6_NO_GOLD_ANSWER_POLICY_NO_GO",
             "method": args.method, "target_key": key, "trained_parent_groups": len(groups), "trained_action_rows": sum(map(len, groups)), "max_groups": args.max_groups,
             "epochs": int(config["epochs"]), "optimizer_steps": optimizer_steps, "mean_final_loss": mean(row["loss"] for row in logs[-20:]),
             "mean_final_posterior_tv": mean(row["posterior_tv"] for row in logs[-20:]), "output_paths": {"adapter": str(adapter), "training_log": str(log_path)},
             "output_hashes": {"adapter_model": sha256(adapter / "adapter_model.safetensors"), "training_log": sha256(log_path)},
             "input_paths": {"freeze": str(freeze_path)}, "input_hashes": {"freeze": sha256(freeze_path)}, "gold_or_qrels_loaded": False,
             "internal_holdout_used": False, "dev_used": False, "test_used": False, "training_run": True}
    audit_path = output / "DAGIG_V6_NO_GOLD_ANSWER_POLICY_TRAIN_AUDIT.json"; audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True)); del model, base, optimizer; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__": main()
