#!/usr/bin/env python3
"""Deterministic resumable listwise trainer for backward query controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

import torch


METHODS = ("no_credit", "local_ig", "outcome", "dagig")
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
                raise ValueError(f"query group sequence exceeds frozen maximum: {row['parent_group_id']}")
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
    logits = model(**batch, use_cache=False).logits[:, :-1].float()
    labels = batch["input_ids"][:, 1:]
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


def recover_atomic_checkpoint(checkpoint_dir: Path) -> None:
    previous = checkpoint_dir.with_name(checkpoint_dir.name + ".previous")
    temporary = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    if not checkpoint_dir.exists() and previous.exists():
        previous.rename(checkpoint_dir)
    if temporary.exists():
        shutil.rmtree(temporary)
    if checkpoint_dir.exists() and previous.exists():
        shutil.rmtree(previous)


def save_checkpoint(
    checkpoint_dir: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    method: str,
    freeze_hash: str,
    completed_epochs: int,
    optimizer_steps: int,
    logs: list[dict[str, Any]],
    trained_groups: int,
    group_universe_sha256: str,
) -> None:
    temporary = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    previous = checkpoint_dir.with_name(checkpoint_dir.name + ".previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    adapter = temporary / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
        },
        temporary / "optimizer_and_rng.pt",
    )
    state = {
        "method": method,
        "freeze_sha256": freeze_hash,
        "completed_epochs": completed_epochs,
        "optimizer_steps": optimizer_steps,
        "logs": logs,
        "trained_groups": trained_groups,
        "group_universe_sha256": group_universe_sha256,
        "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
        "optimizer_and_rng_sha256": sha256(temporary / "optimizer_and_rng.pt"),
    }
    (temporary / "CHECKPOINT_STATE.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if previous.exists():
        shutil.rmtree(previous)
    if checkpoint_dir.exists():
        checkpoint_dir.rename(previous)
    temporary.rename(checkpoint_dir)
    if previous.exists():
        shutil.rmtree(previous)


def load_checkpoint_state(
    checkpoint_dir: Path,
    method: str,
    freeze_hash: str,
    trained_groups: int,
    group_universe_sha256: str,
) -> dict[str, Any] | None:
    recover_atomic_checkpoint(checkpoint_dir)
    state_path = checkpoint_dir / "CHECKPOINT_STATE.json"
    if not state_path.exists():
        return None
    state = read_json(state_path)
    if state.get("method") != method or state.get("freeze_sha256") != freeze_hash:
        raise ValueError("query checkpoint identity differs from frozen run")
    if state.get("trained_groups") != trained_groups or state.get("group_universe_sha256") != group_universe_sha256:
        raise ValueError("query checkpoint group universe differs from frozen run")
    if sha256(checkpoint_dir / "adapter" / "adapter_model.safetensors") != state["adapter_model_sha256"]:
        raise ValueError("query checkpoint adapter is corrupt")
    if sha256(checkpoint_dir / "optimizer_and_rng.pt") != state["optimizer_and_rng_sha256"]:
        raise ValueError("query checkpoint optimizer/RNG payload is corrupt")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_groups", type=int, default=0)
    parser.add_argument("--stop_after_epoch", type=int, default=0)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_BACKWARD_QUERY_TRAINING_FROZEN":
        raise ValueError("backward query training is not frozen")
    if freeze.get("protocol_version") != "dagig_v6_backward_fixed_descendants_equal_query_training_deterministic_v2":
        raise ValueError("query trainer requires deterministic v2")
    if sha256(Path(__file__).resolve()) != freeze["runner_hashes"]["trainer"]:
        raise ValueError("query trainer differs from frozen runner")
    for key, path in freeze["input_paths"].items():
        if sha256(Path(path)) != freeze["input_hashes"][key]:
            raise ValueError(f"frozen query input changed: {key}")
    if tree_hash(Path(freeze["shared_initializer"])) != freeze["shared_initializer_tree_sha256"]:
        raise ValueError("shared query initializer changed")
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA device is required")

    rows = read_jsonl(Path(freeze["input_paths"]["train_data"]))
    rows = rows[: args.max_groups] if args.max_groups else rows
    if not rows or any(not 3 <= len(row["completions"]) <= 5 for row in rows):
        raise ValueError("invalid backward query group universe")
    for row in rows:
        action_count = len(row["completions"])
        if len(row["action_ids"]) != action_count:
            raise ValueError(f"query action/completion count mismatch: {row['parent_group_id']}")
        for target_key in freeze["target_keys"].values():
            probabilities = [float(value) for value in row[target_key]]
            if len(probabilities) != action_count or min(probabilities) <= 0.0 or abs(sum(probabilities) - 1.0) > 1e-8:
                raise ValueError(f"invalid {target_key}: {row['parent_group_id']}")
    group_universe_sha256 = hashlib.sha256(
        "\n".join(row["parent_group_id"] for row in rows).encode("utf-8")
    ).hexdigest()
    config = freeze["training"]
    group_batch_size = int(config["group_batch_size"])
    accumulation_batches = int(config["gradient_accumulation_batches"])
    if group_batch_size * accumulation_batches != int(config["effective_groups_per_optimizer_step"]):
        raise ValueError("query effective batch configuration changed")
    checkpoint_dir = args.checkpoint_dir.resolve()
    freeze_hash = sha256(freeze_path)
    checkpoint_state = load_checkpoint_state(
        checkpoint_dir,
        args.method,
        freeze_hash,
        len(rows),
        group_universe_sha256,
    )
    if args.resume and checkpoint_state is None:
        raise ValueError("--resume requested but no valid query checkpoint exists")
    if checkpoint_state is not None and not args.resume:
        raise ValueError("query checkpoint exists; pass --resume")
    if args.stop_after_epoch and (not args.max_groups or not 1 <= args.stop_after_epoch < int(config["epochs"])):
        raise ValueError("stop_after_epoch is allowed only for a bounded query smoke")

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
    model = PeftModel.from_pretrained(base, freeze["shared_initializer"], is_trainable=True).cuda()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    reference: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), group_batch_size):
            batch_rows = rows[start : start + group_batch_size]
            batch, mask, slices = build_batch(tokenizer, batch_rows, int(config["max_input_tokens"]))
            scores = field_logprobs(model, batch, mask).cpu()
            for row, (begin, end) in zip(batch_rows, slices):
                reference[row["parent_group_id"]] = scores[begin:end]
            if min(start + group_batch_size, len(rows)) % 200 < group_batch_size:
                print(json.dumps({"method": args.method, "reference_groups": min(start + group_batch_size, len(rows)), "total": len(rows)}), flush=True)

    if checkpoint_state is not None:
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            freeze["base_model"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(base, checkpoint_dir / "adapter", is_trainable=True).cuda()
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]), weight_decay=0.0)
    seed = int(config["seed"])
    if checkpoint_state is None:
        torch.manual_seed(seed)
        random.seed(seed)
        logs: list[dict[str, Any]] = []
        optimizer_steps = 0
        start_epoch = 0
    else:
        payload = torch.load(checkpoint_dir / "optimizer_and_rng.pt", map_location="cuda", weights_only=False)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu())
        logs = list(checkpoint_state["logs"])
        optimizer_steps = int(checkpoint_state["optimizer_steps"])
        start_epoch = int(checkpoint_state["completed_epochs"])
        print(json.dumps({"method": args.method, "resumed_from_epoch": start_epoch, "optimizer_steps": optimizer_steps}), flush=True)

    target_key = freeze["target_keys"][args.method]
    # Gradients remain enabled in eval mode. This keeps LoRA dropout disabled so
    # reference and current policies use the same deterministic forward map.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    accumulated_batches = 0
    for epoch in range(start_epoch, int(config["epochs"])):
        ordered = list(rows)
        random.Random(seed + epoch).shuffle(ordered)
        for start in range(0, len(ordered), group_batch_size):
            batch_rows = ordered[start : start + group_batch_size]
            batch, mask, slices = build_batch(tokenizer, batch_rows, int(config["max_input_tokens"]))
            scores = field_logprobs(model, batch, mask)
            losses: list[torch.Tensor] = []
            tvs: list[torch.Tensor] = []
            shifts: list[torch.Tensor] = []
            for row, (begin, end) in zip(batch_rows, slices):
                current = scores[begin:end]
                old = reference[row["parent_group_id"]].cuda()
                behavior = torch.tensor(row["behavior_probabilities"], dtype=torch.float32, device="cuda")
                target = torch.tensor(row[target_key], dtype=torch.float32, device="cuda")
                logits = torch.log(behavior) + float(config["beta"]) * (current - old)
                log_policy = torch.log_softmax(logits, dim=0)
                losses.append(-(target * log_policy).sum())
                tvs.append(0.5 * torch.abs(torch.softmax(logits.detach(), dim=0) - target).sum())
                shifts.append(torch.abs(current.detach() - old).mean())
            batch_loss = torch.stack(losses).mean()
            (batch_loss / accumulation_batches).backward()
            accumulated_batches += 1
            final_batch = start + group_batch_size >= len(ordered)
            if accumulated_batches == accumulation_batches or final_batch:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(config["max_grad_norm"])).item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated_batches = 0
                log = {
                    "epoch": epoch,
                    "optimizer_step": optimizer_steps,
                    "loss": float(batch_loss.detach()),
                    "policy_target_tv": float(torch.stack(tvs).mean()),
                    "mean_abs_policy_shift": float(torch.stack(shifts).mean()),
                    "grad_norm": grad_norm,
                }
                logs.append(log)
                if optimizer_steps % int(config["logging_steps"]) == 0:
                    print(json.dumps({"method": args.method, **log}), flush=True)
        save_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            args.method,
            freeze_hash,
            epoch + 1,
            optimizer_steps,
            logs,
            len(rows),
            group_universe_sha256,
        )
        print(json.dumps({"method": args.method, "checkpoint_completed_epoch": epoch + 1, "optimizer_steps": optimizer_steps}), flush=True)
        if args.stop_after_epoch == epoch + 1:
            return

    output_dir = args.output_dir.resolve()
    output_temporary = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists():
        raise FileExistsError(f"final query output already exists: {output_dir}")
    if output_temporary.exists():
        shutil.rmtree(output_temporary)
    output_temporary.mkdir(parents=True)
    adapter = output_temporary / "adapter"
    adapter.mkdir()
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    log_path = output_temporary / "training_log.jsonl"
    log_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in logs), encoding="utf-8")
    expected_steps = int(config["epochs"]) * math.ceil(len(rows) / int(config["effective_groups_per_optimizer_step"]))
    result = {
        "decision": "DAGIG_V6_BACKWARD_QUERY_POLICY_READY" if not args.max_groups else "DAGIG_V6_BACKWARD_QUERY_POLICY_SMOKE_READY",
        "method": args.method,
        "target_key": target_key,
        "groups": int(freeze["groups"]),
        "trained_groups": len(rows),
        "epochs": int(config["epochs"]),
        "optimizer_steps": optimizer_steps,
        "expected_optimizer_steps": expected_steps,
        "group_batch_size": group_batch_size,
        "effective_groups_per_optimizer_step": int(config["effective_groups_per_optimizer_step"]),
        "mean_final_policy_target_tv": mean(row["policy_target_tv"] for row in logs[-20:]),
        "mean_final_policy_shift": mean(row["mean_abs_policy_shift"] for row in logs[-20:]),
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": sha256(freeze_path)},
        "output_paths": {"adapter": str(output_dir / "adapter"), "training_log": str(output_dir / "training_log.jsonl")},
        "output_hashes": {"adapter_model": sha256(adapter / "adapter_model.safetensors"), "training_log": sha256(log_path)},
        "optimized_field": "search_query",
        "deterministic_policy_logits": True,
        "dropout_active_during_optimization": False,
        "same_group_universe_for_all_methods": True,
        "gold_or_qrels_loaded": False,
        "internal_holdout_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": True,
    }
    audit_path = output_temporary / "DAGIG_V6_BACKWARD_QUERY_POLICY_TRAIN_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_temporary.rename(output_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    del model, base, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
