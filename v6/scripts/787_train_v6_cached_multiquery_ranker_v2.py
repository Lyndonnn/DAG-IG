#!/usr/bin/env python3
"""Train one matched scalar evidence ranker from frozen posterior targets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F


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


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_epoch_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    epoch: int,
    global_step: int,
) -> Path:
    final = output_dir / "checkpoints" / f"epoch_{epoch:03d}"
    temporary = final.with_name(final.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary)
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    (temporary / "trainer_state.json").write_text(
        json.dumps({"completed_epochs": epoch, "global_step": global_step}, indent=2) + "\n",
        encoding="utf-8",
    )
    if final.exists():
        shutil.rmtree(final)
    os.replace(temporary, final)
    atomic_write_json(
        output_dir / "last_checkpoint.json",
        {"completed_epochs": epoch, "global_step": global_step, "checkpoint": str(final)},
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_FROZEN":
        raise ValueError("cached multi-query ranker v2 is not frozen")
    if freeze["code_hashes"]["trainer"] != sha256(Path(__file__).resolve()):
        raise ValueError("ranker trainer changed after freeze")
    for key, raw_path in freeze["input_paths"].items():
        if sha256(Path(raw_path)) != freeze["input_hashes"][key]:
            raise ValueError(f"frozen input changed: {key}")
    model_path = Path(freeze["encoder_model"])
    for relative, expected in freeze["encoder_model_hashes"].items():
        if sha256(model_path / relative) != expected:
            raise ValueError(f"encoder changed: {relative}")
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"ranker worker requires exactly one visible GPU, found {torch.cuda.device_count()}")

    config = freeze["training"]
    target_rows = read_jsonl(Path(freeze["input_paths"]["train_targets"]))
    target_by_state = {row["parent_state_id"]: row for row in target_rows}
    if len(target_by_state) != 946:
        raise ValueError("ranker training target universe is incomplete")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(Path(freeze["input_paths"]["evidence_actions"])):
        if row["query_id"] in target_by_state:
            grouped[row["query_id"]].append(row)
    if set(grouped) != set(target_by_state):
        raise ValueError("ranker action and target universes differ")
    for state_id, rows in grouped.items():
        rows.sort(key=lambda row: STRATEGY_ORDER.index(row["evidence_strategy"]))
        if len(rows) != 5 or tuple(row["evidence_strategy"] for row in rows) != STRATEGY_ORDER:
            raise ValueError(f"invalid five-action group: {state_id}")
        target = target_by_state[state_id]["target_distributions"][args.method]
        if len(target) != 5 or abs(sum(target) - 1.0) > 1e-10:
            raise ValueError(f"invalid {args.method} target: {state_id}")

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_marker = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_TRAINING_COMPLETE.json"
    if completed_marker.exists():
        completed = read_json(completed_marker)
        if completed.get("decision") == "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_METHOD_READY":
            print(json.dumps({"status": "already_complete", "method": args.method, "output": str(output_dir)}, indent=2))
            return

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda")
    checkpoint_state_path = output_dir / "last_checkpoint.json"
    checkpoint_state = read_json(checkpoint_state_path) if checkpoint_state_path.exists() else None
    base = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if checkpoint_state:
        model = PeftModel.from_pretrained(base, checkpoint_state["checkpoint"], is_trainable=True)
        start_epoch = int(checkpoint_state["completed_epochs"])
        global_step = int(checkpoint_state["global_step"])
    else:
        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=list(config["lora_target_modules"]),
            modules_to_save=["classifier"],
            bias="none",
        )
        model = get_peft_model(base, lora)
        start_epoch = 0
        global_step = 0
    model = model.to(device)
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    if checkpoint_state:
        optimizer.load_state_dict(torch.load(Path(checkpoint_state["checkpoint"]) / "optimizer.pt", map_location=device))

    log_path = output_dir / "training_log.jsonl"
    group_ids = sorted(grouped)
    group_batch_size = int(config["group_batch_size"])
    accumulation = int(config["gradient_accumulation_steps"])
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    observed_tokens: list[int] = []
    epoch_summaries: list[dict[str, Any]] = []
    model.train()
    for epoch_index in range(start_epoch, int(config["epochs"])):
        ordered = list(group_ids)
        random.Random(seed + epoch_index * 1000).shuffle(ordered)
        epoch_losses: list[float] = []
        epoch_tvs: list[float] = []
        visits: Counter[str] = Counter()
        optimizer.zero_grad(set_to_none=True)
        micro_batches = math.ceil(len(ordered) / group_batch_size)
        for micro_index, start in enumerate(range(0, len(ordered), group_batch_size)):
            batch_ids = ordered[start : start + group_batch_size]
            rows = [row for state_id in batch_ids for row in grouped[state_id]]
            states = [state_text(row) for row in rows]
            actions = [action_text(row, int(config["max_chars_per_doc"])) for row in rows]
            encoded = tokenizer(
                states,
                actions,
                padding=True,
                truncation="only_second",
                max_length=int(config["max_tokens"]),
                return_tensors="pt",
            ).to(device)
            observed_tokens.extend(int(value) for value in encoded["attention_mask"].sum(dim=1).tolist())
            logits = model(**encoded).logits.squeeze(-1).float().reshape(len(batch_ids), 5)
            targets = torch.tensor(
                [target_by_state[state_id]["target_distributions"][args.method] for state_id in batch_ids],
                dtype=torch.float32,
                device=device,
            )
            log_probabilities = F.log_softmax(logits, dim=-1)
            loss = -(targets * log_probabilities).sum(dim=-1).mean()
            (loss / accumulation).backward()
            predicted = torch.softmax(logits.detach(), dim=-1)
            epoch_losses.append(float(loss.detach()))
            epoch_tvs.extend((0.5 * torch.abs(predicted - targets).sum(dim=-1)).cpu().tolist())
            visits.update(batch_ids)
            step_now = (micro_index + 1) % accumulation == 0 or micro_index + 1 == micro_batches
            if step_now:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, float(config["max_grad_norm"])).item()
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % int(config["logging_steps"]) == 0:
                    log = {
                        "method": args.method,
                        "epoch": epoch_index + 1,
                        "global_step": global_step,
                        "listwise_cross_entropy": epoch_losses[-1],
                        "batch_posterior_tv": mean(epoch_tvs[-len(batch_ids) :]),
                        "grad_norm_before_clip": grad_norm,
                    }
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(log, sort_keys=True) + "\n")
                    print(json.dumps(log), flush=True)
        if set(visits) != set(group_ids) or any(count != 1 for count in visits.values()):
            raise RuntimeError(f"epoch {epoch_index + 1} did not visit every group exactly once")
        epoch_summary = {
            "epoch": epoch_index + 1,
            "groups": len(visits),
            "mean_listwise_cross_entropy": mean(epoch_losses),
            "mean_online_posterior_tv": mean(epoch_tvs),
            "global_step": global_step,
        }
        epoch_summaries.append(epoch_summary)
        checkpoint = save_epoch_checkpoint(model, optimizer, output_dir, epoch_index + 1, global_step)
        print(json.dumps({"method": args.method, "epoch_complete": epoch_summary, "checkpoint": str(checkpoint)}), flush=True)

    final_dir = output_dir / "adapter"
    temporary_final = output_dir / "adapter.tmp"
    if temporary_final.exists():
        shutil.rmtree(temporary_final)
    model.save_pretrained(temporary_final)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temporary_final, final_dir)
    result = {
        "decision": "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_METHOD_READY",
        "method": args.method,
        "training_groups": len(group_ids),
        "training_actions": len(group_ids) * 5,
        "completed_epochs": int(config["epochs"]),
        "optimizer_steps": global_step,
        "epoch_summaries_current_process": epoch_summaries,
        "max_observed_tokens_after_frozen_truncation": max(observed_tokens) if observed_tokens else None,
        "adapter": str(final_dir),
        "adapter_tree_sha256": tree_sha256(final_dir),
        "input_hashes": {"freeze": sha256(freeze_path), "train_targets": sha256(Path(freeze["input_paths"]["train_targets"]))},
        "internal_loaded": False,
        "private_labels_loaded": False,
        "gold_or_qrels_used": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": True,
    }
    atomic_write_json(completed_marker, result)
    print(json.dumps(result, indent=2), flush=True)
    del optimizer, model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
