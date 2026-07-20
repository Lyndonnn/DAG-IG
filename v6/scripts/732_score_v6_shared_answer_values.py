#!/usr/bin/env python3
"""Score one deterministic shard of all answer actions with the shared policy."""

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
from dagig_causal.answer_prompt import answer_completion, build_answer_policy_prompt  # noqa: E402


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


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def field_tokens(tokenizer: Any, completion: str) -> tuple[list[int], list[int]]:
    value = json.loads(completion)["final_answer"]
    marker = json.dumps("final_answer") + ":"
    start = completion.find(marker) + len(marker)
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    end = start + len(serialized)
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    mask = [int(right > start and left < end) for left, right in encoded["offset_mapping"]]
    if not any(mask):
        raise ValueError("empty final_answer mask")
    return list(encoded["input_ids"]), mask


def score_group(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completions: list[str],
    max_tokens: int,
) -> list[float]:
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )
    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    for completion in completions:
        ids, field_mask = field_tokens(tokenizer, completion)
        sequence = prefix + ids + [tokenizer.eos_token_id]
        mask = [0] * len(prefix) + field_mask + [0]
        if len(sequence) > max_tokens:
            raise ValueError(f"score sequence too long: {len(sequence)}")
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
        logits = model(
            input_ids=input_ids.cuda(), attention_mask=attention.cuda(), use_cache=False
        ).logits[:, :-1].float()
    labels = input_ids[:, 1:].cuda()
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = field_masks[:, 1:].cuda()
    return ((token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)).cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_SHARED_ANSWER_VALUE_SCORING_FROZEN":
        raise ValueError("shared answer value scoring is not frozen")
    if sha256(Path(__file__).resolve()) != freeze["runner_hashes"]["scorer"]:
        raise ValueError("scorer does not match frozen runner")
    for key, path in freeze["input_paths"].items():
        if sha256(Path(path)) != freeze["input_hashes"][key]:
            raise ValueError(f"frozen input changed: {key}")
    if tree_sha256(Path(freeze["shared_adapter"])) != freeze["shared_adapter_tree_sha256"]:
        raise ValueError("shared answer adapter changed")
    if not 0 <= args.shard_index < int(freeze["num_shards"]):
        raise ValueError("invalid shard index")

    actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(Path(freeze["input_paths"]["answer_actions"])):
        actions[row["evidence_action_id"]].append(row)
    evidence = {
        row["evidence_action_id"]: row
        for row in read_jsonl(Path(freeze["input_paths"]["evidence_actions"]))
    }
    edges = {
        row["action_id"]: row
        for row in read_jsonl(Path(freeze["input_paths"]["answer_edges"]))
    }
    parent_ids = [
        parent_id
        for index, parent_id in enumerate(sorted(actions))
        if index % int(freeze["num_shards"]) == args.shard_index
    ]

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
    model = PeftModel.from_pretrained(base, freeze["shared_adapter"]).cuda().eval()
    output_rows: list[dict[str, Any]] = []
    for index, parent_id in enumerate(parent_ids, 1):
        group = sorted(actions[parent_id], key=lambda row: row["answer_action_id"])
        prompt = build_answer_policy_prompt(evidence[parent_id])
        completions = [answer_completion(row["candidate_answer"]) for row in group]
        scores = score_group(model, tokenizer, prompt, completions, int(freeze["max_input_tokens"]))
        probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0).tolist()
        child_values = [float(edges[row["answer_action_id"]]["child_success_probability"]) for row in group]
        mode_index = max(range(len(group)), key=probabilities.__getitem__)
        output_rows.append(
            {
                "sample_id": group[0]["sample_id"],
                "partition": group[0]["partition"],
                "evidence_action_id": parent_id,
                "answer_action_ids": [row["answer_action_id"] for row in group],
                "answer_field_logprob_scores": scores,
                "answer_policy_probabilities": probabilities,
                "child_success_probabilities": child_values,
                "shared_answer_value": sum(p * value for p, value in zip(probabilities, child_values)),
                "mode_answer_action_id": group[mode_index]["answer_action_id"],
                "mode_child_success_probability": child_values[mode_index],
            }
        )
        if index % 500 == 0:
            print(json.dumps({"shard": args.shard_index, "scored": index, "total": len(parent_ids)}), flush=True)

    finite = all(
        math.isfinite(value)
        for row in output_rows
        for value in [
            *row["answer_field_logprob_scores"],
            *row["answer_policy_probabilities"],
            row["shared_answer_value"],
        ]
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    values_path = output_dir / f"v6_shared_answer_values_shard{args.shard_index:02d}.jsonl"
    values_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    gates = {
        "complete_deterministic_shard": len(output_rows) == len(parent_ids),
        "finite_scores_and_values": finite,
        "normalized_answer_policies": all(
            abs(sum(row["answer_policy_probabilities"]) - 1.0) <= 1e-8 for row in output_rows
        ),
        "no_gold_or_qrels_loaded": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    result = {
        "decision": "DAGIG_V6_SHARED_ANSWER_VALUE_SHARD_READY" if all(gates.values()) else "DAGIG_V6_SHARED_ANSWER_VALUE_SHARD_FAILED",
        "shard_index": args.shard_index,
        "num_shards": int(freeze["num_shards"]),
        "metrics": {
            "evidence_groups": len(output_rows),
            "answer_actions": sum(len(row["answer_action_ids"]) for row in output_rows),
            "value_min": min(row["shared_answer_value"] for row in output_rows),
            "value_max": max(row["shared_answer_value"] for row in output_rows),
        },
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": sha256(freeze_path)},
        "output_paths": {"values": str(values_path)},
        "output_hashes": {"values": sha256(values_path)},
        "gold_or_qrels_loaded": False,
        "training_run": False,
        "dev_used": False,
        "test_used": False,
    }
    audit_path = output_dir / "DAGIG_V6_SHARED_ANSWER_VALUE_SHARD_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
