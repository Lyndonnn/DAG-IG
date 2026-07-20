#!/usr/bin/env python3
"""Score frozen private per-document-any-hit support-teacher v2 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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


def verify_model(fingerprint: dict[str, Any]) -> Path:
    root = Path(fingerprint["path"])
    for name, expected in fingerprint["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Frozen model changed: {path}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard specification")
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V2_FROZEN":
        raise ValueError("Gold-aware support teacher v2 is not frozen")
    if freeze["input_hashes"]["scorer"] != sha256(Path(__file__).resolve()):
        raise ValueError("v2 scorer changed after freeze")
    input_path = Path(freeze["output_paths"]["private_prompts"])
    if sha256(input_path) != freeze["output_hashes"]["private_prompts"]:
        raise ValueError("Frozen v2 prompts changed")
    model_path = verify_model(freeze["model_fingerprint"])
    all_rows = sorted(read_jsonl(input_path), key=lambda row: row["evidence_action_id"])
    rows = [row for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")}
    if any(len(ids) != 1 for ids in label_ids.values()):
        raise ValueError(f"v2 A/B labels are not one token: {label_ids}")
    label_ids = {key: value[0] for key, value in label_ids.items()}
    texts, counts = [], []
    limit = int(freeze["teacher_contract"]["max_input_tokens"])
    for row in rows:
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt_private"]},
        ], tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(text, add_special_tokens=False))
        if count > limit:
            raise ValueError(f"v2 prompt exceeds limit without truncation: {row['evidence_action_id']}={count}")
        texts.append(text)
        counts.append(count)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=freeze["teacher_contract"]["attn_implementation"],
        local_files_only=True,
    ).eval().cuda()
    scored = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = tokenizer(texts[start : start + args.batch_size], return_tensors="pt", padding=True, truncation=False).to("cuda")
            logits = model(**batch).logits[:, -1, :].float()
            pair = logits[:, [label_ids["A"], label_ids["B"]]]
            probabilities = torch.softmax(pair, dim=-1)[:, 0].tolist()
            differences = (pair[:, 0] - pair[:, 1]).tolist()
            for row, probability, difference in zip(rows[start : start + args.batch_size], probabilities, differences):
                if not math.isfinite(probability) or not math.isfinite(difference):
                    raise ValueError(f"Non-finite v2 score: {row['evidence_action_id']}")
                scored.append({
                    "evidence_action_id": row["evidence_action_id"],
                    "query_id": row["query_id"],
                    "parent_visual_state_id": row["parent_visual_state_id"],
                    "sample_id": row["sample_id"],
                    "partition": row["partition"],
                    "gold_aware_support_logit": float(difference),
                    "gold_aware_support_probability": float(probability),
                    "input_token_count": counts[len(scored)],
                    "prediction_source": "frozen_qwen2_5_vl_7b_gold_aware_any_doc_v2",
                })
            if start and start % (args.batch_size * 250) == 0:
                print(json.dumps({"shard": args.shard_index, "scored": len(scored), "total": len(rows)}), flush=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    score_path = output / f"v6_gold_aware_support_v2_scores_shard{args.shard_index:02d}_of_{args.num_shards:02d}_private.jsonl"
    with score_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "decision": "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V2_SHARD_COMPLETE",
        "freeze_path": str(freeze_path),
        "freeze_sha256": sha256(freeze_path),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rows": len(scored),
        "min_input_tokens": min(counts),
        "max_input_tokens": max(counts),
        "mean_input_tokens": sum(counts) / len(counts),
        "score_path": str(score_path),
        "score_sha256": sha256(score_path),
        "private_reference_answers_used_for_evaluation_labels": True,
        "training_policy_input": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
    }
    manifest_path = output / "SHARD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
