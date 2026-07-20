#!/usr/bin/env python3
"""Score frozen semantic-support prompts without loading evaluation labels."""

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
    model_path = Path(fingerprint["path"])
    for name, expected in fingerprint["files"].items():
        path = model_path / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Frozen model file changed: {path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard specification")

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_FROZEN":
        raise ValueError("Semantic verifier protocol is not frozen")
    if freeze["input_hashes"]["scorer"] != sha256(Path(__file__).resolve()):
        raise ValueError("Scorer changed after protocol freeze")
    input_path = Path(freeze["output_paths"]["verifier_inputs"])
    if sha256(input_path) != freeze["output_hashes"]["verifier_inputs"]:
        raise ValueError("Frozen verifier inputs changed")
    model_path = verify_model(freeze["model_fingerprint"])

    all_rows = sorted(read_jsonl(input_path), key=lambda row: row["query_action_id"])
    rows = [row for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = {}
    for label in ("A", "B"):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Verifier label is not one token: {label} -> {ids}")
        label_ids[label] = ids[0]

    texts, token_counts = [], []
    max_tokens = int(freeze["verifier_contract"]["max_input_tokens"])
    for row in rows:
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(text, add_special_tokens=False))
        if count > max_tokens:
            raise ValueError(f"Prompt exceeds frozen token limit without truncation: {row['query_action_id']}={count}")
        texts.append(text)
        token_counts.append(count)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=freeze["verifier_contract"]["attn_implementation"],
        local_files_only=True,
    ).eval().cuda()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    score_path = output / f"v6_semantic_support_scores_shard{args.shard_index:02d}_of_{args.num_shards:02d}.jsonl"
    scored = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            batch_texts = texts[start : start + args.batch_size]
            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=False).to("cuda")
            logits = model(**encoded).logits[:, -1, :].float()
            pair = logits[:, [label_ids["A"], label_ids["B"]]]
            probabilities = torch.softmax(pair, dim=-1)[:, 0]
            differences = pair[:, 0] - pair[:, 1]
            for row, probability, difference in zip(batch_rows, probabilities.tolist(), differences.tolist()):
                if not math.isfinite(probability) or not math.isfinite(difference):
                    raise ValueError(f"Non-finite semantic support score: {row['query_action_id']}")
                scored.append({
                    "query_action_id": row["query_action_id"],
                    "parent_visual_state_id": row["parent_visual_state_id"],
                    "selected_evidence_action_id": row["selected_evidence_action_id"],
                    "sample_id": row["sample_id"],
                    "partition": row["partition"],
                    "semantic_support_logit": float(difference),
                    "semantic_support_raw_probability": float(probability),
                    "input_token_count": token_counts[len(scored)],
                    "prediction_source": "frozen_qwen2_5_vl_7b_next_token_ab",
                })
    with score_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "decision": "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_SHARD_COMPLETE",
        "freeze_path": str(freeze_path),
        "freeze_sha256": sha256(freeze_path),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rows": len(scored),
        "min_input_tokens": min(token_counts),
        "max_input_tokens": max(token_counts),
        "mean_input_tokens": sum(token_counts) / len(token_counts),
        "score_path": str(score_path),
        "score_sha256": sha256(score_path),
        "private_labels_loaded": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
    }
    manifest_path = output / "SHARD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
