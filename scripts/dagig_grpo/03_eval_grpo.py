#!/usr/bin/env python3
"""Evaluate DAG-IG GRPO models in the frozen offline Pix2Fact environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scripts.dagig_grpo.grpo_utils import (
    BM25Index,
    DEFAULT_ASSET_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    answer_leaks_in_query,
    assert_safe_student_model,
    build_user_messages,
    compute_reward,
    load_corpus,
    parse_policy_output,
    read_jsonl,
    resolve_model_path,
    tokenize,
    write_json,
    write_jsonl,
)

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # noqa: BLE001
    process_vision_info = None
    QWEN_VL_UTILS_IMPORT_ERROR = exc
else:
    QWEN_VL_UTILS_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--corpus_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_path", type=Path, default=None)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--attn_impl", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--max_pixels", type=int, default=1003520)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward_variant", default="dagig_grpo_full")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    model_path = resolve_model_path(args.model_name_or_path)
    assert_safe_student_model(model_path)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=os.path.isdir(model_path),
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
        local_files_only=os.path.isdir(model_path),
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model, processor


def generate_once(model: Any, processor: Any, messages: list[dict[str, Any]], max_new_tokens: int, max_input_tokens: int) -> tuple[str, str | None]:
    if process_vision_info is None:
        raise RuntimeError(f"qwen-vl-utils import failed: {QWEN_VL_UTILS_IMPORT_ERROR}")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    seq_len = int(inputs["input_ids"].shape[1])
    if max_input_tokens > 0 and seq_len > max_input_tokens:
        return "", f"input_tokens_exceed_limit:{seq_len}>{max_input_tokens}"
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    trimmed = generated[:, inputs["input_ids"].shape[-1] :]
    text_out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text_out.strip(), None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    denom = max(1, n)

    def rate(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / denom

    buckets = {
        "format_failure": sum(1 for row in rows if not row.get("format_parse_success")),
        "visual_miss": sum(1 for row in rows if row.get("format_parse_success") and row.get("visual_anchor_overlap", 0) <= 0),
        "query_miss": sum(1 for row in rows if row.get("query_nonempty") and not row.get("retrieval_top5_hit")),
        "retrieval_miss": sum(1 for row in rows if not row.get("retrieval_top5_hit")),
        "retrieval_hit_answer_wrong": sum(1 for row in rows if row.get("retrieval_top5_hit") and not row.get("answer_correct")),
    }
    avg_components: dict[str, float] = {}
    component_keys = set()
    for row in rows:
        component_keys.update((row.get("reward_components") or {}).keys())
    for key in sorted(component_keys):
        avg_components[key] = sum(float((row.get("reward_components") or {}).get(key, 0.0)) for row in rows) / denom
    return {
        "n": n,
        "format_parse_success": rate("format_parse_success"),
        "query_nonempty_rate": rate("query_nonempty"),
        "answer_in_query_rate": rate("answer_in_query"),
        "retrieval_top1_hit": rate("retrieval_top1_hit"),
        "retrieval_top3_hit": rate("retrieval_top3_hit"),
        "retrieval_top5_hit": rate("retrieval_top5_hit"),
        "answer_correct": rate("answer_correct"),
        "evidence_supported": rate("evidence_supported"),
        "strict_success": rate("strict_success"),
        "avg_query_len": sum(len(tokenize(row.get("search_query", ""))) for row in rows) / denom,
        "avg_reward": sum(float(row.get("reward", 0.0)) for row in rows) / denom,
        "avg_reward_components": avg_components,
        "breakdown": buckets,
        "invalid_count": sum(1 for row in rows if row.get("error")),
    }


def main() -> None:
    args = parse_args()
    prompt_text = (args.asset_root / "prompts/grpo_policy_prompt.txt").read_text(encoding="utf-8")
    rows = read_jsonl(args.eval_file)
    if args.eval_samples:
        rows = rows[: args.eval_samples]
    corpus = load_corpus(args.corpus_path)
    bm25 = BM25Index.from_docs(corpus)
    model, processor = load_model(args)

    predictions: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        started = time.perf_counter()
        error = None
        raw = ""
        try:
            raw, error = generate_once(
                model,
                processor,
                build_user_messages(row["image_abs_path"], row["question"], prompt_text, max_pixels=args.max_pixels),
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"generation_error:{type(exc).__name__}:{exc}"
        reward_info = compute_reward(row, raw, bm25, variant=args.reward_variant, top_k=args.top_k)
        parsed = parse_policy_output(raw)
        retrieved = reward_info["retrieved_docs"]
        rec = {
            "sample_id": row.get("sample_id"),
            "split": args.split,
            "question": row.get("question"),
            "gold_answer": row.get("gold_answer"),
            "image_path": row.get("image_abs_path"),
            "raw_generation": raw,
            "visual_observation": parsed["visual_observation"],
            "search_query": parsed["search_query"],
            "final_answer": parsed["final_answer"],
            "format_parse_success": bool(parsed["parsed_json"] and parsed["search_query"] and parsed["final_answer"]),
            "query_nonempty": bool(parsed["search_query"]),
            "answer_in_query": answer_leaks_in_query(parsed["search_query"], str(row.get("gold_answer", ""))),
            "retrieved_docs": retrieved,
            "retrieval_top1_hit": any(str(doc.get("sample_id")) == str(row.get("sample_id")) and doc.get("is_gold") for doc in retrieved[:1]),
            "retrieval_top3_hit": any(str(doc.get("sample_id")) == str(row.get("sample_id")) and doc.get("is_gold") for doc in retrieved[:3]),
            "retrieval_top5_hit": bool(reward_info["retrieval_hit"]),
            "answer_correct": bool(reward_info["answer_correct"]),
            "evidence_supported": bool(reward_info["evidence_supported"]),
            "strict_success": bool(reward_info["strict_success"]),
            "answer_match": reward_info["answer_match"],
            "visual_anchor_overlap": reward_info["visual_anchor_overlap"],
            "reward": reward_info["reward"],
            "reward_components": reward_info["components"],
            "error": error,
            "elapsed_seconds": time.perf_counter() - started,
        }
        predictions.append(rec)
        print(
            f"[{idx}/{len(rows)}] {rec['sample_id']} fmt={rec['format_parse_success']} "
            f"r5={rec['retrieval_top5_hit']} ans={rec['answer_correct']} strict={rec['strict_success']} "
            f"q={rec['search_query'][:80]!r} a={rec['final_answer'][:60]!r}"
        )

    pred_path = args.output_root / "predictions" / f"{args.model_tag}_{args.split}.jsonl"
    metric_path = args.output_root / "metrics" / f"{args.model_tag}_{args.split}.json"
    summary = summarize(predictions)
    summary.update(
        {
            "model_tag": args.model_tag,
            "split": args.split,
            "eval_file": str(args.eval_file.resolve()),
            "corpus_path": str(args.corpus_path.resolve()),
            "adapter_path": str(args.adapter_path) if args.adapter_path else "",
            "reward_variant": args.reward_variant,
            "top_k": args.top_k,
        }
    )
    write_jsonl(pred_path, predictions)
    write_json(metric_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

