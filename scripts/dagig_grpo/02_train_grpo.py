#!/usr/bin/env python3
"""Minimal single-GPU Qwen2.5-VL LoRA GRPO trainer for DAG-IG.

This intentionally avoids depending on TRL. It implements grouped sampled
rollouts with reward-normalized advantages and a frozen reference model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scripts.dagig_grpo.grpo_utils import (
    BM25Index,
    DEFAULT_ASSET_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    anchor_terms,
    assert_safe_student_model,
    answer_leaks_in_query,
    answer_match_details,
    build_user_messages,
    compute_reward,
    extract_json_object,
    load_corpus,
    parse_policy_output,
    query_quality_penalty,
    read_jsonl,
    resolve_model_path,
    stringify,
    support_rank,
    tokenize,
    write_json,
)
from scripts.dagig_7b_extension.audit_qwen25vl7b_reward_v2_rescore import (
    compute_v2_reward as compute_two_stage_reward_v2,
)
from scripts.dagig_7b_extension.reward_v3_utils import (
    DEFAULT_REWARD_V3_SUMMARY,
    DEFAULT_VERIFIER_MODEL,
    compute_v3_reward as compute_two_stage_reward_v3,
    load_reward_v3_state,
)

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # noqa: BLE001
    process_vision_info = None
    QWEN_VL_UTILS_IMPORT_ERROR = exc
else:
    QWEN_VL_UTILS_IMPORT_ERROR = None


STAGE1_PROMPT = """You are a multimodal evidence-search agent.
Given an image and a question, return JSON only with exactly:
{
  "visual_observation": "brief visual evidence you used",
  "search_query": "one concise search query for retrieving supporting evidence"
}
Do not output the final answer. Do not include reasoning, evidence lists, markdown, or extra text.
Do not include the final answer inside the search_query unless it is unavoidable from the question itself."""


READER_PROMPT = """You are an evidence-grounded answer reader.
Given an image, a question, and retrieved evidence documents, return JSON only with exactly:
{
  "final_answer": "short final answer"
}
Use the retrieved evidence when it supports an answer. Keep the answer concise.
Do not output reasoning, citations, search queries, markdown, or extra text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--corpus_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--init_adapter_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=[
        "outcome_grpo",
        "trajectory_grpo",
        "dagig_grpo_no_visual",
        "dagig_grpo_full",
        "dagig_grpo_no_query",
        "dagig_grpo_no_evidence",
        "paper_main_v1",
    ], required=True)
    parser.add_argument("--attn_impl", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means full epoch schedule")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--reader_max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_pixels", type=int, default=1003520)
    parser.add_argument("--two_stage_rollout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--two_stage_loss_scope",
        choices=["both", "stage1", "reader"],
        default="both",
        help="For --two_stage_rollout, choose which generated segment receives GRPO logprob/KL loss.",
    )
    parser.add_argument(
        "--two_stage_reward_version",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="For --two_stage_rollout, choose the reward implementation. Default v1 preserves prior runs.",
    )
    parser.add_argument("--reward_v3_alpha", type=float, default=0.01)
    parser.add_argument("--reward_v3_verifier_model", type=Path, default=DEFAULT_VERIFIER_MODEL)
    parser.add_argument("--reward_v3_summary", type=Path, default=DEFAULT_REWARD_V3_SUMMARY)
    parser.add_argument(
        "--allow_reward_v3_ablation",
        action="store_true",
        help="Required for two_stage_reward_version=v3. v3 is not a mainline 7B reward.",
    )
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    return parser.parse_args()


def load_policy_and_ref(args: argparse.Namespace):
    model_path = resolve_model_path(args.model_name_or_path)
    assert_safe_student_model(model_path)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    base_kwargs = dict(
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
        local_files_only=os.path.isdir(model_path),
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=os.path.isdir(model_path))
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    print("loading policy base model")
    policy_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **base_kwargs)
    policy_base.config.use_cache = False
    policy = PeftModel.from_pretrained(policy_base, str(args.init_adapter_path), is_trainable=True)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        if hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()
    policy.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    policy.train()
    policy.print_trainable_parameters()

    print("loading reference base model")
    ref_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **base_kwargs)
    ref_base.config.use_cache = False
    ref = PeftModel.from_pretrained(ref_base, str(args.init_adapter_path), is_trainable=False)
    ref.to(policy.device)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    return policy, ref, processor, model_path


def generate_once(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    max_new_tokens: int | None = None,
) -> str:
    if process_vision_info is None:
        raise RuntimeError(f"qwen-vl-utils import failed: {QWEN_VL_UTILS_IMPORT_ERROR}")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    seq_len = int(inputs["input_ids"].shape[1])
    if args.max_seq_length > 0 and seq_len > args.max_seq_length:
        raise ValueError(f"input length {seq_len} exceeds max_seq_length {args.max_seq_length}")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens if max_new_tokens is None else max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    trimmed = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def image_message(image_path: str, text: str, max_pixels: int) -> list[dict[str, Any]]:
    image_content: dict[str, Any] = {"type": "image", "image": str(image_path)}
    if max_pixels and max_pixels > 0:
        image_content["max_pixels"] = int(max_pixels)
    return [{"role": "user", "content": [image_content, {"type": "text", "text": text}]}]


def build_stage1_messages(row: dict[str, Any], max_pixels: int) -> list[dict[str, Any]]:
    text = f"{STAGE1_PROMPT.strip()}\n\nQuestion: {str(row.get('question', '')).strip()}"
    return image_message(str(row["image_abs_path"]), text, max_pixels=max_pixels)


def format_docs(docs: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for doc in docs:
        text = str(doc.get("text", "")).strip()
        if len(text) > 1200:
            text = text[:1200].rstrip() + " ..."
        blocks.append(
            "\n".join(
                [
                    f"[Doc {doc.get('rank')}]",
                    f"Title: {str(doc.get('title', '')).strip()}",
                    f"URL: {str(doc.get('url', '')).strip()}",
                    f"Text: {text}",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "[No retrieved documents]"


def build_reader_messages(row: dict[str, Any], docs: list[dict[str, Any]], max_pixels: int) -> list[dict[str, Any]]:
    text = "\n\n".join(
        [
            READER_PROMPT.strip(),
            f"Question: {str(row.get('question', '')).strip()}",
            "Retrieved evidence:",
            format_docs(docs),
        ]
    )
    return image_message(str(row["image_abs_path"]), text, max_pixels=max_pixels)


def parse_reader_answer(raw: str) -> dict[str, Any]:
    obj = extract_json_object(raw)
    answer = ""
    if obj:
        answer = stringify(obj.get("final_answer") or obj.get("answer") or obj.get("pred_answer"))
    if not answer:
        parsed = parse_policy_output(raw)
        answer = parsed["final_answer"]
    if not answer:
        stripped = str(raw or "").strip()
        answer = stripped.splitlines()[0].strip(" `\"'") if stripped else ""
    return {"final_answer": answer, "parsed_json": obj is not None}


def compute_two_stage_reward(
    row: dict[str, Any],
    stage1_text: str,
    reader_text: str,
    retrieved: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    stage1 = parse_policy_output(stage1_text)
    reader = parse_reader_answer(reader_text)
    visual = stage1["visual_observation"].strip()
    query = stage1["search_query"].strip()
    answer = reader["final_answer"].strip()
    rank5 = support_rank(retrieved, str(row.get("sample_id")), top_k)
    rank10 = support_rank(retrieved, str(row.get("sample_id")), 10)
    evidence_supported = rank5 is not None
    answer_match = answer_match_details(answer, str(row.get("gold_answer") or row.get("answer") or ""))
    answer_correct = bool(answer_match["answer_correct"])
    strict_success = bool(answer_correct and evidence_supported)

    format_credit = 0.0
    format_credit += 0.03 if stage1["parsed_json"] else 0.0
    format_credit += 0.02 if visual else 0.0
    format_credit += 0.03 if query else 0.0
    format_credit += 0.02 if reader["parsed_json"] and answer else 0.0
    format_credit = min(format_credit, 0.10)

    visual_terms = anchor_terms(row)
    visual_tokens = set(tokenize(visual))
    visual_overlap = len(visual_terms & visual_tokens)
    visual_credit = min(1.0, visual_overlap / max(3, min(8, len(visual_terms)))) if visual else 0.0
    query_credit = 1.0 / rank10 if rank10 else 0.0
    evidence_credit = 1.0 / rank5 if rank5 else 0.0
    answer_credit = 1.0 if strict_success else (0.35 if answer_correct else 0.0)
    leak_penalty = 0.25 if answer_leaks_in_query(query, str(row.get("gold_answer") or row.get("answer") or "")) else 0.0
    path_penalty = query_quality_penalty(query)
    total = (
        0.10 * format_credit
        + 0.15 * visual_credit
        + 0.40 * query_credit
        + 0.25 * evidence_credit
        + 0.35 * answer_credit
        - leak_penalty
        - path_penalty
    )
    parsed = {
        "raw": json.dumps(
            {
                "stage1": stage1_text,
                "reader": reader_text,
            },
            ensure_ascii=False,
        ),
        "parsed_json": bool(stage1["parsed_json"] and reader["parsed_json"]),
        "visual_observation": visual,
        "search_query": query,
        "final_answer": answer,
        "stage1_raw": stage1_text,
        "reader_raw": reader_text,
        "stage1_parsed_json": bool(stage1["parsed_json"]),
        "reader_parsed_json": bool(reader["parsed_json"]),
    }
    return {
        "reward": float(max(-0.5, total)),
        "components": {
            "format": float(format_credit),
            "visual": float(visual_credit),
            "query": float(query_credit),
            "evidence": float(evidence_credit),
            "answer": float(answer_credit),
            "leakage_penalty": float(-leak_penalty),
            "path_penalty": float(-path_penalty),
        },
        "parsed": parsed,
        "retrieved_docs": retrieved[:top_k],
        "retrieval_hit": bool(evidence_supported),
        "evidence_supported": bool(evidence_supported),
        "answer_correct": bool(answer_correct),
        "strict_success": bool(strict_success),
        "answer_match": answer_match,
        "visual_anchor_overlap": int(visual_overlap),
        "support_rank5": rank5,
        "support_rank10": rank10,
    }


def encode_response(
    processor: Any,
    prompt_messages: list[dict[str, Any]],
    response_text: str,
    max_seq_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if process_vision_info is None:
        raise RuntimeError(f"qwen-vl-utils import failed: {QWEN_VL_UTILS_IMPORT_ERROR}")
    messages = list(prompt_messages) + [{"role": "assistant", "content": response_text}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    seq_len = int(batch["input_ids"].shape[1])
    if max_seq_length > 0 and seq_len > max_seq_length:
        raise ValueError(f"response batch length {seq_len} exceeds max_seq_length {max_seq_length}")

    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_messages)
    prompt_batch = processor(text=[prompt_text], images=prompt_image_inputs, videos=prompt_video_inputs, padding=True, return_tensors="pt")
    prompt_len = int(prompt_batch["attention_mask"].sum(dim=1).item())
    labels = batch["input_ids"].clone()
    labels[:, : min(prompt_len, labels.shape[1])] = -100
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    batch["labels"] = labels
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def sequence_logps(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    target = labels[:, 1:]
    mask = target != -100
    safe_target = target.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)
    seq_logps = (log_probs * mask).sum(dim=-1)
    token_counts = mask.sum(dim=-1).clamp(min=1)
    return seq_logps, token_counts


def save_checkpoint(model: torch.nn.Module, processor: Any, output_dir: Path, step: int) -> None:
    ckpt = output_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    processor.save_pretrained(str(ckpt))


def main() -> None:
    args = parse_args()
    if args.two_stage_reward_version == "v3" and not args.allow_reward_v3_ablation:
        raise ValueError(
            "reward_v3 is an optional/future ablation only and must not be used as the "
            "main 7B reward. Pass --allow_reward_v3_ablation only for an explicit "
            "non-mainline ablation run."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = (args.asset_root / "prompts/grpo_policy_prompt.txt").read_text(encoding="utf-8")
    rows = read_jsonl(args.train_file)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"No training rows in {args.train_file}")
    corpus = load_corpus(args.corpus_path)
    if not corpus:
        raise ValueError(f"Empty corpus: {args.corpus_path}")
    bm25 = BM25Index.from_docs(corpus)
    reward_v3_state = None
    if args.two_stage_reward_version == "v3":
        reward_v3_state = load_reward_v3_state(
            verifier_model_path=args.reward_v3_verifier_model,
            reward_v3_summary_path=args.reward_v3_summary,
            alpha=args.reward_v3_alpha,
        )
    policy, ref, processor, resolved_model_path = load_policy_and_ref(args)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.learning_rate)
    config = {
        "variant": args.variant,
        "train_file": str(args.train_file.resolve()),
        "corpus_path": str(args.corpus_path.resolve()),
        "model_name_or_path": resolved_model_path,
        "init_adapter_path": str(args.init_adapter_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "num_train_epochs": args.num_train_epochs,
        "num_generations": args.num_generations,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "kl_coef": args.kl_coef,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "reader_max_new_tokens": args.reader_max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_pixels": args.max_pixels,
        "two_stage_rollout": args.two_stage_rollout,
        "two_stage_loss_scope": args.two_stage_loss_scope,
        "two_stage_reward_version": args.two_stage_reward_version,
        "reward_v3_alpha": args.reward_v3_alpha,
        "reward_v3_verifier_model": str(args.reward_v3_verifier_model),
        "reward_v3_summary": str(args.reward_v3_summary),
        "reward_v3_state": reward_v3_state,
        "student_model_safety": "Qwen32B/teacher model names are rejected.",
        "trainer": "custom_grouped_sampled_grpo",
    }
    write_json(args.output_dir / "grpo_run_config.json", config)

    train_log = (args.output_dir / "grpo_train_log.jsonl").open("w", encoding="utf-8")
    rollout_log = (args.output_dir / "reward_rollouts.jsonl").open("w", encoding="utf-8")
    total_steps = 0
    micro_steps = 0
    constant_reward_groups = 0
    started = time.perf_counter()
    max_epochs = max(1, int(args.num_train_epochs + 0.999))
    policy.zero_grad(set_to_none=True)
    try:
        for epoch in range(max_epochs):
            if epoch >= args.num_train_epochs:
                break
            random.shuffle(rows)
            for row in rows:
                prompt_messages = build_user_messages(row["image_abs_path"], row["question"], prompt_text, max_pixels=args.max_pixels)
                stage1_messages = build_stage1_messages(row, max_pixels=args.max_pixels) if args.two_stage_rollout else None
                policy.eval()
                completions: list[Any] = []
                if args.two_stage_rollout:
                    assert stage1_messages is not None
                    for _ in range(args.num_generations):
                        stage1_text = generate_once(policy, processor, stage1_messages, args, max_new_tokens=args.max_new_tokens)
                        stage1_parsed = parse_policy_output(stage1_text)
                        query = stage1_parsed["search_query"].strip()
                        retrieved = bm25.search(query, top_k=max(10, args.top_k)) if query else []
                        reader_docs = retrieved[: args.top_k]
                        reader_messages = build_reader_messages(row, reader_docs, max_pixels=args.max_pixels)
                        reader_text = generate_once(
                            policy,
                            processor,
                            reader_messages,
                            args,
                            max_new_tokens=args.reader_max_new_tokens,
                        )
                        completions.append(
                            {
                                "stage1_text": stage1_text,
                                "reader_text": reader_text,
                                "reader_messages": reader_messages,
                                "retrieved": retrieved,
                            }
                        )
                else:
                    completions = [generate_once(policy, processor, prompt_messages, args) for _ in range(args.num_generations)]
                policy.train()
                if args.two_stage_rollout:
                    reward_infos = []
                    for item in completions:
                        info = compute_two_stage_reward(
                            row,
                            item["stage1_text"],
                            item["reader_text"],
                            item["retrieved"],
                            top_k=args.top_k,
                        )
                        if args.two_stage_reward_version == "v2":
                            v2 = compute_two_stage_reward_v2(
                                row,
                                {
                                    "parsed": info["parsed"],
                                    "answer_correct": info["answer_correct"],
                                },
                                bm25,
                                top_k=args.top_k,
                            )
                            info["reward"] = v2["reward_v2"]
                            info["components"] = v2["components_v2"]
                            info["reward_v2_diagnostics"] = v2["diagnostics_v2"]
                            info["retrieval_hit"] = bool(v2["diagnostics_v2"]["retrieval_hit"])
                            info["evidence_supported"] = bool(v2["diagnostics_v2"]["retrieval_hit"])
                            info["strict_success"] = bool(v2["diagnostics_v2"]["strict_success"])
                        elif args.two_stage_reward_version == "v3":
                            assert reward_v3_state is not None
                            v3 = compute_two_stage_reward_v3(row, info, reward_v3_state)
                            info["reward"] = v3["reward_v3"]
                            info["components"] = v3["components_v3"]
                            info["reward_v3_diagnostics"] = v3["diagnostics_v3"]
                        reward_infos.append(info)
                else:
                    reward_infos = [compute_reward(row, text, bm25, variant=args.variant, top_k=args.top_k) for text in completions]
                rewards = torch.tensor([float(info["reward"]) for info in reward_infos], dtype=torch.float32, device=policy.device)
                if float(rewards.max() - rewards.min()) < 1e-8:
                    constant_reward_groups += 1
                    advantages = torch.zeros_like(rewards)
                else:
                    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)

                group_loss = 0.0
                group_kl = 0.0
                response_tokens = 0.0
                for gen_idx, (completion, info, adv) in enumerate(zip(completions, reward_infos, advantages.tolist())):
                    if args.two_stage_rollout:
                        assert stage1_messages is not None
                        stage1_batch = encode_response(
                            processor,
                            stage1_messages,
                            completion["stage1_text"],
                            args.max_seq_length,
                            policy.device,
                        )
                        reader_batch = encode_response(
                            processor,
                            completion["reader_messages"],
                            completion["reader_text"],
                            args.max_seq_length,
                            policy.device,
                        )
                        policy_logp_1, token_count_1 = sequence_logps(policy, stage1_batch)
                        policy_logp_2, token_count_2 = sequence_logps(policy, reader_batch)
                        with torch.no_grad():
                            ref_logp_1, _ = sequence_logps(ref, stage1_batch)
                            ref_logp_2, _ = sequence_logps(ref, reader_batch)
                        if args.two_stage_loss_scope == "stage1":
                            policy_logp = policy_logp_1
                            ref_logp = ref_logp_1
                            token_count = token_count_1
                        elif args.two_stage_loss_scope == "reader":
                            policy_logp = policy_logp_2
                            ref_logp = ref_logp_2
                            token_count = token_count_2
                        else:
                            policy_logp = policy_logp_1 + policy_logp_2
                            ref_logp = ref_logp_1 + ref_logp_2
                            token_count = token_count_1 + token_count_2
                    else:
                        batch = encode_response(processor, prompt_messages, completion, args.max_seq_length, policy.device)
                        policy_logp, token_count = sequence_logps(policy, batch)
                        with torch.no_grad():
                            ref_logp, _ = sequence_logps(ref, batch)
                    logp_per_token = policy_logp / token_count
                    kl_per_token = (policy_logp - ref_logp) / token_count
                    loss = -(float(adv) * logp_per_token).mean() + args.kl_coef * kl_per_token.mean()
                    group_loss = group_loss + loss
                    group_kl += float(kl_per_token.detach().cpu().mean())
                    response_tokens += float(token_count.detach().float().cpu().mean())
                    rollout_log.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "micro_step": micro_steps + 1,
                                "sample_id": row.get("sample_id"),
                                "generation_index": gen_idx,
                                "reward": info["reward"],
                                "advantage": float(adv),
                                "components": info["components"],
                                "parsed": info["parsed"],
                                "retrieval_hit": info["retrieval_hit"],
                                "answer_correct": info["answer_correct"],
                                "strict_success": info["strict_success"],
                                "reward_v2_diagnostics": info.get("reward_v2_diagnostics", {}),
                                "reward_v3_diagnostics": info.get("reward_v3_diagnostics", {}),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                rollout_log.flush()
                loss = group_loss / max(1, args.num_generations)
                (loss / args.gradient_accumulation_steps).backward()
                micro_steps += 1
                do_step = micro_steps % args.gradient_accumulation_steps == 0
                if do_step:
                    torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    total_steps += 1
                    record = {
                        "step": total_steps,
                        "epoch": epoch,
                        "micro_steps": micro_steps,
                        "loss": float(loss.detach().cpu()),
                        "reward_mean": float(rewards.detach().cpu().mean()),
                        "reward_std": float(rewards.detach().cpu().std(unbiased=False)),
                        "reward_min": float(rewards.detach().cpu().min()),
                        "reward_max": float(rewards.detach().cpu().max()),
                        "avg_kl_per_token": group_kl / max(1, args.num_generations),
                        "avg_response_tokens": response_tokens / max(1, args.num_generations),
                        "constant_reward_groups": constant_reward_groups,
                        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3) if torch.cuda.is_available() else 0.0,
                    }
                    train_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    train_log.flush()
                    if total_steps % args.logging_steps == 0:
                        print(json.dumps(record, ensure_ascii=False))
                    if args.save_steps and total_steps % args.save_steps == 0:
                        save_checkpoint(policy, processor, args.output_dir, total_steps)
                    if args.max_steps > 0 and total_steps >= args.max_steps:
                        break
            if args.max_steps > 0 and total_steps >= args.max_steps:
                break
    finally:
        train_log.close()
        rollout_log.close()

    if total_steps == 0:
        raise RuntimeError("GRPO trainer finished without optimizer steps.")
    policy.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    summary = {
        "status": "success",
        "variant": args.variant,
        "optimizer_steps": total_steps,
        "micro_steps": micro_steps,
        "constant_reward_groups": constant_reward_groups,
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(args.output_dir.resolve()),
        "max_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3) if torch.cuda.is_available() else 0.0,
    }
    if constant_reward_groups >= micro_steps:
        summary["status"] = "failed_constant_rewards"
        write_json(args.output_dir / "grpo_train_summary.json", summary)
        raise RuntimeError("All reward groups were constant; refusing to treat GRPO as valid training.")
    write_json(args.output_dir / "grpo_train_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
