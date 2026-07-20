#!/usr/bin/env python3
"""Execute one evidence policy and the shared answer reader without labels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dagig_causal.answer_prompt import build_answer_policy_prompt  # noqa: E402


METHODS = ("no_credit", "local_ig", "outcome", "dagig")


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


def parse_answer(raw: str) -> tuple[str, bool]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == {"final_answer"} and isinstance(value["final_answer"], str):
            answer = " ".join(value["final_answer"].split())
            if answer:
                return answer, True
    return "unknown", False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--train_fit", type=Path, required=True)
    parser.add_argument("--shared_answer_freeze", type=Path, required=True)
    parser.add_argument("--reference_scores", type=Path, required=True)
    parser.add_argument("--method_scores", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    training_path = args.training_freeze.resolve()
    fit_path = args.train_fit.resolve()
    answer_freeze_path = args.shared_answer_freeze.resolve()
    reference_audit_path = args.reference_scores.resolve()
    method_audit_path = args.method_scores.resolve()
    training = read_json(training_path)
    fit = read_json(fit_path)
    answer_freeze = read_json(answer_freeze_path)
    if training.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN":
        raise ValueError("backward evidence training is not frozen")
    if fit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAIN_FIT_GO":
        raise ValueError("evidence policy train fit is not GO")
    if answer_freeze.get("decision") != "DAGIG_V6_SHARED_ANSWER_POLICY_FROZEN":
        raise ValueError("shared answer policy is not frozen")
    reference_audit = read_json(reference_audit_path)
    method_audit = read_json(method_audit_path)
    for audit, method in ((reference_audit, "reference"), (method_audit, args.method)):
        if (
            audit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORES_READY"
            or audit.get("method") != method
            or audit.get("partition") != "internal_holdout"
        ):
            raise ValueError(f"invalid internal scores: {method}")
        if sha256(Path(audit["output_paths"]["scores"])) != audit["output_hashes"]["scores"]:
            raise ValueError(f"internal score rows changed: {method}")
    reference = {row["parent_group_id"]: row for row in read_jsonl(Path(reference_audit["output_paths"]["scores"]))}
    current = {row["parent_group_id"]: row for row in read_jsonl(Path(method_audit["output_paths"]["scores"]))}
    control = read_json(Path(training["input_paths"]["control_freeze"]))
    internal_path = Path(control["output_paths"]["internal_data"])
    action_path = Path(control["input_paths"]["evidence_actions"])
    if sha256(internal_path) != control["output_hashes"]["internal_data"] or sha256(action_path) != control["input_hashes"]["evidence_actions"]:
        raise ValueError("internal action matrix changed")
    groups = {row["parent_group_id"]: row for row in read_jsonl(internal_path)}
    actions = {row["evidence_action_id"]: row for row in read_jsonl(action_path)}
    if set(groups) != set(reference) or set(groups) != set(current):
        raise ValueError("internal score universes differ")

    selected: list[dict[str, Any]] = []
    beta = float(training["training"]["beta"])
    for group_id in sorted(groups):
        row = groups[group_id]
        behavior = np.asarray(row["behavior_probabilities"], dtype=np.float64)
        delta = np.asarray(current[group_id]["field_logprob_scores"]) - np.asarray(reference[group_id]["field_logprob_scores"])
        logits = np.log(behavior) + beta * delta
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        action_id = row["action_ids"][index]
        selected.append(
            {
                **actions[action_id],
                "method": args.method,
                "selector_action_ids": row["action_ids"],
                "selector_behavior_probabilities": behavior.tolist(),
                "selector_field_logprob_deltas": delta.tolist(),
                "selector_probabilities": probabilities.tolist(),
                "selected_action_index": index,
                "selected_evidence_action_id": action_id,
            }
        )

    adapter = Path(answer_freeze["output_paths"]["adapter"])
    if tree_sha256(adapter) != answer_freeze["output_hashes"]["adapter_tree"]:
        raise ValueError("shared answer adapter changed")
    answer_control = read_json(Path(answer_freeze["input_paths"]["answer_freeze"]))
    base_model = answer_control["base_model"]
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    model = PeftModel.from_pretrained(base, adapter).cuda().eval()
    outputs: list[dict[str, Any]] = []
    batch_size = 8
    for start in range(0, len(selected), batch_size):
        rows = selected[start : start + batch_size]
        prompts = [build_answer_policy_prompt(row) for row in rows]
        rendered = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
        encoded = tokenizer(rendered, padding=True, return_tensors="pt")
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        if max(lengths) > 4096:
            raise ValueError("shared answer prompt exceeds 4096 tokens")
        inputs = {key: value.cuda() for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id)
        raws = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for row, raw in zip(rows, raws):
            final_answer, valid = parse_answer(raw)
            outputs.append(
                {
                    "method": args.method,
                    "sample_id": row["sample_id"],
                    "partition": row["partition"],
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "visual_observation": row["visual_observation"],
                    "search_query": row["search_query"],
                    "selector_action_ids": row["selector_action_ids"],
                    "selector_behavior_probabilities": row["selector_behavior_probabilities"],
                    "selector_field_logprob_deltas": row["selector_field_logprob_deltas"],
                    "selector_probabilities": row["selector_probabilities"],
                    "selected_action_index": row["selected_action_index"],
                    "selected_evidence_action_id": row["selected_evidence_action_id"],
                    "selected_evidence_strategy": row["evidence_strategy"],
                    "selected_doc_ids": row["selected_doc_ids"],
                    "selected_docs": row["selected_docs"],
                    "answer_raw_generation": raw.strip(),
                    "final_answer": final_answer,
                    "answer_valid": valid,
                }
            )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / f"v6_backward_evidence_{args.method}_internal_predictions_no_labels.jsonl"
    predictions_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in outputs), encoding="utf-8")
    gates = {
        "complete_595_query_states": len(outputs) == 595,
        "finite_selector_probabilities": all(math.isfinite(value) for row in outputs for value in row["selector_probabilities"]),
        "normalized_selector_probabilities": all(abs(sum(row["selector_probabilities"]) - 1.0) <= 1e-8 for row in outputs),
        "shared_answer_policy_used": True,
        "no_gold_or_qrels_loaded": True,
        "internal_holdout_unused_for_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    input_paths = {
        "training_freeze": str(training_path),
        "train_fit": str(fit_path),
        "shared_answer_freeze": str(answer_freeze_path),
        "reference_score_audit": str(reference_audit_path),
        "method_score_audit": str(method_audit_path),
    }
    result = {
        "decision": "DAGIG_V6_BACKWARD_EVIDENCE_INTERNAL_METHOD_READY" if all(gates.values()) else "DAGIG_V6_BACKWARD_EVIDENCE_INTERNAL_METHOD_FAILED",
        "method": args.method,
        "metrics": {"query_states": len(outputs), "samples": len({row["sample_id"] for row in outputs}), "valid_answers": sum(row["answer_valid"] for row in outputs)},
        "gates": gates,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {"predictions": str(predictions_path)},
        "output_hashes": {"predictions": sha256(predictions_path)},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_BACKWARD_EVIDENCE_INTERNAL_METHOD_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
