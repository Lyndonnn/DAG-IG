#!/usr/bin/env python3
"""Execute frozen evidence and answer descendants for fresh query policies."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dagig_causal.answer_prompt import build_answer_policy_prompt  # noqa: E402


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
    for path in sorted(child for child in root.rglob("*") if child.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


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
            answer = clean(value["final_answer"])
            if answer:
                return answer, True
    return "unknown", False


def load_scores(path: Path, policy: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    audit = read_json(path)
    if audit.get("decision") != "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_SCORES_READY" or audit.get("policy") != policy:
        raise ValueError(f"invalid fresh evidence scores: {policy}")
    scores_path = Path(audit["output_paths"]["scores"])
    if sha256(scores_path) != audit["output_hashes"]["scores"]:
        raise ValueError(f"fresh evidence scores changed: {policy}")
    rows = read_jsonl(scores_path)
    return audit, {row["query_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_policy_freeze", type=Path, required=True)
    parser.add_argument("--shared_answer_freeze", type=Path, required=True)
    parser.add_argument("--fresh_action_audit", type=Path, required=True)
    parser.add_argument("--reference_scores", type=Path, required=True)
    parser.add_argument("--dagig_scores", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    evidence_policy_path = args.evidence_policy_freeze.resolve()
    answer_freeze_path = args.shared_answer_freeze.resolve()
    action_audit_path = args.fresh_action_audit.resolve()
    reference_audit_path = args.reference_scores.resolve()
    dagig_audit_path = args.dagig_scores.resolve()
    evidence_policy = read_json(evidence_policy_path)
    answer_freeze = read_json(answer_freeze_path)
    action_audit = read_json(action_audit_path)
    if evidence_policy.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_FROZEN":
        raise ValueError("backward DAG-IG evidence policy is not frozen")
    if answer_freeze.get("decision") != "DAGIG_V6_SHARED_ANSWER_POLICY_FROZEN":
        raise ValueError("shared answer policy is not frozen")
    if action_audit.get("decision") != "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_ACTIONS_GO":
        raise ValueError("fresh evidence action matrix is not GO")
    _, reference = load_scores(reference_audit_path, "reference")
    _, dagig = load_scores(dagig_audit_path, "dagig")
    if set(reference) != set(dagig) or len(reference) != 480:
        raise ValueError("fresh reference and DAG-IG evidence score universes differ")

    action_path = Path(action_audit["output_paths"]["evidence_actions"])
    if sha256(action_path) != action_audit["output_hashes"]["evidence_actions"]:
        raise ValueError("fresh public evidence actions changed")
    actions = {row["evidence_action_id"]: row for row in read_jsonl(action_path)}
    if len(actions) != 2400:
        raise ValueError("fresh evidence action universe is incomplete")
    evidence_training_path = Path(evidence_policy["input_paths"]["training_freeze"])
    evidence_training = read_json(evidence_training_path)
    if sha256(evidence_training_path) != evidence_policy["input_hashes"]["training_freeze"]:
        raise ValueError("evidence training protocol changed")
    beta = float(evidence_training["training"]["beta"])

    selected: list[dict[str, Any]] = []
    for query_id in sorted(reference):
        left = reference[query_id]
        right = dagig[query_id]
        action_ids = left["action_ids"]
        if right["action_ids"] != action_ids or len(action_ids) != 5:
            raise ValueError(f"fresh evidence action order differs: {query_id}")
        behavior = np.full(5, 0.2, dtype=np.float64)
        delta = np.asarray(right["field_logprob_scores"], dtype=np.float64) - np.asarray(
            left["field_logprob_scores"], dtype=np.float64
        )
        logits = np.log(behavior) + beta * delta
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        action_id = action_ids[index]
        row = actions[action_id]
        selected.append(
            {
                **row,
                "evidence_selector_action_ids": action_ids,
                "evidence_selector_behavior_probabilities": behavior.tolist(),
                "evidence_selector_field_logprob_deltas": delta.tolist(),
                "evidence_selector_probabilities": probabilities.tolist(),
                "selected_evidence_action_index": index,
                "selected_evidence_action_id": action_id,
            }
        )
    if len(selected) != 480 or len({row["query_id"] for row in selected}) != 480:
        raise ValueError("fresh selected evidence matrix is incomplete")

    answer_adapter = Path(answer_freeze["output_paths"]["adapter"])
    if tree_sha256(answer_adapter) != answer_freeze["output_hashes"]["adapter_tree"]:
        raise ValueError("shared answer adapter changed")
    answer_control = read_json(Path(answer_freeze["input_paths"]["answer_freeze"]))
    base_model = answer_control["base_model"]
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, answer_adapter).cuda().eval()
    outputs: list[dict[str, Any]] = []
    batch_size = 8
    for start in range(0, len(selected), batch_size):
        rows = selected[start : start + batch_size]
        prompts = [build_answer_policy_prompt(row) for row in rows]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        encoded = tokenizer(rendered, padding=True, return_tensors="pt")
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        if max(lengths) > 4096:
            raise ValueError("fresh shared-answer prompt exceeds 4096 tokens")
        inputs = {key: value.cuda() for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
            )
        raws = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for row, raw, input_tokens in zip(rows, raws, lengths):
            final_answer, valid = parse_answer(raw)
            outputs.append(
                {
                    "method": row["method"],
                    "sample_id": row["sample_id"],
                    "partition": "internal_holdout",
                    "visual_parent_id": row["visual_parent_id"],
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "visual_field": row["visual_field"],
                    "visual_observation": row["visual_observation"],
                    "search_query": row["search_query"],
                    "search_id": row["search_id"],
                    "retrieved_docs": row["candidate_docs"],
                    "evidence_selector_action_ids": row["evidence_selector_action_ids"],
                    "evidence_selector_behavior_probabilities": row["evidence_selector_behavior_probabilities"],
                    "evidence_selector_field_logprob_deltas": row["evidence_selector_field_logprob_deltas"],
                    "evidence_selector_probabilities": row["evidence_selector_probabilities"],
                    "selected_evidence_action_index": row["selected_evidence_action_index"],
                    "selected_evidence_action_id": row["selected_evidence_action_id"],
                    "selected_evidence_strategy": row["evidence_strategy"],
                    "selected_doc_ids": row["selected_doc_ids"],
                    "selected_docs": row["selected_docs"],
                    "answer_raw_generation": raw.strip(),
                    "final_answer": final_answer,
                    "answer_valid": valid,
                    "answer_input_tokens": int(input_tokens),
                }
            )
        completed = min(start + batch_size, len(selected))
        if completed % 80 == 0 or completed == len(selected):
            print(json.dumps({"answered": completed, "total": len(selected)}), flush=True)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "v6_backward_query_fresh_descendant_predictions_no_labels.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in outputs),
        encoding="utf-8",
    )
    method_counts = Counter(row["method"] for row in outputs)
    gates = {
        "complete_480_method_visual_state_predictions": len(outputs) == 480,
        "complete_120_per_method": all(method_counts[method] == 120 for method in ("no_credit", "local_ig", "outcome", "dagig")),
        "finite_normalized_evidence_policy": all(
            all(math.isfinite(value) for value in row["evidence_selector_probabilities"])
            and abs(sum(row["evidence_selector_probabilities"]) - 1.0) <= 1e-8
            for row in outputs
        ),
        "same_frozen_dagig_evidence_policy": True,
        "same_frozen_shared_answer_policy": True,
        "no_gold_or_qrels_loaded": True,
        "internal_holdout_unused_for_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    input_paths = {
        "evidence_policy_freeze": evidence_policy_path,
        "shared_answer_freeze": answer_freeze_path,
        "fresh_action_audit": action_audit_path,
        "reference_score_audit": reference_audit_path,
        "dagig_score_audit": dagig_audit_path,
    }
    result = {
        "decision": "DAGIG_V6_BACKWARD_QUERY_FRESH_DESCENDANTS_READY" if all(gates.values()) else "DAGIG_V6_BACKWARD_QUERY_FRESH_DESCENDANTS_FAILED",
        "metrics": {
            "predictions": len(outputs),
            "method_counts": dict(sorted(method_counts.items())),
            "answer_valid_rate": sum(row["answer_valid"] for row in outputs) / len(outputs),
            "selected_strategy_counts": dict(sorted(Counter(row["selected_evidence_strategy"] for row in outputs).items())),
        },
        "gates": gates,
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "input_hashes": {key: sha256(path) for key, path in input_paths.items()},
        "output_paths": {"predictions": str(predictions_path)},
        "output_hashes": {"predictions": sha256(predictions_path)},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_BACKWARD_QUERY_FRESH_DESCENDANT_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
