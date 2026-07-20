#!/usr/bin/env python3
"""Score categorical reference/DAG-IG evidence policies on fresh queries."""

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
from dagig_causal.evidence_prompt import build_evidence_selection_prompt  # noqa: E402


STRATEGY_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)
LABELS = tuple("ABCDE")


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


def source_completion(row: dict[str, Any]) -> str:
    docs = sorted(row["candidate_docs"], key=lambda doc: int(doc["rank"]))
    mapping = {doc["doc_id"]: f"D{index}" for index, doc in enumerate(docs, 1)}
    selected = [mapping[doc_id] for doc_id in row["selected_doc_ids"]]
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("fresh evidence action must select exactly three unique docs")
    return json.dumps({"selected_evidence_ids": selected}, ensure_ascii=False, separators=(",", ":"))


def transform_prompt(source: str, evidence_sets: list[list[str]]) -> str:
    start = source.find("Question:")
    if start < 0:
        raise ValueError("fresh source evidence prompt has no Question field")
    body = source[start:].rstrip()
    suffix = "Evidence selection:"
    if body.endswith(suffix):
        body = body[: -len(suffix)].rstrip()
    actions = "\n".join(
        f"[{label}] selected_evidence_ids={json.dumps(ids, ensure_ascii=False, separators=(',', ':'))}"
        for label, ids in zip(LABELS, evidence_sets)
    )
    return (
        "You are the evidence-selection node of a multimodal web-search agent.\n\n"
        "Exactly five legal evidence-set actions are listed below. Select one action using the question, "
        "frozen visual observation, executed query, and retrieved document content.\n\n"
        "Return only compact valid JSON with exactly one field, for example {\"action\":\"A\"}. "
        "The action must be one of A, B, C, D, or E. Do not answer the question and do not add reasoning.\n\n"
        f"{body}\n\nLegal candidate evidence-set actions:\n{actions}\n\nEvidence action:"
    )


def categorical_prompt(rows: list[dict[str, Any]]) -> str:
    evidence_sets = [json.loads(source_completion(row))["selected_evidence_ids"] for row in rows]
    return transform_prompt(build_evidence_selection_prompt(rows[0]), evidence_sets)


def completion(label: str) -> str:
    return json.dumps({"action": label}, separators=(",", ":"))


def field_token_mask(tokenizer: Any, value: str) -> tuple[list[int], list[int]]:
    parsed = json.loads(value)
    if set(parsed) != {"action"} or parsed["action"] not in LABELS:
        raise ValueError("fresh categorical evidence completion schema changed")
    marker = json.dumps("action") + ":"
    start = value.find(marker)
    if start < 0:
        raise ValueError("categorical action marker missing")
    value_start = start + len(marker)
    serialized = json.dumps(parsed["action"], ensure_ascii=False, separators=(",", ":"))
    value_end = value_start + len(serialized)
    if value[value_start:value_end] != serialized:
        raise ValueError("categorical action field span mismatch")
    action_start = value_start + 1
    action_end = value_end - 1
    encoded = tokenizer(value, add_special_tokens=False, return_offsets_mapping=True)
    mask = [int(end > action_start and begin < action_end) for begin, end in encoded["offset_mapping"]]
    if sum(mask) != 1:
        raise ValueError("fresh categorical action must score exactly one token")
    return list(encoded["input_ids"]), mask


def build_batch(
    tokenizer: Any,
    groups: list[list[dict[str, Any]]],
    max_tokens: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[tuple[int, int]]]:
    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    slices: list[tuple[int, int]] = []
    for rows in groups:
        prompt = categorical_prompt(rows)
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        begin = len(sequences)
        for label in LABELS:
            tokens, field_mask = field_token_mask(tokenizer, completion(label))
            sequence = prefix + tokens + [tokenizer.eos_token_id]
            mask = [0] * len(prefix) + field_mask + [0]
            if len(sequence) > max_tokens:
                raise ValueError(f"fresh categorical evidence scoring sequence too long: {rows[0]['query_id']}")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_policy_freeze", type=Path, required=True)
    parser.add_argument("--query_training_freeze", type=Path, required=True)
    parser.add_argument("--fresh_action_audit", type=Path, required=True)
    parser.add_argument("--policy", choices=("reference", "dagig"), required=True)
    parser.add_argument("--group_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    evidence_policy_path = args.evidence_policy_freeze.resolve()
    query_training_path = args.query_training_freeze.resolve()
    action_audit_path = args.fresh_action_audit.resolve()
    evidence_policy = read_json(evidence_policy_path)
    action_audit = read_json(action_audit_path)
    if evidence_policy.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_FROZEN":
        raise ValueError("backward DAG-IG evidence policy is not frozen")
    query_training = read_json(query_training_path)
    if query_training.get("protocol_version") != "dagig_v6_backward_fixed_descendants_equal_query_training_deterministic_v2":
        raise ValueError("fresh categorical scorer requires deterministic query v2")
    if sha256(Path(__file__).resolve()) != query_training["runner_hashes"]["fresh_evidence_scorer"]:
        raise ValueError("fresh categorical evidence scorer differs from frozen query runner")
    if action_audit.get("decision") != "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_ACTIONS_GO":
        raise ValueError("fresh evidence actions are not GO")
    action_path = Path(action_audit["output_paths"]["evidence_actions"])
    if sha256(action_path) != action_audit["output_hashes"]["evidence_actions"]:
        raise ValueError("fresh public evidence actions changed")

    training_path = Path(evidence_policy["input_paths"]["training_freeze"])
    training = read_json(training_path)
    if sha256(training_path) != evidence_policy["input_hashes"]["training_freeze"]:
        raise ValueError("frozen evidence training protocol changed")
    if training.get("protocol_version") != "dagig_v6_backward_evidence_explicit_categorical_deterministic_microbatch_v3":
        raise ValueError("fresh evidence scoring requires categorical deterministic evidence v3")
    if args.policy == "reference":
        adapter = Path(training["shared_sft_adapter"])
        if sha256(adapter / "adapter_model.safetensors") != training["input_hashes"]["sft_adapter_model"]:
            raise ValueError("reference evidence adapter changed")
    else:
        adapter = Path(evidence_policy["output_paths"]["adapter"])
        if tree_sha256(adapter) != evidence_policy["output_hashes"]["adapter_tree"]:
            raise ValueError("frozen DAG-IG evidence adapter changed")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(action_path):
        grouped[row["query_id"]].append(row)
    ordered_groups: list[list[dict[str, Any]]] = []
    for query_id in sorted(grouped):
        rows = sorted(grouped[query_id], key=lambda row: STRATEGY_ORDER.index(row["evidence_strategy"]))
        if len(rows) != 5 or [row["evidence_strategy"] for row in rows] != list(STRATEGY_ORDER):
            raise ValueError(f"fresh evidence strategy universe changed: {query_id}")
        ordered_groups.append(rows)
    if len(ordered_groups) != 480:
        raise ValueError("expected 480 fresh query parents")

    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(training["base_model"], local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if any(len(tokenizer(label, add_special_tokens=False).input_ids) != 1 for label in LABELS):
        raise ValueError("A-E are not single-token actions under the frozen evidence tokenizer")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        training["base_model"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, adapter).cuda().eval()
    output_rows: list[dict[str, Any]] = []
    max_tokens = int(training["training"]["max_input_tokens"])
    for start in range(0, len(ordered_groups), args.group_batch_size):
        groups = ordered_groups[start : start + args.group_batch_size]
        batch, mask, slices = build_batch(tokenizer, groups, max_tokens)
        with torch.inference_mode():
            logits = model(**batch, use_cache=False).logits[:, :-1].float()
        labels = batch["input_ids"][:, 1:]
        token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        scores = ((token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)).cpu().tolist()
        for rows, (begin, end) in zip(groups, slices):
            output_rows.append(
                {
                    "policy": args.policy,
                    "partition": "internal_holdout",
                    "method": rows[0]["method"],
                    "sample_id": rows[0]["sample_id"],
                    "visual_parent_id": rows[0]["visual_parent_id"],
                    "query_id": rows[0]["query_id"],
                    "action_ids": [row["evidence_action_id"] for row in rows],
                    "field_logprob_scores": scores[begin:end],
                }
            )
        completed = min(start + args.group_batch_size, len(ordered_groups))
        if completed % 100 < args.group_batch_size or completed == len(ordered_groups):
            print(json.dumps({"policy": args.policy, "scored": completed, "total": len(ordered_groups)}), flush=True)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scores_path = output / f"v6_backward_query_fresh_evidence_{args.policy}_scores_no_labels.jsonl"
    scores_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    gates = {
        "complete_480_query_parents": len(output_rows) == 480,
        "complete_five_action_scores": all(len(row["field_logprob_scores"]) == 5 for row in output_rows),
        "finite_scores": all(math.isfinite(value) for row in output_rows for value in row["field_logprob_scores"]),
        "one_categorical_action_token_only": True,
        "private_support_not_loaded": True,
        "internal_holdout_unused_for_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    inputs = {
        "evidence_policy_freeze": evidence_policy_path,
        "query_training_freeze": query_training_path,
        "fresh_action_audit": action_audit_path,
        "fresh_public_actions": action_path,
        "adapter_model": adapter / "adapter_model.safetensors",
        "scorer": Path(__file__).resolve(),
    }
    result = {
        "decision": "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_SCORES_READY" if all(gates.values()) else "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_SCORES_FAILED",
        "policy": args.policy,
        "metrics": {"query_parents": len(output_rows), "evidence_actions": sum(len(row["action_ids"]) for row in output_rows)},
        "action_representation": "one_of_five_single_token_labels_A_to_E",
        "deterministic_policy_logits": True,
        "gates": gates,
        "input_paths": {key: str(path) for key, path in inputs.items()},
        "input_hashes": {key: sha256(path) for key, path in inputs.items()},
        "output_paths": {"scores": str(scores_path)},
        "output_hashes": {"scores": sha256(scores_path)},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_SCORE_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
