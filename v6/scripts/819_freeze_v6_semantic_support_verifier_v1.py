#!/usr/bin/env python3
"""Freeze answer-independent semantic support-verifier inputs and protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict evidence support verifier for web research.
Do not answer the question. Judge only whether the supplied evidence is sufficient to determine a correct answer to the question.
Respect the entity, date, location, comparison, and other constraints in the question.
The visual observation and executed search query are context, not evidence by themselves.
Choose A only when the evidence states or strongly entails the requested answer. Mere topical relevance, an entity mention, weak hints, conflicting evidence, or missing constraints require B.
Return exactly one character: A for sufficient support, or B for insufficient support."""

USER_TEMPLATE = """Question:
{question}

Visual observation:
{visual_observation}

Executed search query:
{search_query}

Selected evidence:
{evidence}

Decision:"""


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


def model_fingerprint(model_path: Path) -> dict[str, Any]:
    required = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
    shards = sorted(model_path.glob("model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No local model shards under {model_path}")
    files = [model_path / name for name in required] + shards
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "path": str(model_path),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }


def format_evidence(docs: list[dict[str, Any]]) -> str:
    blocks = []
    for index, doc in enumerate(docs, 1):
        fields = [
            f"Document {index}",
            f"Title: {str(doc.get('title') or '').strip()}",
            f"Source: {str(doc.get('domain') or '').strip()}",
            f"Date: {str(doc.get('date') or '').strip() or 'not provided'}",
            f"Snippet: {str(doc.get('snippet') or '').strip()}",
        ]
        blocks.append("\n".join(fields))
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_value_freeze", type=Path, required=True)
    parser.add_argument("--query_actions", type=Path, required=True)
    parser.add_argument("--evidence_actions", type=Path, required=True)
    parser.add_argument("--factorized_predictions", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--fitter", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "query_value_freeze": args.query_value_freeze.resolve(),
        "query_actions": args.query_actions.resolve(),
        "evidence_actions": args.evidence_actions.resolve(),
        "factorized_predictions": args.factorized_predictions.resolve(),
        "scorer": args.scorer.resolve(),
        "fitter": args.fitter.resolve(),
        "auditor": args.auditor.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    query_freeze = read_json(inputs["query_value_freeze"])
    if query_freeze.get("decision") != "DAGIG_V6_HYBRID_QUERY_VALUES_V1_FROZEN":
        raise ValueError("The downstream evidence/query contract is not frozen")
    for key, raw_path in query_freeze["output_paths"].items():
        if sha256(Path(raw_path)) != query_freeze["output_hashes"][key]:
            raise ValueError(f"Frozen query-value output changed: {key}")

    query_actions = {row["query_id"]: row for row in read_jsonl(inputs["query_actions"])}
    evidence_actions = {row["evidence_action_id"]: row for row in read_jsonl(inputs["evidence_actions"])}
    factorized = {row["evidence_action_id"]: row for row in read_jsonl(inputs["factorized_predictions"])}
    diagnostics = read_jsonl(Path(query_freeze["output_paths"]["diagnostics"]))

    rows = []
    for group in diagnostics:
        for query_id, evidence_id in zip(group["query_action_ids"], group["selected_evidence_action_ids"]):
            query = query_actions[query_id]
            evidence = evidence_actions[evidence_id]
            baseline = factorized[evidence_id]
            if evidence["query_id"] != query_id or baseline["query_id"] != query_id:
                raise ValueError(f"Query/evidence mismatch for {query_id}")
            docs = evidence.get("selected_docs") or []
            if not docs or len(docs) > 5:
                raise ValueError(f"Illegal selected evidence count for {evidence_id}: {len(docs)}")
            user_prompt = USER_TEMPLATE.format(
                question=str(query["question"]).strip(),
                visual_observation=str(query["visual_observation"]).strip(),
                search_query=str(query["search_query"]).strip(),
                evidence=format_evidence(docs),
            )
            rows.append({
                "query_action_id": query_id,
                "parent_visual_state_id": group["parent_state_id"],
                "selected_evidence_action_id": evidence_id,
                "sample_id": query["sample_id"],
                "partition": group["partition"],
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "selected_doc_ids": [doc["doc_id"] for doc in docs],
                "baseline_support_probability": float(baseline["support_probability"]),
                "answer_correct_given_support_probability": float(baseline["answer_correct_given_support_probability"]),
            })

    train = sum(row["partition"] == "policy_train" for row in rows)
    internal = sum(row["partition"] == "internal_holdout" for row in rows)
    if len(rows) != 2954 or len({row["query_action_id"] for row in rows}) != 2954:
        raise ValueError(f"Expected 2954 unique query states, got {len(rows)}")
    if (train, internal) != (2359, 595):
        raise ValueError(f"Partition mismatch: {(train, internal)}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_path = output / "v6_semantic_support_inputs_no_labels.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["query_action_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    forbidden_runtime_fields = [
        "gold", "oracle", "teacher", "support_label", "evidence_hit",
        "answer_correct", "target_doc", "ground_truth", "final_answer",
    ]
    serialized_keys = set().union(*(row.keys() for row in rows))
    gates = {
        "exact_2954_query_states": len(rows) == 2954,
        "exact_2359_595_partition": (train, internal) == (2359, 595),
        "one_frozen_evidence_action_per_query": len({row["selected_evidence_action_id"] for row in rows}) == 2954,
        "runtime_records_have_no_forbidden_fields": not any(name in serialized_keys for name in forbidden_runtime_fields),
        "prompt_excludes_candidate_answer": all("Candidate answer:" not in row["user_prompt"] for row in rows),
        "prompt_excludes_gold_and_qrels": True,
        "scorer_cannot_load_private_labels": True,
        "cached_search_only": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    fit = {
        "folds": 5,
        "repeats": 5,
        "seed_prefix": "dagig_v6_semantic_support_verifier_v1",
        "l2_grid": [0.003, 0.01, 0.03, 0.1, 0.3],
        "candidate_features": [
            ["semantic_logit"],
            ["semantic_logit", "baseline_support_logit"],
            ["semantic_logit", "baseline_support_logit", "semantic_x_baseline"],
        ],
        "selection_rule": "first feature family in listed order with any l2 passing every train-OOF gate; within family choose lowest Brier then smallest l2",
        "newton_steps": 60,
        "probability_clip": [1e-5, 0.99999],
    }
    train_gates = {
        "support_auc_min": 0.80,
        "brier_improvement_vs_baseline_min": 0.01,
        "within_visual_pair_order_min": 0.68,
        "nonconstant_parent_group_rate_min": 0.95,
        "semantic_coefficient_positive": True,
    }
    development_gates = {
        "support_delta_vs_no_credit_min": 0.0,
        "support_noninferiority_vs_local_tolerance": 0.01,
        "support_noninferiority_vs_outcome_tolerance": 0.01,
        "strict_noninferiority_vs_no_credit_tolerance": 0.0,
        "strict_noninferiority_vs_local_tolerance": 0.015,
        "strict_noninferiority_vs_outcome_tolerance": 0.015,
        "top_action_disagreement_vs_outcome_min": 0.05,
        "selected_query_strategies_min": 4,
    }
    protocol = {
        "decision": "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_FROZEN" if all(gates.values()) else "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_NO_GO",
        "protocol_version": "dagig_v6_answer_independent_semantic_support_verifier_v1",
        "semantics": "P(selected evidence is sufficient to determine the requested answer | question, visual observation, executed query, evidence)",
        "verifier_contract": {
            "model": "frozen local Qwen2.5-VL-7B-Instruct in text-only mode",
            "labels": {"A": "sufficient support", "B": "insufficient support"},
            "score": "next-token logit(A)-logit(B) after the frozen chat prompt",
            "attn_implementation": "sdpa",
            "dtype": "bfloat16",
            "max_input_tokens": 2048,
            "truncation": False,
            "answer_independent": True,
        },
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "fit": fit,
        "train_oof_gates": train_gates,
        "development_gates": development_gates,
        "input_paths": {key: str(path) for key, path in inputs.items()},
        "input_hashes": {key: sha256(path) for key, path in inputs.items()},
        "model_fingerprint": model_fingerprint(model_path),
        "output_paths": {"verifier_inputs": str(input_path)},
        "output_hashes": {"verifier_inputs": sha256(input_path)},
        "gates": gates,
        "private_labels_loaded": False,
        "internal_used_for_fit_or_selection": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_FREEZE.json"
    freeze_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "rows": len(rows), "train": train, "internal": internal, "freeze": str(freeze_path), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
