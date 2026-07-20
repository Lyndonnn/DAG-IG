#!/usr/bin/env python3
"""Freeze the per-document-any-hit gold-aware support teacher v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict fact-entailment annotator for top-k web retrieval.
You receive a question, a private reference answer, optional equivalent aliases, visual context, an executed query, and three selected evidence snippets.
Evaluate each document separately. Choose A when at least one document directly states or strongly entails the reference answer for the exact entity and every relevant date, location, comparison, and other condition in the question.
Irrelevant or incorrect documents do not cancel a genuinely supporting document. Choose B only when no document supports the reference answer.
Allow semantically equivalent formatting and transliteration for addresses, phone numbers, dates, times, and units. Allow a stated source value that yields the reference through the rounding or simple conversion explicitly requested by the question.
An incidental occurrence of the same number or words for the wrong entity/context, topical relevance, or a weak hint is not support.
The visual context and query identify intent but are not evidence by themselves.
Return exactly one character: A for at-least-one supporting document, or B for no supporting document."""

USER_TEMPLATE = """Question:
{question}

Private reference answer:
{gold_answer}

Equivalent aliases:
{aliases}

Visual context:
{visual_observation}

Executed search query:
{search_query}

Selected evidence:
{evidence}

Decision:"""


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
    files = [model_path / name for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")]
    files.extend(sorted(model_path.glob("model-*.safetensors")))
    if len(files) < 4 or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"Incomplete local model: {model_path}")
    return {"path": str(model_path), "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files}}


def evidence_text(docs: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        "\n".join([
            f"Document {index}",
            f"Title: {str(doc.get('title') or '').strip()}",
            f"Source: {str(doc.get('domain') or '').strip()}",
            f"Date: {str(doc.get('date') or '').strip() or 'not provided'}",
            f"Snippet: {str(doc.get('snippet') or '').strip()}",
        ])
        for index, doc in enumerate(docs, 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_actions", type=Path, required=True)
    parser.add_argument("--private_labels", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {key: value.resolve() for key, value in {
        "evidence_actions": args.evidence_actions,
        "private_labels": args.private_labels,
        "scorer": args.scorer,
        "auditor": args.auditor,
        "auditor_helper": args.auditor.resolve().with_name("826_build_audit_v6_gold_aware_support_labels_v1.py"),
    }.items()}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    labels = {row["sample_id"]: row for row in read_jsonl(inputs["private_labels"])}
    rows = []
    for action in read_jsonl(inputs["evidence_actions"]):
        label = labels[action["sample_id"]]
        docs = action.get("selected_docs") or []
        if len(docs) != 3:
            raise ValueError(f"Expected three documents: {action['evidence_action_id']}")
        aliases = label.get("aliases") or []
        rows.append({
            "evidence_action_id": action["evidence_action_id"],
            "query_id": action["query_id"],
            "parent_visual_state_id": action["query_id"].rsplit("::", 1)[0],
            "sample_id": action["sample_id"],
            "partition": action["partition"],
            "legacy_evidence_strategy": action["evidence_strategy"],
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_private": USER_TEMPLATE.format(
                question=str(action["question"]).strip(),
                gold_answer=str(label["gold_answer"]).strip(),
                aliases="; ".join(str(value) for value in aliases) if aliases else "none",
                visual_observation=str(action["visual_observation"]).strip(),
                search_query=str(action["search_query"]).strip(),
                evidence=evidence_text(docs),
            ),
            "selected_doc_ids": action["selected_doc_ids"],
        })
    train = sum(row["partition"] == "policy_train" for row in rows)
    internal = sum(row["partition"] == "internal_holdout" for row in rows)
    if len(rows) != 14770 or (train, internal) != (11795, 2975):
        raise ValueError(f"v2 teacher universe mismatch: {len(rows)}/{train}/{internal}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompt_path = output / "v6_gold_aware_support_teacher_v2_inputs_private.jsonl"
    with prompt_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["evidence_action_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    gates = {
        "exact_14770_evidence_actions": len(rows) == 14770,
        "exact_11795_2975_partition": (train, internal) == (11795, 2975),
        "three_docs_per_action": all(len(row["selected_doc_ids"]) == 3 for row in rows),
        "support_semantics_is_any_document_hit": True,
        "equivalence_and_question_requested_rounding_allowed": True,
        "private_prompts_not_training_data": True,
        "annotation_notes_not_used": True,
        "cached_evidence_only": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    model_path = args.model_path.resolve()
    protocol = {
        "decision": "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V2_FROZEN" if all(gates.values()) else "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V2_NO_GO",
        "protocol_version": "dagig_v6_private_gold_aware_any_document_support_teacher_v2",
        "semantics": "Support@3: at least one selected snippet entails the private answer under all question conditions",
        "teacher_contract": {
            "model": "frozen local Qwen2.5-VL-7B-Instruct text-only",
            "score": "next-token logit(A)-logit(B)",
            "attn_implementation": "sdpa",
            "dtype": "bfloat16",
            "max_input_tokens": 2048,
            "truncation": False,
            "hard_label_threshold": 0.5,
        },
        "quality_gate": {
            "independent_stratified_audit_samples_min": 300,
            "independent_balanced_accuracy_min": 0.90,
            "independent_precision_min": 0.90,
            "independent_recall_min": 0.85,
            "short_numeric_subset_accuracy_min": 0.90,
            "address_subset_accuracy_min": 0.85,
        },
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "input_paths": {key: str(path) for key, path in inputs.items()},
        "input_hashes": {key: sha256(path) for key, path in inputs.items()},
        "model_fingerprint": model_fingerprint(model_path),
        "output_paths": {"private_prompts": str(prompt_path)},
        "output_hashes": {"private_prompts": sha256(prompt_path)},
        "gates": gates,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V2_FREEZE.json"
    freeze_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "rows": len(rows), "train": train, "internal": internal, "freeze": str(freeze_path), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
