#!/usr/bin/env python3
"""Freeze a fresh structured support-label teacher/auditor pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict fact-entailment annotator for top-k web evidence.
For each case, decide whether at least one supplied document directly states or strongly entails the private reference answer for the exact entity and every date, location, comparison, and other constraint in the question.
Judge each document separately. Irrelevant or incorrect documents do not cancel a genuinely supporting document. Visual context and the executed query may resolve the intended entity but are not evidence by themselves.
Accept equivalent formatting or transliteration of addresses, phone numbers, dates, times, identifiers, and units. Accept a source value that yields the answer through rounding or a simple conversion explicitly requested by the question.
Reject incidental answer-string occurrences, topical relevance without the answer, wrong entities, wrong conditions, and unsupported guesses.
For a positive decision, identify every supporting document index and copy one short verbatim supporting span from a supplied title or snippet. For rounding, conversion, or strong entailment, also state the short derivation. For a negative decision, return no document indices and an empty supporting span."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_answer_type_helper() -> Any:
    path = Path(__file__).with_name("826_build_audit_v6_gold_aware_support_labels_v1.py")
    spec = importlib.util.spec_from_file_location("dagig_support_helper_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def state_key(action: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return action["sample_id"], tuple(str(value) for value in action["selected_doc_ids"])


def select_fresh_pilot(
    actions: list[dict[str, Any]], private: dict[str, dict[str, Any]], excluded: set[tuple[str, tuple[str, ...]]]
) -> list[dict[str, Any]]:
    helper = load_answer_type_helper()
    quotas = {
        "text_or_entity": 159,
        "phone_or_identifier": 70,
        "short_numeric": 70,
        "email": 45,
        "address": 44,
        "time": 12,
    }
    rng = random.Random("dagig_v6_structured_support_teacher_pilot_v3_fresh400")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_states: set[tuple[str, tuple[str, ...]]] = set()
    for action in actions:
        key = state_key(action)
        if action["partition"] != "policy_train" or key in excluded or key in seen_states:
            continue
        seen_states.add(key)
        grouped[helper.answer_type(private[action["sample_id"]]["gold_answer"])].append(action)
    selected: list[dict[str, Any]] = []
    sample_counts: Counter[str] = Counter()
    for answer_type, quota in quotas.items():
        candidates = grouped[answer_type]
        rng.shuffle(candidates)
        strategy_counts: Counter[str] = Counter()
        chosen: list[dict[str, Any]] = []
        while len(chosen) < quota:
            eligible = [
                row for row in candidates
                if row not in chosen and sample_counts[row["sample_id"]] < 4
            ]
            if not eligible:
                raise ValueError(f"Cannot satisfy fresh v3 pilot quota for {answer_type}: {len(chosen)}/{quota}")
            minimum = min(strategy_counts[row["evidence_strategy"]] for row in eligible)
            row = next(value for value in eligible if strategy_counts[value["evidence_strategy"]] == minimum)
            chosen.append(row)
            sample_counts[row["sample_id"]] += 1
            strategy_counts[row["evidence_strategy"]] += 1
        selected.extend(chosen)
    rng.shuffle(selected)
    if len(selected) != 400 or len({state_key(row) for row in selected}) != 400:
        raise ValueError("Fresh v3 pilot must contain exactly 400 unique evidence states")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_actions", type=Path, required=True)
    parser.add_argument("--private_labels", type=Path, required=True)
    parser.add_argument("--previous_audit_key", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: path.resolve() for name, path in {
        "evidence_actions": args.evidence_actions,
        "private_labels": args.private_labels,
        "previous_audit_key": args.previous_audit_key,
        "runner": args.runner,
        "evaluator": args.evaluator,
    }.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    actions = read_jsonl(paths["evidence_actions"])
    by_id = {row["evidence_action_id"]: row for row in actions}
    private = {row["sample_id"]: row for row in read_jsonl(paths["private_labels"])}
    previous = read_jsonl(paths["previous_audit_key"])
    previous_ids = {row["evidence_action_id"] for row in previous}
    excluded_states = {state_key(by_id[action_id]) for action_id in previous_ids}
    selected = select_fresh_pilot(actions, private, excluded_states)
    helper = load_answer_type_helper()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    item_path = output / "v6_structured_support_v3_pilot_items_blinded_private.jsonl"
    key_path = output / "v6_structured_support_v3_pilot_key_private.jsonl"
    with item_path.open("w", encoding="utf-8") as item_handle, key_path.open("w", encoding="utf-8") as key_handle:
        for index, action in enumerate(selected):
            audit_id = f"support_v3_pilot_{index:04d}"
            label = private[action["sample_id"]]
            aliases = label.get("aliases") or []
            prompt = "\n\n".join([
                f"Question:\n{str(action['question']).strip()}",
                f"Private reference answer:\n{str(label['gold_answer']).strip()}",
                f"Equivalent aliases:\n{'; '.join(str(value) for value in aliases) if aliases else 'none'}",
                f"Visual context:\n{str(action['visual_observation']).strip()}",
                f"Executed search query:\n{str(action['search_query']).strip()}",
                f"Selected evidence:\n{evidence_text(action['selected_docs'])}",
            ])
            item_handle.write(json.dumps({"audit_id": audit_id, "user_prompt_private": prompt}, ensure_ascii=False, sort_keys=True) + "\n")
            key_handle.write(json.dumps({
                "audit_id": audit_id,
                "answer_type": helper.answer_type(label["gold_answer"]),
                "evidence_action_id": action["evidence_action_id"],
                "evidence_strategy": action["evidence_strategy"],
                "sample_id": action["sample_id"],
                "selected_doc_ids": action["selected_doc_ids"],
            }, sort_keys=True) + "\n")
    answer_counts = Counter(helper.answer_type(private[row["sample_id"]]["gold_answer"]) for row in selected)
    strategy_counts = Counter(row["evidence_strategy"] for row in selected)
    sample_counts = Counter(row["sample_id"] for row in selected)
    gates = {
        "exact_400_fresh_policy_train_states": len(selected) == 400,
        "no_previous_audit_action_overlap": not ({row["evidence_action_id"] for row in selected} & previous_ids),
        "no_previous_audit_state_overlap": not ({state_key(row) for row in selected} & excluded_states),
        "max_four_states_per_sample": max(sample_counts.values()) <= 4,
        "all_six_answer_types_present": len(answer_counts) == 6,
        "all_five_evidence_actions_present": len(strategy_counts) == 5,
        "private_reference_used_only_for_label_audit": True,
        "local_predictions_absent": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    protocol = {
        "decision": "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_FROZEN" if all(gates.values()) else "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_NO_GO",
        "protocol_version": "dagig_v6_structured_cited_support_teacher_pilot_v3",
        "semantics": "Support@3: any selected document entails the private answer under every question condition",
        "system_prompt": SYSTEM_PROMPT,
        "models": {
            "teacher": "gpt-5.4-mini-2026-03-17",
            "auditor": "gpt-5.4-2026-03-05",
        },
        "generation": {
            "batch_size": 10,
            "max_completion_tokens": 6000,
            "response_format": "strict_json_schema",
            "reasoning_effort": {"teacher": "low", "auditor": "medium"},
            "temperature": None,
            "seed": None,
        },
        "budget": {
            "expected_requests_per_role": 40,
            "max_requests_per_role": 55,
            "max_input_tokens_per_role": 750000,
            "max_output_tokens_per_role": 250000,
        },
        "quality_gates": {
            "samples": 400,
            "balanced_accuracy_min": 0.90,
            "precision_min": 0.90,
            "recall_min": 0.90,
            "citation_validity_min": 0.97,
            "short_numeric_accuracy_min": 0.90,
            "phone_or_identifier_accuracy_min": 0.90,
            "email_accuracy_min": 0.90,
            "address_accuracy_min": 0.85,
        },
        "sample_counts": {
            "answer_type": dict(sorted(answer_counts.items())),
            "evidence_strategy": dict(sorted(strategy_counts.items())),
            "unique_samples": len(sample_counts),
        },
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_hashes": {name: sha256(path) for name, path in paths.items()},
        "output_paths": {"pilot_items": str(item_path), "private_key": str(key_path)},
        "output_hashes": {"pilot_items": sha256(item_path), "private_key": sha256(key_path)},
        "gates": gates,
        "serper_calls": 0,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_FREEZE.json"
    freeze_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "sample_counts": protocol["sample_counts"], "gates": gates, "freeze": str(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
