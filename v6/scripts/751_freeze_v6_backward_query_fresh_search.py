#!/usr/bin/env python3
"""Freeze fresh Serper execution for matched backward query policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        method, separator, path = value.partition("=")
        if not separator or method not in METHODS or method in result:
            raise ValueError(f"invalid internal score mapping: {value}")
        result[method] = Path(path).resolve()
    if set(result) != set(METHODS):
        raise ValueError("all four internal query score audits are required")
    return result


def load_scores(path: Path, method: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    audit = read_json(path)
    if (
        audit.get("decision") != "DAGIG_V6_BACKWARD_QUERY_POLICY_SCORES_READY"
        or audit.get("method") != method
        or audit.get("partition") != "internal_holdout"
    ):
        raise ValueError(f"invalid internal query score audit: {method}")
    score_path = Path(audit["output_paths"]["scores"])
    if sha256(score_path) != audit["output_hashes"]["scores"]:
        raise ValueError(f"internal query scores changed: {method}")
    rows = read_jsonl(score_path)
    return audit, {row["parent_group_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--train_fit", type=Path, required=True)
    parser.add_argument("--reference_scores", type=Path, required=True)
    parser.add_argument("--method_scores", action="append", required=True, help="method=/path/to/internal_score_audit.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    training_path = args.training_freeze.resolve()
    fit_path = args.train_fit.resolve()
    training = read_json(training_path)
    fit = read_json(fit_path)
    if training.get("decision") != "DAGIG_V6_BACKWARD_QUERY_TRAINING_FROZEN":
        raise ValueError("backward query training is not frozen")
    if fit.get("decision") != "DAGIG_V6_BACKWARD_QUERY_TRAIN_FIT_GO":
        raise ValueError("query train fit is not GO")
    if fit["input_hashes"].get("training_freeze") != sha256(training_path):
        raise ValueError("query train-fit audit belongs to another protocol")

    control_path = Path(training["input_paths"]["control_freeze"])
    control = read_json(control_path)
    internal_path = Path(control["output_paths"]["internal_data"])
    query_action_path = Path(control["input_paths"]["query_actions"])
    if sha256(internal_path) != control["output_hashes"]["internal_data"]:
        raise ValueError("internal query action matrix changed")
    if sha256(query_action_path) != control["input_hashes"]["query_actions"]:
        raise ValueError("structured query source actions changed")
    groups = {row["parent_group_id"]: row for row in read_jsonl(internal_path)}
    actions = {row["query_id"]: row for row in read_jsonl(query_action_path)}
    if len(groups) != 120 or len({row["sample_id"] for row in groups.values()}) != 40:
        raise ValueError("expected 120 sealed visual parent states from 40 internal samples")

    reference_audit_path = args.reference_scores.resolve()
    _, reference = load_scores(reference_audit_path, "reference")
    method_paths = parse_mapping(args.method_scores)
    method_scores: dict[str, dict[str, dict[str, Any]]] = {}
    for method, path in method_paths.items():
        _, method_scores[method] = load_scores(path, method)
    if any(set(scores) != set(groups) for scores in [reference, *method_scores.values()]):
        raise ValueError("internal query score universes differ")

    beta = float(training["training"]["beta"])
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_action_ids: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(groups):
        group = groups[group_id]
        action_ids = group["action_ids"]
        if reference[group_id]["action_ids"] != action_ids:
            raise ValueError(f"reference action order changed: {group_id}")
        behavior = np.asarray(group["behavior_probabilities"], dtype=np.float64)
        for method in METHODS:
            score = method_scores[method][group_id]
            if score["action_ids"] != action_ids:
                raise ValueError(f"method action order changed: {method}/{group_id}")
            delta = np.asarray(score["field_logprob_scores"], dtype=np.float64) - np.asarray(
                reference[group_id]["field_logprob_scores"], dtype=np.float64
            )
            logits = np.log(behavior) + beta * delta
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            index = int(np.argmax(probabilities))
            action_id = action_ids[index]
            action = actions[action_id]
            selected_action_ids[method].append(action_id)
            predictions[method].append(
                {
                    "member_query_id": f"{method}::{group_id}",
                    "method": method,
                    "sample_id": group["sample_id"],
                    "partition": "internal_holdout",
                    "visual_parent_id": group_id,
                    "query_action_id": action_id,
                    "source_dataset": action.get("source_dataset", "pix2fact"),
                    "question": action["question"],
                    "visual_field": action["visual_field"],
                    "visual_observation": action["visual_observation"],
                    "query_strategy": action["query_strategy"],
                    "entity_quote": action.get("entity_quote") or "",
                    "information_need": action.get("information_need") or "",
                    "constraints": action.get("constraints") or [],
                    "search_query": action["search_query"],
                    "selector_action_ids": action_ids,
                    "selector_behavior_probabilities": behavior.tolist(),
                    "selector_field_logprob_deltas": delta.tolist(),
                    "selector_probabilities": probabilities.tolist(),
                    "selected_action_index": index,
                }
            )

    if any(len(rows) != 120 for rows in predictions.values()):
        raise ValueError("incomplete method-by-visual-state query selections")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_paths: dict[str, Path] = {}
    for method in METHODS:
        path = output / f"v6_backward_query_{method}_internal_selected_no_labels.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions[method]),
            encoding="utf-8",
        )
        prediction_paths[method] = path

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for method in METHODS:
        for row in predictions[method]:
            query = normalized(row["search_query"])
            if not query:
                raise ValueError(f"empty normalized selected query: {row['member_query_id']}")
            grouped[(row["sample_id"], query)].append(row)
    plan: list[dict[str, Any]] = []
    for (sample_id, normalized_query), members in sorted(grouped.items()):
        search_id = "search_" + hashlib.sha256(f"{sample_id}\n{normalized_query}".encode("utf-8")).hexdigest()[:20]
        plan.append(
            {
                "search_id": search_id,
                "sample_id": sample_id,
                "split": "internal_holdout",
                "query": members[0]["search_query"],
                "normalized_query": normalized_query,
                "num": 10,
                "member_query_ids": sorted(row["member_query_id"] for row in members),
                "member_methods": sorted({row["method"] for row in members}),
                "member_visual_parent_ids": sorted({row["visual_parent_id"] for row in members}),
            }
        )
    if not 120 <= len(plan) <= 480 or len({row["search_id"] for row in plan}) != len(plan):
        raise ValueError(f"unexpected fresh-search call universe: {len(plan)}")
    plan_path = output / "v6_backward_query_internal_fresh_search_plan_no_labels.jsonl"
    plan_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in plan),
        encoding="utf-8",
    )
    ledger_path = output / "serper_attempts.jsonl"
    ledger_path.write_text("", encoding="utf-8")

    runner = Path(__file__).with_name("531_run_v6_serper_search.py").resolve()
    input_paths: dict[str, Path] = {
        "training_freeze": training_path,
        "train_fit": fit_path,
        "control_freeze": control_path,
        "internal_data": internal_path,
        "query_actions": query_action_path,
        "reference_score_audit": reference_audit_path,
        **{f"{method}_score_audit": path for method, path in method_paths.items()},
    }
    input_paths.update({f"{method}_predictions": path for method, path in prediction_paths.items()})
    retry_budget = max(20, math.ceil(len(plan) * 0.10))
    freeze = {
        "decision": "DAGIG_V6_SERPER_SEARCH_PLAN_FROZEN",
        "protocol_version": "dagig_v6_backward_query_all_visual_states_internal_fresh_search_v1",
        "samples": 40,
        "visual_parent_states": 120,
        "methods": list(METHODS),
        "method_roles": {"no_credit": "no_credit", "local_ig": "local_ig", "outcome": "outcome", "dagig": "dagig"},
        "query_actions": 120 * len(METHODS),
        "unique_search_calls": len(plan),
        "new_api_calls_planned": len(plan),
        "serper_num_results": 10,
        "serper_key_env": "SERPER_API_KEY",
        "reserved_final_eval_key_env": "SERPER_API_KEY_EVAL",
        "reserved_final_eval_key_used": False,
        "http_attempt_hard_cap": len(plan) + retry_budget,
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "input_hashes": {key: sha256(path) for key, path in input_paths.items()},
        "output_paths": {
            "search_plan": str(plan_path),
            "seeded_ledger": str(ledger_path),
            **{f"{method}_predictions": str(path) for method, path in prediction_paths.items()},
        },
        "output_hashes": {
            "search_plan": sha256(plan_path),
            "seeded_ledger": sha256(ledger_path),
            **{f"{method}_predictions": sha256(path) for method, path in prediction_paths.items()},
        },
        "runner_hashes": {"runner": sha256(runner)},
        "selection_uses_gold_terminal_or_qrels": False,
        "old_retrieved_docs_reused": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_BACKWARD_QUERY_FRESH_SEARCH_FREEZE.json"
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "freeze": str(freeze_path),
                "visual_parent_states": 120,
                "method_query_actions": 480,
                "unique_search_calls": len(plan),
                "key_env": freeze["serper_key_env"],
                "reserved_eval_key_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
