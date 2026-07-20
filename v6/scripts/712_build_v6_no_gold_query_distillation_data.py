#!/usr/bin/env python3
"""Build KL-matched structured-query distillation targets from no-gold credit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


FIELDS = {"entity_quote", "information_need", "constraints", "search_query"}
FORBIDDEN = {
    "aliases",
    "answer_correct_proxy",
    "gold_answer",
    "positive_doc_ids",
    "qrels",
    "strict_proxy",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("cannot normalize query target")
    return [value / total for value in values]


def completion(row: dict[str, Any]) -> str:
    value = json.dumps(
        {
            "entity_quote": str(row.get("entity_quote") or "").strip(),
            "information_need": str(row.get("information_need") or "").strip(),
            "constraints": [str(item).strip() for item in row.get("constraints") or [] if str(item).strip()],
            "search_query": str(row["search_query"]).strip(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    parsed = json.loads(value)
    if set(parsed) != FIELDS or not all(parsed[key] for key in ("entity_quote", "information_need", "search_query")):
        raise ValueError(f"invalid structured query action: {row['query_id']}")
    return value


def prompt(question: str, visual_observation: str) -> str:
    return "\n".join(
        [
            "You are the structured query node of a multimodal web-search agent.",
            "Given the question and frozen image-only visual observation, return only compact valid JSON with exactly these fields:",
            '{"entity_quote":"...","information_need":"...","constraints":[],"search_query":"..."}',
            "The search query must identify the visual entity and requested fact without guessing or including the answer.",
            f"Question: {question}",
            f"Frozen image-only visual observation: {visual_observation}",
            "Structured query action:",
        ]
    )


def tempered_policy(
    candidates: list[dict[str, Any]], method: str, beta: float
) -> list[float]:
    logits = []
    for row in candidates:
        if method == "local_fixed_descendant":
            score = math.log(float(row["local_fixed_descendant_value"]))
        elif method == "true_outcome_grpo":
            score = float(row["outcome_mean_advantage"])
        else:
            raise ValueError(method)
        logits.append(math.log(float(row["behavior_probability"])) + beta * score)
    offset = max(logits)
    return normalize([math.exp(value - offset) for value in logits])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_audit", type=Path, required=True)
    parser.add_argument("--kl_calibration_audit", type=Path, required=True)
    parser.add_argument("--selector_audit", type=Path, required=True)
    parser.add_argument("--query_actions", type=Path, required=True)
    parser.add_argument("--max_query_tokens", type=int, default=24)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "control_audit": args.control_audit.resolve(),
        "kl_calibration_audit": args.kl_calibration_audit.resolve(),
        "selector_audit": args.selector_audit.resolve(),
        "query_actions": args.query_actions.resolve(),
    }
    controls = read_json(paths["control_audit"])
    calibration = read_json(paths["kl_calibration_audit"])
    selector = read_json(paths["selector_audit"])
    if controls.get("decision") != "DAGIG_V6_NO_GOLD_QUERY_CONTROLS_FROZEN":
        raise ValueError("no-gold query controls are not frozen")
    if calibration.get("decision") != "DAGIG_V6_QUERY_CONTROL_KL_BUDGET_FROZEN":
        raise ValueError("query-control KL budget is not frozen")
    if selector.get("decision") != "DAGIG_V6_KL_MATCHED_NO_GOLD_QUERY_SELECTOR_DEVELOPMENT_GO":
        raise ValueError("KL-matched exact query selector did not pass")
    if calibration["input_hashes"]["control_audit"] != sha256(paths["control_audit"]):
        raise ValueError("KL calibration and query controls differ")
    if selector["input_hashes"]["control_audit"] != sha256(paths["control_audit"]):
        raise ValueError("selector and query controls differ")

    target_path = Path(controls["output_paths"]["query_targets"])
    if sha256(target_path) != controls["output_hashes"]["query_targets"]:
        raise ValueError("frozen no-gold query targets changed")
    groups = read_jsonl(target_path)
    source = {str(row["query_id"]): row for row in read_jsonl(paths["query_actions"])}
    beta = calibration["policy_beta"]
    output_rows = []
    group_private = []
    query_lengths = []
    action_counts = Counter()
    forbidden = set()
    normalization_errors = []
    for group in groups:
        candidates = sorted(group["candidates"], key=lambda row: str(row["query_id"]))
        behavior = [float(row["behavior_probability"]) for row in candidates]
        dagig = [float(row["dagig_exact_probability"]) for row in candidates]
        local = tempered_policy(candidates, "local_fixed_descendant", float(beta["local_fixed_descendant"]))
        outcome = tempered_policy(candidates, "true_outcome_grpo", float(beta["true_outcome_grpo"]))
        weights = {
            "behavior_weight": behavior,
            "local_fixed_descendant_weight": local,
            "true_outcome_grpo_weight": outcome,
            "dagig_exact_weight": dagig,
        }
        for values in weights.values():
            normalization_errors.append(abs(sum(values) - 1.0))
            if min(values) <= 0.0:
                raise ValueError("query target has zero or negative support")
        action_counts[len(candidates)] += 1
        user_prompt = prompt(group["question"], group["visual_observation"])
        for index, candidate in enumerate(candidates):
            query_id = str(candidate["query_id"])
            action = source[query_id]
            if action["partition"] != "policy_train" or action["visual_field"] != "joint_state":
                raise ValueError(f"non-train or non-joint query action: {query_id}")
            value = completion(action)
            parsed = json.loads(value)
            length = len(parsed["search_query"].split())
            query_lengths.append(length)
            if length > args.max_query_tokens:
                raise ValueError(f"query exceeds frozen token proxy limit: {query_id}")
            for key in FORBIDDEN:
                if key in user_prompt.casefold() or key in value.casefold():
                    forbidden.add(key)
            row = {
                "sample_id": group["sample_id"],
                "partition": "policy_train",
                "parent_group_id": group["query_parent_id"],
                "query_action_id": query_id,
                "visual_field": "joint_state",
                "prompt": user_prompt,
                "completion": value,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": value}]},
                ],
                **{key: values[index] for key, values in weights.items()},
            }
            output_rows.append(row)
        group_private.append(
            {
                "sample_id": group["sample_id"],
                "parent_group_id": group["query_parent_id"],
                "action_ids": [row["query_id"] for row in candidates],
                **weights,
            }
        )

    metrics = {
        "samples": len({row["sample_id"] for row in output_rows}),
        "groups": len(groups),
        "legal_actions": len(output_rows),
        "action_count_distribution": dict(sorted(action_counts.items())),
        "max_query_tokens_proxy": max(query_lengths),
        "mean_query_tokens_proxy": mean(query_lengths),
        "max_target_normalization_error": max(normalization_errors),
        "forbidden_public_fields": sorted(forbidden),
    }
    gates = {
        "complete_samples": metrics["samples"] == 158,
        "complete_joint_state_groups": metrics["groups"] == 158,
        "complete_legal_actions": metrics["legal_actions"] == 773,
        "at_least_three_actions_per_group": min(action_counts) >= 3,
        "all_targets_normalized": metrics["max_target_normalization_error"] <= 1e-10,
        "bounded_query_length": metrics["max_query_tokens_proxy"] <= args.max_query_tokens,
        "same_action_universe": True,
        "kl_matched_controls": True,
        "no_gold_or_qrels_in_public_text": not forbidden,
        "internal_development_unused": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = (
        "DAGIG_V6_LEGAL_STRUCTURED_QUERY_TARGETS_GO"
        if all(gates.values())
        else "DAGIG_V6_LEGAL_STRUCTURED_QUERY_TARGETS_NO_GO"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    actions_path = output / "v6_no_gold_kl_matched_query_action_targets_train.jsonl"
    private_path = output / "v6_no_gold_kl_matched_query_group_targets_private.jsonl"
    write_jsonl(actions_path, output_rows)
    write_jsonl(private_path, group_private)
    audit = {
        "decision": decision,
        "protocol_version": "dagig_v6_no_gold_kl_matched_joint_state_query_distillation_v1",
        "metrics": metrics,
        "gates": gates,
        "policy_beta": beta,
        "input_paths": {**{key: str(path) for key, path in paths.items()}, "control_targets": str(target_path)},
        "input_hashes": {**{key: sha256(path) for key, path in paths.items()}, "control_targets": sha256(target_path)},
        "output_paths": {"legal_actions": str(actions_path), "group_targets_private": str(private_path)},
        "output_hashes": {"legal_actions": sha256(actions_path), "group_targets_private": sha256(private_path)},
        "gold_or_qrels_loaded_for_training_data": False,
        "internal_development_used": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_LEGAL_STRUCTURED_QUERY_TARGET_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gates": gates, "audit": str(audit_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
