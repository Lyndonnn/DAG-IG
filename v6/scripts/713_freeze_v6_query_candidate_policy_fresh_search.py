#!/usr/bin/env python3
"""Freeze fresh search for learned listwise query-candidate policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = ("no_credit", "local_fixed_descendant", "true_outcome_grpo", "dagig_exact")


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
            raise ValueError(f"invalid score audit mapping: {value}")
        result[method] = Path(path).resolve()
    if set(result) != set(METHODS):
        raise ValueError("score audits must cover all matched methods")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_audit", type=Path, required=True)
    parser.add_argument("--score_audit", action="append", required=True, help="method=/path/to/audit.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_audit_path = args.candidate_audit.resolve()
    candidate_audit = read_json(candidate_audit_path)
    if candidate_audit.get("decision") != "DAGIG_V6_QUERY_SELECTOR_CANDIDATES_GO":
        raise ValueError("shared query candidate universe is not GO")
    score_paths = parse_mapping(args.score_audit)
    input_paths: dict[str, str] = {"candidate_audit": str(candidate_audit_path)}
    method_predictions: dict[str, list[dict[str, Any]]] = {}
    for method, audit_path in score_paths.items():
        audit = read_json(audit_path)
        if audit.get("decision") != "DAGIG_V6_QUERY_SELECTOR_SCORES_READY" or audit.get("method") != method:
            raise ValueError(f"candidate-policy scores are not ready: {method}")
        if audit["input_hashes"]["candidate_audit"] != sha256(candidate_audit_path):
            raise ValueError(f"candidate universe differs: {method}")
        score_path = Path(audit["output_paths"]["scores"])
        if sha256(score_path) != audit["output_hashes"]["scores"]:
            raise ValueError(f"candidate-policy scores changed: {method}")
        rows = read_jsonl(score_path)
        if len(rows) != 40 or len({row["sample_id"] for row in rows}) != 40:
            raise ValueError(f"incomplete internal candidate-policy rows: {method}")
        method_predictions[method] = [
            {
                "method": method,
                "sample_id": row["sample_id"],
                "query_parent_id": row["query_parent_id"],
                "question": row["question"],
                "visual_observation": row["visual_observation"],
                "search_query": row["selected_search_query"],
                "selected_query_id": row["selected_query_id"],
                "selected_strategy": row["selected_strategy"],
                "selection_mode": "learned_listwise_candidate_policy",
            }
            for row in rows
        ]
        input_paths[f"{method}_score_audit"] = str(audit_path)
        input_paths[f"{method}_scores"] = str(score_path)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for method, rows in method_predictions.items():
        for row in rows:
            query = str(row["search_query"]).strip()
            if not query:
                raise ValueError(f"empty selected query: {method}/{row['sample_id']}")
            grouped[(row["sample_id"], normalized(query))].append({"method": method, "query": query})
    plan = []
    for (sample_id, normalized_query), members in sorted(grouped.items()):
        search_id = "search_" + hashlib.sha256(f"{sample_id}\n{normalized_query}".encode()).hexdigest()[:20]
        plan.append(
            {
                "search_id": search_id,
                "sample_id": sample_id,
                "split": "internal_holdout",
                "query": members[0]["query"],
                "normalized_query": normalized_query,
                "num": 10,
                "member_query_ids": sorted(f"{member['method']}::{sample_id}" for member in members),
                "member_methods": sorted(member["method"] for member in members),
            }
        )
    if not 40 <= len(plan) <= 160 or len({row["search_id"] for row in plan}) != len(plan):
        raise ValueError(f"unexpected fresh-search universe: {len(plan)}")

    runner = Path(__file__).with_name("531_run_v6_serper_search.py").resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_paths: dict[str, str] = {}
    for method, rows in method_predictions.items():
        path = output / f"v6_{method}_candidate_policy_selected_queries_no_labels.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        prediction_paths[method] = str(path)
    plan_path = output / "v6_query_candidate_policy_fresh_search_plan_no_labels.jsonl"
    plan_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in plan),
        encoding="utf-8",
    )
    ledger_path = output / "serper_attempts.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    input_paths.update({f"{method}_predictions": path for method, path in prediction_paths.items()})
    freeze = {
        "decision": "DAGIG_V6_SERPER_SEARCH_PLAN_FROZEN",
        "protocol_version": "dagig_v6_query_candidate_policy_internal_holdout_fresh_search_v1",
        "samples": 40,
        "methods": list(METHODS),
        "method_roles": {
            "no_credit": "no_credit",
            "local": "local_fixed_descendant",
            "outcome": "true_outcome_grpo",
            "dagig": "dagig_exact",
        },
        "query_actions": 160,
        "unique_search_calls": len(plan),
        "seeded_cache_hits": 0,
        "new_api_calls_planned": len(plan),
        "serper_num_results": 10,
        "serper_key_env": "SERPER_API_KEY_EVAL",
        "http_attempt_hard_cap": len(plan) + 20,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {
            "search_plan": str(plan_path),
            "seeded_ledger": str(ledger_path),
            **{f"{method}_predictions": path for method, path in prediction_paths.items()},
        },
        "output_hashes": {
            "search_plan": sha256(plan_path),
            "seeded_ledger": sha256(ledger_path),
            **{f"{method}_predictions": sha256(Path(path)) for method, path in prediction_paths.items()},
        },
        "runner_hashes": {"runner": sha256(runner)},
        "selection_uses_gold_terminal_or_qrels": False,
        "old_retrieved_docs_reused": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_QUERY_CANDIDATE_POLICY_FRESH_SEARCH_FREEZE.json"
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "freeze": str(freeze_path),
                "unique_search_calls": len(plan),
                "new_api_calls_planned": len(plan),
                "key_env": freeze["serper_key_env"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
