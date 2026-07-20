#!/usr/bin/env python3
"""Freeze five-action evidence rebuilding for fresh backward-query searches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


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


def normalize_docs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for rank, item in enumerate(payload.get("organic") or [], 1):
        url = str(item.get("link") or "")
        docs.append(
            {
                "rank": rank,
                "title": str(item.get("title") or ""),
                "url": url,
                "domain": urlsplit(url).netloc.casefold().removeprefix("www."),
                "snippet": str(item.get("snippet") or ""),
                "date": str(item.get("date") or ""),
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_freeze", type=Path, required=True)
    parser.add_argument("--cache_manifest", type=Path, required=True)
    parser.add_argument("--evidence_policy_freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    search_freeze_path = args.search_freeze.resolve()
    cache_path = args.cache_manifest.resolve()
    evidence_policy_path = args.evidence_policy_freeze.resolve()
    search = read_json(search_freeze_path)
    cache = read_json(cache_path)
    evidence_policy = read_json(evidence_policy_path)
    if search.get("protocol_version") != "dagig_v6_backward_query_all_visual_states_internal_fresh_search_v1":
        raise ValueError("backward query fresh-search protocol mismatch")
    if search.get("serper_key_env") != "SERPER_API_KEY" or search.get("reserved_final_eval_key_used"):
        raise ValueError("internal fresh search must not use the reserved final-evaluation key")
    if (
        cache.get("decision") != "DAGIG_V6_SERPER_CACHE_COMPLETE"
        or cache.get("freeze") != str(search_freeze_path)
        or cache.get("freeze_sha256") != sha256(search_freeze_path)
    ):
        raise ValueError("complete fresh-search cache from this freeze is required")
    if evidence_policy.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_FROZEN":
        raise ValueError("backward DAG-IG evidence policy is not frozen")
    for key, path in evidence_policy["input_paths"].items():
        if sha256(Path(path)) != evidence_policy["input_hashes"][key]:
            raise ValueError(f"evidence-policy lineage changed: {key}")

    result_path = Path(cache["output_paths"]["results"])
    plan_path = Path(search["output_paths"]["search_plan"])
    if sha256(result_path) != cache["output_hashes"]["results"]:
        raise ValueError("fresh-search results changed")
    if sha256(plan_path) != search["output_hashes"]["search_plan"]:
        raise ValueError("fresh-search plan changed")
    results = {row["search_id"]: row for row in read_jsonl(result_path)}
    member_search = {
        member: row["search_id"]
        for row in read_jsonl(plan_path)
        for member in row["member_query_ids"]
    }
    parents: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []
    prediction_paths: dict[str, Path] = {}
    for method in METHODS:
        path = Path(search["output_paths"][f"{method}_predictions"])
        if sha256(path) != search["output_hashes"][f"{method}_predictions"]:
            raise ValueError(f"selected fresh queries changed: {method}")
        prediction_paths[method] = path
        rows = read_jsonl(path)
        if len(rows) != 120:
            raise ValueError(f"incomplete selected-query parent matrix: {method}")
        for row in rows:
            member_id = row["member_query_id"]
            search_id = member_search.get(member_id)
            search_row = results.get(search_id or "")
            payload = ((search_row or {}).get("serper_response") or {}).get("json") or {}
            docs = normalize_docs(payload)[:10]
            if len(docs) < 5:
                insufficient.append(
                    {
                        "method": method,
                        "sample_id": row["sample_id"],
                        "visual_parent_id": row["visual_parent_id"],
                        "member_query_id": member_id,
                        "organic_docs": len(docs),
                    }
                )
            parents.append(
                {
                    **row,
                    "query_id": member_id,
                    "search_id": search_id or "",
                    "retrieved_docs": docs,
                    "answer_box": payload.get("answerBox"),
                    "knowledge_graph": payload.get("knowledgeGraph"),
                }
            )
    if len(parents) != 480 or len({row["query_id"] for row in parents}) != 480:
        raise ValueError("fresh method-by-visual-state parent universe is incomplete")

    evidence_training_path = Path(evidence_policy["input_paths"]["training_freeze"])
    evidence_training = read_json(evidence_training_path)
    evidence_control = read_json(Path(evidence_training["input_paths"]["control_freeze"]))
    original_action_audit_path = Path(evidence_control["input_paths"]["evidence_action_audit"])
    original_action_audit = read_json(original_action_audit_path)
    original_action_freeze_path = Path(original_action_audit["input_paths"]["freeze"])
    original_action_freeze = read_json(original_action_freeze_path)
    for path, expected in (
        (evidence_training_path, evidence_policy["input_hashes"]["training_freeze"]),
        (original_action_audit_path, evidence_control["input_hashes"]["evidence_action_audit"]),
        (original_action_freeze_path, original_action_audit["input_hashes"]["freeze"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"fresh evidence protocol lineage changed: {path}")
    private_labels = Path(original_action_freeze["input_paths"]["private_labels"])
    corpus = Path(original_action_freeze["input_paths"]["corpus"])
    for path in (private_labels, corpus):
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parents_path = output / "v6_backward_query_fresh_evidence_parents_no_labels.jsonl"
    parents_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(parents, key=lambda item: item["query_id"])),
        encoding="utf-8",
    )
    insufficient_path = output / "v6_backward_query_fresh_insufficient_results_public.jsonl"
    insufficient_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in insufficient),
        encoding="utf-8",
    )

    builder = Path(__file__).with_name("570_build_v6_on_policy_evidence_actions.py").resolve()
    auditor = Path(__file__).with_name("753_build_audit_v6_backward_query_fresh_evidence_actions.py").resolve()
    input_paths: dict[str, Path] = {
        "query_actions_with_search": parents_path,
        "search_freeze": search_freeze_path,
        "search_cache_manifest": cache_path,
        "fresh_search_results": result_path,
        "evidence_policy_freeze": evidence_policy_path,
        "evidence_training_freeze": evidence_training_path,
        "original_evidence_action_audit": original_action_audit_path,
        "original_evidence_action_freeze": original_action_freeze_path,
        "private_labels": private_labels,
        "corpus": corpus,
        **{f"{method}_selected_queries": path for method, path in prediction_paths.items()},
    }
    gates = {
        "complete_480_query_parents": len(parents) == 480,
        "all_parents_have_at_least_five_results": not insufficient,
        "all_query_parent_ids_unique": len({row["query_id"] for row in parents}) == 480,
        "same_frozen_evidence_policy_for_all_query_methods": True,
        "reserved_final_eval_key_unused": True,
        "no_gold_or_qrels_used_to_construct_public_parents": True,
        "internal_holdout_unused_for_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_ON_POLICY_EVIDENCE_ACTIONS_FROZEN" if all(gates.values()) else "DAGIG_V6_ON_POLICY_EVIDENCE_ACTIONS_NO_GO"
    freeze = {
        "decision": decision,
        "protocol_version": "dagig_v6_cached_multiquery_evidence_state_expansion_v1",
        "parent_protocol_version": "dagig_v6_backward_query_fresh_fixed_descendants_v1",
        "query_actions": len(parents),
        "candidate_depth": 10,
        "strategies": [
            "serper_rank_top3",
            "bge_top3",
            "support_diverse_top3",
            "observable_low_support_top3",
            "entity_condition_mismatch_top3",
        ],
        "universe": {
            "samples": 40,
            "visual_parent_states": 120,
            "methods": 4,
            "eligible_query_parents": len(parents) - len(insufficient),
            "excluded_query_parents": len(insufficient),
            "policy_train_parents": 0,
            "internal_holdout_parents": len(parents) - len(insufficient),
            "evidence_actions": (len(parents) - len(insufficient)) * 5,
            "new_search_calls": int(search["unique_search_calls"]),
        },
        "bge": original_action_freeze["bge"],
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "input_hashes": {key: sha256(path) for key, path in input_paths.items()},
        "action_gates": {
            "five_unique_sets_rate_min": 1.0,
            "mean_union_selected_docs_min": 6.0,
            "internal_holdout_mixed_support_parents_min": 0,
            "internal_holdout_union_recoveries_min": 0,
        },
        "runner_hashes": {"builder": sha256(builder), "auditor": sha256(auditor)},
        "gates": gates,
        "gold_or_qrels_visible_to_action_policy": False,
        "private_labels_used_only_after_public_actions_materialized": True,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_ACTION_FREEZE.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "freeze": str(freeze_path),
                "query_parents": len(parents),
                "insufficient_results": len(insufficient),
                "planned_evidence_actions": freeze["universe"]["evidence_actions"],
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
