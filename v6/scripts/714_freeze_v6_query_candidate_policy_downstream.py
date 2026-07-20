#!/usr/bin/env python3
"""Freeze fixed evidence/reader evaluation for candidate-policy queries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(child.relative_to(root)).encode())
        digest.update(sha256(child).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_freeze", type=Path, required=True)
    parser.add_argument("--search_manifest", type=Path, required=True)
    parser.add_argument("--private_labels", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--bge_model", type=Path, default=Path("/root/dagig_models/bge-reranker-v2-m3"))
    parser.add_argument("--reader_model", type=Path, default=Path("/root/autodl-tmp/hf_models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    search_freeze_path = args.search_freeze.resolve()
    search_manifest_path = args.search_manifest.resolve()
    search = read_json(search_freeze_path)
    manifest = read_json(search_manifest_path)
    if search.get("protocol_version") != "dagig_v6_query_candidate_policy_internal_holdout_fresh_search_v1":
        raise ValueError("candidate-policy fresh-search protocol mismatch")
    if manifest.get("decision") != "DAGIG_V6_SERPER_CACHE_COMPLETE" or manifest.get("freeze") != str(search_freeze_path):
        raise ValueError("complete candidate-policy fresh-search cache required")
    result_path = Path(manifest["output_paths"]["results"])
    if sha256(result_path) != manifest["output_hashes"]["results"]:
        raise ValueError("fresh-search results changed")

    methods = list(search["methods"])
    public_inputs: dict[str, str] = {
        "search_freeze": str(search_freeze_path),
        "search_manifest": str(search_manifest_path),
        "search_plan": search["output_paths"]["search_plan"],
        "search_results": str(result_path),
    }
    for method in methods:
        prediction_path = Path(search["output_paths"][f"{method}_predictions"])
        if sha256(prediction_path) != search["output_hashes"][f"{method}_predictions"]:
            raise ValueError(f"selected candidate-policy queries changed: {method}")
        public_inputs[f"{method}_predictions"] = str(prediction_path)

    private_labels = args.private_labels.resolve()
    corpus = args.corpus.resolve()
    bge_model = args.bge_model.resolve()
    reader_model = args.reader_model.resolve()
    for path in (private_labels, corpus):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (bge_model, reader_model):
        if not path.is_dir():
            raise FileNotFoundError(path)
    runner = Path(__file__).with_name("563_run_v6_query_policy_downstream.py").resolve()
    auditor = Path(__file__).with_name("564_audit_v6_query_policy_downstream.py").resolve()
    for path in (runner, auditor):
        if not path.is_file():
            raise FileNotFoundError(path)

    freeze = {
        "decision": "DAGIG_V6_QUERY_POLICY_DOWNSTREAM_EVAL_FROZEN",
        "protocol_version": "dagig_v6_query_candidate_policy_fixed_evidence_reader_internal_holdout_v1",
        "methods": methods,
        "main_method": search["method_roles"]["dagig"],
        "method_roles": search["method_roles"],
        "samples": 40,
        "query_actions": 160,
        "visual_field": "joint_state",
        "candidate_depth": 10,
        "evidence_selector": {
            "name": "support_diverse_top3",
            "selected_docs": 3,
            "normalized_bge_weight": 0.65,
            "question_keyword_overlap_weight": 0.20,
            "answer_type_pattern_weight": 0.15,
            "unseen_domain_bonus": 0.10,
            "gold_or_answer_content_used": False,
        },
        "bge": {
            "model": str(bge_model),
            "model_tree_sha256": tree_hash(bge_model),
            "max_tokens": 512,
            "batch_size": 64,
            "dtype": "bfloat16",
        },
        "reader": {
            "model": str(reader_model),
            "model_tree_sha256": tree_hash(reader_model),
            "operator": "type_constrained",
            "generation": "deterministic_greedy",
            "attn_implementation": "sdpa",
            "batch_size": 4,
            "max_input_tokens": 4096,
            "max_new_tokens": 96,
            "evidence_snippet_chars": 1200,
            "raw_image_visible": False,
            "visual_state_visible": True,
            "gold_or_qrels_visible": False,
        },
        "primary_metric": "retrieval_support_at5",
        "strict_metric": "answer_correct_and_selected_answer_bearing_evidence",
        "gates": {
            "dagig_support_at5_delta_vs_no_credit_min": 1 / 40,
            "dagig_support_at5_noninferior_strongest_tolerance": 0.0,
            "dagig_strict_noninferior_no_credit_tolerance": 1 / 40,
            "dagig_strict_noninferior_outcome_tolerance": 1 / 40,
            "dagig_valid_rate_min": 0.95,
        },
        "public_input_paths": public_inputs,
        "public_input_hashes": {key: sha256(Path(path)) for key, path in public_inputs.items()},
        "private_audit_input_paths": {"private_labels": str(private_labels), "corpus": str(corpus)},
        "private_audit_input_hashes": {"private_labels": sha256(private_labels), "corpus": sha256(corpus)},
        "runner_hashes": {"runner": sha256(runner), "auditor": sha256(auditor)},
        "candidate_policy_selection_uses_gold_terminal_or_qrels": False,
        "reader_and_evidence_selector_fixed_across_methods": True,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "DAGIG_V6_QUERY_CANDIDATE_POLICY_DOWNSTREAM_EVAL_FREEZE.json"
    path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"freeze": str(path), "methods": methods, "query_actions": 160}, indent=2))


if __name__ == "__main__":
    main()
