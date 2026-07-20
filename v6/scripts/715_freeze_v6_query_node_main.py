#!/usr/bin/env python3
"""Freeze the protocol-aligned learned DAG-IG query node."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--dagig_fit_audit", type=Path, required=True)
    parser.add_argument("--dagig_score_audit", type=Path, required=True)
    parser.add_argument("--candidate_audit", type=Path, required=True)
    parser.add_argument("--learned_selector_audit", type=Path, required=True)
    parser.add_argument("--fresh_search_audit", type=Path, required=True)
    parser.add_argument("--free_generation_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "training_freeze": args.training_freeze.resolve(),
        "dagig_fit_audit": args.dagig_fit_audit.resolve(),
        "dagig_score_audit": args.dagig_score_audit.resolve(),
        "candidate_audit": args.candidate_audit.resolve(),
        "learned_selector_audit": args.learned_selector_audit.resolve(),
        "fresh_search_audit": args.fresh_search_audit.resolve(),
        "free_generation_audit": args.free_generation_audit.resolve(),
    }
    values = {key: read_json(path) for key, path in paths.items()}
    expected = {
        "training_freeze": "DAGIG_V6_TRUE_CONTROL_QUERY_TRAINING_FROZEN",
        "dagig_fit_audit": "DAGIG_V6_TRUE_CONTROL_QUERY_TRAIN_FIT_GO",
        "dagig_score_audit": "DAGIG_V6_QUERY_SELECTOR_SCORES_READY",
        "candidate_audit": "DAGIG_V6_QUERY_SELECTOR_CANDIDATES_GO",
        "learned_selector_audit": "DAGIG_V6_LEARNED_QUERY_SELECTOR_GO",
        "fresh_search_audit": "DAGIG_V6_QUERY_POLICY_FRESH_SEARCH_GO",
    }
    for key, decision in expected.items():
        if values[key].get("decision") != decision:
            raise ValueError(f"query main input is not GO: {key}")
    if values["dagig_fit_audit"].get("method") != "dagig_exact":
        raise ValueError("query fit is not the exact DAG-IG method")
    if values["dagig_score_audit"].get("method") != "dagig_exact":
        raise ValueError("query scores are not the exact DAG-IG method")
    if values["free_generation_audit"].get("decision") != "DAGIG_V6_QUERY_POLICY_FRESH_SEARCH_NO_GO":
        raise ValueError("free-generation ablation status changed")
    score_path = Path(values["dagig_score_audit"]["output_paths"]["scores"])
    if sha256(score_path) != values["dagig_score_audit"]["output_hashes"]["scores"]:
        raise ValueError("DAG-IG candidate scores changed")
    adapter_path = Path(values["dagig_fit_audit"]["input_paths"]["adapter_model"])
    if sha256(adapter_path) != values["dagig_fit_audit"]["input_hashes"]["adapter_model"]:
        raise ValueError("DAG-IG query adapter changed")

    learned_metrics = values["learned_selector_audit"]["metrics"]
    fresh_metrics = values["fresh_search_audit"]["metrics"]
    freeze = {
        "decision": "DAGIG_V6_QUERY_NODE_MAIN_FROZEN",
        "protocol_version": "dagig_v6_exact_no_gold_listwise_query_candidate_policy_v1",
        "interface": {
            "candidate_generator": "frozen five-strategy structured query action generator",
            "policy": "Qwen2.5-VL-3B reference-free mean search_query field log-probability",
            "selection": "argmax over the shared legal candidate set",
            "free_generation_is_main_interface": False,
            "search_executes_only_selected_query": True,
        },
        "main_method": "dagig_exact",
        "controls": ["no_credit", "local_fixed_descendant", "true_outcome_grpo"],
        "adapter_model": str(adapter_path),
        "candidate_score_rows": str(score_path),
        "internal_selector_metrics": learned_metrics,
        "fresh_search_downstream_metrics": fresh_metrics,
        "failed_ablation": {
            "name": "free-form single-query generation",
            "decision": values["free_generation_audit"]["decision"],
            "interpretation": "sequence-level mode mixing is not the trained listwise policy interface",
        },
        "gates": {
            "train_fit_go": True,
            "shared_candidate_selector_go": True,
            "fresh_search_go": True,
            "selection_uses_gold_terminal_or_qrels": False,
            "internal_holdout_never_fit": True,
            "dev_sealed": True,
            "test_sealed": True,
        },
        "input_paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "runtime_uses_gold_or_qrels": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "DAGIG_V6_QUERY_NODE_MAIN_FREEZE.json"
    path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": freeze["decision"], "freeze": str(path), "fresh_metrics": fresh_metrics}, indent=2))


if __name__ == "__main__":
    main()
