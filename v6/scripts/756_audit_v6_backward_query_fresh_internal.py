#!/usr/bin/env python3
"""Private clustered audit for fresh backward query-node execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def phrase_contains(text: Any, phrase: Any) -> bool:
    source = normalized(text).split()
    target = normalized(phrase).split()
    return bool(target and any(source[index : index + len(target)] == target for index in range(len(source) - len(target) + 1)))


def canonical_url(value: Any) -> str:
    parsed = urlsplit(str(value or ""))
    host = parsed.netloc.casefold().removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    return host + path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descendant_audit", type=Path, required=True)
    parser.add_argument("--fresh_action_freeze", type=Path, required=True)
    parser.add_argument("--answer_amendment", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    descendant_path = args.descendant_audit.resolve()
    action_freeze_path = args.fresh_action_freeze.resolve()
    amendment_path = args.answer_amendment.resolve()
    descendant = read_json(descendant_path)
    action_freeze = read_json(action_freeze_path)
    amendment = read_json(amendment_path)
    if descendant.get("decision") != "DAGIG_V6_BACKWARD_QUERY_FRESH_DESCENDANTS_READY":
        raise ValueError("fresh descendant predictions are not ready")
    if action_freeze.get("parent_protocol_version") != "dagig_v6_backward_query_fresh_fixed_descendants_v1":
        raise ValueError("fresh action freeze protocol mismatch")
    if amendment.get("decision") != "DAGIG_V6_ANSWER_NORMALIZATION_AMENDMENT_FROZEN":
        raise ValueError("answer normalization amendment is not frozen")
    prediction_path = Path(descendant["output_paths"]["predictions"])
    if sha256(prediction_path) != descendant["output_hashes"]["predictions"]:
        raise ValueError("fresh descendant predictions changed")

    private_labels_path = Path(action_freeze["input_paths"]["private_labels"])
    corpus_path = Path(action_freeze["input_paths"]["corpus"])
    if sha256(private_labels_path) != action_freeze["input_hashes"]["private_labels"]:
        raise ValueError("internal private labels changed")
    if sha256(corpus_path) != action_freeze["input_hashes"]["corpus"]:
        raise ValueError("frozen corpus changed")
    labels = {row["sample_id"]: row for row in read_jsonl(private_labels_path)}
    corpus = {row["doc_id"]: row for row in read_jsonl(corpus_path)}
    baseline = load_module("dagig_query_internal_baseline", Path(amendment["input_paths"]["baseline_eval_utils"]))
    matcher = load_module("dagig_query_internal_matcher", Path(amendment["input_paths"]["amendment_module"]))

    private_rows: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_method_sample: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(prediction_path):
        method = row["method"]
        if method not in METHODS:
            raise ValueError(f"unknown query method: {method}")
        label = labels[row["sample_id"]]
        accepted = [label["gold_answer"], *(label.get("aliases") or [])]
        positive_docs = [corpus[doc_id] for doc_id in label.get("positive_doc_ids") or [] if doc_id in corpus]
        positive_urls = {
            value
            for doc in positive_docs
            if (value := canonical_url(doc.get("final_url") or doc.get("url")))
        }
        docs = row["retrieved_docs"]
        exact_rank = next(
            (
                index + 1
                for index, doc in enumerate(docs)
                if (url := canonical_url(doc.get("url"))) and url in positive_urls
            ),
            0,
        )
        bearing_rank = next(
            (
                index + 1
                for index, doc in enumerate(docs)
                if any(phrase_contains(f"{doc.get('title', '')} {doc.get('snippet', '')}", answer) for answer in accepted)
            ),
            0,
        )
        selected_exact = any(
            (url := canonical_url(doc.get("url"))) and url in positive_urls
            for doc in row["selected_docs"]
        )
        selected_bearing = any(
            any(phrase_contains(f"{doc.get('title', '')} {doc.get('snippet', '')}", answer) for answer in accepted)
            for doc in row["selected_docs"]
        )
        match = matcher.answer_match_details(
            baseline,
            row["final_answer"],
            label["gold_answer"],
            label.get("aliases") or [],
        )
        answer_correct = bool(match["answer_correct"])
        retrieval_support = bool((exact_rank and exact_rank <= 5) or (bearing_rank and bearing_rank <= 5))
        selected_support = bool(selected_exact or selected_bearing)
        evaluated = {
            "method": method,
            "sample_id": row["sample_id"],
            "visual_parent_id": row["visual_parent_id"],
            "visual_field": row["visual_field"],
            "query_id": row["query_id"],
            "search_query": row["search_query"],
            "selected_evidence_strategy": row["selected_evidence_strategy"],
            "final_answer": row["final_answer"],
            "answer_valid": bool(row["answer_valid"]),
            "answer_correct": answer_correct,
            "answer_match_type": match.get("answer_match_type"),
            "exact_qrel_rank": exact_rank,
            "answer_bearing_rank": bearing_rank,
            "retrieval_support_at5": retrieval_support,
            "selected_exact_qrel_support": selected_exact,
            "selected_answer_bearing": selected_bearing,
            "selected_support": selected_support,
            "strict": bool(answer_correct and selected_support),
        }
        private_rows.append(evaluated)
        by_method[method].append(evaluated)
        per_method_sample[method][row["sample_id"]].append(evaluated)

    metrics: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows = by_method[method]
        if not rows:
            raise ValueError(f"no fresh query rows for {method}")
        metrics[method] = {
            "n": len(rows),
            "samples": len(per_method_sample[method]),
            "valid_rate": mean(row["answer_valid"] for row in rows),
            "exact_qrel_r1": mean(0 < row["exact_qrel_rank"] <= 1 for row in rows),
            "exact_qrel_r5": mean(0 < row["exact_qrel_rank"] <= 5 for row in rows),
            "answer_bearing_r5": mean(0 < row["answer_bearing_rank"] <= 5 for row in rows),
            "retrieval_support_at5": mean(row["retrieval_support_at5"] for row in rows),
            "selected_support": mean(row["selected_support"] for row in rows),
            "answer_correct": mean(row["answer_correct"] for row in rows),
            "strict": mean(row["strict"] for row in rows),
            "sample_macro_retrieval_support_at5": mean(
                mean(row["retrieval_support_at5"] for row in sample_rows)
                for sample_rows in per_method_sample[method].values()
            ),
            "sample_macro_strict": mean(
                mean(row["strict"] for row in sample_rows)
                for sample_rows in per_method_sample[method].values()
            ),
            "selected_strategy_counts": dict(sorted(Counter(row["selected_evidence_strategy"] for row in rows).items())),
        }

    sample_ids = sorted(per_method_sample["dagig"])
    rng = random.Random(761943)
    bootstrap: dict[str, list[float]] = {control: [] for control in ("no_credit", "local_ig", "outcome")}
    for _ in range(20000):
        draw = [rng.choice(sample_ids) for _ in sample_ids]
        for control in bootstrap:
            differences = []
            for sample_id in draw:
                dag = mean(row["strict"] for row in per_method_sample["dagig"][sample_id])
                other = mean(row["strict"] for row in per_method_sample[control][sample_id])
                differences.append(dag - other)
            bootstrap[control].append(mean(differences))
    pairwise = {
        f"dagig_minus_{control}_sample_cluster_strict": {
            "point": metrics["dagig"]["sample_macro_strict"] - metrics[control]["sample_macro_strict"],
            "bootstrap_95ci": [percentile(values, 0.025), percentile(values, 0.975)],
        }
        for control, values in bootstrap.items()
    }

    strongest_support = max(metrics[method]["retrieval_support_at5"] for method in ("local_ig", "outcome"))
    strongest_strict = max(metrics[method]["strict"] for method in ("local_ig", "outcome"))
    action_tolerance = 1 / 120
    gates = {
        "complete_equal_120_visual_states_per_method": all(metrics[method]["n"] == 120 for method in METHODS),
        "complete_equal_40_samples_per_method": all(metrics[method]["samples"] == 40 for method in METHODS),
        "answer_valid_at_least_0p98": all(metrics[method]["valid_rate"] >= 0.98 for method in METHODS),
        "dagig_support_improves_no_credit_by_one_state": metrics["dagig"]["retrieval_support_at5"] - metrics["no_credit"]["retrieval_support_at5"] + 1e-12 >= action_tolerance,
        "dagig_support_noninferior_to_strong_control": metrics["dagig"]["retrieval_support_at5"] + action_tolerance + 1e-12 >= strongest_support,
        "dagig_strict_noninferior_to_strong_control": metrics["dagig"]["strict"] + action_tolerance + 1e-12 >= strongest_strict,
        "dagig_strict_not_below_no_credit": metrics["dagig"]["strict"] + 1e-12 >= metrics["no_credit"]["strict"],
        "same_frozen_evidence_and_answer_descendants": True,
        "fresh_search_used_general_not_reserved_eval_key": True,
        "internal_holdout_unused_for_training_or_tuning": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_BACKWARD_QUERY_INTERNAL_GO" if all(gates.values()) else "DAGIG_V6_BACKWARD_QUERY_INTERNAL_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases_path = output / "v6_backward_query_fresh_internal_private_cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(private_rows, key=lambda item: (item["sample_id"], item["visual_parent_id"], item["method"]))),
        encoding="utf-8",
    )
    input_paths = {
        "descendant_audit": descendant_path,
        "fresh_action_freeze": action_freeze_path,
        "answer_amendment": amendment_path,
        "private_labels": private_labels_path,
        "corpus": corpus_path,
    }
    result = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_query_fresh_fixed_descendants_internal_audit_v1",
        "metrics": metrics,
        "pairwise": pairwise,
        "gates": gates,
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "input_hashes": {key: sha256(path) for key, path in input_paths.items()},
        "output_paths": {"private_cases": str(cases_path)},
        "output_hashes": {"private_cases": sha256(cases_path)},
        "gold_or_qrels_loaded_only_by_private_auditor": True,
        "internal_holdout_used_for_training_or_tuning": False,
        "reserved_final_eval_key_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_BACKWARD_QUERY_FRESH_INTERNAL_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
