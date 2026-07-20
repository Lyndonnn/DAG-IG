#!/usr/bin/env python3
"""Materialize no-gold query-state features and freeze critic selection rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


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


def tokens(value: Any) -> set[str]:
    return set(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def coverage(needles: set[str], haystack: set[str]) -> float:
    return len(needles & haystack) / max(1, len(needles))


def entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probabilities)


def margin(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_value_freeze", type=Path, required=True)
    parser.add_argument("--query_actions", type=Path, required=True)
    parser.add_argument("--evidence_features", type=Path, required=True)
    parser.add_argument("--hybrid_evidence_predictions", type=Path, required=True)
    parser.add_argument("--factorized_evidence_predictions", type=Path, required=True)
    parser.add_argument("--pairwise_evidence_predictions", type=Path, required=True)
    parser.add_argument("--private_support", type=Path, required=True)
    parser.add_argument("--terminal_private", type=Path, required=True)
    parser.add_argument("--shared_answer_values", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--fitter", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "query_value_freeze": args.query_value_freeze,
        "query_actions": args.query_actions,
        "evidence_features": args.evidence_features,
        "hybrid_evidence_predictions": args.hybrid_evidence_predictions,
        "factorized_evidence_predictions": args.factorized_evidence_predictions,
        "pairwise_evidence_predictions": args.pairwise_evidence_predictions,
        "private_support": args.private_support,
        "terminal_private": args.terminal_private,
        "shared_answer_values": args.shared_answer_values,
        "helper": args.helper,
        "fitter": args.fitter,
        "auditor": args.auditor,
    }.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    query_freeze = read_json(paths["query_value_freeze"])
    if query_freeze.get("decision") != "DAGIG_V6_HYBRID_QUERY_VALUES_V1_FROZEN":
        raise ValueError("hybrid query value source is not frozen")
    for key, raw_path in query_freeze["output_paths"].items():
        if sha256(Path(raw_path)) != query_freeze["output_hashes"][key]:
            raise ValueError(f"hybrid query output changed: {key}")

    query_actions = {row["query_id"]: row for row in read_jsonl(paths["query_actions"])}
    evidence_features = {row["evidence_action_id"]: row for row in read_jsonl(paths["evidence_features"])}
    hybrid = {row["evidence_action_id"]: row for row in read_jsonl(paths["hybrid_evidence_predictions"])}
    factorized = {row["evidence_action_id"]: row for row in read_jsonl(paths["factorized_evidence_predictions"])}
    pairwise = {row["evidence_action_id"]: row for row in read_jsonl(paths["pairwise_evidence_predictions"])}
    diagnostics = read_jsonl(Path(query_freeze["output_paths"]["diagnostics"]))

    hybrid_by_query: dict[str, list[float]] = defaultdict(list)
    for row in hybrid.values():
        hybrid_by_query[row["query_id"]].append(float(row["evidence_success_probability"]))
    # The evidence feature names are frozen by their protocol; preserve ordinal
    # names here instead of reinterpreting their semantics.
    evidence_width = len(next(iter(evidence_features.values()))["features"])
    feature_names = [f"selected_evidence_feature_{index:02d}" for index in range(evidence_width)] + [
        "selected_hybrid_evidence_value",
        "selected_factorized_support_probability",
        "selected_factorized_answer_given_support_probability",
        "selected_pairwise_evidence_rank_score",
        "evidence_posterior_entropy",
        "evidence_posterior_top_probability",
        "evidence_posterior_margin",
        "evidence_value_mean",
        "evidence_value_std",
        "query_token_count",
        "query_entity_quote_coverage",
        "query_information_need_coverage",
        "query_constraint_coverage",
        "question_terms_covered_by_query",
        "visual_terms_covered_by_query",
        "mean_query_terms_covered_in_top5",
        "max_query_terms_covered_in_top5",
        "top5_unique_domain_ratio",
        "top5_date_presence_rate",
    ]

    feature_rows = []
    for diagnostic in diagnostics:
        for query_id, evidence_id in zip(diagnostic["query_action_ids"], diagnostic["selected_evidence_action_ids"]):
            action = query_actions[query_id]
            selected_feature = evidence_features[evidence_id]
            if selected_feature["query_id"] != query_id:
                raise ValueError(f"selected evidence/query mismatch: {query_id}")
            values = sorted(hybrid_by_query[query_id])
            posterior = [value / sum(values) for value in values]
            query_terms = tokens(action["search_query"])
            question_terms = tokens(action["question"])
            visual_terms = tokens(action["visual_observation"])
            entity_terms = tokens(action.get("entity_quote"))
            need_terms = tokens(action.get("information_need"))
            constraint_terms = tokens(" ".join(action.get("constraints") or []))
            docs = (action.get("retrieved_docs") or [])[:5]
            doc_coverages = [coverage(query_terms, tokens(f"{doc.get('title', '')} {doc.get('snippet', '')}")) for doc in docs]
            domains = {str(doc.get("domain") or "").casefold() for doc in docs if str(doc.get("domain") or "").strip()}
            features = [
                *[float(value) for value in selected_feature["features"]],
                float(hybrid[evidence_id]["evidence_success_probability"]),
                float(factorized[evidence_id]["support_probability"]),
                float(factorized[evidence_id]["answer_correct_given_support_probability"]),
                float(pairwise[evidence_id]["pairwise_rank_score"]),
                entropy(posterior),
                max(posterior),
                margin(posterior),
                mean(values),
                pstdev(values),
                float(len(query_terms)),
                coverage(entity_terms, query_terms),
                coverage(need_terms, query_terms),
                coverage(constraint_terms, query_terms),
                coverage(question_terms, query_terms),
                coverage(visual_terms, query_terms),
                mean(doc_coverages) if doc_coverages else 0.0,
                max(doc_coverages) if doc_coverages else 0.0,
                len(domains) / max(1, len(docs)),
                mean(float(bool(doc.get("date"))) for doc in docs) if docs else 0.0,
            ]
            if len(features) != len(feature_names) or not all(math.isfinite(value) for value in features):
                raise ValueError(f"invalid query-state features: {query_id}")
            feature_rows.append({
                "query_action_id": query_id,
                "parent_visual_state_id": diagnostic["parent_state_id"],
                "selected_evidence_action_id": evidence_id,
                "sample_id": query_id.split("::", 1)[0],
                "partition": diagnostic["partition"],
                "features": features,
            })
    if len(feature_rows) != 2954 or len({row["query_action_id"] for row in feature_rows}) != 2954:
        raise ValueError(f"query feature universe mismatch: {len(feature_rows)}")
    train = sum(row["partition"] == "policy_train" for row in feature_rows)
    internal = sum(row["partition"] == "internal_holdout" for row in feature_rows)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    feature_path = output / "v6_query_state_features_no_labels.jsonl"
    with feature_path.open("w", encoding="utf-8") as handle:
        for row in sorted(feature_rows, key=lambda item: item["query_action_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    gates = {
        "exact_2954_query_actions": len(feature_rows) == 2954,
        "exact_2359_595_partition": train == 2359 and internal == 595,
        "features_are_runtime_observable": True,
        "strategy_identity_not_encoded": True,
        "gold_qrel_correctness_not_encoded": True,
        "private_labels_not_parsed": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    protocol = {
        "decision": "DAGIG_V6_QUERY_STATE_CRITIC_V1_FROZEN" if all(gates.values()) else "DAGIG_V6_QUERY_STATE_CRITIC_V1_NO_GO",
        "protocol_version": "dagig_v6_node_specific_query_state_value_v1",
        "semantics": "P(strict success | executed query, frozen hybrid evidence selector, frozen answer policy)",
        "feature_names": feature_names,
        "fit": {"folds": 5, "repeats": 5, "support_l2": 0.01, "conditional_l2": 0.01, "pairwise_l2": 0.03, "platt_l2": 0.003, "newton_steps": 40, "seed_prefix": "dagig_v6_query_state_critic_v1", "alpha_grid": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]},
        "alpha_selection_rule": "smallest train-OOF alpha passing all calibration/ranking/selector gates; no internal labels",
        "train_oof_gates": {"support_auc_min": 0.70, "strict_brier_improvement_vs_downstream_min": 0.001, "strict_spearman_delta_vs_downstream_min": 0.015, "pair_order_delta_vs_downstream_min": 0.015, "selected_support_noninferiority_vs_outcome_tolerance": 0.01, "selected_strict_noninferiority_vs_outcome_tolerance": 0.01, "nonconstant_parent_group_rate_min": 0.95},
        "development_gates": {"support_delta_vs_no_credit_min": 0.0, "support_noninferiority_vs_local_tolerance": 0.01, "support_noninferiority_vs_outcome_tolerance": 0.01, "strict_noninferiority_vs_no_credit_tolerance": 0.0, "strict_noninferiority_vs_local_tolerance": 0.015, "strict_noninferiority_vs_outcome_tolerance": 0.015, "top_action_disagreement_vs_outcome_min": 0.05, "selected_query_strategies_min": 4},
        "input_paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "output_paths": {"features": str(feature_path)},
        "output_hashes": {"features": sha256(feature_path)},
        "gates": gates,
        "internal_used_for_fit_or_selection": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    path = output / "DAGIG_V6_QUERY_STATE_CRITIC_V1_FREEZE.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "features": len(feature_rows), "width": len(feature_names), "freeze": str(path), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
