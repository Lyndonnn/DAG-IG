#!/usr/bin/env python3
"""Freeze runtime features and protocol for a factorized evidence-state critic.

The critic estimates

    P(success | evidence state)
      = P(evidence supports the question)
        * P(answer is correct | supporting evidence).

Only deployable, no-gold signals are materialized here. Private labels are hashed
for later audit integrity but are not parsed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


FEATURE_NAMES = (
    "old_shared_answer_value",
    "old_mode_child_success_probability",
    "answer_policy_entropy",
    "answer_policy_mode_probability",
    "answer_policy_effective_count",
    "answer_action_count",
    "weighted_support_probability",
    "max_support_probability",
    "support_probability_std",
    "weighted_reader_logprob",
    "max_reader_logprob",
    "reader_logprob_std",
    "weighted_unknown_probability",
    "answer_policy_logprob_margin",
    "child_value_margin",
    "mean_selected_normalized_bge",
    "max_selected_normalized_bge",
    "mean_question_keyword_overlap",
    "max_question_keyword_overlap",
    "mean_answer_type_pattern_match",
    "unique_domain_ratio",
    "mean_reciprocal_rank",
    "mean_query_doc_token_coverage",
    "max_query_doc_token_coverage",
)


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


def sigmoid(value: float) -> float:
    value = min(40.0, max(-40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def tokens(value: Any) -> set[str]:
    return set(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def margin(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else 0.0


def weighted(values: list[float], probabilities: list[float]) -> float:
    return sum(value * probability for value, probability in zip(values, probabilities))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_actions", type=Path, required=True)
    parser.add_argument("--shared_answer_values", type=Path, required=True)
    parser.add_argument("--score_files", type=Path, nargs="+", required=True)
    parser.add_argument("--private_support", type=Path, required=True)
    parser.add_argument("--terminal_private", type=Path, required=True)
    parser.add_argument("--categorical_train", type=Path, required=True)
    parser.add_argument("--categorical_internal", type=Path, required=True)
    parser.add_argument("--fitter", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "evidence_actions": args.evidence_actions.resolve(),
        "shared_answer_values": args.shared_answer_values.resolve(),
        "private_support": args.private_support.resolve(),
        "terminal_private": args.terminal_private.resolve(),
        "categorical_train": args.categorical_train.resolve(),
        "categorical_internal": args.categorical_internal.resolve(),
        "fitter": args.fitter.resolve(),
        "auditor": args.auditor.resolve(),
    }
    score_paths = [path.resolve() for path in args.score_files]
    for path in [*paths.values(), *score_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Explicit projection prevents the gold-derived equivalence_logit field in
    # the score shards from entering the runtime feature map.
    scores: dict[str, dict[str, float | bool]] = {}
    for path in score_paths:
        for row in read_jsonl(path):
            answer_id = str(row["answer_action_id"])
            if answer_id in scores:
                raise ValueError(f"duplicate terminal score: {answer_id}")
            scores[answer_id] = {
                "support_logit": float(row["support_logit"]),
                "reader_candidate_mean_logprob": float(row["reader_candidate_mean_logprob"]),
                "is_unknown": bool(row["is_unknown"]),
            }

    values = {
        str(row["evidence_action_id"]): row
        for row in read_jsonl(paths["shared_answer_values"])
    }
    actions = read_jsonl(paths["evidence_actions"])
    if len(actions) != 14770 or len(values) != 14770 or len(scores) != 41273:
        raise ValueError(
            f"unexpected action universe: evidence={len(actions)}, values={len(values)}, scores={len(scores)}"
        )

    feature_rows: list[dict[str, Any]] = []
    missing_scores = set()
    for action in actions:
        evidence_id = str(action["evidence_action_id"])
        value = values[evidence_id]
        answer_ids = [str(item) for item in value["answer_action_ids"]]
        if any(answer_id not in scores for answer_id in answer_ids):
            missing_scores.update(answer_id for answer_id in answer_ids if answer_id not in scores)
            continue
        probabilities = [float(item) for item in value["answer_policy_probabilities"]]
        if len(probabilities) != len(answer_ids) or abs(sum(probabilities) - 1.0) > 1e-8:
            raise ValueError(f"invalid answer policy: {evidence_id}")
        support_probabilities = [sigmoid(float(scores[answer_id]["support_logit"])) for answer_id in answer_ids]
        reader_scores = [float(scores[answer_id]["reader_candidate_mean_logprob"]) for answer_id in answer_ids]
        unknown = [float(bool(scores[answer_id]["is_unknown"])) for answer_id in answer_ids]
        entropy = -sum(probability * math.log(max(probability, 1e-12)) for probability in probabilities)

        docs = action.get("selected_docs") or []
        if len(docs) != 3:
            raise ValueError(f"expected three selected docs: {evidence_id}")
        normalized_bge = [float(doc.get("normalized_bge_score", 0.0)) for doc in docs]
        question_overlap = [float(doc.get("question_keyword_overlap", 0.0)) for doc in docs]
        type_match = [float(doc.get("answer_type_pattern_match", 0.0)) for doc in docs]
        reciprocal_rank = [1.0 / max(1, int(doc.get("rank", 1))) for doc in docs]
        domains = {str(doc.get("domain") or "").casefold() for doc in docs if str(doc.get("domain") or "").strip()}
        query_tokens = tokens(action.get("search_query"))
        query_coverage = []
        for doc in docs:
            doc_tokens = tokens(f"{doc.get('title', '')} {doc.get('snippet', '')}")
            query_coverage.append(len(query_tokens & doc_tokens) / max(1, len(query_tokens)))

        answer_logprobs = [float(item) for item in value["answer_field_logprob_scores"]]
        child_values = [float(item) for item in value["child_success_probabilities"]]
        features = [
            float(value["shared_answer_value"]),
            float(value["mode_child_success_probability"]),
            entropy,
            max(probabilities),
            math.exp(entropy),
            float(len(answer_ids)),
            weighted(support_probabilities, probabilities),
            max(support_probabilities),
            pstdev(support_probabilities) if len(support_probabilities) > 1 else 0.0,
            weighted(reader_scores, probabilities),
            max(reader_scores),
            pstdev(reader_scores) if len(reader_scores) > 1 else 0.0,
            weighted(unknown, probabilities),
            margin(answer_logprobs),
            margin(child_values),
            mean(normalized_bge),
            max(normalized_bge),
            mean(question_overlap),
            max(question_overlap),
            mean(type_match),
            len(domains) / 3.0,
            mean(reciprocal_rank),
            mean(query_coverage),
            max(query_coverage),
        ]
        if len(features) != len(FEATURE_NAMES) or not all(math.isfinite(item) for item in features):
            raise ValueError(f"invalid runtime features: {evidence_id}")
        feature_rows.append(
            {
                "evidence_action_id": evidence_id,
                "query_id": str(action["query_id"]),
                "sample_id": str(action["sample_id"]),
                "partition": str(action["partition"]),
                "evidence_strategy": str(action["evidence_strategy"]),
                "features": features,
            }
        )
    if missing_scores or len(feature_rows) != 14770:
        raise ValueError(f"missing score coverage: {len(missing_scores)} / features={len(feature_rows)}")

    train_rows = sum(row["partition"] == "policy_train" for row in feature_rows)
    internal_rows = sum(row["partition"] == "internal_holdout" for row in feature_rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    feature_path = output / "v6_factorized_evidence_state_features_no_labels.jsonl"
    with feature_path.open("w", encoding="utf-8") as handle:
        for row in sorted(feature_rows, key=lambda item: item["evidence_action_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    gates = {
        "exact_14770_evidence_actions": len(feature_rows) == 14770,
        "exact_11795_2975_partition": train_rows == 11795 and internal_rows == 2975,
        "complete_41273_terminal_scores": len(scores) == 41273,
        "runtime_features_are_no_gold": True,
        "equivalence_logit_not_projected": True,
        "private_labels_not_parsed": True,
        "feature_values_finite": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    input_paths = {key: str(path) for key, path in paths.items()}
    input_paths["score_files"] = [str(path) for path in score_paths]
    input_hashes = {key: sha256(path) for key, path in paths.items()}
    input_hashes["score_files"] = [sha256(path) for path in score_paths]
    protocol = {
        "decision": "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_FROZEN" if all(gates.values()) else "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_NO_GO",
        "protocol_version": "dagig_v6_factorized_no_gold_evidence_state_critic_v1",
        "semantics": "P_success(e)=P_support(e)*P_answer_correct(e|support) under the frozen shared answer policy",
        "feature_names": list(FEATURE_NAMES),
        "feature_families": {
            "frozen_answer_policy": list(FEATURE_NAMES[:15]),
            "retrieval_and_text_support": list(FEATURE_NAMES[15:]),
        },
        "fit": {
            "model": "two standardized logistic regressions",
            "support_head_target": "private evidence support on policy_train only",
            "conditional_answer_head_target": "shared-policy expected answer correctness, fit only where support=1",
            "folds": 5,
            "repeats": 5,
            "l2": 0.01,
            "newton_steps": 40,
            "seed_prefix": "dagig_v6_factorized_evidence_state_v1",
            "probability_clip": [1e-5, 0.99999],
        },
        "train_oof_gates": {
            "support_auc_min": 0.72,
            "strict_brier_improvement_vs_old_min": 0.001,
            "strict_spearman_delta_vs_old_min": 0.02,
            "pair_order_delta_vs_old_min": 0.02,
            "selected_support_noninferiority_vs_outcome_tolerance": 0.01,
            "selected_strict_noninferiority_vs_outcome_tolerance": 0.01,
            "nonconstant_query_group_rate_min": 0.95,
        },
        "development_gates": {
            "support_delta_vs_no_credit_min": 0.0,
            "support_noninferiority_vs_local_tolerance": 0.01,
            "support_noninferiority_vs_outcome_tolerance": 0.01,
            "strict_noninferiority_vs_no_credit_tolerance": 0.0,
            "strict_noninferiority_vs_local_tolerance": 0.015,
            "strict_noninferiority_vs_outcome_tolerance": 0.015,
            "top_action_disagreement_vs_outcome_min": 0.05,
            "selected_evidence_strategies_min": 3,
        },
        "input_paths": input_paths,
        "input_hashes": input_hashes,
        "output_paths": {"features": str(feature_path)},
        "output_hashes": {"features": sha256(feature_path)},
        "gates": gates,
        "gold_or_qrels_in_runtime_features": False,
        "private_label_files_hashed_but_not_parsed": True,
        "internal_holdout_used_for_fit_or_selection": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    protocol_path = output / "DAGIG_V6_FACTORIZED_EVIDENCE_STATE_CRITIC_FREEZE.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "features": len(feature_rows), "freeze": str(protocol_path), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
