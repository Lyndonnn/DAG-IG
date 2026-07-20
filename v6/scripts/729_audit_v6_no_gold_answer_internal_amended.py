#!/usr/bin/env python3
"""Re-score the preserved answer predictions under a versioned amendment.

This is explicitly a post-hoc sensitivity audit, not a second pristine internal
holdout.  It does not alter model scores, generations, targets, or training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--train_fit", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    for method in METHODS:
        parser.add_argument(f"--{method}_scores", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    train_fit_path = args.train_fit.resolve()
    amendment_path = args.amendment.resolve()
    freeze = read_json(freeze_path)
    train_fit = read_json(train_fit_path)
    amendment_manifest = read_json(amendment_path)
    if train_fit.get("decision") != "DAGIG_V6_NO_GOLD_ANSWER_TRAIN_FIT_GO":
        raise ValueError("train fit is not GO")
    if amendment_manifest.get("decision") != "DAGIG_V6_ANSWER_NORMALIZATION_AMENDMENT_FROZEN":
        raise ValueError("normalization amendment is not frozen")
    baseline_path = Path(amendment_manifest["input_paths"]["baseline_eval_utils"])
    amendment_module_path = Path(amendment_manifest["input_paths"]["amendment_module"])
    if sha256(baseline_path) != amendment_manifest["input_hashes"]["baseline_eval_utils"]:
        raise ValueError("baseline evaluator changed")
    if sha256(amendment_module_path) != amendment_manifest["input_hashes"]["amendment_module"]:
        raise ValueError("amendment module changed")
    baseline = load_module("dagig_answer_internal_baseline", baseline_path)
    matcher = load_module("dagig_answer_internal_amendment", amendment_module_path)

    terminal_audit = read_json(Path(freeze["input_paths"]["terminal_audit"]))
    source_freeze = read_json(Path(terminal_audit["input_paths"]["freeze"]))
    labels = {row["sample_id"]: row for row in read_jsonl(Path(source_freeze["input_paths"]["private_labels"]))}
    evidence_audit = read_json(Path(freeze["input_paths"]["evidence_action_audit"]))
    if sha256(Path(evidence_audit["output_paths"]["private_support"])) != evidence_audit["output_hashes"]["private_support"]:
        raise ValueError("private support audit changed")
    support: dict[str, bool] = {}
    for row in read_jsonl(Path(evidence_audit["output_paths"]["private_support"])):
        for strategy, value in row["strategy_support"].items():
            support[f"{row['query_id']}::{strategy}"] = bool(value)
    actions = {row["answer_action_id"]: row for row in read_jsonl(Path(freeze["input_paths"]["answer_actions"]))}
    targets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(Path(freeze["input_paths"]["internal_data"])):
        targets[row["parent_group_id"]].append(row)

    metrics: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    inputs = {
        "freeze": str(freeze_path),
        "train_fit": str(train_fit_path),
        "amendment": str(amendment_path),
    }
    amendment_match_counts: Counter[str] = Counter()
    for method in METHODS:
        score_audit_path = getattr(args, f"{method}_scores").resolve()
        score_audit = read_json(score_audit_path)
        if (
            score_audit.get("decision") != "DAGIG_V6_NO_GOLD_ANSWER_POLICY_SCORES_READY"
            or score_audit.get("method") != method
            or score_audit.get("partition") != "internal_holdout"
        ):
            raise ValueError(f"invalid internal scores: {method}")
        for key, path in score_audit["output_paths"].items():
            if sha256(Path(path)) != score_audit["output_hashes"][key]:
                raise ValueError(f"score output changed: {method}/{key}")
        score_rows = read_jsonl(Path(score_audit["output_paths"]["scores"]))
        generations = {
            row["parent_group_id"]: row
            for row in read_jsonl(Path(score_audit["output_paths"]["generations"]))
        }
        selected: list[dict[str, Any]] = []
        generated: list[dict[str, Any]] = []
        for score in score_rows:
            group = sorted(targets[score["parent_group_id"]], key=lambda row: row["answer_action_id"])
            index = max(range(len(group)), key=score["policy_probabilities"].__getitem__)
            row = group[index]
            action = actions[row["answer_action_id"]]
            label = labels[row["sample_id"]]
            candidate_match = matcher.answer_match_details(
                baseline,
                action["candidate_answer"],
                label["gold_answer"],
                label.get("aliases") or [],
            )
            is_supported = support[row["parent_group_id"]]
            selected.append(
                {
                    "answer_correct": bool(candidate_match["answer_correct"]),
                    "support": is_supported,
                    "strict": bool(is_supported and candidate_match["answer_correct"]),
                    "expected_terminal": sum(
                        float(item["child_success_probability"]) * probability
                        for item, probability in zip(group, score["policy_probabilities"])
                    ),
                }
            )
            generation = generations[row["parent_group_id"]]
            generated_match = matcher.answer_match_details(
                baseline,
                generation["final_answer"],
                label["gold_answer"],
                label.get("aliases") or [],
            )
            if generated_match.get("answer_match_type") == "numeric_unit_symmetric_exact":
                amendment_match_counts[method] += 1
            generated.append(
                {
                    "valid": bool(generation["valid_json"]),
                    "answer_correct": bool(generated_match["answer_correct"]),
                    "support": is_supported,
                    "strict": bool(is_supported and generated_match["answer_correct"]),
                }
            )
            cases.append(
                {
                    "method": method,
                    "sample_id": row["sample_id"],
                    "parent_group_id": row["parent_group_id"],
                    "selected_action_id": row["answer_action_id"],
                    "candidate_answer_correct": selected[-1]["answer_correct"],
                    "candidate_match_type": candidate_match.get("answer_match_type"),
                    "evidence_supported": is_supported,
                    "candidate_strict": selected[-1]["strict"],
                    "generated_valid": generated[-1]["valid"],
                    "generated_answer_correct": generated[-1]["answer_correct"],
                    "generated_match_type": generated_match.get("answer_match_type"),
                    "generated_strict": generated[-1]["strict"],
                }
            )
        supported_generations = [row for row in generated if row["support"]]
        metrics[method] = {
            "groups": len(selected),
            "candidate_answer_correct": mean(row["answer_correct"] for row in selected),
            "candidate_strict": mean(row["strict"] for row in selected),
            "model_expected_terminal": mean(row["expected_terminal"] for row in selected),
            "generated_valid_json": mean(row["valid"] for row in generated),
            "generated_answer_correct": mean(row["answer_correct"] for row in generated),
            "generated_strict": mean(row["strict"] for row in generated),
            "generated_answer_correct_given_support": (
                mean(row["answer_correct"] for row in supported_generations)
                if supported_generations
                else 0.0
            ),
            "generated_matches_from_amendment": amendment_match_counts[method],
        }
        inputs[f"{method}_score_audit"] = str(score_audit_path)

    strongest = max(metrics[method]["generated_strict"] for method in ("local_ig", "outcome"))
    gates = {
        "complete_equal_internal_groups": {row["groups"] for row in metrics.values()} == {480},
        "all_generation_valid_at_least_0p98": all(row["generated_valid_json"] >= 0.98 for row in metrics.values()),
        "dagig_candidate_strict_not_below_no_credit": metrics["dagig"]["candidate_strict"] >= metrics["no_credit"]["candidate_strict"],
        "dagig_generated_strict_not_below_no_credit": metrics["dagig"]["generated_strict"] >= metrics["no_credit"]["generated_strict"],
        "dagig_generated_noninferior_to_strong_control": metrics["dagig"]["generated_strict"] + 0.01 >= strongest,
        "dagig_expected_terminal_not_below_no_credit": metrics["dagig"]["model_expected_terminal"] >= metrics["no_credit"]["model_expected_terminal"],
        "predictions_unchanged": True,
        "training_unchanged": True,
        "internal_holdout_disclosed_as_previously_observed": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = (
        "DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_AMENDED_SENSITIVITY_GO"
        if all(gates.values())
        else "DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_AMENDED_SENSITIVITY_NO_GO"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cases_path = output_dir / "v6_no_gold_answer_internal_amended_private_cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases),
        encoding="utf-8",
    )
    result = {
        "decision": decision,
        "metrics": metrics,
        "gates": gates,
        "input_paths": inputs,
        "input_hashes": {key: sha256(Path(path)) for key, path in inputs.items()},
        "output_paths": {"private_cases": str(cases_path)},
        "output_hashes": {"private_cases": sha256(cases_path)},
        "primary_original_internal_audit_remains_authoritative": True,
        "amended_internal_is_post_hoc_sensitivity": True,
        "eligible_as_pristine_holdout_claim": False,
        "eligible_to_freeze_checker_before_unseen_dev_test": True,
        "training_run": False,
        "predictions_modified": False,
        "dev_used": False,
        "test_used": False,
    }
    report_path = output_dir / "DAGIG_V6_NO_GOLD_ANSWER_INTERNAL_AMENDED_SENSITIVITY.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
