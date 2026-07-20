#!/usr/bin/env python3
"""Audit the failed evidence train-fit without opening any held-out labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


METHODS = ("no_credit", "local_ig", "outcome", "dagig")
MARGIN_BINS = (
    ("[0,0.01)", 0.0, 0.01),
    ("[0.01,0.03)", 0.01, 0.03),
    ("[0.03,0.05)", 0.03, 0.05),
    ("[0.05,0.10)", 0.05, 0.10),
    ("[0.10,inf)", 0.10, float("inf")),
)


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


def score_rows(audit_path: Path, method: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    audit = read_json(audit_path)
    if (
        audit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_POLICY_SCORES_READY"
        or audit.get("method") != method
        or audit.get("partition") != "policy_train"
    ):
        raise ValueError(f"invalid policy-train scores for {method}")
    path = Path(audit["output_paths"]["scores"])
    if sha256(path) != audit["output_hashes"]["scores"]:
        raise ValueError(f"score rows changed for {method}")
    return audit, {row["parent_group_id"]: row for row in read_jsonl(path)}


def margin_bin(value: float) -> str:
    for name, lower, upper in MARGIN_BINS:
        if lower <= value < upper:
            return name
    raise AssertionError(value)


def summarize_method(
    rows: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    target_key: str,
    beta: float,
) -> dict[str, Any]:
    bins: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    tvs: list[float] = []
    regrets: list[float] = []
    selected_target_mass: list[float] = []
    pairwise: list[float] = []
    agreements: list[int] = []
    for group_id in sorted(rows):
        row = rows[group_id]
        target = np.asarray(row[target_key], dtype=np.float64)
        behavior = np.asarray(row["behavior_probabilities"], dtype=np.float64)
        delta = np.asarray(current[group_id]["field_logprob_scores"]) - np.asarray(
            reference[group_id]["field_logprob_scores"]
        )
        logits = np.log(behavior) + beta * delta
        policy = np.exp(logits - logits.max())
        policy /= policy.sum()
        target_top = int(target.argmax())
        policy_top = int(policy.argmax())
        ordered = np.sort(target)
        margin = float(ordered[-1] - ordered[-2])
        label = margin_bin(margin)
        regret = float(target[target_top] - target[policy_top])
        tv = float(0.5 * np.abs(target - policy).sum())
        comparisons = [
            int((policy[i] - policy[j]) * (target[i] - target[j]) > 0)
            for i in range(len(target))
            for j in range(i + 1, len(target))
            if abs(float(target[i] - target[j])) > 1e-12
        ]
        agreement = int(target_top == policy_top)
        tvs.append(tv)
        regrets.append(regret)
        selected_target_mass.append(float(target[policy_top]))
        pairwise.append(mean(comparisons) if comparisons else 1.0)
        agreements.append(agreement)
        bins[label]["agreement"].append(agreement)
        bins[label]["regret"].append(regret)
        bins[label]["tv"].append(tv)
    return {
        "groups": len(rows),
        "mean_tv": mean(tvs),
        "mean_top_action_regret": mean(regrets),
        "p95_top_action_regret": float(np.quantile(regrets, 0.95)),
        "mean_selected_target_mass": mean(selected_target_mass),
        "mean_pairwise_order_accuracy": mean(pairwise),
        "top_action_agreement": mean(agreements),
        "margin_bins": {
            name: {
                "groups": len(bins[name]["agreement"]),
                "top_action_agreement": mean(bins[name]["agreement"]) if bins[name]["agreement"] else None,
                "mean_top_action_regret": mean(bins[name]["regret"]) if bins[name]["regret"] else None,
                "mean_tv": mean(bins[name]["tv"]) if bins[name]["tv"] else None,
            }
            for name, _, _ in MARGIN_BINS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--train_fit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.training_freeze.resolve()
    fit_path = args.train_fit.resolve()
    freeze = read_json(freeze_path)
    fit = read_json(fit_path)
    if freeze.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAINING_FROZEN":
        raise ValueError("evidence training protocol is not frozen")
    if fit.get("decision") != "DAGIG_V6_BACKWARD_EVIDENCE_TRAIN_FIT_NO_GO":
        raise ValueError("representation audit is only valid after the frozen train-fit NO-GO")
    if fit.get("internal_holdout_used") or fit.get("dev_used") or fit.get("test_used"):
        raise ValueError("held-out data was opened before the representation audit")

    control = read_json(Path(freeze["input_paths"]["control_freeze"]))
    train_path = Path(control["output_paths"]["train_data"])
    rows_list = read_jsonl(train_path)
    rows = {row["parent_group_id"]: row for row in rows_list}
    _, reference = score_rows(Path(fit["input_paths"]["reference_score_audit"]), "reference")
    if set(reference) != set(rows):
        raise ValueError("reference score universe differs from policy-train groups")

    metrics: dict[str, Any] = {}
    score_audits: dict[str, str] = {}
    for method in METHODS:
        path = Path(fit["input_paths"][f"{method}_score_audit"])
        _, current = score_rows(path, method)
        if set(current) != set(rows):
            raise ValueError(f"score universe differs for {method}")
        metrics[method] = summarize_method(
            rows,
            reference,
            current,
            freeze["target_keys"][method],
            float(freeze["training"]["beta"]),
        )
        score_audits[method] = str(path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(freeze["base_model"], local_files_only=True)
    token_lengths: Counter[int] = Counter()
    overlaps: list[float] = []
    for row in rows_list:
        token_sets: list[set[int]] = []
        for completion in row["completions"]:
            token_ids = tokenizer(completion, add_special_tokens=False).input_ids
            token_lengths[len(token_ids)] += 1
            token_sets.append(set(token_ids))
        for left in range(len(token_sets)):
            for right in range(left + 1, len(token_sets)):
                union = token_sets[left] | token_sets[right]
                overlaps.append(len(token_sets[left] & token_sets[right]) / len(union))

    prompt_enumerates_legal_actions = sum(
        "Legal candidate evidence-set actions" in row["prompt"] for row in rows_list
    )
    high_margin_needed = {}
    for method in ("local_ig", "outcome", "dagig"):
        high = fit["metrics"][method]
        count = int(high["margin_ge_0p05_groups"])
        observed = float(high["margin_ge_0p05_top_action_agreement"])
        high_margin_needed[method] = max(0, int(np.ceil(0.85 * count - observed * count - 1e-12)))

    representation = {
        "source_completion_token_length_counts": {str(key): value for key, value in sorted(token_lengths.items())},
        "mean_pairwise_completion_token_jaccard": mean(overlaps),
        "min_pairwise_completion_token_jaccard": min(overlaps),
        "max_pairwise_completion_token_jaccard": max(overlaps),
        "prompts_enumerating_the_five_legal_actions": prompt_enumerates_legal_actions,
        "policy_train_groups": len(rows_list),
        "single_token_categorical_labels": {
            label: tokenizer(label, add_special_tokens=False).input_ids for label in "ABCDE"
        },
        "diagnosis": (
            "The scorer treats five legal evidence sets as separate actions, but the prompt does not enumerate "
            "those legal sets and the optimized JSON arrays share most output tokens. This creates a structured, "
            "partially coupled output space instead of five explicit categorical action logits."
        ),
    }
    gates = {
        "source_train_fit_was_no_go": True,
        "policy_train_only": True,
        "internal_holdout_sealed": True,
        "dev_sealed": True,
        "test_sealed": True,
        "five_legal_actions_not_explicit_in_prompt": prompt_enumerates_legal_actions == 0,
        "source_action_completions_are_multitoken": min(token_lengths) > 1,
        "single_token_categorical_labels_available": all(
            len(tokenizer(label, add_special_tokens=False).input_ids) == 1 for label in "ABCDE"
        ),
    }
    decision = (
        "DAGIG_V6_BACKWARD_EVIDENCE_CATEGORICAL_REPAIR_JUSTIFIED"
        if all(gates.values())
        else "DAGIG_V6_BACKWARD_EVIDENCE_CATEGORICAL_REPAIR_NOT_JUSTIFIED"
    )
    input_paths = {
        "training_freeze": str(freeze_path),
        "train_fit": str(fit_path),
        "control_freeze": str(Path(freeze["input_paths"]["control_freeze"])),
        "train_data": str(train_path),
        "reference_score_audit": fit["input_paths"]["reference_score_audit"],
        **{f"{method}_score_audit": path for method, path in score_audits.items()},
    }
    result = {
        "decision": decision,
        "gates": gates,
        "metrics": metrics,
        "representation": representation,
        "high_margin_additional_correct_groups_needed_for_old_gate": high_margin_needed,
        "repair_contract": {
            "same_five_action_universe": True,
            "same_targets": True,
            "same_initializer_and_optimizer_controls": True,
            "new_prompt_explicitly_enumerates_A_to_E_actions": True,
            "optimized_output_is_one_categorical_token": True,
            "old_frozen_thresholds_remain_unchanged": True,
            "all_four_methods_must_be_retrained": True,
            "internal_dev_test_remain_sealed_until_new_train_fit_go": True,
        },
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "DAGIG_V6_BACKWARD_EVIDENCE_ACTION_REPRESENTATION_AUDIT.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Backward Evidence Action Representation Audit",
        "",
        f"- Decision: `{decision}`",
        f"- Policy-train groups: `{len(rows_list)}`",
        f"- Mean completion-token Jaccard: `{representation['mean_pairwise_completion_token_jaccard']:.4f}`",
        f"- Existing prompt enumerates legal five actions: `{prompt_enumerates_legal_actions}/{len(rows_list)}`",
        f"- Existing completion token lengths: `{dict(sorted(token_lengths.items()))}`",
        f"- Old high-margin gate additional correct groups needed: `{high_margin_needed}`",
        "",
        "## Train-only Fit Diagnostics",
        "",
        "| method | TV | top agreement | mean regret | p95 regret | pairwise order |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = metrics[method]
        report.append(
            f"| {method} | {value['mean_tv']:.4f} | {value['top_action_agreement']:.4f} | "
            f"{value['mean_top_action_regret']:.4f} | {value['p95_top_action_regret']:.4f} | "
            f"{value['mean_pairwise_order_accuracy']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Diagnosis",
            "",
            representation["diagnosis"],
            "",
            "The repair must preserve the same five evidence sets and target distributions. It may only make the "
            "legal action universe explicit and replace the overlapping multi-token set output with one A-E token. "
            "All four methods must restart from the shared initializer, and held-out partitions remain sealed.",
        ]
    )
    (output / "DAGIG_V6_BACKWARD_EVIDENCE_ACTION_REPRESENTATION_AUDIT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
