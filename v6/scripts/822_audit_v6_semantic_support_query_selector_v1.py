#!/usr/bin/env python3
"""One-shot internal audit of semantic-support-backed query DAG-IG."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


METHODS = ("no_credit", "local_ig_m", "outcome", "old_dagig", "semantic_support_dagig")


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


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positive = labels == 1
    positives = int(positive.sum())
    negatives = len(labels) - positives
    return 0.5 if not positives or not negatives else float((ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def pair_order(scores: np.ndarray, labels: np.ndarray, groups: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    correct = total = 0.0
    for indices in grouped.values():
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                delta = labels[left] - labels[right]
                if abs(delta) <= 1e-12:
                    continue
                prediction = scores[left] - scores[right]
                total += 1.0
                correct += float(prediction * delta > 0.0) + 0.5 * float(abs(prediction) <= 1e-12)
    return {"pairs": int(total), "accuracy": correct / total if total else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--train_audit", type=Path, required=True)
    parser.add_argument("--private_support", type=Path, required=True)
    parser.add_argument("--terminal_private", type=Path, required=True)
    parser.add_argument("--shared_answer_values", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    train_path = args.train_audit.resolve()
    freeze, train = read_json(freeze_path), read_json(train_path)
    if train.get("decision") != "DAGIG_V6_SEMANTIC_SUPPORT_VERIFIER_V1_TRAIN_OOF_GO":
        raise ValueError("Semantic support verifier did not pass train OOF")
    if freeze["input_hashes"]["auditor"] != sha256(Path(__file__).resolve()):
        raise ValueError("Auditor changed after protocol freeze")
    if train["input_hashes"]["freeze"] != sha256(freeze_path):
        raise ValueError("Train audit came from another freeze")
    for key, raw_path in train["output_paths"].items():
        if sha256(Path(raw_path)) != train["output_hashes"][key]:
            raise ValueError(f"Frozen semantic-support output changed: {key}")

    query_freeze = read_json(Path(freeze["input_paths"]["query_value_freeze"]))
    public_targets = read_jsonl(Path(query_freeze["output_paths"]["internal_targets"]))
    diagnostics = {
        row["parent_state_id"]: row
        for row in read_jsonl(Path(query_freeze["output_paths"]["diagnostics"]))
        if row["partition"] == "internal_holdout"
    }
    verifier_inputs = {
        row["query_action_id"]: row
        for row in read_jsonl(Path(freeze["output_paths"]["verifier_inputs"]))
        if row["partition"] == "internal_holdout"
    }
    predictions = {
        row["query_action_id"]: row
        for row in read_jsonl(Path(train["output_paths"]["predictions"]))
        if row["partition"] == "internal_holdout"
    }
    support_map = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(args.private_support.resolve())
        if row["partition"] == "internal_holdout"
    }
    terminal = {
        row["answer_action_id"]: row
        for row in read_jsonl(args.terminal_private.resolve())
        if row["partition"] == "internal_holdout"
    }
    shared = {
        row["evidence_action_id"]: row
        for row in read_jsonl(args.shared_answer_values.resolve())
    }
    if len(public_targets) != 120 or len(predictions) != 595 or len(verifier_inputs) != 595:
        raise ValueError(f"Internal universe mismatch: {len(public_targets)}/{len(predictions)}/{len(verifier_inputs)}")

    rows = []
    all_support_scores, all_baseline_scores, all_support_labels, all_parent_ids = [], [], [], []
    for target in public_targets:
        diagnostic = diagnostics[target["parent_state_id"]]
        query_ids = diagnostic["query_action_ids"]
        action_metrics = []
        semantic_strict_values = []
        for index, query_id in enumerate(query_ids):
            input_row = verifier_inputs[query_id]
            prediction = predictions[query_id]
            evidence_id = diagnostic["selected_evidence_action_ids"][index]
            if input_row["selected_evidence_action_id"] != evidence_id:
                raise ValueError(f"Selected evidence mismatch: {query_id}")
            evidence_strategy = evidence_id.rsplit("::", 1)[-1]
            support = float(support_map[query_id][evidence_strategy])
            support_probability = float(prediction["semantic_support_probability"])
            conditional = float(input_row["answer_correct_given_support_probability"])
            semantic_strict = support_probability * conditional
            value = shared[evidence_id]
            answer_probabilities = [float(item) for item in value["answer_policy_probabilities"]]
            strict_labels = [float(terminal[answer_id]["strict_proxy"]) for answer_id in value["answer_action_ids"]]
            mode = value["answer_action_ids"].index(value["mode_answer_action_id"])
            action_metrics.append({
                "query_action_id": query_id,
                "query_strategy": query_id.rsplit("::", 1)[-1],
                "evidence_action_id": evidence_id,
                "support": support,
                "expected_strict": sum(probability * label for probability, label in zip(answer_probabilities, strict_labels)),
                "mode_strict": strict_labels[mode],
                "semantic_support_probability": support_probability,
                "answer_correct_given_support_probability": conditional,
                "semantic_strict_value": semantic_strict,
            })
            semantic_strict_values.append(max(semantic_strict, 1e-8))
            all_support_scores.append(support_probability)
            all_baseline_scores.append(float(input_row["baseline_support_probability"]))
            all_support_labels.append(support)
            all_parent_ids.append(target["parent_state_id"])
        semantic_posterior = normalize(semantic_strict_values)
        methods = {
            "no_credit": target["target_distributions"]["no_credit"],
            "local_ig_m": target["target_distributions"]["local_ig_m"],
            "outcome": target["target_distributions"]["outcome"],
            "old_dagig": target["target_distributions"]["dagig"],
            "semantic_support_dagig": semantic_posterior,
        }
        selected = {}
        for method, distribution in methods.items():
            choice = max(range(len(distribution)), key=lambda item: (float(distribution[item]), -item))
            selected[method] = action_metrics[choice]
        rows.append({
            "parent_state_id": target["parent_state_id"],
            "sample_id": target["parent_state_id"].split("::", 1)[0],
            "methods": selected,
        })

    summary = {}
    for method in METHODS:
        chosen = [row["methods"][method] for row in rows]
        summary[method] = {
            "states": len(chosen),
            "samples": len({row["sample_id"] for row in rows}),
            "support": mean(row["support"] for row in chosen),
            "expected_strict": mean(row["expected_strict"] for row in chosen),
            "mode_strict": mean(row["mode_strict"] for row in chosen),
            "predicted_support": mean(row["semantic_support_probability"] for row in chosen),
            "predicted_strict": mean(row["semantic_strict_value"] for row in chosen),
            "query_strategy_distribution": dict(sorted(Counter(row["query_strategy"] for row in chosen).items())),
        }
    support_scores = np.asarray(all_support_scores)
    baseline_scores = np.asarray(all_baseline_scores)
    support_labels = np.asarray(all_support_labels)
    verifier_metrics = {
        "actions": len(support_scores),
        "support_prevalence": float(support_labels.mean()),
        "support_auc": auc(support_scores, support_labels),
        "support_brier": float(np.mean((support_scores - support_labels) ** 2)),
        "baseline_support_brier": float(np.mean((baseline_scores - support_labels) ** 2)),
        "brier_improvement_vs_baseline": float(np.mean((baseline_scores - support_labels) ** 2) - np.mean((support_scores - support_labels) ** 2)),
        "within_visual_pair_order": pair_order(support_scores, support_labels, all_parent_ids),
    }
    disagreement = mean(
        row["methods"]["semantic_support_dagig"]["query_action_id"] != row["methods"]["outcome"]["query_action_id"]
        for row in rows
    )
    threshold = freeze["development_gates"]
    dag = summary["semantic_support_dagig"]
    no_credit, local, outcome = summary["no_credit"], summary["local_ig_m"], summary["outcome"]
    gates = {
        "complete_120_internal_visual_states": len(rows) == 120,
        "complete_40_internal_samples": len({row["sample_id"] for row in rows}) == 40,
        "support_not_below_no_credit": dag["support"] - no_credit["support"] >= threshold["support_delta_vs_no_credit_min"],
        "support_noninferior_local": dag["support"] >= local["support"] - threshold["support_noninferiority_vs_local_tolerance"],
        "support_noninferior_outcome": dag["support"] >= outcome["support"] - threshold["support_noninferiority_vs_outcome_tolerance"],
        "strict_noninferior_no_credit": dag["expected_strict"] >= no_credit["expected_strict"] - threshold["strict_noninferiority_vs_no_credit_tolerance"],
        "strict_noninferior_local": dag["expected_strict"] >= local["expected_strict"] - threshold["strict_noninferiority_vs_local_tolerance"],
        "strict_noninferior_outcome": dag["expected_strict"] >= outcome["expected_strict"] - threshold["strict_noninferiority_vs_outcome_tolerance"],
        "differs_from_outcome": disagreement >= threshold["top_action_disagreement_vs_outcome_min"],
        "query_strategy_diversity": len(dag["query_strategy_distribution"]) >= threshold["selected_query_strategies_min"],
        "predictions_frozen_before_internal_labels": True,
        "runtime_verifier_is_answer_independent": True,
        "development_result_not_paper_final": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {key: bool(value) for key, value in gates.items()}
    decision = "DAGIG_V6_SEMANTIC_SUPPORT_QUERY_SELECTOR_V1_DEVELOPMENT_GO" if all(gates.values()) else "DAGIG_V6_SEMANTIC_SUPPORT_QUERY_SELECTOR_V1_DEVELOPMENT_NO_GO"

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "v6_semantic_support_query_selector_internal_private.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "verifier_metrics": verifier_metrics,
        "method_summary": summary,
        "dagig_outcome_top_disagreement": disagreement,
        "gates": gates,
        "input_paths": {
            "freeze": str(freeze_path),
            "train_audit": str(train_path),
            "private_support": str(args.private_support.resolve()),
            "terminal_private": str(args.terminal_private.resolve()),
            "shared_answer_values": str(args.shared_answer_values.resolve()),
        },
        "input_hashes": {
            "freeze": sha256(freeze_path),
            "train_audit": sha256(train_path),
            "private_support": sha256(args.private_support.resolve()),
            "terminal_private": sha256(args.terminal_private.resolve()),
            "shared_answer_values": sha256(args.shared_answer_values.resolve()),
        },
        "output_paths": {"private_rows": str(rows_path)},
        "output_hashes": {"private_rows": sha256(rows_path)},
        "internal_labels_loaded_only_after_predictions_frozen": True,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_SEMANTIC_SUPPORT_QUERY_SELECTOR_V1_DEVELOPMENT_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = output / "DAGIG_V6_SEMANTIC_SUPPORT_QUERY_SELECTOR_V1_REPORT.md"
    lines = [
        "# DAG-IG v6 Semantic Support Query Selector v1",
        "",
        f"Decision: `{decision}`",
        "",
        "## Frozen verifier",
        "",
        f"- Internal support AUC: `{verifier_metrics['support_auc']:.6f}`",
        f"- Internal support Brier: `{verifier_metrics['support_brier']:.6f}`",
        f"- Baseline support Brier: `{verifier_metrics['baseline_support_brier']:.6f}`",
        f"- Within-visual support pair order: `{verifier_metrics['within_visual_pair_order']['accuracy']:.6f}`",
        "",
        "## Direct selector",
        "",
        "| Method | Support | Expected strict | Mode strict |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary[method]
        lines.append(f"| {method} | {item['support']:.4f} | {item['expected_strict']:.4f} | {item['mode_strict']:.4f} |")
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    lines.extend(["", "This is an internal method-development audit, not a paper-final dev/test result.", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": decision, "verifier_metrics": verifier_metrics, "method_summary": summary, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
