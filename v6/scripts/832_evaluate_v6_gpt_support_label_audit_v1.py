#!/usr/bin/env python3
"""Evaluate the frozen blinded GPT audit and gate provisional support labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


GO = "DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_V2_INDEPENDENT_AUDIT_GO"
NO_GO = "DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_V2_INDEPENDENT_AUDIT_NO_GO"


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


def divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def classification_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    tp = sum(row["local_label"] and row["independent_label"] for row in rows)
    tn = sum(not row["local_label"] and not row["independent_label"] for row in rows)
    fp = sum(row["local_label"] and not row["independent_label"] for row in rows)
    fn = sum(not row["local_label"] and row["independent_label"] for row in rows)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    precision = divide(tp, tp + fp)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {
        "n": len(rows),
        "reference_positive": tp + fn,
        "reference_negative": tn + fp,
        "local_positive": tp + fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": divide(tp + tn, len(rows)),
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def group_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: classification_metrics(value) for key, value in sorted(grouped.items())}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_table(groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["| group | n | accuracy | balanced accuracy | precision | recall |", "|---|---:|---:|---:|---:|---:|"]
    for name, metric in groups.items():
        lines.append(
            f"| {name} | {metric['n']} | {pct(metric['accuracy'])} | {pct(metric['balanced_accuracy'])} | {pct(metric['precision'])} | {pct(metric['recall'])} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_GPT_SUPPORT_LABEL_AUDIT_V1_FROZEN":
        raise ValueError("GPT support-label audit protocol is not frozen")
    if freeze["input_hashes"]["evaluator"] != sha256(Path(__file__).resolve()):
        raise ValueError("GPT audit evaluator changed after protocol freeze")
    for field, path_key in (("private_audit_key", "private_audit_key"), ("provisional_labels", "provisional_labels")):
        path = Path(freeze["input_paths"][path_key])
        if sha256(path) != freeze["input_hashes"][field]:
            raise ValueError(f"Frozen evaluator input changed: {field}")

    output = args.run_dir.resolve()
    manifest_path = output / "GPT_SUPPORT_LABEL_AUDIT_V1_RUN_MANIFEST.json"
    manifest = read_json(manifest_path)
    if manifest.get("decision") != "DAGIG_V6_GPT_SUPPORT_LABEL_AUDIT_V1_RUN_COMPLETE":
        raise ValueError("Independent GPT audit run is incomplete")
    if manifest["freeze_sha256"] != sha256(freeze_path):
        raise ValueError("Independent GPT audit used another frozen protocol")
    decisions_path = Path(manifest["decisions_path"])
    if sha256(decisions_path) != manifest["decisions_sha256"]:
        raise ValueError("Independent GPT decision file changed")
    decisions = read_jsonl(decisions_path)
    key_path = Path(freeze["input_paths"]["private_audit_key"])
    key = read_jsonl(key_path)
    if len(decisions) != 350 or len(key) != 350:
        raise ValueError("Independent audit does not contain exactly 350 items")
    decision_by_id = {row["audit_id"]: row for row in decisions}
    key_by_id = {row["audit_id"]: row for row in key}
    if len(decision_by_id) != 350 or len(key_by_id) != 350 or set(decision_by_id) != set(key_by_id):
        raise ValueError("Independent decisions and private audit key do not match exactly")

    joined = []
    for audit_id in sorted(key_by_id):
        local, independent = key_by_id[audit_id], decision_by_id[audit_id]
        joined.append({
            **local,
            "independent_label": bool(independent["supported"]),
            "independent_reason": independent["reason"],
            "independent_model": independent["model"],
        })
    overall = classification_metrics(joined)
    by_answer_type = group_metrics(joined, "answer_type")
    by_category = group_metrics(joined, "audit_category")
    gates_spec = freeze["quality_gates"]
    short_numeric = by_answer_type.get("short_numeric", {"accuracy": None})
    address = by_answer_type.get("address", {"accuracy": None})
    gates = {
        "exact_independent_audit_sample_min": len(joined) >= int(gates_spec["independent_stratified_audit_samples_min"]),
        "balanced_accuracy": overall["balanced_accuracy"] is not None and overall["balanced_accuracy"] >= float(gates_spec["independent_balanced_accuracy_min"]),
        "precision": overall["precision"] is not None and overall["precision"] >= float(gates_spec["independent_precision_min"]),
        "recall": overall["recall"] is not None and overall["recall"] >= float(gates_spec["independent_recall_min"]),
        "short_numeric_subset_accuracy": short_numeric["accuracy"] is not None and short_numeric["accuracy"] >= float(gates_spec["short_numeric_subset_accuracy_min"]),
        "address_subset_accuracy": address["accuracy"] is not None and address["accuracy"] >= float(gates_spec["address_subset_accuracy_min"]),
        "runner_blinded_to_local_labels": manifest.get("local_labels_visible_to_runner") is False and manifest.get("private_audit_key_visible_to_runner") is False,
        "dev_sealed": manifest.get("dev_used") is False,
        "test_sealed": manifest.get("test_used") is False,
    }
    passed = all(gates.values())
    decision = GO if passed else NO_GO

    disagreements = [row for row in joined if row["local_label"] != row["independent_label"]]
    disagreement_path = output / "gpt_support_label_audit_disagreements_private.jsonl"
    with disagreement_path.open("w", encoding="utf-8") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    final_label_path = output / "v6_gold_aware_support_v2_labels_frozen_private.jsonl"
    if final_label_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing final labels: {final_label_path}")
    final_label_hash = None
    if passed:
        provisional_path = Path(freeze["input_paths"]["provisional_labels"])
        with final_label_path.open("w", encoding="utf-8") as handle:
            for row in read_jsonl(provisional_path):
                row["label_status"] = "frozen_after_independent_blinded_gpt_audit"
                row["independent_audit_decision"] = GO
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        final_label_hash = sha256(final_label_path)

    result = {
        "decision": decision,
        "reference": "independent_blinded_gpt5mini_audit",
        "overall": overall,
        "by_answer_type": by_answer_type,
        "by_audit_category": by_category,
        "gates": gates,
        "gate_thresholds": gates_spec,
        "disagreements": len(disagreements),
        "disagreement_category_counts": dict(sorted(Counter(row["audit_category"] for row in disagreements).items())),
        "input_paths": {
            "freeze": str(freeze_path),
            "run_manifest": str(manifest_path),
            "decisions": str(decisions_path),
            "private_audit_key": str(key_path),
            "provisional_labels": freeze["input_paths"]["provisional_labels"],
        },
        "input_hashes": {
            "freeze": sha256(freeze_path),
            "run_manifest": sha256(manifest_path),
            "decisions": sha256(decisions_path),
            "private_audit_key": sha256(key_path),
            "provisional_labels": freeze["input_hashes"]["provisional_labels"],
        },
        "output_paths": {
            "disagreements": str(disagreement_path),
            "frozen_labels": str(final_label_path) if passed else None,
        },
        "output_hashes": {
            "disagreements": sha256(disagreement_path),
            "frozen_labels": final_label_hash,
        },
        "labels_allowed_for_downstream_credit_recomputation": passed,
        "labels_allowed_as_runtime_policy_features": False,
        "failure_action": None if passed else "Do not train or recompute final credit. Inspect disagreement strata and create a new untouched blinded audit after revising the support teacher; do not lower frozen thresholds.",
        "serper_calls": 0,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_V2_INDEPENDENT_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = [
        "# DAG-IG v6 Gold-Aware Support Label Independent Audit",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "This is a blinded model-based independent audit on 350 policy-train actions. The GPT runner saw the private reference answer and evidence needed for semantic adjudication, but did not see the local prediction, legacy label, audit category, dev, or test data.",
        "",
        "## Overall",
        "",
        f"- n: `{overall['n']}`",
        f"- accuracy: `{pct(overall['accuracy'])}`",
        f"- balanced accuracy: `{pct(overall['balanced_accuracy'])}`",
        f"- precision: `{pct(overall['precision'])}`",
        f"- recall: `{pct(overall['recall'])}`",
        f"- specificity: `{pct(overall['specificity'])}`",
        f"- F1: `{pct(overall['f1'])}`",
        f"- disagreements: `{len(disagreements)}`",
        "",
        "## Answer-Type Strata",
        "",
        *render_table(by_answer_type),
        "",
        "## Audit Strata",
        "",
        *render_table(by_category),
        "",
        "## Frozen Gates",
        "",
    ]
    for name, value in gates.items():
        report.append(f"- {name}: `{'PASS' if value else 'FAIL'}`")
    report.extend([
        "",
        "## Use Contract",
        "",
        f"- downstream credit recomputation allowed: `{passed}`",
        "- runtime policy feature use allowed: `False`",
        "- dev used: `False`",
        "- test used: `False`",
        "- Serper calls: `0`",
        "",
    ])
    if passed:
        report.extend([
            "The sampled semantic contract passed every pre-registered gate. Freeze the labels for evaluation/value supervision, then train a separate no-gold runtime verifier and recompute evidence-node terminal values before reopening query-node work.",
        ])
    else:
        report.extend([
            "The provisional labels are not permitted for downstream training or claims. Inspect disagreement strata, revise the support-teacher semantics, and evaluate a new untouched blinded sample without relaxing thresholds.",
        ])
    report_path = output / "DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_V2_INDEPENDENT_AUDIT_REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "overall": overall,
        "gates": gates,
        "disagreements": len(disagreements),
        "audit": str(audit_path),
        "report": str(report_path),
        "frozen_labels": str(final_label_path) if passed else None,
    }, indent=2))


if __name__ == "__main__":
    main()
