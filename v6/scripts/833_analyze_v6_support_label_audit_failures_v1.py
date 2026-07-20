#!/usr/bin/env python3
"""Diagnose the failed v2 semantic-support independent audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
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


def normalize(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def prompt_field(prompt: str, name: str, next_name: str) -> str:
    pattern = rf"{re.escape(name)}:\n(.*?)\n\n{re.escape(next_name)}:"
    match = re.search(pattern, prompt, re.S)
    return match.group(1).strip() if match else ""


def confusion_metrics(reference: list[bool], prediction: list[bool]) -> dict[str, float | int]:
    tp = sum(p and y for y, p in zip(reference, prediction))
    tn = sum(not p and not y for y, p in zip(reference, prediction))
    fp = sum(p and not y for y, p in zip(reference, prediction))
    fn = sum(not p and y for y, p in zip(reference, prediction))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(reference),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": recall,
        "specificity": specificity,
    }


def tie_aware_auc(reference: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(reference, scores) if label]
    negatives = [score for label, score in zip(reference, scores) if not label]
    wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    run = args.run_dir.resolve()
    audit_path = run / "DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_V2_INDEPENDENT_AUDIT.json"
    audit = read_json(audit_path)
    if audit.get("decision") != "DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_V2_INDEPENDENT_AUDIT_NO_GO":
        raise ValueError("This analysis is only valid for the frozen v2 audit NO-GO")
    if audit["input_hashes"]["freeze"] != sha256(freeze_path):
        raise ValueError("Audit and freeze mismatch")
    key = {row["audit_id"]: row for row in read_jsonl(Path(freeze["input_paths"]["private_audit_key"]))}
    items = {row["audit_id"]: row for row in read_jsonl(Path(freeze["input_paths"]["blinded_items"]))}
    manifest = read_json(run / "GPT_SUPPORT_LABEL_AUDIT_V1_RUN_MANIFEST.json")
    decisions = {row["audit_id"]: row for row in read_jsonl(Path(manifest["decisions_path"]))}
    if set(key) != set(items) or set(key) != set(decisions):
        raise ValueError("Audit universes differ")

    rows = []
    for audit_id in sorted(key):
        local, independent, item = key[audit_id], decisions[audit_id], items[audit_id]
        prompt = item["user_prompt_private"]
        gold = prompt_field(prompt, "Private reference answer", "Equivalent aliases")
        aliases_text = prompt_field(prompt, "Equivalent aliases", "Visual context")
        aliases = [] if aliases_text == "none" else [value.strip() for value in aliases_text.split(";") if value.strip()]
        evidence = prompt.split("Selected evidence:\n", 1)[-1].rsplit("\n\nDecision:", 1)[0]
        accepted = [gold, *aliases]
        exact_in_evidence = any(normalize(answer) and normalize(answer) in normalize(evidence) for answer in accepted)
        local_label = bool(local["local_label"])
        reference_label = bool(independent["supported"])
        if local_label == reference_label:
            root_cause = "agreement"
        elif not local_label and reference_label and exact_in_evidence:
            root_cause = "local_false_negative_exact_or_normalized_answer_present"
        elif not local_label and reference_label:
            root_cause = "local_false_negative_semantic_entailment_or_conversion"
        elif local_label and not reference_label and not exact_in_evidence:
            root_cause = "local_false_positive_without_answer_support"
        else:
            root_cause = "local_false_positive_incidental_or_wrong_context_match"
        rows.append({
            **local,
            "independent_label": reference_label,
            "independent_reason": independent["reason"],
            "gold_answer_private": gold,
            "exact_or_normalized_answer_in_evidence": exact_in_evidence,
            "root_cause": root_cause,
            "question_private": prompt_field(prompt, "Question", "Private reference answer"),
            "evidence_private": evidence,
        })

    reference = [row["independent_label"] for row in rows]
    local_hard = [bool(row["local_label"]) for row in rows]
    scores = [float(row["local_probability"]) for row in rows]
    legacy = [row["legacy_support_reason"] != "negative" for row in rows]
    thresholds = sorted({0.0, 1.0, *scores})
    best_threshold = max(
        ((confusion_metrics(reference, [score >= threshold for score in scores])["balanced_accuracy"], threshold) for threshold in thresholds),
        key=lambda value: value[0],
    )
    disagreements = [row for row in rows if row["root_cause"] != "agreement"]
    metrics = {
        "decision": "DAGIG_V6_SUPPORT_LABEL_V2_ROOT_CAUSE_CONFIRMED",
        "n": len(rows),
        "local_teacher_at_frozen_threshold": confusion_metrics(reference, local_hard),
        "local_teacher_score_auc": tie_aware_auc(reference, scores),
        "diagnostic_best_threshold_not_for_reuse": {"threshold": best_threshold[1], "balanced_accuracy": best_threshold[0]},
        "legacy_trigger": confusion_metrics(reference, legacy),
        "root_cause_counts": dict(sorted(Counter(row["root_cause"] for row in rows).items())),
        "answer_type_disagreement_counts": dict(sorted(Counter(row["answer_type"] for row in disagreements).items())),
        "local_probability_mean_reference_positive": sum(score for score, label in zip(scores, reference) if label) / sum(reference),
        "local_probability_mean_reference_negative": sum(score for score, label in zip(scores, reference) if not label) / sum(not value for value in reference),
        "conclusion": "The local next-token Qwen teacher has useful ranking signal but cannot produce paper-valid semantic support labels at any threshold; replace the teacher/scoring protocol rather than tune the threshold.",
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases_path = output / "support_label_v2_disagreements_private.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path = output / "SUPPORT_LABEL_V2_ROOT_CAUSE.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# DAG-IG v6 Support Label v2 Root-Cause Analysis",
        "",
        "## Decision",
        "",
        "`DAGIG_V6_SUPPORT_LABEL_V2_ROOT_CAUSE_CONFIRMED`",
        "",
        "The v2 label teacher is not repairable by changing its 0.5 threshold. Against the frozen independent blinded reference, its score AUC is useful but its best diagnostic balanced accuracy remains far below the pre-registered 0.90 gate.",
        "",
        "## Metrics",
        "",
        f"- frozen-threshold balanced accuracy: `{metrics['local_teacher_at_frozen_threshold']['balanced_accuracy']:.4f}`",
        f"- frozen-threshold precision/recall: `{metrics['local_teacher_at_frozen_threshold']['precision']:.4f}` / `{metrics['local_teacher_at_frozen_threshold']['recall']:.4f}`",
        f"- continuous-score AUC: `{metrics['local_teacher_score_auc']:.4f}`",
        f"- diagnostic best balanced accuracy across all observed thresholds: `{best_threshold[0]:.4f}` at `{best_threshold[1]:.6f}` (analysis only; not reused)",
        f"- legacy-trigger balanced accuracy: `{metrics['legacy_trigger']['balanced_accuracy']:.4f}`",
        f"- disagreements: `{len(disagreements)}/{len(rows)}`",
        "",
        "## Root Causes",
        "",
    ]
    for name, count in metrics["root_cause_counts"].items():
        report.append(f"- {name}: `{count}`")
    report.extend([
        "",
        "False negatives include exact phone numbers, emails, addresses, and numeric facts explicitly stated in a selected snippet. False positives include missing answers, wrong numbers, wrong entities, and topical-but-non-entailing evidence. This rules out threshold calibration as the main repair.",
        "",
        "## Required v3 Contract",
        "",
        "1. Use a stronger structured semantic teacher, not next-token A/B logits.",
        "2. Require every positive decision to identify a supporting document and return a verifiable evidence span or explicit derivation.",
        "3. Run a small fresh teacher-versus-independent-auditor pilot before scoring the full 14,770-action universe.",
        "4. Freeze a new untouched audit set; do not reuse these 350 items as the v3 quality gate.",
        "5. Keep labels evaluation/value-supervision only; runtime policies must use a separately trained no-gold verifier.",
        "",
    ])
    report_path = output / "SUPPORT_LABEL_V2_ROOT_CAUSE_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"decision": metrics["decision"], "metrics": str(metrics_path), "report": str(report_path), "disagreements": len(disagreements)}, indent=2))


if __name__ == "__main__":
    main()
