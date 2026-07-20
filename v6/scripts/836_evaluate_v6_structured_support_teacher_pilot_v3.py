#!/usr/bin/env python3
"""Evaluate the fresh structured support teacher against its blinded auditor."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


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


def metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    tp = sum(row["teacher_supported"] and row["auditor_supported"] for row in rows)
    tn = sum(not row["teacher_supported"] and not row["auditor_supported"] for row in rows)
    fp = sum(row["teacher_supported"] and not row["auditor_supported"] for row in rows)
    fn = sum(not row["teacher_supported"] and row["auditor_supported"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "recall": recall, "specificity": specificity,
    }


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: values[str(row[field])].append(row)
    return {key: metrics(value) for key, value in sorted(values.items())}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--teacher_dir", type=Path, required=True)
    parser.add_argument("--auditor_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_FROZEN":
        raise ValueError("Structured support pilot is not frozen")
    if freeze["input_hashes"]["evaluator"] != sha256(Path(__file__).resolve()):
        raise ValueError("Structured support pilot evaluator changed after freeze")
    manifests = {}
    decisions = {}
    for role, directory in (("teacher", args.teacher_dir.resolve()), ("auditor", args.auditor_dir.resolve())):
        manifest = read_json(directory / "RUN_MANIFEST.json")
        if manifest.get("decision") != "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_ROLE_COMPLETE" or manifest.get("role") != role:
            raise ValueError(f"Incomplete {role} run")
        if manifest["freeze_sha256"] != sha256(freeze_path):
            raise ValueError(f"{role} used another protocol")
        path = Path(manifest["decisions_path"])
        if sha256(path) != manifest["decisions_sha256"]:
            raise ValueError(f"{role} decisions changed")
        manifests[role] = manifest
        decisions[role] = {row["audit_id"]: row for row in read_jsonl(path)}
    key_path = Path(freeze["output_paths"]["private_key"])
    if sha256(key_path) != freeze["output_hashes"]["private_key"]:
        raise ValueError("Pilot private key changed")
    key = {row["audit_id"]: row for row in read_jsonl(key_path)}
    if len(key) != 400 or set(key) != set(decisions["teacher"]) or set(key) != set(decisions["auditor"]):
        raise ValueError("Teacher/auditor pilot universes do not match")
    rows = []
    for audit_id in sorted(key):
        teacher, auditor = decisions["teacher"][audit_id], decisions["auditor"][audit_id]
        rows.append({
            **key[audit_id],
            "teacher_supported": bool(teacher["supported"]),
            "auditor_supported": bool(auditor["supported"]),
            "teacher_citation_valid": bool(teacher["citation_valid"]),
            "auditor_citation_valid": bool(auditor["citation_valid"]),
            "teacher_reason": teacher["reason"],
            "auditor_reason": auditor["reason"],
            "teacher_supporting_doc_indices": teacher["supporting_doc_indices"],
            "auditor_supporting_doc_indices": auditor["supporting_doc_indices"],
            "teacher_supporting_span": teacher["supporting_span"],
            "auditor_supporting_span": auditor["supporting_span"],
        })
    overall = metrics(rows)
    by_type = grouped(rows, "answer_type")
    by_strategy = grouped(rows, "evidence_strategy")
    teacher_positive = [row for row in rows if row["teacher_supported"]]
    auditor_positive = [row for row in rows if row["auditor_supported"]]
    teacher_citation = sum(row["teacher_citation_valid"] for row in teacher_positive) / len(teacher_positive) if teacher_positive else 0.0
    auditor_citation = sum(row["auditor_citation_valid"] for row in auditor_positive) / len(auditor_positive) if auditor_positive else 0.0
    q = freeze["quality_gates"]
    gates = {
        "exact_samples": len(rows) == int(q["samples"]),
        "balanced_accuracy": overall["balanced_accuracy"] >= float(q["balanced_accuracy_min"]),
        "precision": overall["precision"] >= float(q["precision_min"]),
        "recall": overall["recall"] >= float(q["recall_min"]),
        "teacher_citation_validity": teacher_citation >= float(q["citation_validity_min"]),
        "auditor_citation_validity": auditor_citation >= float(q["citation_validity_min"]),
        "short_numeric_accuracy": by_type["short_numeric"]["accuracy"] >= float(q["short_numeric_accuracy_min"]),
        "phone_or_identifier_accuracy": by_type["phone_or_identifier"]["accuracy"] >= float(q["phone_or_identifier_accuracy_min"]),
        "email_accuracy": by_type["email"]["accuracy"] >= float(q["email_accuracy_min"]),
        "address_accuracy": by_type["address"]["accuracy"] >= float(q["address_accuracy_min"]),
        "auditor_blinded_to_teacher": manifests["auditor"].get("teacher_outputs_visible") is False,
        "dev_sealed": manifests["teacher"].get("dev_used") is False and manifests["auditor"].get("dev_used") is False,
        "test_sealed": manifests["teacher"].get("test_used") is False and manifests["auditor"].get("test_used") is False,
    }
    passed = all(gates.values())
    decision = "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_GO" if passed else "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    disagreement_path = output / "structured_support_v3_pilot_disagreements_private.jsonl"
    with disagreement_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["teacher_supported"] != row["auditor_supported"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = {
        "decision": decision,
        "overall": overall,
        "by_answer_type": by_type,
        "by_evidence_strategy": by_strategy,
        "teacher_citation_validity": teacher_citation,
        "auditor_citation_validity": auditor_citation,
        "gates": gates,
        "gate_thresholds": q,
        "teacher_manifest": manifests["teacher"],
        "auditor_manifest": manifests["auditor"],
        "disagreements": sum(row["teacher_supported"] != row["auditor_supported"] for row in rows),
        "full_v3_label_generation_allowed": passed,
        "next_action": "Freeze and score the full deduplicated 14,770-action support universe with the exact teacher contract, then run another untouched audit." if passed else "Do not scale labels. Adjudicate pilot disagreements and revise the structured contract on a new untouched pilot without lowering gates.",
        "serper_calls": 0,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    result_path = output / "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_AUDIT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# DAG-IG v6 Structured Support Teacher Pilot v3",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "## Overall",
        "",
        f"- n: `{overall['n']}`",
        f"- accuracy: `{pct(overall['accuracy'])}`",
        f"- balanced accuracy: `{pct(overall['balanced_accuracy'])}`",
        f"- precision: `{pct(overall['precision'])}`",
        f"- recall: `{pct(overall['recall'])}`",
        f"- disagreements: `{result['disagreements']}`",
        f"- teacher/auditor citation validity: `{pct(teacher_citation)}` / `{pct(auditor_citation)}`",
        "",
        "## Answer Types",
        "",
        "| type | n | accuracy | balanced accuracy | precision | recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in by_type.items():
        report.append(f"| {name} | {value['n']} | {pct(value['accuracy'])} | {pct(value['balanced_accuracy'])} | {pct(value['precision'])} | {pct(value['recall'])} |")
    report.extend(["", "## Gates", ""])
    for name, value in gates.items(): report.append(f"- {name}: `{'PASS' if value else 'FAIL'}`")
    report.extend(["", "## Next Action", "", result["next_action"], ""])
    report_path = output / "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"decision": decision, "overall": overall, "gates": gates, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
