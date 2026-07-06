#!/usr/bin/env python3
"""Audit progress against the DAG-IG / Pix2Fact paper-main objective.

This is intentionally broader than the asset audit: it checks whether the
current filesystem proves the main experimental chain and paper package are in
place, and it separates remaining paper-production blockers from experiment
blockers.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
GRPO_DERIVED = Path("outputs/dagig_grpo_main/derived_assets")
ASSETS = ROOT / "paper_assets"
REPORTS = ROOT / "reports"

OUT_JSON = ASSETS / "goal_completion_audit.json"
OUT_MD = ASSETS / "GOAL_COMPLETION_AUDIT.md"

CONSOLIDATED = REPORTS / "paper_main_v1_consolidated_results.json"
NODE_SUMMARY = REPORTS / "node_credit_component_analysis/node_credit_component_summary.json"
PAPER_AUDIT = ASSETS / "paper_asset_audit.json"
PDF_PREFLIGHT = ASSETS / "pdf_build_preflight.json"
POST_COMPILE_PDF_AUDIT = ASSETS / "post_compile_pdf_audit.json"
PDF_LAYOUT_AUDIT = ASSETS / "pdf_layout_audit.json"
VENUE_DECISION_AUDIT = ASSETS / "venue_decision_audit.json"
FINAL_SUBMISSION_GATE = ASSETS / "final_submission_gate.json"
MAINLINE_SCHEMA_CONTRACT = ASSETS / "MAINLINE_SCHEMA_CONTRACT_AUDIT.json"
DERIVED_MANIFEST = GRPO_DERIVED / "derived_manifest.json"

MAIN_CKPT = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60"
MAIN_TRAIN_SUMMARY = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/grpo_train_summary.json"
SEED43_TRAIN_SUMMARY = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/grpo_train_summary.json"
GOLDFIXED_TRAIN_SUMMARY = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/grpo_train_summary.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def status(passed: bool) -> str:
    return "complete" if passed else "incomplete"


def check_path(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "nonempty": path.exists() and path.stat().st_size > 0}


def check_data_and_schema() -> dict[str, Any]:
    expected_counts = {
        "train": (GRPO_DERIVED / "grpo_train.jsonl", 458),
        "dev": (GRPO_DERIVED / "grpo_dev.jsonl", 98),
        "test": (GRPO_DERIVED / "grpo_test.jsonl", 64),
        "train_corpus": (GRPO_DERIVED / "bm25_train_corpus.jsonl", 610),
        "eval_corpus": (GRPO_DERIVED / "bm25_eval_corpus.jsonl", 201),
        "goldfixed_train_corpus": (ROOT / "derived_assets/bm25_train_corpus_goldfixed.jsonl", 610),
    }
    observed: dict[str, int] = {}
    mismatches: list[str] = []
    for name, (path, expected) in expected_counts.items():
        if not path.exists():
            mismatches.append(f"{name}: missing {path}")
            continue
        count = line_count(path)
        observed[name] = count
        if count != expected:
            mismatches.append(f"{name}: observed {count}, expected {expected}")

    required_fields = {"sample_id", "split", "question", "image_path", "gold_answer", "reward_fields"}
    reward_flags = {
        "use_for_visual_credit",
        "use_for_query_credit",
        "use_for_evidence_credit",
        "use_for_answer_credit",
    }
    schema_errors: list[str] = []
    image_missing = 0
    reward_flag_missing = 0
    checked_rows = 0
    for split_name in ["train", "dev", "test"]:
        path = expected_counts[split_name][0]
        if not path.exists():
            continue
        for row in read_jsonl(path):
            checked_rows += 1
            missing = required_fields - set(row)
            if missing:
                schema_errors.append(f"{path}:{row.get('sample_id', '<unknown>')} missing {sorted(missing)}")
            rf = row.get("reward_fields") or {}
            if reward_flags - set(rf):
                reward_flag_missing += 1
            image_path = Path(row.get("image_abs_path") or row.get("image_path", ""))
            if row.get("image_abs_path") and not image_path.exists():
                image_missing += 1

    manifest_ok = DERIVED_MANIFEST.exists() and not load_json(DERIVED_MANIFEST).get("hard_fail", True)
    passed = not mismatches and not schema_errors and reward_flag_missing == 0 and image_missing == 0 and manifest_ok
    return {
        "requirement": "Unified GRPO data, frozen corpora, and rollout/reward schema are present.",
        "status": status(passed),
        "passed": passed,
        "observed_counts": observed,
        "mismatches": mismatches,
        "checked_rows": checked_rows,
        "schema_error_count": len(schema_errors),
        "schema_errors_sample": schema_errors[:10],
        "reward_flag_missing_rows": reward_flag_missing,
        "missing_images": image_missing,
        "derived_manifest_hard_fail": None if not DERIVED_MANIFEST.exists() else load_json(DERIVED_MANIFEST).get("hard_fail"),
        "evidence": [str(DERIVED_MANIFEST), str(GRPO_DERIVED)],
    }


def check_two_stage_predictions() -> dict[str, Any]:
    files = [
        ROOT / "two_stage_predictions/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.jsonl",
        ROOT / "two_stage_predictions/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.jsonl",
        ROOT / "two_stage_predictions/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.jsonl",
        ROOT / "two_stage_predictions/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test.jsonl",
    ]
    required = {
        "stage1_raw_generation",
        "visual_observation",
        "search_query",
        "retrieved_docs",
        "reader_raw_generation",
        "final_answer",
        "retrieval_top5_hit",
        "answer_correct",
        "evidence_supported",
        "strict_success",
    }
    missing_files = [str(p) for p in files if not p.exists()]
    schema_errors: list[str] = []
    counts: dict[str, int] = {}
    for path in files:
        if not path.exists():
            continue
        count = 0
        for row in read_jsonl(path):
            count += 1
            missing = required - set(row)
            if missing:
                schema_errors.append(f"{path}:{row.get('sample_id', '<unknown>')} missing {sorted(missing)}")
            if not isinstance(row.get("retrieved_docs"), list):
                schema_errors.append(f"{path}:{row.get('sample_id', '<unknown>')} retrieved_docs is not a list")
        counts[str(path)] = count
    passed = not missing_files and not schema_errors
    return {
        "requirement": "Two-stage rollout outputs expose visual/query/evidence/answer nodes for evaluation.",
        "status": status(passed),
        "passed": passed,
        "prediction_counts": counts,
        "missing_files": missing_files,
        "schema_error_count": len(schema_errors),
        "schema_errors_sample": schema_errors[:10],
        "evidence": [str(p) for p in files],
    }


def check_mainline_schema_contract() -> dict[str, Any]:
    audit = load_json(MAINLINE_SCHEMA_CONTRACT) if MAINLINE_SCHEMA_CONTRACT.exists() else {}
    checks = audit.get("checks", {})
    failed_checks = sorted(name for name, item in checks.items() if not item.get("passed"))
    passed = bool(audit.get("overall_pass")) and not failed_checks
    return {
        "requirement": "Mainline schema contract proves split/corpus isolation, four-node rollout schema, node-credit formula, and two-stage prediction schema.",
        "status": status(passed),
        "passed": passed,
        "overall_pass": audit.get("overall_pass"),
        "failed_checks": failed_checks,
        "contract": audit.get("contract", {}),
        "evidence": [str(MAINLINE_SCHEMA_CONTRACT), str(ASSETS / "MAINLINE_SCHEMA_CONTRACT_AUDIT.md")],
    }


def check_reward_audit() -> dict[str, Any]:
    summary = load_json(NODE_SUMMARY) if NODE_SUMMARY.exists() else {}
    required_runs = ["seed42_main", "seed43_confirm", "goldfixed_control"]
    missing_runs = [run for run in required_runs if run not in summary]
    thresholds: dict[str, Any] = {}
    failures: list[str] = []
    for run in required_runs:
        if run not in summary:
            continue
        item = summary[run]
        auc_hit = item.get("reward_auc_retrieval_hit", 0)
        auc_strict = item.get("reward_auc_strict_success", 0)
        constant_rate = item.get("groups", {}).get("constant_group_rate", 1)
        query_auc = item.get("components", {}).get("query", {}).get("auc_retrieval_hit", 0)
        evidence_auc = item.get("components", {}).get("evidence", {}).get("auc_retrieval_hit", 0)
        answer_auc = item.get("components", {}).get("answer", {}).get("auc_strict_success", 0)
        thresholds[run] = {
            "reward_auc_hit": auc_hit,
            "reward_auc_strict": auc_strict,
            "constant_group_rate": constant_rate,
            "query_auc_hit": query_auc,
            "evidence_auc_hit": evidence_auc,
            "answer_auc_strict": answer_auc,
        }
        if auc_hit < 0.95:
            failures.append(f"{run}: reward_auc_hit {auc_hit:.3f} < 0.95")
        if auc_strict < 0.90:
            failures.append(f"{run}: reward_auc_strict {auc_strict:.3f} < 0.90")
        if constant_rate > 0.05:
            failures.append(f"{run}: constant_group_rate {constant_rate:.3f} > 0.05")
        if query_auc < 0.90 or evidence_auc < 0.90 or answer_auc < 0.90:
            failures.append(f"{run}: node component AUC below threshold")
    passed = NODE_SUMMARY.exists() and not missing_runs and not failures
    return {
        "requirement": "Node-level DAG-IG reward audit is discriminative and non-collapsed.",
        "status": status(passed),
        "passed": passed,
        "missing_runs": missing_runs,
        "thresholds": thresholds,
        "failures": failures,
        "evidence": [str(NODE_SUMMARY), str(ASSETS / "node_credit_diagnostic_table.tex")],
    }


def check_grpo_results() -> dict[str, Any]:
    consolidated = load_json(CONSOLIDATED) if CONSOLIDATED.exists() else {}
    metrics = consolidated.get("metrics", {})
    summaries = {
        "seed42": MAIN_TRAIN_SUMMARY,
        "seed43": SEED43_TRAIN_SUMMARY,
        "goldfixed": GOLDFIXED_TRAIN_SUMMARY,
    }
    train_failures: list[str] = []
    train_observed: dict[str, Any] = {}
    for name, path in summaries.items():
        if not path.exists():
            train_failures.append(f"{name}: missing {path}")
            continue
        obj = load_json(path)
        train_observed[name] = obj
        if obj.get("status") != "success":
            train_failures.append(f"{name}: status={obj.get('status')}")
        if obj.get("optimizer_steps") != 60:
            train_failures.append(f"{name}: optimizer_steps={obj.get('optimizer_steps')}, expected 60")
        if obj.get("constant_reward_groups", 999) > 5:
            train_failures.append(f"{name}: constant_reward_groups={obj.get('constant_reward_groups')} > 5")

    required_metric_keys = [
        "format_dev",
        "format_test",
        "seed42_dev",
        "seed42_test",
        "seed43_dev",
        "seed43_test",
        "goldfixed_dev",
        "goldfixed_test",
    ]
    missing_metric_keys = [k for k in required_metric_keys if k not in metrics]
    comparisons: dict[str, float] = {}
    metric_failures: list[str] = []
    if not missing_metric_keys:
        comparisons = {
            "seed42_dev_strict_gain": metrics["seed42_dev"]["strict"] - metrics["format_dev"]["strict"],
            "seed42_test_strict_gain": metrics["seed42_test"]["strict"] - metrics["format_test"]["strict"],
            "seed43_dev_strict_gain": metrics["seed43_dev"]["strict"] - metrics["format_dev"]["strict"],
            "seed43_test_strict_gain": metrics["seed43_test"]["strict"] - metrics["format_test"]["strict"],
            "seed42_dev_r5_gain": metrics["seed42_dev"]["r5"] - metrics["format_dev"]["r5"],
            "seed42_test_r5_gain": metrics["seed42_test"]["r5"] - metrics["format_test"]["r5"],
        }
        for key, value in comparisons.items():
            if key.startswith("seed42") and value <= 0:
                metric_failures.append(f"{key}: {value:.4f} <= 0")
        if comparisons["seed43_dev_strict_gain"] <= 0 or comparisons["seed43_test_strict_gain"] <= 0:
            metric_failures.append("seed43 does not confirm strict improvement on both splits")
    ckpt_ok = all((MAIN_CKPT / name).exists() for name in ["adapter_model.safetensors", "adapter_config.json"])
    passed = CONSOLIDATED.exists() and ckpt_ok and not train_failures and not missing_metric_keys and not metric_failures
    return {
        "requirement": "GRPO main training exists, is healthy, and improves Format-SFT on dev/test.",
        "status": status(passed),
        "passed": passed,
        "checkpoint": check_path(MAIN_CKPT),
        "checkpoint_adapter_exists": ckpt_ok,
        "training_failures": train_failures,
        "missing_metric_keys": missing_metric_keys,
        "metric_failures": metric_failures,
        "comparisons": comparisons,
        "main_numbers": {
            key: {
                "r5": metrics[key]["r5"],
                "strict": metrics[key]["strict"],
            }
            for key in required_metric_keys
            if key in metrics
        },
        "evidence": [str(CONSOLIDATED), str(MAIN_TRAIN_SUMMARY), str(MAIN_CKPT)],
    }


def check_paper_package() -> dict[str, Any]:
    audit = load_json(PAPER_AUDIT) if PAPER_AUDIT.exists() else {}
    required = [
        ASSETS / "latex/main.tex",
        ASSETS / "latex/appendix.tex",
        ASSETS / "latex/references.bib",
        ASSETS / "main_results_table.tex",
        ASSETS / "node_credit_diagnostic_table.tex",
        ASSETS / "FINAL_HANDOFF_PROMPT.md",
        ASSETS / "SUBMISSION_READINESS_REPORT.md",
        ASSETS / "PDF_BUILD_PREFLIGHT_REPORT.md",
        ASSETS / "PAPER_LENGTH_AUDIT.md",
        ASSETS / "VENUE_TEMPLATE_CONVERSION_GUIDE.md",
        ASSETS / "venue_template_parts/body_sections.tex",
        ASSETS / "venue_template_parts/abstract.tex",
        ASSETS / "SUBMISSION_PACKAGE_INDEX.md",
        ASSETS / "SUBMISSION_PACKAGE_INDEX.json",
        ASSETS / "DAGIG_Pix2Fact_review_clean_source_bundle.zip",
        ASSETS / "REVIEW_CLEAN_BUNDLE_REPORT.md",
        ASSETS / "REVIEW_CLEAN_BUNDLE_REPORT.json",
        ASSETS / "REVIEW_CLEAN_ANONYMITY_AUDIT.md",
        ASSETS / "REVIEW_CLEAN_ANONYMITY_AUDIT.json",
        ASSETS / "PACKAGE_EXTRACT_VERIFICATION_REPORT.md",
        ASSETS / "package_extract_verification.json",
        ASSETS / "DAGIG_Pix2Fact_paper_source_bundle.tar.gz",
        ASSETS / "DAGIG_Pix2Fact_overleaf_source_bundle.zip",
        ASSETS / "submission_bundle/main.tex",
    ]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.is_file() and path.stat().st_size == 0]
    passed = bool(audit.get("overall_pass")) and not missing and not empty
    return {
        "requirement": "Paper assets and self-contained submission bundle are mechanically consistent.",
        "status": status(passed),
        "passed": passed,
        "paper_asset_audit_overall_pass": audit.get("overall_pass"),
        "pdflatex_available": shutil.which("pdflatex") is not None,
        "missing": missing,
        "empty": empty,
        "evidence": [str(PAPER_AUDIT), str(ASSETS / "DAGIG_Pix2Fact_overleaf_source_bundle.zip")],
    }


def check_compiled_pdf() -> dict[str, Any]:
    pdf_path = ASSETS / "submission_bundle/main.pdf"
    preflight = load_json(PDF_PREFLIGHT) if PDF_PREFLIGHT.exists() else {}
    post_compile = load_json(POST_COMPILE_PDF_AUDIT) if POST_COMPILE_PDF_AUDIT.exists() else {}
    layout = load_json(PDF_LAYOUT_AUDIT) if PDF_LAYOUT_AUDIT.exists() else {}
    pdf_ok = pdf_path.exists() and pdf_path.stat().st_size > 0
    preflight_ok = bool(preflight.get("source_check_passed")) and bool(preflight.get("pdf_compilation_passed"))
    post_compile_ok = bool(post_compile.get("passed"))
    layout_ok = bool(layout.get("passed"))
    passed = pdf_ok and preflight_ok and post_compile_ok and layout_ok
    return {
        "requirement": "Compiled source PDF is built and rendered-content audited.",
        "status": status(passed),
        "passed": passed,
        "pdf_path": str(pdf_path),
        "pdf_exists": pdf_ok,
        "pdf_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "pdf_pages_known": layout.get("pdfinfo", {}).get("pages"),
        "preflight_pdf_compilation_passed": preflight.get("pdf_compilation_passed"),
        "post_compile_pdf_audit_passed": post_compile.get("passed"),
        "pdf_layout_audit_passed": layout.get("passed"),
        "post_compile_checks": post_compile.get("checks", {}),
        "pdf_layout_checks": layout.get("checks", {}),
        "evidence": [
            str(pdf_path),
            str(ASSETS / "PDF_BUILD_PREFLIGHT_REPORT.md"),
            str(ASSETS / "POST_COMPILE_PDF_AUDIT.md"),
            str(ASSETS / "PDF_LAYOUT_AUDIT.md"),
        ],
    }


def check_venue_decision() -> dict[str, Any]:
    audit = load_json(VENUE_DECISION_AUDIT) if VENUE_DECISION_AUDIT.exists() else {}
    ready = bool(audit.get("ready_for_target_conversion"))
    missing: list[str] = []
    if not VENUE_DECISION_AUDIT.exists():
        missing.append(str(VENUE_DECISION_AUDIT))
    return {
        "requirement": "Target venue/template/review/page-rule decisions are filled for final conversion.",
        "status": status(ready),
        "passed": ready,
        "ready_for_target_conversion": ready,
        "ready_for_final_submission": bool(audit.get("ready_for_final_submission")),
        "required_blank_fields": audit.get("required_blank_fields", []),
        "checkbox_groups_not_single_choice": audit.get("checkbox_groups_not_single_choice", []),
        "missing": missing,
        "evidence": [str(VENUE_DECISION_AUDIT), str(ASSETS / "VENUE_DECISION_FORM.md")],
    }


def check_final_submission_gate() -> dict[str, Any]:
    audit = load_json(FINAL_SUBMISSION_GATE) if FINAL_SUBMISSION_GATE.exists() else {}
    ready = bool(audit.get("final_submission_ready"))
    missing: list[str] = []
    if not FINAL_SUBMISSION_GATE.exists():
        missing.append(str(FINAL_SUBMISSION_GATE))
    return {
        "requirement": "Final venue-formatted submission gate passes before upload.",
        "status": status(ready),
        "passed": ready,
        "final_submission_ready": ready,
        "checks": audit.get("checks", {}),
        "blockers": audit.get("blockers", []),
        "missing": missing,
        "evidence": [str(FINAL_SUBMISSION_GATE), str(ASSETS / "FINAL_SUBMISSION_GATE.md")],
    }


def build_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# DAG-IG / Pix2Fact Goal Completion Audit\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"- experimental mainline complete: `{audit['experimental_mainline_complete']}`\n")
    lines.append(f"- paper package ready for template conversion: `{audit['paper_package_ready_for_template']}`\n")
    lines.append(f"- compiled source PDF verified: `{audit['compiled_source_pdf_verified']}`\n")
    lines.append(f"- venue decision ready for target conversion: `{audit['venue_decision_ready_for_target_conversion']}`\n")
    lines.append(f"- final submission gate ready: `{audit['final_submission_gate_ready']}`\n")
    lines.append(f"- final paper complete: `{audit['final_paper_complete']}`\n")
    lines.append(f"- pdflatex available in this environment: `{audit['environment']['pdflatex_available']}`\n")
    lines.append("\n")
    lines.append(
        "Interpretation: the experiment, source package, and generic compiled PDF are verified. "
        "The remaining paper-production work is venue-template formatting and author/anonymous metadata.\n\n"
    )

    lines.append("## Requirement Checks\n\n")
    lines.append("| requirement | status | evidence |\n")
    lines.append("|---|---|---|\n")
    for item in audit["checks"]:
        evidence = "<br>".join(item.get("evidence", []))
        lines.append(f"| {item['requirement']} | `{item['status']}` | {evidence} |\n")
    lines.append("\n")

    grpo = audit["checks_by_key"]["grpo_results"]
    lines.append("## Main Result Evidence\n\n")
    lines.append("| comparison | gain |\n")
    lines.append("|---|---:|\n")
    for key, value in grpo["comparisons"].items():
        lines.append(f"| {key} | {100.0 * value:.1f} pts |\n")
    lines.append("\n")

    reward = audit["checks_by_key"]["reward_audit"]
    lines.append("## Reward Audit Evidence\n\n")
    lines.append("| run | reward AUC hit | reward AUC strict | constant group rate | query AUC hit | evidence AUC hit | answer AUC strict |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for run, vals in reward["thresholds"].items():
        lines.append(
            f"| {run} | {vals['reward_auc_hit']:.3f} | {vals['reward_auc_strict']:.3f} | "
            f"{pct(vals['constant_group_rate'])} | {vals['query_auc_hit']:.3f} | "
            f"{vals['evidence_auc_hit']:.3f} | {vals['answer_auc_strict']:.3f} |\n"
        )
    lines.append("\n")

    venue = audit["checks_by_key"]["venue_decision"]
    lines.append("## Venue Decision Evidence\n\n")
    lines.append(f"- ready for target conversion: `{venue['ready_for_target_conversion']}`\n")
    lines.append(f"- blank required fields: `{venue['required_blank_fields']}`\n")
    lines.append(f"- checkbox groups needing exactly one choice: `{venue['checkbox_groups_not_single_choice']}`\n\n")

    final_gate = audit["checks_by_key"]["final_submission_gate"]
    lines.append("## Final Submission Gate Evidence\n\n")
    lines.append(f"- final submission ready: `{final_gate['final_submission_ready']}`\n")
    lines.append(f"- blockers: `{final_gate['blockers']}`\n\n")

    lines.append("## Remaining Work\n\n")
    for item in audit["remaining_work"]:
        lines.append(f"- {item}\n")
    lines.append("\n")

    lines.append("## Do Not Reopen As Mainline\n\n")
    for item in audit["do_not_reopen"]:
        lines.append(f"- {item}\n")
    lines.append("\n")
    return "".join(lines)


def main() -> None:
    checks_by_key = {
        "data_schema": check_data_and_schema(),
        "two_stage_predictions": check_two_stage_predictions(),
        "mainline_schema_contract": check_mainline_schema_contract(),
        "reward_audit": check_reward_audit(),
        "grpo_results": check_grpo_results(),
        "paper_package": check_paper_package(),
        "compiled_pdf": check_compiled_pdf(),
        "venue_decision": check_venue_decision(),
        "final_submission_gate": check_final_submission_gate(),
    }
    checks = list(checks_by_key.values())
    experimental_keys = ["data_schema", "two_stage_predictions", "mainline_schema_contract", "reward_audit", "grpo_results"]
    experimental_mainline_complete = all(checks_by_key[key]["passed"] for key in experimental_keys)
    paper_package_ready = checks_by_key["paper_package"]["passed"]
    compiled_source_pdf_verified = checks_by_key["compiled_pdf"]["passed"]
    venue_decision_ready = checks_by_key["venue_decision"]["passed"]
    final_submission_gate_ready = checks_by_key["final_submission_gate"]["passed"]
    pdflatex_available = shutil.which("pdflatex") is not None
    final_paper_complete = final_submission_gate_ready
    remaining_work = []
    if not venue_decision_ready:
        remaining_work.append("Fill VENUE_DECISION_FORM.md and pass VENUE_DECISION_AUDIT.md with --require-ready.")
    remaining_work.extend(
        [
            "Convert the source to the target venue template.",
            "Set author/anonymous metadata according to the target venue.",
            "Adjust appendix length to the target venue page limit.",
            "Compile the venue PDF and rerun post-compile content/layout audits.",
            "Pass FINAL_SUBMISSION_GATE.md with --require-ready before upload.",
        ]
    )
    audit = {
        "experimental_mainline_complete": experimental_mainline_complete,
        "paper_package_ready_for_template": paper_package_ready,
        "compiled_source_pdf_verified": compiled_source_pdf_verified,
        "venue_decision_ready_for_target_conversion": venue_decision_ready,
        "final_submission_gate_ready": final_submission_gate_ready,
        "final_paper_complete": final_paper_complete,
        "environment": {
            "pdflatex_available": pdflatex_available,
            "bibtex_available": shutil.which("bibtex") is not None,
        },
        "checks_by_key": checks_by_key,
        "checks": checks,
        "remaining_work": remaining_work,
        "do_not_reopen": [
            "DAG-SFT trace imitation as the main method.",
            "Query reranking/switching as the main method.",
            "No-teacher fusion as the main method.",
            "Broad answer repair as the main method.",
            "Same-recipe GRPO reruns without a new mechanism.",
        ],
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_markdown(audit), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not experimental_mainline_complete or not paper_package_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
