#!/usr/bin/env python3
"""Audit paper-facing assets for path, number, citation, and command drift."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
REPORTS = ROOT / "reports"
DERIVED = Path("outputs/dagig_grpo_main/derived_assets")

MANIFEST = ASSETS / "paper_experiment_manifest.json"
CONSOLIDATED = REPORTS / "paper_main_v1_consolidated_results.json"
NODE_SUMMARY = REPORTS / "node_credit_component_analysis/node_credit_component_summary.json"
MAIN_TABLE = ASSETS / "main_results_table.csv"
NODE_TABLE = ASSETS / "node_credit_diagnostic_table.csv"
DRAFT = ASSETS / "PAPER_DRAFT_V0.md"
BRIEF = ASSETS / "PAPER_MAIN_EVIDENCE_BRIEF.md"
REPRO = ASSETS / "REPRODUCIBILITY_APPENDIX.md"
CLAIMS = ASSETS / "CLAIMS_EVIDENCE_MATRIX.md"
RISKS = ASSETS / "REVIEWER_RISK_REGISTER.md"
COMMANDS = ASSETS / "reproduce_main_commands.sh"
RELEASE_CHECKS = ASSETS / "run_release_checks.sh"
MAINLINE_EVIDENCE_JSON = ASSETS / "MAINLINE_EVIDENCE_CHAIN.json"
MAINLINE_EVIDENCE_MD = ASSETS / "MAINLINE_EVIDENCE_CHAIN.md"
MAINLINE_SCHEMA_CONTRACT_JSON = ASSETS / "MAINLINE_SCHEMA_CONTRACT_AUDIT.json"
MAINLINE_SCHEMA_CONTRACT_MD = ASSETS / "MAINLINE_SCHEMA_CONTRACT_AUDIT.md"
LATEX_MAIN = ASSETS / "latex/main.tex"
BIB = ASSETS / "latex/references.bib"
LATEX_DIR = ASSETS / "latex"
LATEX_MAKEFILE = LATEX_DIR / "Makefile"
SUBMISSION_BUNDLE = ASSETS / "submission_bundle"
SUBMISSION_BUNDLE_MANIFEST = SUBMISSION_BUNDLE / "SUBMISSION_BUNDLE_MANIFEST.json"
SUBMISSION_TARBALL = ASSETS / "DAGIG_Pix2Fact_paper_source_bundle.tar.gz"
SUBMISSION_ZIP = ASSETS / "DAGIG_Pix2Fact_overleaf_source_bundle.zip"
REVIEW_CLEAN_ZIP = ASSETS / "DAGIG_Pix2Fact_review_clean_source_bundle.zip"
REVIEW_CLEAN_BUNDLE = ASSETS / "review_clean_bundle"
REVIEW_CLEAN_REPORT_JSON = ASSETS / "REVIEW_CLEAN_BUNDLE_REPORT.json"
REVIEW_CLEAN_REPORT_MD = ASSETS / "REVIEW_CLEAN_BUNDLE_REPORT.md"
REVIEW_CLEAN_ANON_JSON = ASSETS / "REVIEW_CLEAN_ANONYMITY_AUDIT.json"
REVIEW_CLEAN_ANON_MD = ASSETS / "REVIEW_CLEAN_ANONYMITY_AUDIT.md"
REVIEW_CLEAN_COMPILE_JSON = ASSETS / "review_clean_compile_verification.json"
REVIEW_CLEAN_COMPILE_MD = ASSETS / "REVIEW_CLEAN_COMPILE_VERIFICATION.md"
PACKAGE_INDEX_JSON = ASSETS / "SUBMISSION_PACKAGE_INDEX.json"
PACKAGE_INDEX_MD = ASSETS / "SUBMISSION_PACKAGE_INDEX.md"
PACKAGE_EXTRACT_VERIFY_JSON = ASSETS / "package_extract_verification.json"
PACKAGE_EXTRACT_VERIFY_MD = ASSETS / "PACKAGE_EXTRACT_VERIFICATION_REPORT.md"
SOURCE_BUNDLE_COMPILE_JSON = ASSETS / "source_bundle_compile_verification.json"
SOURCE_BUNDLE_COMPILE_MD = ASSETS / "SOURCE_BUNDLE_COMPILE_VERIFICATION.md"
ARTIFACT_CHECKSUMS_JSON = ASSETS / "ARTIFACT_CHECKSUMS.json"
ARTIFACT_CHECKSUMS_MD = ASSETS / "ARTIFACT_CHECKSUMS.md"
SHA256SUMS = ASSETS / "SHA256SUMS.txt"
TEXT_FINALIZATION_JSON = ASSETS / "text_finalization_audit.json"
TEXT_FINALIZATION_MD = ASSETS / "TEXT_FINALIZATION_AUDIT.md"
VENUE_WORKSPACE_AUDIT_JSON = ASSETS / "venue_workspace_audit.json"
VENUE_WORKSPACE_AUDIT_MD = ASSETS / "VENUE_WORKSPACE_AUDIT.md"
SUBMISSION_PAYLOAD_JSON = ASSETS / "submission_payload_index.json"
SUBMISSION_PAYLOAD_MD = ASSETS / "SUBMISSION_PAYLOAD_INDEX.md"
STATUS = REPORTS / "PAPER_MAIN_V1_CURRENT_STATUS.md"

AUDIT_JSON = ASSETS / "paper_asset_audit.json"
AUDIT_MD = ASSETS / "PAPER_ASSET_AUDIT_REPORT.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def rel(path: Path | str) -> str:
    return str(path)


def flatten_manifest_paths(obj: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            paths.extend(flatten_manifest_paths(value))
    elif isinstance(obj, list):
        for value in obj:
            paths.extend(flatten_manifest_paths(value))
    elif isinstance(obj, str):
        if obj.startswith("outputs/") or obj.startswith("scripts/"):
            paths.append(obj)
    return paths


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_bundle_paths(text: str) -> str:
    replacements = {
        "../figures/dagig_method_diagram.tex": "figures/dagig_method_diagram.tex",
        "../figures/dagig_reward_equations.tex": "figures/dagig_reward_equations.tex",
        "../main_results_table.tex": "tables/main_results_table.tex",
        "../node_credit_diagnostic_table.tex": "tables/node_credit_diagnostic_table.tex",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def extract_latex_part(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"could not extract {label}")
    return match.group(1).strip() + "\n"


def check_main_table(consolidated: dict[str, Any]) -> dict[str, Any]:
    table_rows = read_csv(MAIN_TABLE)
    expected_keys = {
        ("Format-SFT", "dev"): "format_dev",
        ("Format-SFT", "test"): "format_test",
        ("DAG-IG seed42 main", "dev"): "seed42_dev",
        ("DAG-IG seed42 main", "test"): "seed42_test",
        ("DAG-IG seed43 confirm", "dev"): "seed43_dev",
        ("DAG-IG seed43 confirm", "test"): "seed43_test",
        ("Goldfixed control", "dev"): "goldfixed_dev",
        ("Goldfixed control", "test"): "goldfixed_test",
    }
    metrics = consolidated["metrics"]
    mismatches: list[str] = []
    for row in table_rows:
        key = (row["method"], row["split"])
        metric_key = expected_keys.get(key)
        if not metric_key:
            mismatches.append(f"unexpected main table row {key}")
            continue
        m = metrics[metric_key]
        checks = {
            "r1": pct(m["r1"]),
            "r3": pct(m["r3"]),
            "r5": pct(m["r5"]),
            "answer_correct": pct(m["answer"]),
            "strict_success": pct(m["strict"]),
            "format_success": pct(m["format"]),
            "retrieval_miss": str(m["retrieval_miss"]),
            "hit_answer_wrong": str(m["hit_answer_wrong"]),
        }
        for field, expected in checks.items():
            if row[field] != expected:
                mismatches.append(f"{key} field {field}: table={row[field]} expected={expected}")
    if len(table_rows) != len(expected_keys):
        mismatches.append(f"main table row count {len(table_rows)} expected {len(expected_keys)}")
    return {"rows": len(table_rows), "mismatches": mismatches, "passed": not mismatches}


def check_node_table(node_summary: dict[str, Any]) -> dict[str, Any]:
    rows = read_csv(NODE_TABLE)
    mismatches: list[str] = []
    expected_runs = ["seed42_main", "seed43_confirm", "goldfixed_control"]
    by_run = {row["run"]: row for row in rows}
    for run_name in expected_runs:
        row = by_run.get(run_name)
        if not row:
            mismatches.append(f"missing node table row {run_name}")
            continue
        run = node_summary[run_name]
        checks = {
            "reward_auc_hit": f"{run['reward_auc_retrieval_hit']:.3f}",
            "reward_auc_strict": f"{run['reward_auc_strict_success']:.3f}",
            "constant_groups": f"{run['groups']['constant_groups']}/{run['groups']['groups']}",
            "top_hit": pct(run["groups"]["top_retrieval_hit"]),
            "bottom_hit": pct(run["groups"]["bottom_retrieval_hit"]),
            "top_strict": pct(run["groups"]["top_strict_success"]),
            "bottom_strict": pct(run["groups"]["bottom_strict_success"]),
            "query_auc_hit": f"{run['components']['query']['auc_retrieval_hit']:.3f}",
            "evidence_auc_hit": f"{run['components']['evidence']['auc_retrieval_hit']:.3f}",
            "answer_auc_strict": f"{run['components']['answer']['auc_strict_success']:.3f}",
        }
        for field, expected in checks.items():
            if row[field] != expected:
                mismatches.append(f"{run_name} field {field}: table={row[field]} expected={expected}")
    if len(rows) != len(expected_runs):
        mismatches.append(f"node table row count {len(rows)} expected {len(expected_runs)}")
    return {"rows": len(rows), "mismatches": mismatches, "passed": not mismatches}


def check_text_claims(consolidated: dict[str, Any]) -> dict[str, Any]:
    texts = {
        "draft": DRAFT.read_text(encoding="utf-8"),
        "brief": BRIEF.read_text(encoding="utf-8"),
        "repro": REPRO.read_text(encoding="utf-8"),
        "claims": CLAIMS.read_text(encoding="utf-8"),
        "risks": RISKS.read_text(encoding="utf-8"),
        "status": STATUS.read_text(encoding="utf-8"),
    }
    metrics = consolidated["metrics"]
    required = {
        "Format-SFT dev strict": pct(metrics["format_dev"]["strict"]) + "%",
        "seed42 dev strict": pct(metrics["seed42_dev"]["strict"]) + "%",
        "Format-SFT test strict": pct(metrics["format_test"]["strict"]) + "%",
        "seed42 test strict": pct(metrics["seed42_test"]["strict"]) + "%",
        "seed42 dev R@5": pct(metrics["seed42_dev"]["r5"]) + "%",
        "seed42 test R@5": pct(metrics["seed42_test"]["r5"]) + "%",
        "DAG-SFT not main": "DAG-SFT is not",
    }
    missing: list[str] = []
    for label, needle in required.items():
        if not any(needle in text for text in texts.values()):
            missing.append(f"{label}: `{needle}` not found in tracked paper text")
    return {"required_claims": required, "missing": missing, "passed": not missing}


def check_cross_bundle_consistency() -> dict[str, Any]:
    mismatches: list[str] = []
    missing: list[str] = []

    def must_read(path: Path) -> str:
        if not path.exists():
            missing.append(str(path))
            return ""
        return path.read_text(encoding="utf-8")

    source_main = normalize_bundle_paths(must_read(LATEX_DIR / "main.tex"))
    submission_main = must_read(SUBMISSION_BUNDLE / "main.tex")
    clean_main = must_read(REVIEW_CLEAN_BUNDLE / "main.tex")
    if source_main and submission_main and source_main != submission_main:
        mismatches.append("submission_bundle/main.tex differs from normalized source latex/main.tex")
    if submission_main and clean_main and submission_main != clean_main:
        mismatches.append("review_clean_bundle/main.tex differs from submission_bundle/main.tex")

    mirrored_files = [
        (LATEX_DIR / "appendix.tex", SUBMISSION_BUNDLE / "appendix.tex", REVIEW_CLEAN_BUNDLE / "appendix.tex"),
        (LATEX_DIR / "algorithm_dagig_grpo.tex", SUBMISSION_BUNDLE / "algorithm_dagig_grpo.tex", REVIEW_CLEAN_BUNDLE / "algorithm_dagig_grpo.tex"),
        (LATEX_DIR / "diagnostic_branches_table.tex", SUBMISSION_BUNDLE / "diagnostic_branches_table.tex", REVIEW_CLEAN_BUNDLE / "diagnostic_branches_table.tex"),
        (LATEX_DIR / "references.bib", SUBMISSION_BUNDLE / "references.bib", REVIEW_CLEAN_BUNDLE / "references.bib"),
        (ASSETS / "main_results_table.tex", SUBMISSION_BUNDLE / "tables/main_results_table.tex", REVIEW_CLEAN_BUNDLE / "tables/main_results_table.tex"),
        (ASSETS / "node_credit_diagnostic_table.tex", SUBMISSION_BUNDLE / "tables/node_credit_diagnostic_table.tex", REVIEW_CLEAN_BUNDLE / "tables/node_credit_diagnostic_table.tex"),
        (ASSETS / "figures/dagig_method_diagram.tex", SUBMISSION_BUNDLE / "figures/dagig_method_diagram.tex", REVIEW_CLEAN_BUNDLE / "figures/dagig_method_diagram.tex"),
        (ASSETS / "figures/dagig_reward_equations.tex", SUBMISSION_BUNDLE / "figures/dagig_reward_equations.tex", REVIEW_CLEAN_BUNDLE / "figures/dagig_reward_equations.tex"),
    ]
    for source, submission, clean in mirrored_files:
        source_text = must_read(source)
        submission_text = must_read(submission)
        clean_text = must_read(clean)
        if source_text and submission_text and source_text != submission_text:
            mismatches.append(f"{submission} differs from {source}")
        if submission_text and clean_text and submission_text != clean_text:
            mismatches.append(f"{clean} differs from {submission}")

    venue_dir = ASSETS / "venue_template_parts"
    submission_venue_dir = SUBMISSION_BUNDLE / "venue_template_parts"
    clean_venue_dir = REVIEW_CLEAN_BUNDLE / "venue_template_parts"
    main_for_parts = source_main
    try:
        expected_parts = {
            "title.tex": extract_latex_part(r"\\title\{(.+?)\}\s*\\author", main_for_parts, "title"),
            "abstract.tex": extract_latex_part(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", main_for_parts, "abstract"),
            "body_sections.tex": extract_latex_part(r"(\\section\{Introduction\}.+?)\\appendix", main_for_parts, "body"),
            "appendix_sections.tex": must_read(LATEX_DIR / "appendix.tex").strip() + "\n",
        }
        expected_parts["abstract_environment.tex"] = "\\begin{abstract}\n" + expected_parts["abstract.tex"] + "\\end{abstract}\n"
        for name, expected in expected_parts.items():
            for path in [venue_dir / name, submission_venue_dir / name, clean_venue_dir / name]:
                actual = must_read(path)
                if actual and actual != expected:
                    mismatches.append(f"{path} differs from generated expected {name}")
    except Exception as exc:
        mismatches.append(f"venue template part extraction failed: {exc}")

    return {"missing": missing, "mismatches": mismatches, "passed": not missing and not mismatches}


def check_claim_boundaries(consolidated: dict[str, Any]) -> dict[str, Any]:
    main = LATEX_MAIN.read_text(encoding="utf-8")
    appendix = (LATEX_DIR / "appendix.tex").read_text(encoding="utf-8")
    body = main + "\n" + appendix
    abstract_match = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", main, flags=re.S)
    abstract = abstract_match.group(1) if abstract_match else ""
    metrics = consolidated["metrics"]
    required_in_abstract = [
        pct(metrics["format_dev"]["strict"]) + r"\%",
        pct(metrics["seed42_dev"]["strict"]) + r"\%",
        pct(metrics["format_test"]["strict"]) + r"\%",
        pct(metrics["seed42_test"]["strict"]) + r"\%",
        "frozen offline BM25",
    ]
    required_boundaries = {
        "DAG-SFT diagnostic boundary": "DAG-SFT, DPO, query-reranking, fusion, and answer-repair experiments are diagnostics rather than the main result",
        "live web boundary": "does not establish live web-search generalization",
        "future end-to-end boundary": "Full end-to-end optimization of visual grounding, retrieval, evidence selection, and answer extraction remains future work",
        "reader/retrieval bottleneck": "the stage-1 policy still sometimes fails to formulate a query that retrieves support, and the reader sometimes extracts the wrong span",
    }
    missing: list[str] = []
    for needle in required_in_abstract:
        if needle not in abstract:
            missing.append(f"abstract missing `{needle}`")
    for label, needle in required_boundaries.items():
        if needle not in body:
            missing.append(f"{label}: missing `{needle}`")

    forbidden_patterns = [
        r"DAG-SFT\s+is\s+the\s+main\s+method",
        r"answer extraction\s+is\s+solved",
        r"solves\s+answer extraction",
        r"establishes\s+live\s+web-search\s+generalization",
        r"(?<!not a )complete\s+web-search\s+agent",
        r"goldfixed\s+(?:is|as)\s+the\s+(?:main|best|final)\s+checkpoint",
    ]
    forbidden_hits: list[dict[str, str]] = []
    for pattern in forbidden_patterns:
        for match in re.finditer(pattern, body, flags=re.I):
            start = max(0, match.start() - 80)
            end = min(len(body), match.end() + 80)
            forbidden_hits.append({"pattern": pattern, "context": body[start:end].replace("\n", " ")})
    return {
        "missing": missing,
        "forbidden_hits": forbidden_hits,
        "passed": not missing and not forbidden_hits,
    }


def check_citations() -> dict[str, Any]:
    main = "\n".join(path.read_text(encoding="utf-8") for path in sorted(LATEX_DIR.glob("*.tex")))
    bib = BIB.read_text(encoding="utf-8")
    cites: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", main):
        cites.update(key.strip() for key in match.group(1).split(",") if key.strip())
    entries = set(re.findall(r"@\w+\{([^,]+),", bib))
    return {
        "n_cites": len(cites),
        "n_entries": len(entries),
        "missing": sorted(cites - entries),
        "unused": sorted(entries - cites),
        "passed": not (cites - entries),
    }


def check_latex_structure() -> dict[str, Any]:
    roots = [LATEX_DIR, SUBMISSION_BUNDLE]
    problems: list[str] = []
    summaries: dict[str, Any] = {}
    for root in roots:
        if not root.exists():
            problems.append(f"missing latex root {root}")
            continue
        groups: dict[str, dict[str, list[Any]]] = {
            "compile": {"labels": [], "refs": [], "inputs": []},
            "venue_template_parts": {"labels": [], "refs": [], "inputs": []},
        }
        tex_files = sorted(root.rglob("*.tex"))
        for tex in tex_files:
            group = "venue_template_parts" if "venue_template_parts" in tex.parts else "compile"
            text = tex.read_text(encoding="utf-8")
            groups[group]["labels"].extend((str(tex), label) for label in re.findall(r"\\label\{([^}]+)\}", text))
            groups[group]["refs"].extend((str(tex), ref) for ref in re.findall(r"\\(?:ref|pageref|autoref)\{([^}]+)\}", text))
            groups[group]["inputs"].extend((tex, target) for target in re.findall(r"\\input\{([^}]+)\}", text))
        group_summaries: dict[str, Any] = {}
        for group_name, group_data in groups.items():
            labels = group_data["labels"]
            refs = group_data["refs"]
            inputs = group_data["inputs"]
            label_values = [label for _, label in labels]
            duplicate_labels = sorted({label for label in label_values if label_values.count(label) > 1})
            undefined_refs = sorted({ref for _, ref in refs if ref not in set(label_values)})
            missing_inputs: list[str] = []
            for tex, target in inputs:
                base = root if group_name == "venue_template_parts" else tex.parent
                target_path = (base / target).resolve()
                if not target_path.exists():
                    missing_inputs.append(f"{tex}:{target}")
            if duplicate_labels:
                problems.append(f"{root}:{group_name}: duplicate labels {duplicate_labels}")
            if undefined_refs:
                problems.append(f"{root}:{group_name}: undefined refs {undefined_refs}")
            if missing_inputs:
                problems.append(f"{root}:{group_name}: missing inputs {missing_inputs}")
            group_summaries[group_name] = {
                "labels": len(labels),
                "refs": len(refs),
                "inputs": len(inputs),
                "duplicate_labels": duplicate_labels,
                "undefined_refs": undefined_refs,
                "missing_inputs": missing_inputs,
            }
        summaries[str(root)] = {
            "tex_files": len(tex_files),
            "groups": group_summaries,
        }
    return {"summaries": summaries, "problems": problems, "passed": not problems}


def check_shell() -> dict[str, Any]:
    checks = []
    for path in [COMMANDS, RELEASE_CHECKS]:
        if not path.exists():
            checks.append({"path": str(path), "returncode": 1, "stderr": "missing"})
            continue
        result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
        checks.append({"path": str(path), "returncode": result.returncode, "stderr": result.stderr.strip()})
    return {
        "command": "bash -n paper shell scripts",
        "checks": checks,
        "returncode": 0 if all(item["returncode"] == 0 for item in checks) else 1,
        "stderr": "; ".join(f"{item['path']}: {item['stderr']}" for item in checks if item["stderr"]),
        "passed": all(item["returncode"] == 0 for item in checks),
    }


def check_latex_makefile() -> dict[str, Any]:
    if not LATEX_MAKEFILE.exists():
        return {"command": "make check", "returncode": 1, "stderr": "Makefile missing", "passed": False}
    if shutil.which("make") is None:
        return {"command": "make check", "returncode": None, "stderr": "make unavailable", "passed": False}
    result = subprocess.run(["make", "check"], cwd=str(LATEX_DIR), text=True, capture_output=True)
    return {
        "command": f"cd {LATEX_DIR} && make check",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "passed": result.returncode == 0,
    }


def check_submission_bundle() -> dict[str, Any]:
    required = [
        SUBMISSION_BUNDLE / "main.tex",
        SUBMISSION_BUNDLE / "appendix.tex",
        SUBMISSION_BUNDLE / "diagnostic_branches_table.tex",
        SUBMISSION_BUNDLE / "algorithm_dagig_grpo.tex",
        SUBMISSION_BUNDLE / "references.bib",
        SUBMISSION_BUNDLE / "Makefile",
        SUBMISSION_BUNDLE / "tables/main_results_table.tex",
        SUBMISSION_BUNDLE / "tables/node_credit_diagnostic_table.tex",
        SUBMISSION_BUNDLE / "figures/dagig_method_diagram.tex",
        SUBMISSION_BUNDLE / "figures/dagig_reward_equations.tex",
        SUBMISSION_BUNDLE / "venue_template_parts/body_sections.tex",
        SUBMISSION_BUNDLE / "venue_template_parts/abstract.tex",
        SUBMISSION_BUNDLE / "README.md",
        SUBMISSION_BUNDLE / "docs/HANDOFF_README.md",
        SUBMISSION_BUNDLE / "docs/FINAL_HANDOFF_PROMPT.md",
        SUBMISSION_BUNDLE / "docs/MAINLINE_EVIDENCE_CHAIN.md",
        SUBMISSION_BUNDLE / "docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.md",
        SUBMISSION_BUNDLE / "docs/MAINLINE_SCHEMA_CONTRACT_AUDIT.json",
        SUBMISSION_BUNDLE / "docs/GOAL_COMPLETION_AUDIT.md",
        SUBMISSION_BUNDLE / "docs/PAPER_MAIN_V1_CURRENT_STATUS.md",
        SUBMISSION_BUNDLE / "docs/SUBMISSION_READINESS_REPORT.md",
        SUBMISSION_BUNDLE / "docs/PDF_BUILD_PREFLIGHT_REPORT.md",
        SUBMISSION_BUNDLE / "docs/POST_COMPILE_PDF_AUDIT.md",
        SUBMISSION_BUNDLE / "docs/PDF_LAYOUT_AUDIT.md",
        SUBMISSION_BUNDLE / "docs/TEXT_FINALIZATION_AUDIT.md",
        SUBMISSION_BUNDLE / "docs/PAPER_LENGTH_AUDIT.md",
        SUBMISSION_BUNDLE / "docs/SUBMISSION_ROUTE_GUIDE.md",
        SUBMISSION_BUNDLE / "docs/VENUE_DECISION_AUDIT.md",
        SUBMISSION_BUNDLE / "docs/FINAL_SUBMISSION_GATE.md",
        SUBMISSION_BUNDLE / "docs/VENUE_TEMPLATE_CONVERSION_GUIDE.md",
        SUBMISSION_BUNDLE / "docs/VENUE_DECISION_FORM.md",
        SUBMISSION_BUNDLE / "scripts/reproduce_main_commands.sh",
        SUBMISSION_BUNDLE / "scripts/run_release_checks.sh",
        SUBMISSION_BUNDLE / "scripts/post_compile_pdf_audit.py",
        SUBMISSION_BUNDLE / "scripts/prepare_venue_workspace.py",
        SUBMISSION_BUNDLE / "scripts/pdf_layout_audit.py",
        SUBMISSION_BUNDLE / "scripts/audit_venue_decision_form.py",
        SUBMISSION_BUNDLE / "scripts/audit_final_submission_gate.py",
        SUBMISSION_BUNDLE_MANIFEST,
        SUBMISSION_TARBALL,
        SUBMISSION_ZIP,
    ]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.is_file() and path.stat().st_size == 0]
    parent_refs: list[str] = []
    if SUBMISSION_BUNDLE.exists():
        for tex in SUBMISSION_BUNDLE.rglob("*.tex"):
            if "../" in tex.read_text(encoding="utf-8"):
                parent_refs.append(str(tex.relative_to(SUBMISSION_BUNDLE)))
    doc_mismatches: list[str] = []
    for name in [
        "PDF_BUILD_PREFLIGHT_REPORT.md",
        "POST_COMPILE_PDF_AUDIT.md",
        "PDF_LAYOUT_AUDIT.md",
        "MAINLINE_EVIDENCE_CHAIN.md",
        "MAINLINE_SCHEMA_CONTRACT_AUDIT.md",
        "MAINLINE_SCHEMA_CONTRACT_AUDIT.json",
        "TEXT_FINALIZATION_AUDIT.md",
        "VENUE_DECISION_FORM.md",
        "SUBMISSION_ROUTE_GUIDE.md",
        "VENUE_DECISION_AUDIT.md",
        "FINAL_SUBMISSION_GATE.md",
    ]:
        root_doc = ASSETS / name
        bundle_doc = SUBMISSION_BUNDLE / "docs" / name
        if root_doc.exists() and bundle_doc.exists() and root_doc.read_text(encoding="utf-8") != bundle_doc.read_text(encoding="utf-8"):
            doc_mismatches.append(f"docs/{name} differs from root {name}")
    manifest_passed = False
    manifest_error = ""
    if SUBMISSION_BUNDLE_MANIFEST.exists():
        try:
            manifest = load_json(SUBMISSION_BUNDLE_MANIFEST)
            manifest_passed = bool(manifest.get("validation", {}).get("passed"))
        except Exception as exc:  # pragma: no cover - defensive report path
            manifest_error = str(exc)
    make_result = None
    if shutil.which("make") and (SUBMISSION_BUNDLE / "Makefile").exists():
        make = subprocess.run(["make", "check"], cwd=str(SUBMISSION_BUNDLE), text=True, capture_output=True)
        make_result = {"returncode": make.returncode, "stdout": make.stdout.strip(), "stderr": make.stderr.strip()}
    else:
        make_result = {"returncode": None, "stdout": "", "stderr": "make or Makefile unavailable"}
    bundled_shell_checks = []
    for bundled_commands in [
        SUBMISSION_BUNDLE / "scripts/reproduce_main_commands.sh",
        SUBMISSION_BUNDLE / "scripts/run_release_checks.sh",
    ]:
        if bundled_commands.exists():
            command_check = subprocess.run(["bash", "-n", str(bundled_commands)], text=True, capture_output=True)
            bundled_shell_checks.append(
                {
                    "path": str(bundled_commands),
                    "returncode": command_check.returncode,
                    "stdout": command_check.stdout.strip(),
                    "stderr": command_check.stderr.strip(),
                }
            )
        else:
            bundled_shell_checks.append({"path": str(bundled_commands), "returncode": None, "stdout": "", "stderr": "missing"})
    bundled_python_checks = []
    for bundled_script in [
        SUBMISSION_BUNDLE / "scripts/post_compile_pdf_audit.py",
        SUBMISSION_BUNDLE / "scripts/prepare_venue_workspace.py",
        SUBMISSION_BUNDLE / "scripts/pdf_layout_audit.py",
        SUBMISSION_BUNDLE / "scripts/audit_venue_decision_form.py",
        SUBMISSION_BUNDLE / "scripts/audit_final_submission_gate.py",
    ]:
        if bundled_script.exists():
            script_check = subprocess.run(["python", "-m", "py_compile", str(bundled_script)], text=True, capture_output=True)
            bundled_python_checks.append(
                {
                    "path": str(bundled_script),
                    "returncode": script_check.returncode,
                    "stdout": script_check.stdout.strip(),
                    "stderr": script_check.stderr.strip(),
                }
            )
        else:
            bundled_python_checks.append({"path": str(bundled_script), "returncode": None, "stdout": "", "stderr": "missing"})
    python_result = {
        "returncode": 0 if all(item["returncode"] == 0 for item in bundled_python_checks) else 1,
        "stdout": "",
        "stderr": "; ".join(f"{item['path']}: {item['stderr']}" for item in bundled_python_checks if item["stderr"]),
        "checks": bundled_python_checks,
    }
    command_result = {
        "returncode": 0 if all(item["returncode"] == 0 for item in bundled_shell_checks) else 1,
        "stdout": "",
        "stderr": "; ".join(f"{item['path']}: {item['stderr']}" for item in bundled_shell_checks if item["stderr"]),
        "checks": bundled_shell_checks,
    }
    passed = (
        not missing
        and not empty
        and not parent_refs
        and not doc_mismatches
        and manifest_passed
        and make_result["returncode"] == 0
        and command_result["returncode"] == 0
        and python_result["returncode"] == 0
    )
    return {
        "missing": missing,
        "empty": empty,
        "tex_files_with_parent_refs": parent_refs,
        "doc_mismatches": doc_mismatches,
        "manifest_validation_passed": manifest_passed,
        "manifest_error": manifest_error,
        "make_check": make_result,
        "bundled_command_check": command_result,
        "bundled_python_check": python_result,
        "passed": passed,
    }


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_package_index() -> dict[str, Any]:
    required = [PACKAGE_INDEX_JSON, PACKAGE_INDEX_MD, SUBMISSION_TARBALL, SUBMISSION_ZIP, REVIEW_CLEAN_ZIP]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    if PACKAGE_INDEX_JSON.exists():
        try:
            index = load_json(PACKAGE_INDEX_JSON)
            expected = {
                "tarball": (SUBMISSION_TARBALL, index.get("tarball", {}).get("sha256")),
                "zipfile": (SUBMISSION_ZIP, index.get("zipfile", {}).get("sha256")),
                "review_clean_zip": (REVIEW_CLEAN_ZIP, index.get("review_clean_zip", {}).get("sha256")),
                "compiled_pdf": (SUBMISSION_BUNDLE / "main.pdf", index.get("compiled_pdf", {}).get("sha256")),
            }
            for label, (path, recorded_sha) in expected.items():
                if path.exists():
                    actual_sha = sha256_file(path)
                    if actual_sha != recorded_sha:
                        mismatches.append(f"{label}: recorded sha {recorded_sha} != actual {actual_sha}")
            if not index.get("bundle_validation_passed"):
                mismatches.append("bundle_validation_passed is false")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse package index: {exc}")
    return {"missing": missing, "empty": empty, "mismatches": mismatches, "passed": not missing and not empty and not mismatches}


def check_artifact_checksums() -> dict[str, Any]:
    required = [ARTIFACT_CHECKSUMS_JSON, ARTIFACT_CHECKSUMS_MD, SHA256SUMS]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    sha256sum_check: dict[str, Any] = {"returncode": None, "stdout": "", "stderr": "not run"}

    expected_labels = {
        "compiled_pdf",
        "full_source_zip",
        "full_source_tarball",
        "review_clean_zip",
        "package_index_json",
        "package_index_md",
    }
    if ARTIFACT_CHECKSUMS_JSON.exists() and SHA256SUMS.exists():
        try:
            manifest = load_json(ARTIFACT_CHECKSUMS_JSON)
            artifacts = manifest.get("artifacts", [])
            observed_labels = {item.get("label") for item in artifacts if isinstance(item, dict)}
            if observed_labels != expected_labels:
                mismatches.append(f"artifact labels {sorted(observed_labels)} != expected {sorted(expected_labels)}")

            sha_lines: dict[str, str] = {}
            for lineno, line in enumerate(SHA256SUMS.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    mismatches.append(f"SHA256SUMS line {lineno} is not standard sha/path format")
                    continue
                sha, path_str = parts
                sha_lines[path_str.strip()] = sha

            forbidden_manifest_targets = ["paper_asset_audit", "ARTIFACT_CHECKSUMS", "SHA256SUMS"]
            for item in artifacts:
                path = Path(item.get("path", ""))
                if any(pattern in str(path) for pattern in forbidden_manifest_targets):
                    mismatches.append(f"checksum manifest includes self-mutating audit/checksum target {path}")
                    continue
                if not path.exists():
                    mismatches.append(f"{item.get('label')}: missing {path}")
                    continue
                actual_sha = sha256_file(path)
                actual_bytes = path.stat().st_size
                if item.get("sha256") != actual_sha:
                    mismatches.append(f"{item.get('label')}: recorded sha {item.get('sha256')} != actual {actual_sha}")
                if item.get("bytes") != actual_bytes:
                    mismatches.append(f"{item.get('label')}: recorded bytes {item.get('bytes')} != actual {actual_bytes}")
                sha_line_value = sha_lines.get(str(path))
                if sha_line_value != actual_sha:
                    mismatches.append(f"{item.get('label')}: SHA256SUMS value {sha_line_value} != actual {actual_sha}")

            if shutil.which("sha256sum") is None:
                mismatches.append("sha256sum command is unavailable")
            else:
                result = subprocess.run(["sha256sum", "-c", str(SHA256SUMS)], text=True, capture_output=True)
                sha256sum_check = {
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                }
                if result.returncode != 0:
                    mismatches.append("sha256sum -c failed")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse artifact checksum manifest: {exc}")
    return {
        "missing": missing,
        "empty": empty,
        "mismatches": mismatches,
        "sha256sum_check": sha256sum_check,
        "passed": not missing and not empty and not mismatches,
    }


def check_mainline_evidence_chain() -> dict[str, Any]:
    required = [MAINLINE_EVIDENCE_JSON, MAINLINE_EVIDENCE_MD]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    if MAINLINE_EVIDENCE_JSON.exists():
        try:
            chain = load_json(MAINLINE_EVIDENCE_JSON)
            if not chain.get("overall_pass"):
                mismatches.append("mainline evidence chain overall_pass is false")
            expected_stages = {
                "data_and_corpora",
                "rollout_schema",
                "reward_audit",
                "main_grpo_training",
                "main_dev_test_result",
            }
            observed_stages = {stage.get("stage") for stage in chain.get("stages", [])}
            if observed_stages != expected_stages:
                mismatches.append(f"mainline evidence stages {sorted(observed_stages)} != expected {sorted(expected_stages)}")
            for stage in chain.get("stages", []):
                if not stage.get("passed"):
                    mismatches.append(f"mainline stage {stage.get('stage')} did not pass")
            reward_runs = chain.get("checks", {}).get("reward_audit", {}).get("runs", {})
            for run_name in ["seed42_main", "seed43_confirm", "goldfixed_control"]:
                run = reward_runs.get(run_name, {})
                if run.get("reward_auc_strict_success", 0.0) < 0.90:
                    mismatches.append(f"{run_name}: strict reward AUC below 0.90")
                if run.get("constant_group_rate", 1.0) > 0.02:
                    mismatches.append(f"{run_name}: constant group rate above 2%")
            results = chain.get("checks", {}).get("main_results", {}).get("comparisons", {})
            for split in ["dev", "test"]:
                item = results.get(split, {})
                if item.get("seed42_strict_gain", 0.0) <= 0:
                    mismatches.append(f"{split}: seed42 strict gain is not positive")
                if item.get("seed42_r5_gain", 0.0) <= 0:
                    mismatches.append(f"{split}: seed42 R@5 gain is not positive")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse mainline evidence chain: {exc}")
    return {"missing": missing, "empty": empty, "mismatches": mismatches, "passed": not missing and not empty and not mismatches}


def check_mainline_schema_contract() -> dict[str, Any]:
    required = [MAINLINE_SCHEMA_CONTRACT_JSON, MAINLINE_SCHEMA_CONTRACT_MD]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    if MAINLINE_SCHEMA_CONTRACT_JSON.exists():
        try:
            audit = load_json(MAINLINE_SCHEMA_CONTRACT_JSON)
            if not audit.get("overall_pass"):
                mismatches.append("mainline schema contract audit overall_pass is false")
            expected_checks = {
                "data_split_contract",
                "corpus_contract",
                "unified_rollout_contract",
                "training_reward_rollout_contract",
                "node_credit_summary_contract",
                "two_stage_prediction_contract",
            }
            observed_checks = set(audit.get("checks", {}))
            if observed_checks != expected_checks:
                mismatches.append(f"schema contract checks {sorted(observed_checks)} != expected {sorted(expected_checks)}")
            for check_name, check in audit.get("checks", {}).items():
                if not check.get("passed"):
                    mismatches.append(f"schema contract check {check_name} did not pass")
            corpus = audit.get("checks", {}).get("corpus_contract", {})
            if corpus.get("train_eval_doc_id_overlap") != 0:
                mismatches.append("schema contract found train/eval doc_id overlap")
            if corpus.get("train_eval_url_overlap") != 0:
                mismatches.append("schema contract found train/eval URL overlap")
            rollout = audit.get("checks", {}).get("unified_rollout_contract", {})
            if rollout.get("forbidden_generation_rows") != 0:
                mismatches.append("schema contract found forbidden generated markers")
            if rollout.get("reward_formula_mismatches") != 0:
                mismatches.append("schema contract found rollout reward formula mismatches")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse mainline schema contract audit: {exc}")
    return {"missing": missing, "empty": empty, "mismatches": mismatches, "passed": not missing and not empty and not mismatches}


def check_text_finalization_audit() -> dict[str, Any]:
    required = [TEXT_FINALIZATION_JSON, TEXT_FINALIZATION_MD]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    if TEXT_FINALIZATION_JSON.exists():
        try:
            audit = load_json(TEXT_FINALIZATION_JSON)
            if not audit.get("overall_pass"):
                mismatches.append("text finalization audit overall_pass is false")
            for name, check in audit.get("checks", {}).items():
                if not check.get("passed"):
                    mismatches.append(f"text finalization check {name} did not pass")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse text finalization audit: {exc}")
    return {"missing": missing, "empty": empty, "mismatches": mismatches, "passed": not missing and not empty and not mismatches}


def check_venue_workspace_audit() -> dict[str, Any]:
    required = [VENUE_WORKSPACE_AUDIT_JSON, VENUE_WORKSPACE_AUDIT_MD]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    if VENUE_WORKSPACE_AUDIT_JSON.exists():
        try:
            audit = load_json(VENUE_WORKSPACE_AUDIT_JSON)
            if not audit.get("overall_pass"):
                mismatches.append("venue workspace audit overall_pass is false")
            for name, check in audit.get("checks", {}).items():
                if not check.get("passed"):
                    mismatches.append(f"venue workspace check {name} did not pass")
            manifest_check = audit.get("checks", {}).get("manifest", {})
            if manifest_check.get("compile_check_returncode") != 0:
                mismatches.append("venue workspace generic compile check did not pass")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse venue workspace audit: {exc}")
    return {"missing": missing, "empty": empty, "mismatches": mismatches, "passed": not missing and not empty and not mismatches}


def check_submission_payload_index() -> dict[str, Any]:
    required = [SUBMISSION_PAYLOAD_JSON, SUBMISSION_PAYLOAD_MD]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    expected_routes = {"anonymous_review_generic", "full_handoff_or_overleaf", "target_venue_conversion", "generic_preprint"}
    if SUBMISSION_PAYLOAD_JSON.exists():
        try:
            index = load_json(SUBMISSION_PAYLOAD_JSON)
            if not index.get("overall_pass"):
                mismatches.append("submission payload index overall_pass is false")
            observed_routes = set(index.get("routes", {}))
            if observed_routes != expected_routes:
                mismatches.append(f"payload routes {sorted(observed_routes)} != expected {sorted(expected_routes)}")
            anon = index.get("routes", {}).get("anonymous_review_generic", {})
            anon_uploads = [item.get("label") for item in anon.get("upload_or_share_artifacts", [])]
            if "full_source_zip" in anon_uploads or "full_source_tarball" in anon_uploads:
                mismatches.append("anonymous_review_generic uploads full source artifacts")
            review_audit = index.get("zip_audits", {}).get("review_clean_source_zip", {})
            if review_audit.get("has_docs") or review_audit.get("has_scripts") or review_audit.get("disallowed_review_entries"):
                mismatches.append("review-clean zip boundary failed in payload index")
            for label, artifact in index.get("artifacts", {}).items():
                path = Path(artifact.get("path", ""))
                if not path.exists():
                    mismatches.append(f"payload artifact {label} missing: {path}")
                elif artifact.get("sha256") != sha256_file(path):
                    mismatches.append(f"payload artifact {label} sha mismatch")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse submission payload index: {exc}")
    return {"missing": missing, "empty": empty, "mismatches": mismatches, "passed": not missing and not empty and not mismatches}


def check_review_clean_bundle() -> dict[str, Any]:
    required = [
        REVIEW_CLEAN_ZIP,
        REVIEW_CLEAN_REPORT_JSON,
        REVIEW_CLEAN_REPORT_MD,
        REVIEW_CLEAN_ANON_JSON,
        REVIEW_CLEAN_ANON_MD,
        REVIEW_CLEAN_COMPILE_JSON,
        REVIEW_CLEAN_COMPILE_MD,
    ]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    if REVIEW_CLEAN_REPORT_JSON.exists():
        try:
            report = load_json(REVIEW_CLEAN_REPORT_JSON)
            if not report.get("validation", {}).get("passed"):
                mismatches.append("clean bundle validation did not pass")
            if not report.get("zip_verification", {}).get("passed"):
                mismatches.append("clean bundle zip verification did not pass")
            if REVIEW_CLEAN_ZIP.exists() and report.get("clean_zip_sha256") != sha256_file(REVIEW_CLEAN_ZIP):
                mismatches.append("clean bundle zip sha mismatch")
            anon = load_json(REVIEW_CLEAN_ANON_JSON) if REVIEW_CLEAN_ANON_JSON.exists() else {}
            if not anon.get("overall_pass"):
                mismatches.append("clean bundle anonymity audit did not pass")
            compile_report = load_json(REVIEW_CLEAN_COMPILE_JSON) if REVIEW_CLEAN_COMPILE_JSON.exists() else {}
            if not compile_report.get("passed"):
                mismatches.append("review-clean clean-extract compile verification did not pass")
            if compile_report.get("make_all", {}).get("returncode") != 0:
                mismatches.append("review-clean make all did not pass")
            if compile_report.get("post_compile_audit", {}).get("returncode") != 0:
                mismatches.append("review-clean post-compile audit did not pass")
            if compile_report.get("layout_audit", {}).get("returncode") != 0:
                mismatches.append("review-clean layout audit did not pass")
            if REVIEW_CLEAN_ZIP.exists():
                with zipfile.ZipFile(REVIEW_CLEAN_ZIP) as zf:
                    names = zf.namelist()
                    if any(name.endswith(".json") for name in names):
                        mismatches.append("clean bundle zip contains JSON metadata")
                    if any(name.startswith("docs/") or name.startswith("scripts/") for name in names):
                        mismatches.append("clean bundle zip contains docs/ or scripts/")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse clean bundle report: {exc}")
    return {"missing": missing, "empty": empty, "mismatches": mismatches, "passed": not missing and not empty and not mismatches}


def check_package_extract_verification() -> dict[str, Any]:
    required = [PACKAGE_EXTRACT_VERIFY_JSON, PACKAGE_EXTRACT_VERIFY_MD, PACKAGE_INDEX_JSON]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    if PACKAGE_EXTRACT_VERIFY_JSON.exists() and PACKAGE_INDEX_JSON.exists():
        try:
            verify = load_json(PACKAGE_EXTRACT_VERIFY_JSON)
            index = load_json(PACKAGE_INDEX_JSON)
            if not verify.get("overall_pass"):
                mismatches.append("package extraction verification overall_pass is false")
            for label in ["zipfile", "tarball"]:
                if verify.get(label, {}).get("expected_sha256") != index.get(label, {}).get("sha256"):
                    mismatches.append(f"{label}: verification sha does not match package index")
                if not verify.get(label, {}).get("passed"):
                    mismatches.append(f"{label}: extraction verification did not pass")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse package extraction verification: {exc}")
    return {"missing": missing, "empty": empty, "mismatches": mismatches, "passed": not missing and not empty and not mismatches}


def check_source_bundle_compile_verification() -> dict[str, Any]:
    required = [SOURCE_BUNDLE_COMPILE_JSON, SOURCE_BUNDLE_COMPILE_MD, PACKAGE_INDEX_JSON]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.stat().st_size == 0]
    mismatches: list[str] = []
    if SOURCE_BUNDLE_COMPILE_JSON.exists():
        try:
            report = load_json(SOURCE_BUNDLE_COMPILE_JSON)
            if not report.get("overall_pass"):
                mismatches.append("source bundle compile verification overall_pass is false")
            for label in ["zipfile", "tarball"]:
                item = report.get(label, {})
                if not item.get("passed"):
                    mismatches.append(f"{label}: clean-extract compile verification did not pass")
                if item.get("make_all", {}).get("returncode") != 0:
                    mismatches.append(f"{label}: make all did not pass")
                if item.get("post_compile_audit", {}).get("returncode") != 0:
                    mismatches.append(f"{label}: post-compile PDF audit did not pass")
                if item.get("layout_audit", {}).get("returncode") != 0:
                    mismatches.append(f"{label}: PDF layout audit did not pass")
        except Exception as exc:  # pragma: no cover - defensive report path
            mismatches.append(f"failed to parse source bundle compile verification: {exc}")
    return {"missing": missing, "empty": empty, "mismatches": mismatches, "passed": not missing and not empty and not mismatches}


def check_tex_path_leaks() -> dict[str, Any]:
    forbidden = ["/root/", "outputs/", "scripts/"]
    roots = [LATEX_DIR, ASSETS / "figures", SUBMISSION_BUNDLE]
    leaks: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for tex in sorted(root.rglob("*.tex")):
            for lineno, line in enumerate(tex.read_text(encoding="utf-8").splitlines(), start=1):
                hits = [pattern for pattern in forbidden if pattern in line]
                if hits:
                    leaks.append(
                        {
                            "file": str(tex),
                            "line": lineno,
                            "patterns": hits,
                            "text": line.strip(),
                        }
                    )
    return {"forbidden_patterns": forbidden, "leaks": leaks, "passed": not leaks}


def check_tex_compile_risks() -> dict[str, Any]:
    roots = [LATEX_DIR, ASSETS / "figures", SUBMISSION_BUNDLE]
    linebreak_join_re = re.compile(r"\\\\[A-Za-z]")
    risks: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for tex in sorted(root.rglob("*.tex")):
            for lineno, line in enumerate(tex.read_text(encoding="utf-8").splitlines(), start=1):
                if linebreak_join_re.search(line):
                    risks.append(
                        {
                            "file": str(tex),
                            "line": lineno,
                            "risk": "latex_linebreak_joined_to_text",
                            "text": line.strip(),
                        }
                    )
    return {"risks": risks, "passed": not risks}


def check_counts() -> dict[str, Any]:
    expected = {
        "grpo_train": ("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl", 458),
        "grpo_dev": ("outputs/dagig_grpo_main/derived_assets/grpo_dev.jsonl", 98),
        "grpo_test": ("outputs/dagig_grpo_main/derived_assets/grpo_test.jsonl", 64),
        "bm25_train_corpus": ("outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl", 610),
        "bm25_eval_corpus": ("outputs/dagig_grpo_main/derived_assets/bm25_eval_corpus.jsonl", 201),
        "bm25_train_corpus_goldfixed": ("outputs/dagig_paper_main_v1/derived_assets/bm25_train_corpus_goldfixed.jsonl", 610),
    }
    mismatches: list[str] = []
    observed: dict[str, int] = {}
    for name, (path_str, expected_count) in expected.items():
        path = Path(path_str)
        if not path.exists():
            mismatches.append(f"{name}: missing {path_str}")
            continue
        count = line_count(path)
        observed[name] = count
        if count != expected_count:
            mismatches.append(f"{name}: observed {count} expected {expected_count}")
    return {"observed": observed, "mismatches": mismatches, "passed": not mismatches}


def check_manifest_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = flatten_manifest_paths(manifest)
    missing = sorted(path for path in paths if not Path(path).exists())
    empty = sorted(path for path in paths if Path(path).is_file() and Path(path).stat().st_size == 0)
    return {"n_paths": len(paths), "missing": missing, "empty": empty, "passed": not missing and not empty}


def build_report(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Paper Asset Audit Report\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"- overall pass: `{audit['overall_pass']}`\n")
    lines.append(f"- manifest paths checked: `{audit['manifest_paths']['n_paths']}`\n")
    lines.append(f"- pdflatex available: `{audit['environment']['pdflatex_available']}`\n")
    lines.append("\n")

    lines.append("## Checks\n\n")
    lines.append("| check | passed | notes |\n")
    lines.append("|---|---:|---|\n")
    for key in [
        "manifest_paths",
        "main_table",
        "node_table",
        "text_claims",
        "cross_bundle_consistency",
        "mainline_evidence_chain",
        "mainline_schema_contract",
        "text_finalization_audit",
        "venue_workspace_audit",
        "submission_payload_index",
        "claim_boundaries",
        "citations",
        "latex_structure",
        "shell_commands",
        "latex_makefile",
        "submission_bundle",
        "package_index",
        "artifact_checksums",
        "review_clean_bundle",
        "package_extract_verification",
        "source_bundle_compile_verification",
        "tex_path_leaks",
        "tex_compile_risks",
        "counts",
    ]:
        item = audit[key]
        notes: list[str] = []
        for field in ["missing", "empty", "mismatches", "unused"]:
            if item.get(field):
                notes.append(f"{field}={item[field]}")
        if item.get("stderr"):
            notes.append(f"stderr={item['stderr']}")
        if item.get("stdout") and item.get("stdout") != "latex source check passed":
            notes.append(f"stdout={item['stdout']}")
        if item.get("tex_files_with_parent_refs"):
            notes.append(f"parent_refs={item['tex_files_with_parent_refs']}")
        if item.get("leaks"):
            notes.append(f"leaks={item['leaks']}")
        if item.get("risks"):
            notes.append(f"risks={item['risks']}")
        if item.get("problems"):
            notes.append(f"problems={item['problems']}")
        if item.get("forbidden_hits"):
            notes.append(f"forbidden_hits={item['forbidden_hits']}")
        if item.get("manifest_validation_passed") is False:
            notes.append("bundle_manifest_validation=False")
        if item.get("manifest_error"):
            notes.append(f"manifest_error={item['manifest_error']}")
        if item.get("make_check", {}).get("returncode") not in (None, 0):
            notes.append(f"make_check={item['make_check']}")
        if item.get("bundled_command_check", {}).get("returncode") not in (None, 0):
            notes.append(f"bundled_command_check={item['bundled_command_check']}")
        if item.get("sha256sum_check", {}).get("returncode") not in (None, 0):
            notes.append(f"sha256sum_check={item['sha256sum_check']}")
        if not notes:
            notes.append("ok")
        lines.append(f"| {key} | `{item['passed']}` | {'; '.join(notes)} |\n")
    lines.append("\n")

    lines.append("## Dataset And Corpus Counts\n\n")
    lines.append("| item | observed |\n")
    lines.append("|---|---:|\n")
    for name, count in audit["counts"]["observed"].items():
        lines.append(f"| {name} | {count} |\n")
    lines.append("\n")

    lines.append("## Citation Check\n\n")
    lines.append(f"- citation keys in `main.tex`: `{audit['citations']['n_cites']}`\n")
    lines.append(f"- BibTeX entries: `{audit['citations']['n_entries']}`\n")
    lines.append(f"- missing citation keys: `{audit['citations']['missing']}`\n")
    lines.append(f"- unused BibTeX entries: `{audit['citations']['unused']}`\n")
    lines.append("\n")

    lines.append("## Interpretation\n\n")
    if audit["overall_pass"]:
        lines.append(
            "The current paper-facing assets are internally consistent: tracked paths exist, "
            "tables match the consolidated metric JSON, citations resolve, command syntax is valid, "
            "the LaTeX source and self-contained submission bundle validate, and dataset/corpus "
            "counts match the reproducibility appendix.\n"
        )
    else:
        lines.append(
            "At least one paper-facing consistency check failed. Fix the failing fields before "
            "treating the paper package as ready for template conversion.\n"
        )
    if not audit["environment"]["pdflatex_available"]:
        lines.append("\nNote: `pdflatex` is not available in this environment, so PDF compilation was not audited.\n")
    return "".join(lines)


def main() -> None:
    manifest = load_json(MANIFEST)
    consolidated = load_json(CONSOLIDATED)
    node_summary = load_json(NODE_SUMMARY)
    audit: dict[str, Any] = {
        "manifest_paths": check_manifest_paths(manifest),
        "main_table": check_main_table(consolidated),
        "node_table": check_node_table(node_summary),
        "text_claims": check_text_claims(consolidated),
        "cross_bundle_consistency": check_cross_bundle_consistency(),
        "mainline_evidence_chain": check_mainline_evidence_chain(),
        "mainline_schema_contract": check_mainline_schema_contract(),
        "text_finalization_audit": check_text_finalization_audit(),
        "venue_workspace_audit": check_venue_workspace_audit(),
        "submission_payload_index": check_submission_payload_index(),
        "claim_boundaries": check_claim_boundaries(consolidated),
        "citations": check_citations(),
        "latex_structure": check_latex_structure(),
        "shell_commands": check_shell(),
        "latex_makefile": check_latex_makefile(),
        "submission_bundle": check_submission_bundle(),
        "package_index": check_package_index(),
        "artifact_checksums": check_artifact_checksums(),
        "review_clean_bundle": check_review_clean_bundle(),
        "package_extract_verification": check_package_extract_verification(),
        "source_bundle_compile_verification": check_source_bundle_compile_verification(),
        "tex_path_leaks": check_tex_path_leaks(),
        "tex_compile_risks": check_tex_compile_risks(),
        "counts": check_counts(),
        "environment": {"pdflatex_available": shutil.which("pdflatex") is not None},
    }
    audit["overall_pass"] = all(
        audit[key]["passed"]
        for key in [
            "manifest_paths",
            "main_table",
            "node_table",
            "text_claims",
            "cross_bundle_consistency",
            "mainline_evidence_chain",
            "mainline_schema_contract",
            "text_finalization_audit",
            "venue_workspace_audit",
            "submission_payload_index",
            "claim_boundaries",
            "citations",
            "latex_structure",
            "shell_commands",
            "latex_makefile",
            "submission_bundle",
            "package_index",
            "artifact_checksums",
            "review_clean_bundle",
            "package_extract_verification",
            "source_bundle_compile_verification",
            "tex_path_leaks",
            "tex_compile_risks",
            "counts",
        ]
    )
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    AUDIT_MD.write_text(build_report(audit), encoding="utf-8")
    print(f"wrote {AUDIT_MD}")
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
