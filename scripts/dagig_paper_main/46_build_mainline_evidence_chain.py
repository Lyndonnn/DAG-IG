#!/usr/bin/env python3
"""Build an auditable evidence chain for the DAG-IG paper mainline."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
REPORTS = ROOT / "reports"
DERIVED = Path("outputs/dagig_grpo_main/derived_assets")

SCHEMA = ROOT / "protocol/PAPER_MAIN_V1_SCHEMA.md"
DERIVED_MANIFEST = DERIVED / "derived_manifest.json"
UNIFIED_ROLLOUTS = ROOT / "rollouts/train_rollouts_unified_scored.jsonl"
CONSOLIDATED_RESULTS = REPORTS / "paper_main_v1_consolidated_results.json"
NODE_CREDIT_SUMMARY = REPORTS / "node_credit_component_analysis/node_credit_component_summary.json"
MAIN_CKPT = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320"
MAIN_CKPT60 = MAIN_CKPT / "checkpoint-60"
TRAIN_SUMMARY = MAIN_CKPT / "grpo_train_summary.json"
TRAIN_CONFIG = MAIN_CKPT / "grpo_run_config.json"
OUT_JSON = ASSETS / "MAINLINE_EVIDENCE_CHAIN.json"
OUT_MD = ASSETS / "MAINLINE_EVIDENCE_CHAIN.md"

EXPECTED_DATA_COUNTS = {
    "derived_grpo_train": 458,
    "derived_grpo_dev": 98,
    "derived_grpo_test": 64,
    "derived_bm25_train_docs": 610,
    "derived_bm25_eval_docs": 201,
}

REQUIRED_ROW_KEYS = {
    "sample_id",
    "split",
    "source_run",
    "question",
    "gold_answer",
    "image_path",
    "rollout",
    "retrieval",
    "metrics",
    "node_credits",
}
REQUIRED_ROLLOUT_KEYS = {"visual_observation", "search_query", "final_answer", "raw", "parsed_json"}
REQUIRED_RETRIEVAL_KEYS = {"top_k", "support_rank5", "support_rank10", "mrr10", "hit5", "top_docs"}
REQUIRED_METRIC_KEYS = {
    "format_valid",
    "query_nonempty",
    "evidence_supported",
    "answer_correct",
    "strict_success",
    "answer_in_query",
}
REQUIRED_CREDIT_KEYS = {
    "format_credit",
    "visual_credit",
    "query_credit",
    "evidence_credit",
    "answer_credit",
    "leak_penalty",
    "path_penalty",
    "total_reward",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def check_paths() -> dict[str, Any]:
    required = [
        SCHEMA,
        DERIVED_MANIFEST,
        UNIFIED_ROLLOUTS,
        CONSOLIDATED_RESULTS,
        NODE_CREDIT_SUMMARY,
        TRAIN_SUMMARY,
        TRAIN_CONFIG,
        MAIN_CKPT60 / "adapter_model.safetensors",
        MAIN_CKPT60 / "adapter_config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    empty = [str(path) for path in required if path.exists() and path.is_file() and path.stat().st_size == 0]
    return {"missing": missing, "empty": empty, "passed": not missing and not empty}


def check_data_manifest() -> dict[str, Any]:
    manifest = load_json(DERIVED_MANIFEST)
    counts = manifest.get("derived_counts", {})
    mismatches = []
    for key, expected in EXPECTED_DATA_COUNTS.items():
        actual = counts.get(key)
        if actual != expected:
            mismatches.append(f"{key}: observed {actual} expected {expected}")
    if manifest.get("hard_fail"):
        mismatches.append("derived manifest hard_fail is true")
    return {
        "manifest": str(DERIVED_MANIFEST),
        "derived_counts": counts,
        "expected_counts": EXPECTED_DATA_COUNTS,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def check_rollout_schema() -> dict[str, Any]:
    n = 0
    invalid_rows: list[dict[str, Any]] = []
    source_runs: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    parsed_json_count = 0
    top_k_counts: Counter[str] = Counter()
    for line in UNIFIED_ROLLOUTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        row = json.loads(line)
        source_runs[str(row.get("source_run"))] += 1
        split_counts[str(row.get("split"))] += 1
        problems: list[str] = []
        missing_top = sorted(REQUIRED_ROW_KEYS - set(row))
        if missing_top:
            problems.append(f"missing row keys {missing_top}")
        rollout = row.get("rollout", {}) if isinstance(row.get("rollout"), dict) else {}
        retrieval = row.get("retrieval", {}) if isinstance(row.get("retrieval"), dict) else {}
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        credits = row.get("node_credits", {}) if isinstance(row.get("node_credits"), dict) else {}
        for label, obj, required in [
            ("rollout", rollout, REQUIRED_ROLLOUT_KEYS),
            ("retrieval", retrieval, REQUIRED_RETRIEVAL_KEYS),
            ("metrics", metrics, REQUIRED_METRIC_KEYS),
            ("node_credits", credits, REQUIRED_CREDIT_KEYS),
        ]:
            missing = sorted(required - set(obj))
            if missing:
                problems.append(f"missing {label} keys {missing}")
        if rollout.get("parsed_json") is True:
            parsed_json_count += 1
        if retrieval.get("top_k") is not None:
            top_k_counts[str(retrieval.get("top_k"))] += 1
        if problems and len(invalid_rows) < 20:
            invalid_rows.append({"line": n, "sample_id": row.get("sample_id"), "problems": problems})
    expected_min_rows = 458 * 4
    passed = n >= expected_min_rows and not invalid_rows and "5" in top_k_counts
    return {
        "path": str(UNIFIED_ROLLOUTS),
        "rows": n,
        "expected_min_rows": expected_min_rows,
        "source_runs": dict(source_runs),
        "split_counts": dict(split_counts),
        "parsed_json_rate": parsed_json_count / n if n else 0.0,
        "top_k_counts": dict(top_k_counts),
        "invalid_row_examples": invalid_rows,
        "passed": passed,
    }


def check_reward_audit() -> dict[str, Any]:
    summary = load_json(NODE_CREDIT_SUMMARY)
    runs: dict[str, Any] = {}
    failures: list[str] = []
    for run_name in ["seed42_main", "seed43_confirm", "goldfixed_control"]:
        run = summary.get(run_name, {})
        constant_group_rate = run.get("groups", {}).get("constant_group_rate")
        hit_auc = run.get("reward_auc_retrieval_hit")
        strict_auc = run.get("reward_auc_strict_success")
        groups = run.get("groups", {}).get("groups")
        if groups != 240:
            failures.append(f"{run_name}: groups={groups} expected 240")
        if hit_auc is None or hit_auc < 0.95:
            failures.append(f"{run_name}: retrieval-hit reward AUC {hit_auc} < 0.95")
        if strict_auc is None or strict_auc < 0.90:
            failures.append(f"{run_name}: strict-success reward AUC {strict_auc} < 0.90")
        if constant_group_rate is None or constant_group_rate > 0.02:
            failures.append(f"{run_name}: constant_group_rate {constant_group_rate} > 0.02")
        components = run.get("components", {})
        required_components = {"visual", "query", "evidence", "answer", "format", "leakage_penalty", "path_penalty"}
        missing_components = sorted(required_components - set(components))
        if missing_components:
            failures.append(f"{run_name}: missing components {missing_components}")
        runs[run_name] = {
            "reward_auc_retrieval_hit": hit_auc,
            "reward_auc_strict_success": strict_auc,
            "constant_group_rate": constant_group_rate,
            "top_strict_success": run.get("groups", {}).get("top_strict_success"),
            "bottom_strict_success": run.get("groups", {}).get("bottom_strict_success"),
            "query_auc_retrieval_hit": components.get("query", {}).get("auc_retrieval_hit"),
            "evidence_auc_retrieval_hit": components.get("evidence", {}).get("auc_retrieval_hit"),
            "answer_auc_strict_success": components.get("answer", {}).get("auc_strict_success"),
        }
    return {"path": str(NODE_CREDIT_SUMMARY), "runs": runs, "failures": failures, "passed": not failures}


def check_training_checkpoint() -> dict[str, Any]:
    summary = load_json(TRAIN_SUMMARY)
    config = load_json(TRAIN_CONFIG)
    failures = []
    if summary.get("status") != "success":
        failures.append(f"training status is {summary.get('status')}")
    if summary.get("optimizer_steps") != 60:
        failures.append(f"optimizer_steps={summary.get('optimizer_steps')} expected 60")
    if summary.get("micro_steps") != 240:
        failures.append(f"micro_steps={summary.get('micro_steps')} expected 240")
    if summary.get("constant_reward_groups", 10**9) > 2:
        failures.append(f"constant_reward_groups={summary.get('constant_reward_groups')} expected <=2")
    if not config.get("two_stage_rollout"):
        failures.append("two_stage_rollout is not true")
    if config.get("two_stage_loss_scope") != "stage1":
        failures.append(f"two_stage_loss_scope={config.get('two_stage_loss_scope')} expected stage1")
    if config.get("kl_coef") != 0.1:
        failures.append(f"kl_coef={config.get('kl_coef')} expected 0.1")
    if not (MAIN_CKPT60 / "adapter_model.safetensors").exists():
        failures.append("checkpoint-60 adapter_model.safetensors missing")
    return {
        "summary_path": str(TRAIN_SUMMARY),
        "config_path": str(TRAIN_CONFIG),
        "checkpoint": str(MAIN_CKPT60),
        "status": summary.get("status"),
        "optimizer_steps": summary.get("optimizer_steps"),
        "micro_steps": summary.get("micro_steps"),
        "constant_reward_groups": summary.get("constant_reward_groups"),
        "two_stage_rollout": config.get("two_stage_rollout"),
        "two_stage_loss_scope": config.get("two_stage_loss_scope"),
        "kl_coef": config.get("kl_coef"),
        "failures": failures,
        "passed": not failures,
    }


def check_main_results() -> dict[str, Any]:
    consolidated = load_json(CONSOLIDATED_RESULTS)
    metrics = consolidated.get("metrics", {})
    failures: list[str] = []
    comparisons: dict[str, Any] = {}
    for split in ["dev", "test"]:
        fmt = metrics[f"format_{split}"]
        seed42 = metrics[f"seed42_{split}"]
        seed43 = metrics[f"seed43_{split}"]
        strict_gain = seed42["strict"] - fmt["strict"]
        r5_gain = seed42["r5"] - fmt["r5"]
        seed43_strict_gain = seed43["strict"] - fmt["strict"]
        comparisons[split] = {
            "format_strict": fmt["strict"],
            "seed42_strict": seed42["strict"],
            "seed43_strict": seed43["strict"],
            "seed42_strict_gain": strict_gain,
            "seed43_strict_gain": seed43_strict_gain,
            "format_r5": fmt["r5"],
            "seed42_r5": seed42["r5"],
            "seed42_r5_gain": r5_gain,
            "n": seed42["n"],
        }
        if strict_gain <= 0:
            failures.append(f"{split}: seed42 strict gain {strict_gain:.4f} <= 0")
        if r5_gain <= 0:
            failures.append(f"{split}: seed42 R@5 gain {r5_gain:.4f} <= 0")
        if seed43_strict_gain <= 0:
            failures.append(f"{split}: seed43 strict gain {seed43_strict_gain:.4f} <= 0")
    return {"path": str(CONSOLIDATED_RESULTS), "comparisons": comparisons, "failures": failures, "passed": not failures}


def build_chain() -> dict[str, Any]:
    checks = {
        "paths": check_paths(),
        "data_manifest": check_data_manifest(),
        "rollout_schema": check_rollout_schema(),
        "reward_audit": check_reward_audit(),
        "training_checkpoint": check_training_checkpoint(),
        "main_results": check_main_results(),
    }
    stages = [
        {
            "stage": "data_and_corpora",
            "claim": "GRPO train/dev/test data and frozen BM25 train/eval corpora are present and counted.",
            "evidence": [str(DERIVED_MANIFEST), str(DERIVED)],
            "passed": checks["data_manifest"]["passed"],
        },
        {
            "stage": "rollout_schema",
            "claim": "Unified rollouts expose visual, query, evidence, answer, metrics, and node-credit fields.",
            "evidence": [str(SCHEMA), str(UNIFIED_ROLLOUTS)],
            "passed": checks["rollout_schema"]["passed"],
        },
        {
            "stage": "reward_audit",
            "claim": "DAG-IG node-level reward is discriminative and non-collapsed before main GRPO training.",
            "evidence": [str(NODE_CREDIT_SUMMARY), str(ASSETS / "node_credit_diagnostic_table.tex")],
            "passed": checks["reward_audit"]["passed"],
        },
        {
            "stage": "main_grpo_training",
            "claim": "The selected two-stage DAG-IG GRPO checkpoint trained successfully under the paper-main config.",
            "evidence": [str(TRAIN_CONFIG), str(TRAIN_SUMMARY), str(MAIN_CKPT60)],
            "passed": checks["training_checkpoint"]["passed"],
        },
        {
            "stage": "main_dev_test_result",
            "claim": "The selected DAG-IG checkpoint improves over Format-SFT on both dev and test strict success and R@5.",
            "evidence": [str(CONSOLIDATED_RESULTS), str(ASSETS / "main_results_table.tex")],
            "passed": checks["main_results"]["passed"],
        },
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Single-entry audit trail for the DAG-IG Pix2Fact paper mainline.",
        "overall_pass": all(item["passed"] for item in checks.values()) and all(stage["passed"] for stage in stages),
        "checks": checks,
        "stages": stages,
        "do_not_reopen_as_mainline": [
            "DAG-SFT trace imitation",
            "query reranking or switch selection",
            "no-teacher evidence fusion",
            "broad answer repair",
            "same-recipe GRPO reruns without a new mechanism",
        ],
    }


def build_markdown(chain: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Mainline Evidence Chain\n\n")
    lines.append("This is the single-entry audit trail for the DAG-IG / Pix2Fact paper mainline. It links the frozen data/corpus, unified rollout schema, node-level credit audit, selected GRPO checkpoint, and final dev/test result.\n\n")
    lines.append(f"- created_at_utc: `{chain['created_at_utc']}`\n")
    lines.append(f"- overall pass: `{chain['overall_pass']}`\n\n")

    lines.append("## Stages\n\n")
    lines.append("| stage | passed | claim | evidence |\n")
    lines.append("|---|---:|---|---|\n")
    for stage in chain["stages"]:
        evidence = "<br>".join(f"`{path}`" for path in stage["evidence"])
        lines.append(f"| {stage['stage']} | `{stage['passed']}` | {stage['claim']} | {evidence} |\n")
    lines.append("\n")

    data = chain["checks"]["data_manifest"]
    lines.append("## Data And Corpus Counts\n\n")
    lines.append("| item | observed | expected |\n")
    lines.append("|---|---:|---:|\n")
    for key, expected in data["expected_counts"].items():
        lines.append(f"| {key} | {data['derived_counts'].get(key)} | {expected} |\n")
    lines.append("\n")

    rollout = chain["checks"]["rollout_schema"]
    lines.append("## Unified Rollout Schema\n\n")
    lines.append(f"- rows: `{rollout['rows']}`\n")
    lines.append(f"- parsed_json_rate: `{pct(rollout['parsed_json_rate'])}`\n")
    lines.append(f"- source_runs: `{rollout['source_runs']}`\n")
    lines.append(f"- top_k_counts: `{rollout['top_k_counts']}`\n")
    lines.append(f"- invalid row examples: `{rollout['invalid_row_examples']}`\n\n")

    reward = chain["checks"]["reward_audit"]
    lines.append("## Reward Audit\n\n")
    lines.append("| run | hit AUC | strict AUC | constant groups | top strict | bottom strict | query AUC hit | evidence AUC hit | answer AUC strict |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, item in reward["runs"].items():
        lines.append(
            f"| {name} | {item['reward_auc_retrieval_hit']:.3f} | {item['reward_auc_strict_success']:.3f} | "
            f"{pct(item['constant_group_rate'])} | {pct(item['top_strict_success'])} | {pct(item['bottom_strict_success'])} | "
            f"{item['query_auc_retrieval_hit']:.3f} | {item['evidence_auc_retrieval_hit']:.3f} | {item['answer_auc_strict_success']:.3f} |\n"
        )
    lines.append("\n")

    train = chain["checks"]["training_checkpoint"]
    lines.append("## Selected Checkpoint\n\n")
    lines.append(f"- checkpoint: `{train['checkpoint']}`\n")
    lines.append(f"- status: `{train['status']}`\n")
    lines.append(f"- optimizer_steps: `{train['optimizer_steps']}`\n")
    lines.append(f"- micro_steps: `{train['micro_steps']}`\n")
    lines.append(f"- constant_reward_groups: `{train['constant_reward_groups']}`\n")
    lines.append(f"- two_stage_loss_scope: `{train['two_stage_loss_scope']}`\n")
    lines.append(f"- kl_coef: `{train['kl_coef']}`\n\n")

    results = chain["checks"]["main_results"]
    lines.append("## Main Result\n\n")
    lines.append("| split | Format-SFT strict | DAG-IG seed42 strict | seed42 strict gain | Format-SFT R@5 | DAG-IG seed42 R@5 | seed42 R@5 gain | seed43 strict gain |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for split, item in results["comparisons"].items():
        lines.append(
            f"| {split} | {pct(item['format_strict'])} | {pct(item['seed42_strict'])} | "
            f"{100.0 * item['seed42_strict_gain']:.1f} pts | {pct(item['format_r5'])} | "
            f"{pct(item['seed42_r5'])} | {100.0 * item['seed42_r5_gain']:.1f} pts | "
            f"{100.0 * item['seed43_strict_gain']:.1f} pts |\n"
        )
    lines.append("\n")

    failures: list[str] = []
    for key, item in chain["checks"].items():
        for field in ["missing", "empty", "mismatches", "failures"]:
            if item.get(field):
                failures.append(f"{key}.{field}: {item[field]}")
    if failures:
        lines.append("## Failures\n\n")
        for failure in failures:
            lines.append(f"- {failure}\n")
        lines.append("\n")

    lines.append("## Boundary\n\n")
    lines.append("This chain supports the current paper mainline only: DAG-IG node-level GRPO for a two-stage multimodal search agent. It does not promote DAG-SFT trace imitation, query reranking, no-teacher fusion, or broad answer repair to the main method.\n\n")
    return "".join(lines)


def main() -> None:
    chain = build_chain()
    OUT_JSON.write_text(json.dumps(chain, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_markdown(chain), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if not chain["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
