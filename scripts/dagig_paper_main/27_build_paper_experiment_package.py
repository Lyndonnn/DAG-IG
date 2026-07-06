#!/usr/bin/env python3
"""Build compact paper-facing experiment assets from current main reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
REPORTS = ROOT / "reports"
ASSETS = ROOT / "paper_assets"

CONSOLIDATED = REPORTS / "paper_main_v1_consolidated_results.json"
NODE_CREDIT = REPORTS / "node_credit_component_analysis/node_credit_component_summary.json"
TRAIN_CONFIG = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/grpo_run_config.json"
TRAIN_SUMMARY = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/grpo_train_summary.json"
SEED43_SUMMARY = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/grpo_train_summary.json"
GOLDFIXED_SUMMARY = ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/grpo_train_summary.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.1f}"


def tex_pct(value: float | None) -> str:
    return pct(value) + "\\%"


def esc(text: str) -> str:
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main_results_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    spec = [
        ("Format-SFT", "dev", "format_dev"),
        ("Format-SFT", "test", "format_test"),
        ("DAG-IG seed42 main", "dev", "seed42_dev"),
        ("DAG-IG seed42 main", "test", "seed42_test"),
        ("DAG-IG seed43 confirm", "dev", "seed43_dev"),
        ("DAG-IG seed43 confirm", "test", "seed43_test"),
        ("Goldfixed control", "dev", "goldfixed_dev"),
        ("Goldfixed control", "test", "goldfixed_test"),
    ]
    rows = []
    for name, split, key in spec:
        m = metrics[key]
        rows.append(
            {
                "method": name,
                "split": split,
                "r1": pct(m["r1"]),
                "r3": pct(m["r3"]),
                "r5": pct(m["r5"]),
                "answer_correct": pct(m["answer"]),
                "strict_success": pct(m["strict"]),
                "format_success": pct(m["format"]),
                "retrieval_miss": m["retrieval_miss"],
                "hit_answer_wrong": m["hit_answer_wrong"],
            }
        )
    return rows


def write_main_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{llrrrrr}\n",
        "\\toprule\n",
        "Method & Split & R@1 & R@3 & R@5 & Ans. & Strict \\\\\n",
        "\\midrule\n",
    ]
    for row in rows:
        lines.append(
            f"{esc(row['method'])} & {row['split']} & {row['r1']} & {row['r3']} & {row['r5']} & "
            f"{row['answer_correct']} & {row['strict_success']} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    path.write_text("".join(lines), encoding="utf-8")


def node_credit_rows(node_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run_name in ["seed42_main", "seed43_confirm", "goldfixed_control"]:
        run = node_summary[run_name]
        rows.append(
            {
                "run": run_name,
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
        )
    return rows


def write_node_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}\n",
        "\\toprule\n",
        "Run & AUC(hit) & AUC(strict) & Top hit & Bottom hit & Top strict & Bottom strict \\\\\n",
        "\\midrule\n",
    ]
    for row in rows:
        lines.append(
            f"{esc(row['run'])} & {row['reward_auc_hit']} & {row['reward_auc_strict']} & "
            f"{row['top_hit']} & {row['bottom_hit']} & {row['top_strict']} & {row['bottom_strict']} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    consolidated = load_json(CONSOLIDATED)
    node_summary = load_json(NODE_CREDIT)
    train_config = load_json(TRAIN_CONFIG)
    train_summary = load_json(TRAIN_SUMMARY)
    seed43_summary = load_json(SEED43_SUMMARY)
    goldfixed_summary = load_json(GOLDFIXED_SUMMARY)

    main_rows = main_results_rows(consolidated["metrics"])
    node_rows = node_credit_rows(node_summary)
    write_csv(
        ASSETS / "main_results_table.csv",
        main_rows,
        ["method", "split", "r1", "r3", "r5", "answer_correct", "strict_success", "format_success", "retrieval_miss", "hit_answer_wrong"],
    )
    write_csv(
        ASSETS / "node_credit_diagnostic_table.csv",
        node_rows,
        [
            "run",
            "reward_auc_hit",
            "reward_auc_strict",
            "constant_groups",
            "top_hit",
            "bottom_hit",
            "top_strict",
            "bottom_strict",
            "query_auc_hit",
            "evidence_auc_hit",
            "answer_auc_strict",
        ],
    )
    write_main_latex(ASSETS / "main_results_table.tex", main_rows)
    write_node_latex(ASSETS / "node_credit_diagnostic_table.tex", node_rows)

    manifest = {
        "current_main_checkpoint": "outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60",
        "confirmation_checkpoint": "outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/checkpoint-60",
        "goldfixed_control_checkpoint": "outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/checkpoint-60",
        "main_result_table": str(ASSETS / "main_results_table.csv"),
        "node_credit_table": str(ASSETS / "node_credit_diagnostic_table.csv"),
        "training_config": str(TRAIN_CONFIG),
        "training_summary": str(TRAIN_SUMMARY),
        "seed43_training_summary": str(SEED43_SUMMARY),
        "goldfixed_training_summary": str(GOLDFIXED_SUMMARY),
        "primary_reports": [
            str(REPORTS / "PAPER_MAIN_V1_CONSOLIDATED_RESULTS.md"),
            str(REPORTS / "node_credit_component_analysis/NODE_CREDIT_COMPONENT_ANALYSIS.md"),
            str(REPORTS / "GOLDFIXED_GRPO_60_REPORT.md"),
            str(REPORTS / "SEED_CONFIRMATION_REPORT.md"),
        ],
    }
    (ASSETS / "paper_experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# DAG-IG Paper Experiment Package\n\n")
    lines.append("## Current Claim\n\n")
    lines.append(
        "The current paper-usable claim is: in the Pix2Fact two-stage offline retrieval setting, "
        "DAG-IG node-level GRPO improves over the Format-SFT baseline and is confirmed by a second seed. "
        "The best single checkpoint remains seed42 `scale60_s320` checkpoint-60.\n\n"
    )
    lines.append("## Main Numbers\n\n")
    lines.append("| Method | Split | R@5 | Answer | Strict | Retrieval miss | Hit-answer-wrong |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for row in main_rows:
        lines.append(
            f"| {row['method']} | {row['split']} | {row['r5']}% | {row['answer_correct']}% | "
            f"{row['strict_success']}% | {row['retrieval_miss']} | {row['hit_answer_wrong']} |\n"
        )
    lines.append("\n")
    lines.append("## Node-Credit Diagnostic\n\n")
    lines.append("| Run | AUC(hit) | AUC(strict) | Top strict | Bottom strict | Constant groups |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for row in node_rows:
        lines.append(
            f"| {row['run']} | {row['reward_auc_hit']} | {row['reward_auc_strict']} | "
            f"{row['top_strict']}% | {row['bottom_strict']}% | {row['constant_groups']} |\n"
        )
    lines.append("\n")
    lines.append("## Training Recipe\n\n")
    lines.append(f"- base model: `{train_config['model_name_or_path']}`\n")
    lines.append(f"- initializer: `{train_config['init_adapter_path']}`\n")
    lines.append(f"- reward variant: `{train_config['variant']}`\n")
    lines.append(f"- two-stage rollout: `{train_config['two_stage_rollout']}`\n")
    lines.append(f"- loss scope: `{train_config['two_stage_loss_scope']}`\n")
    lines.append(f"- num generations: `{train_config['num_generations']}`\n")
    lines.append(f"- KL coefficient: `{train_config['kl_coef']}`\n")
    lines.append(f"- learning rate: `{train_config['learning_rate']}`\n")
    lines.append(f"- top-k retrieval: `{train_config['top_k']}`\n")
    lines.append(f"- main run constant reward groups: `{train_summary['constant_reward_groups']} / {train_summary['micro_steps']}`\n")
    lines.append(f"- seed43 constant reward groups: `{seed43_summary['constant_reward_groups']} / {seed43_summary['micro_steps']}`\n")
    lines.append(f"- goldfixed control constant reward groups: `{goldfixed_summary['constant_reward_groups']} / {goldfixed_summary['micro_steps']}`\n\n")
    lines.append("## Claim Boundaries\n\n")
    lines.append("- Claim DAG-IG improves the Format-SFT two-stage agent under the offline BM25 Pix2Fact setup.\n")
    lines.append("- Claim node-level reward components are discriminative and non-collapsed.\n")
    lines.append("- Do not claim fixed-corpus control is the best model; it is a robustness/control run.\n")
    lines.append("- Do not claim answer extraction is solved; retrieval misses and hit-answer-wrong remain the main bottlenecks.\n")
    lines.append("- Do not claim web-search generalization; all reported numbers use the frozen offline BM25 corpus.\n\n")
    lines.append("## Generated Assets\n\n")
    lines.append(f"- `main_results_table.csv`: `{ASSETS / 'main_results_table.csv'}`\n")
    lines.append(f"- `main_results_table.tex`: `{ASSETS / 'main_results_table.tex'}`\n")
    lines.append(f"- `node_credit_diagnostic_table.csv`: `{ASSETS / 'node_credit_diagnostic_table.csv'}`\n")
    lines.append(f"- `node_credit_diagnostic_table.tex`: `{ASSETS / 'node_credit_diagnostic_table.tex'}`\n")
    lines.append(f"- manifest: `{ASSETS / 'paper_experiment_manifest.json'}`\n")
    (ASSETS / "PAPER_EXPERIMENT_PACKAGE.md").write_text("".join(lines), encoding="utf-8")
    print(f"wrote {ASSETS / 'PAPER_EXPERIMENT_PACKAGE.md'}")


if __name__ == "__main__":
    main()
