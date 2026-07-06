#!/usr/bin/env python3
"""Build paper-facing case studies from frozen two-stage predictions.

This script is analysis-only: it reads existing predictions/metrics and writes
compact qualitative artifacts for paper drafting. It does not modify pools,
predictions, checkpoints, or metrics.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
PRED = ROOT / "two_stage_predictions"
PRED_V3 = ROOT / "two_stage_predictions_rescored_v3"
REPORTS = ROOT / "reports"
ASSETS = ROOT / "paper_assets"
CASE_DIR = ASSETS / "case_studies"

CONSOLIDATED = REPORTS / "paper_main_v1_consolidated_results.json"
NODE_SUMMARY = REPORTS / "node_credit_component_analysis/node_credit_component_summary.json"
MANIFEST = ASSETS / "paper_experiment_manifest.json"

FORMAT_PRED = {
    "dev": PRED_V3 / "format_sft_two_stage_own_full_dev.jsonl",
    "test": PRED_V3 / "format_sft_two_stage_own_full_test.jsonl",
}
MAIN_PRED = {
    "dev": PRED / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.jsonl",
    "test": PRED / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.jsonl",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows[obj["sample_id"]] = obj
    return rows


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def b(value: Any) -> bool:
    return bool(value)


def doc_summary(row: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
    docs = []
    for doc in row.get("retrieved_docs", [])[:k]:
        docs.append(
            {
                "rank": doc.get("rank"),
                "domain": doc.get("domain") or "",
                "title": doc.get("title") or "",
                "is_gold": bool(doc.get("is_gold")),
            }
        )
    return docs


def category(format_row: dict[str, Any], main_row: dict[str, Any]) -> str:
    f_strict = b(format_row.get("strict_success"))
    m_strict = b(main_row.get("strict_success"))
    f_hit = b(format_row.get("retrieval_top5_hit"))
    m_hit = b(main_row.get("retrieval_top5_hit"))
    m_answer = b(main_row.get("answer_correct"))
    f_answer = b(format_row.get("answer_correct"))

    if m_strict and not f_strict:
        return "dagig_strict_win"
    if f_strict and not m_strict:
        return "format_strict_win"
    if m_hit and not f_hit:
        return "dagig_retrieval_gain_only" if not m_strict else "dagig_retrieval_gain_success"
    if f_hit and not m_hit:
        return "dagig_retrieval_loss"
    if m_hit and f_hit and m_answer and not f_answer:
        return "dagig_answer_gain_same_retrieval"
    if m_hit and f_hit and f_answer and not m_answer:
        return "dagig_answer_loss_same_retrieval"
    if m_hit and f_hit and not m_answer and not f_answer:
        return "both_hit_answer_wrong"
    if not m_hit and not f_hit:
        return "both_retrieval_miss"
    if m_strict and f_strict:
        return "both_strict"
    return "other"


def row_for(split: str, sid: str, format_row: dict[str, Any], main_row: dict[str, Any]) -> dict[str, Any]:
    cat = category(format_row, main_row)
    return {
        "split": split,
        "sample_id": sid,
        "category": cat,
        "question": main_row.get("question") or format_row.get("question") or "",
        "gold_answer": main_row.get("gold_answer") or format_row.get("gold_answer") or "",
        "format_visual_observation": format_row.get("visual_observation") or "",
        "dagig_visual_observation": main_row.get("visual_observation") or "",
        "format_search_query": format_row.get("search_query") or "",
        "dagig_search_query": main_row.get("search_query") or "",
        "format_answer": format_row.get("final_answer") or "",
        "dagig_answer": main_row.get("final_answer") or "",
        "format_r5": b(format_row.get("retrieval_top5_hit")),
        "dagig_r5": b(main_row.get("retrieval_top5_hit")),
        "format_answer_correct": b(format_row.get("answer_correct")),
        "dagig_answer_correct": b(main_row.get("answer_correct")),
        "format_strict": b(format_row.get("strict_success")),
        "dagig_strict": b(main_row.get("strict_success")),
        "format_top_docs": doc_summary(format_row),
        "dagig_top_docs": doc_summary(main_row),
        "format_stage1": format_row.get("stage1_raw_generation") or "",
        "dagig_stage1": main_row.get("stage1_raw_generation") or "",
        "format_reader": format_row.get("reader_raw_generation") or "",
        "dagig_reader": main_row.get("reader_raw_generation") or "",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "sample_id",
        "category",
        "gold_answer",
        "format_search_query",
        "dagig_search_query",
        "format_answer",
        "dagig_answer",
        "format_r5",
        "dagig_r5",
        "format_strict",
        "dagig_strict",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def short(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def docs_md(docs: list[dict[str, Any]]) -> str:
    parts = []
    for doc in docs:
        gold = "gold" if doc.get("is_gold") else "non-gold"
        label = doc.get("domain") or doc.get("title") or "untitled"
        parts.append(f"r{doc.get('rank')} {label} ({gold})")
    return "; ".join(parts) if parts else "-"


def representative(rows: list[dict[str, Any]], category_name: str, limit: int = 3) -> list[dict[str, Any]]:
    return [row for row in rows if row["category"] == category_name][:limit]


def build_case_summary(all_rows: list[dict[str, Any]], consolidated: dict[str, Any], node_summary: dict[str, Any]) -> str:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_split[row["split"]].append(row)

    lines: list[str] = []
    lines.append("# Paper Case Studies\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This is a read-only qualitative analysis of the frozen Format-SFT baseline and the current "
        "DAG-IG seed42 main checkpoint. It uses gold labels only for post-hoc categorization, not for "
        "training, scoring, or prediction changes.\n\n"
    )
    lines.append("## Strict Comparison Counts\n\n")
    lines.append("| split | DAG-IG only strict | Format only strict | both strict | both fail | DAG retrieval gain | DAG retrieval loss |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for split in ["dev", "test"]:
        comp = consolidated["comparisons"][f"seed42_vs_format_{split}"]
        lines.append(
            f"| {split} | {comp['method_only_strict']} | {comp['base_only_strict']} | "
            f"{comp['both_strict']} | {comp['both_fail_strict']} | "
            f"{comp['method_retrieval_gain']} | {comp['method_retrieval_loss']} |\n"
        )
    lines.append("\n")

    lines.append("## Representative Wins And Losses\n\n")
    for split in ["dev", "test"]:
        split_rows = by_split[split]
        lines.append(f"### {split}\n\n")
        for cat, title in [
            ("dagig_strict_win", "DAG-IG strict wins"),
            ("format_strict_win", "Format-SFT strict wins"),
            ("both_hit_answer_wrong", "Both retrieve support but answer wrong"),
            ("both_retrieval_miss", "Both miss retrieval"),
        ]:
            reps = representative(split_rows, cat, 3)
            lines.append(f"#### {title}\n\n")
            if not reps:
                lines.append("- none in this split.\n\n")
                continue
            for row in reps:
                lines.append(f"- `{row['sample_id']}` gold=`{row['gold_answer']}`\n")
                lines.append(f"  - question: {short(row['question'])}\n")
                lines.append(
                    f"  - Format query=`{short(row['format_search_query'], 120)}` answer=`{short(row['format_answer'], 120)}` "
                    f"R@5={row['format_r5']} strict={row['format_strict']}\n"
                )
                lines.append(
                    f"  - DAG-IG query=`{short(row['dagig_search_query'], 120)}` answer=`{short(row['dagig_answer'], 120)}` "
                    f"R@5={row['dagig_r5']} strict={row['dagig_strict']}\n"
                )
                lines.append(f"  - DAG-IG top docs: {docs_md(row['dagig_top_docs'])}\n")
            lines.append("\n")

    seed42 = node_summary["seed42_main"]
    lines.append("## Node-Credit Link\n\n")
    lines.append(
        f"For the main seed42 reward rollouts, reward AUC is `{seed42['reward_auc_retrieval_hit']:.3f}` "
        f"for retrieval hit and `{seed42['reward_auc_strict_success']:.3f}` for strict success. "
        f"Top-ranked samples in each GRPO group have strict success `{pct(seed42['groups']['top_strict_success'])}`, "
        f"while bottom-ranked samples have `{pct(seed42['groups']['bottom_strict_success'])}`. "
        "This supports the paper claim that the node-level DAG-IG reward is discriminative rather than collapsed.\n\n"
    )
    lines.append("## Paper Use\n\n")
    lines.append(
        "Use these cases to illustrate that the main gains are usually query/retrieval improvements, "
        "while the remaining failures are dominated by retrieval misses and retrieved-evidence answer errors. "
        "Do not present these cases as additional test tuning or as evidence of web-search generalization.\n"
    )
    return "".join(lines)


def build_evidence_brief(consolidated: dict[str, Any], node_summary: dict[str, Any]) -> str:
    m = consolidated["metrics"]
    c_dev = consolidated["comparisons"]["seed42_vs_format_dev"]
    c_test = consolidated["comparisons"]["seed42_vs_format_test"]
    seed42 = node_summary["seed42_main"]
    seed43 = node_summary["seed43_confirm"]
    goldfixed = node_summary["goldfixed_control"]

    lines: list[str] = []
    lines.append("# DAG-IG Paper Main Evidence Brief\n\n")
    lines.append("## Paper Position\n\n")
    lines.append(
        "The main method should be positioned as node-level DAG-IG credit for a two-stage multimodal "
        "search agent, optimized with grouped GRPO over the stage-1 policy that emits "
        "`visual_observation` and `search_query`. The fixed reader consumes the image, question, and "
        "top-5 BM25 evidence to produce `final_answer`.\n\n"
    )
    lines.append("DAG-SFT is not the main claim. Preference-style planner tuning and query reranking were useful diagnostics, but the paper-facing result is the DAG-IG GRPO agent.\n\n")

    lines.append("## Main Claim\n\n")
    lines.append(
        "In the frozen Pix2Fact offline BM25 setting, DAG-IG node-level GRPO improves the Format-SFT "
        "two-stage agent on both dev and test, with a second seed confirming the recipe.\n\n"
    )
    lines.append("| method | dev R@5 | dev strict | test R@5 | test strict |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    lines.append(f"| Format-SFT | {pct(m['format_dev']['r5'])} | {pct(m['format_dev']['strict'])} | {pct(m['format_test']['r5'])} | {pct(m['format_test']['strict'])} |\n")
    lines.append(f"| DAG-IG seed42 main | {pct(m['seed42_dev']['r5'])} | {pct(m['seed42_dev']['strict'])} | {pct(m['seed42_test']['r5'])} | {pct(m['seed42_test']['strict'])} |\n")
    lines.append(f"| DAG-IG seed43 confirm | {pct(m['seed43_dev']['r5'])} | {pct(m['seed43_dev']['strict'])} | {pct(m['seed43_test']['r5'])} | {pct(m['seed43_test']['strict'])} |\n")
    lines.append(f"| Goldfixed control | {pct(m['goldfixed_dev']['r5'])} | {pct(m['goldfixed_dev']['strict'])} | {pct(m['goldfixed_test']['r5'])} | {pct(m['goldfixed_test']['strict'])} |\n\n")

    lines.append("## Effect Size\n\n")
    lines.append(
        f"Seed42 improves strict success by `{pct(m['seed42_dev']['strict'] - m['format_dev']['strict'])}` on dev "
        f"and `{pct(m['seed42_test']['strict'] - m['format_test']['strict'])}` on test. "
        f"Against Format-SFT, seed42 has `{c_dev['method_only_strict']}` dev strict wins versus `{c_dev['base_only_strict']}` dev losses, "
        f"and `{c_test['method_only_strict']}` test strict wins versus `{c_test['base_only_strict']}` test losses.\n\n"
    )

    lines.append("## Why The Reward Is Trustworthy\n\n")
    lines.append("| run | reward AUC hit | reward AUC strict | constant groups | top strict | bottom strict |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for name, run in [("seed42", seed42), ("seed43", seed43), ("goldfixed", goldfixed)]:
        lines.append(
            f"| {name} | {run['reward_auc_retrieval_hit']:.3f} | {run['reward_auc_strict_success']:.3f} | "
            f"{run['groups']['constant_groups']}/{run['groups']['groups']} | "
            f"{pct(run['groups']['top_strict_success'])} | {pct(run['groups']['bottom_strict_success'])} |\n"
        )
    lines.append(
        "\nThe query and evidence components have AUC(hit)=1.000 in the main seed42 reward analysis, "
        "and the answer component has AUC(strict)=1.000. The format term is intentionally low-variance "
        "and does not drive the ranking.\n\n"
    )

    lines.append("## Reproducibility Anchors\n\n")
    lines.append("- main checkpoint: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/checkpoint-60`\n")
    lines.append("- seed confirmation: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/checkpoint-60`\n")
    lines.append("- goldfixed control: `outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/checkpoint-60`\n")
    lines.append("- main result table: `outputs/dagig_paper_main_v1/paper_assets/main_results_table.tex`\n")
    lines.append("- node-credit table: `outputs/dagig_paper_main_v1/paper_assets/node_credit_diagnostic_table.tex`\n")
    lines.append("- case studies: `outputs/dagig_paper_main_v1/paper_assets/case_studies/CASE_STUDY_SUMMARY.md`\n\n")

    lines.append("## Claim Boundaries\n\n")
    lines.append("- Do not claim DAG-SFT is the main method; it is a diagnostic/pretraining baseline.\n")
    lines.append("- Do not claim the goldfixed control is the best model; it is a robustness/control run.\n")
    lines.append("- Do not claim real web search generalization; all results use the frozen offline BM25 corpus.\n")
    lines.append("- Do not claim answer extraction is solved. The remaining bottlenecks are retrieval misses and retrieval-hit-answer-wrong cases.\n")
    lines.append("- Do not launch more same-recipe GRPO runs without a new mechanism. The current next step is paper writing plus targeted qualitative/error presentation.\n\n")

    lines.append("## Paper-Completion Next Step\n\n")
    lines.append(
        "Write the paper around the current evidence chain: problem formulation, DAG-IG node credit, "
        "two-stage GRPO training, reward audit, main results, seed confirmation, goldfixed control, "
        "and failure analysis. Additional experiments should be limited to paper-essential presentation gaps, "
        "not new method branches.\n"
    )
    return "".join(lines)


def main() -> None:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    consolidated = load_json(CONSOLIDATED)
    node_summary = load_json(NODE_SUMMARY)

    all_rows: list[dict[str, Any]] = []
    for split in ["dev", "test"]:
        format_rows = load_jsonl(FORMAT_PRED[split])
        main_rows = load_jsonl(MAIN_PRED[split])
        common_ids = sorted(set(format_rows) & set(main_rows))
        split_rows = [row_for(split, sid, format_rows[sid], main_rows[sid]) for sid in common_ids]
        write_jsonl(CASE_DIR / f"{split}_case_table.jsonl", split_rows)
        write_csv(CASE_DIR / f"{split}_case_table.csv", split_rows)
        all_rows.extend(split_rows)

    counts = {
        split: Counter(row["category"] for row in all_rows if row["split"] == split)
        for split in ["dev", "test"]
    }
    (CASE_DIR / "case_category_counts.json").write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    (CASE_DIR / "CASE_STUDY_SUMMARY.md").write_text(
        build_case_summary(all_rows, consolidated, node_summary),
        encoding="utf-8",
    )
    (ASSETS / "PAPER_MAIN_EVIDENCE_BRIEF.md").write_text(
        build_evidence_brief(consolidated, node_summary),
        encoding="utf-8",
    )
    if MANIFEST.exists():
        manifest = load_json(MANIFEST)
    else:
        manifest = {}
    manifest.update(
        {
            "evidence_brief": str(ASSETS / "PAPER_MAIN_EVIDENCE_BRIEF.md"),
            "draft_outline": str(ASSETS / "PAPER_DRAFT_OUTLINE.md"),
            "case_study_summary": str(CASE_DIR / "CASE_STUDY_SUMMARY.md"),
            "case_study_tables": {
                "dev_jsonl": str(CASE_DIR / "dev_case_table.jsonl"),
                "test_jsonl": str(CASE_DIR / "test_case_table.jsonl"),
                "dev_csv": str(CASE_DIR / "dev_case_table.csv"),
                "test_csv": str(CASE_DIR / "test_case_table.csv"),
            },
        }
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {CASE_DIR / 'CASE_STUDY_SUMMARY.md'}")
    print(f"wrote {ASSETS / 'PAPER_MAIN_EVIDENCE_BRIEF.md'}")


if __name__ == "__main__":
    main()
