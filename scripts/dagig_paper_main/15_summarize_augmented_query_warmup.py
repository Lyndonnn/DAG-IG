#!/usr/bin/env python3
"""Summarize augmented query-node warmup for paper-main v1."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs/dagig_paper_main_v1")
METRICS = ROOT / "two_stage_metrics"
METRICS_V3 = ROOT / "two_stage_metrics_rescored_v3"
REPORT = ROOT / "reports/AUGMENTED_QUERY_WARMUP_REPORT.md"
SUMMARY = ROOT / "reports/augmented_query_warmup_summary.json"


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(x) -> str:
    return "-" if x is None else f"{100.0 * float(x):.1f}%"


def row(name: str, path: Path) -> dict:
    m = load(path)
    b = m.get("breakdown", {})
    return {
        "method": name,
        "path": str(path),
        "n": m.get("n"),
        "r1": m.get("retrieval_top1_hit"),
        "r3": m.get("retrieval_top3_hit"),
        "r5": m.get("retrieval_top5_hit"),
        "answer": m.get("answer_correct"),
        "strict": m.get("strict_success"),
        "format": m.get("format_parse_success"),
        "retrieval_miss": b.get("retrieval_miss"),
        "hit_answer_wrong": b.get("retrieval_hit_answer_wrong"),
    }


def main() -> None:
    data_report = load(ROOT / "query_node_sft_aug/query_node_sft_aug_summary.json")
    nohit_report = load(ROOT / "reports/nohit_query_candidate_mining/nohit_query_recovery_summary.json")
    rows = [
        row("Format-SFT full dev", METRICS_V3 / "format_sft_two_stage_own_full_dev.json"),
        row(
            "Query-node SFT smoke + fixed reader",
            METRICS / "query_node_sft_format_init_smoke20_full__reader_format_sft_dev.json",
        ),
        row(
            "Augmented query-node SFT smoke + fixed reader",
            METRICS / "query_node_sft_aug_format_init_smoke20_full__reader_format_sft_dev.json",
        ),
        row(
            "DAG-IG GRPO seed42 ckpt60 dev",
            METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.json",
        ),
        row(
            "DAG-IG GRPO seed43 ckpt60 dev",
            METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.json",
        ),
        row(
            "Query-warm GRPO ckpt30 + fixed Format reader",
            METRICS / "paper_main_v1_querywarm_stage1loss_kl01_30_s320_ckpt30_formatreader__reader_format_sft_dev.json",
        ),
    ]
    aug = rows[2]
    base_warm = rows[1]
    current = rows[3]
    summary = {
        "data": data_report,
        "nohit_mining": nohit_report,
        "metrics": rows,
        "decision": {
            "augmented_warmup_is_useful": True,
            "promote_as_main": False,
            "run_one_augmented_init_grpo": True,
            "reason": (
                "Augmented warmup recovers strict success versus the earlier query-node warmup, "
                "but still does not beat the current seed42/seed43 GRPO checkpoints."
            ),
        },
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Augmented Query Warmup Report\n")
    lines.append("## Scope\n")
    lines.append(
        "This tests whether train-only no-hit recovery queries improve stage-1 query-node warmup. "
        "It does not use dev/test labels for training and does not train a reader.\n"
    )
    lines.append("\n## Data\n")
    lines.append(f"- base hit-vs-miss rows: `{data_report['base_rows']}`\n")
    lines.append(f"- no-hit recovery rows: `{data_report['nohit_recovery_rows']}`\n")
    lines.append(f"- augmented rows: `{data_report['output_rows']}`\n")
    lines.append(f"- no-hit train samples recovered by mining: `{nohit_report['recovered_samples']}` / `{nohit_report['nohit_samples']}` (`{100.0 * nohit_report['recovery_rate']:.1f}%`)\n")
    lines.append(f"- recovered support-rank counts: `{nohit_report['support_rank_counts']}`\n")
    lines.append("\n## Evaluation\n")
    lines.append("| Method | n | R@1 | R@3 | R@5 | answer | strict | format | retrieval miss | hit-answer-wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['n']} | {pct(r['r1'])} | {pct(r['r3'])} | {pct(r['r5'])} | "
            f"{pct(r['answer'])} | {pct(r['strict'])} | {pct(r['format'])} | {r['retrieval_miss']} | {r['hit_answer_wrong']} |\n"
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        f"- Compared with the earlier query-node warmup, augmented warmup improves strict from `{pct(base_warm['strict'])}` to `{pct(aug['strict'])}` and keeps hit-answer-wrong at `{aug['hit_answer_wrong']}`.\n"
    )
    lines.append(
        f"- It still trails the current seed42 main checkpoint on dev strict (`{pct(aug['strict'])}` vs `{pct(current['strict'])}`) and R@5 (`{pct(aug['r5'])}` vs `{pct(current['r5'])}`).\n"
    )
    lines.append(
        "- The data change is useful but insufficient as a standalone adapter. Its value is as a cleaner initialization for exactly one controlled GRPO run, not as another branch to tune indefinitely.\n"
    )
    lines.append("\n## Decision\n")
    lines.append(
        "Run one short paper-main v1 GRPO initialized from `query_node_sft_aug_format_init_smoke20`, using the same two-stage stage1-loss settings as the current main recipe. "
        "Evaluate dev only first. Promote or run test only if dev strict exceeds the current `49.0%` or if retrieval improves without increasing hit-answer-wrong.\n"
    )
    lines.append("\n## Artifacts\n")
    lines.append("- no-hit mining: `outputs/dagig_paper_main_v1/reports/nohit_query_candidate_mining/NOHIT_QUERY_CANDIDATE_MINING_REPORT.md`\n")
    lines.append("- augmented data: `outputs/dagig_paper_main_v1/query_node_sft_aug/query_node_sft_aug_train.jsonl`\n")
    lines.append("- augmented adapter: `outputs/dagig_paper_main_v1/checkpoints/query_node_sft_aug_format_init_smoke20`\n")
    lines.append(f"- machine summary: `{SUMMARY}`\n")
    REPORT.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
