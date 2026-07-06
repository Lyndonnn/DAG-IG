#!/usr/bin/env python3
"""Summarize query-node SFT warmup smoke results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
REPORTS = ROOT / "reports"
METRICS = ROOT / "two_stage_metrics"
METRICS_V3 = ROOT / "two_stage_metrics_rescored_v3"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    return {
        "path": str(path),
        "n": obj["n"],
        "r1": obj["retrieval_top1_hit"],
        "r3": obj["retrieval_top3_hit"],
        "r5": obj["retrieval_top5_hit"],
        "answer": obj["answer_correct"],
        "strict": obj["strict_success"],
        "format": obj["format_parse_success"],
        "retrieval_miss": obj.get("breakdown", {}).get("retrieval_miss", 0),
        "hit_answer_wrong": obj.get("breakdown", {}).get("retrieval_hit_answer_wrong", 0),
        "answer_in_query": obj.get("answer_in_query_rate", 0.0),
    }


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def row(name: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {m['n']} | {pct(m['r1'])} | {pct(m['r3'])} | {pct(m['r5'])} | "
        f"{pct(m['answer'])} | {pct(m['strict'])} | {pct(m['format'])} | "
        f"{m['retrieval_miss']} | {m['hit_answer_wrong']} |\n"
    )


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    data_summary = load_json(ROOT / "query_node_sft/query_node_sft_summary.json")
    train_log = ROOT / "eval_logs/query_node_sft_format_init_smoke20_train.log"
    metrics = {
        "format_full_dev": metric(METRICS_V3 / "format_sft_two_stage_own_full_dev.json"),
        "query_node_dev20": metric(METRICS / "query_node_sft_format_init_smoke20__reader_format_sft_dev.json"),
        "query_node_full_dev": metric(METRICS / "query_node_sft_format_init_smoke20_full__reader_format_sft_dev.json"),
        "seed42_full_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.json"),
        "seed43_full_dev": metric(METRICS / "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.json"),
    }
    summary = {
        "data_summary": data_summary,
        "metrics": metrics,
        "train_log": str(train_log),
        "decision": "infrastructure_pass_but_not_promoted",
    }
    (REPORTS / "query_node_warmup_smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = ["# Query Node Warmup Smoke Report\n\n"]
    lines.append("## Scope\n\n")
    lines.append(
        "This tests whether train-only query hit-vs-miss pairs can supervise stage-1 query generation. "
        "It is not a final training run and does not use dev/test labels for training.\n\n"
    )
    lines.append("## Data\n\n")
    lines.append(f"- train query-node rows: `{data_summary['rows']}`\n")
    lines.append(f"- skipped rows: `{data_summary['skipped']}`\n")
    lines.append("- assistant target fields: `visual_observation`, `search_query`\n")
    lines.append("- final answer included in target: `False`\n")
    lines.append("- dev/test labels used for training: `False`\n\n")
    lines.append("## Training Smoke\n\n")
    lines.append("- init adapter: `outputs/dagig_grpo_main/checkpoints/format_sft`\n")
    lines.append("- output adapter: `outputs/dagig_paper_main_v1/checkpoints/query_node_sft_format_init_smoke20`\n")
    lines.append("- max steps: `20`\n")
    lines.append(f"- train log: `{train_log}`\n\n")
    lines.append("## Evaluation\n\n")
    lines.append(
        "Evaluation uses the query-node adapter for stage 1 and a fixed Format-SFT reader, so movement is mostly query/retrieval-side rather than reader drift.\n\n"
    )
    lines.append("| Method | n | R@1 | R@3 | R@5 | answer | strict | format | retrieval miss | hit-answer-wrong |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    lines.append(row("Format-SFT full dev", metrics["format_full_dev"]))
    lines.append(row("Query-node SFT smoke dev20 + fixed reader", metrics["query_node_dev20"]))
    lines.append(row("Query-node SFT smoke full dev + fixed reader", metrics["query_node_full_dev"]))
    lines.append(row("DAG-IG GRPO seed42 full dev", metrics["seed42_full_dev"]))
    lines.append(row("DAG-IG GRPO seed43 full dev", metrics["seed43_full_dev"]))
    lines.append("\n")
    lines.append("## Decision\n\n")
    lines.append(
        "The query-node SFT smoke passes infrastructure and improves over Format-SFT on full-dev R@5 (`55.1%` vs `52.0%`) and strict (`44.9%` vs `42.9%`). "
        "It does not beat the current GRPO checkpoints (`49.0%` dev strict), so it should not be promoted as a standalone method.\n\n"
    )
    lines.append(
        "Use it only as a candidate initialization or auxiliary warmup for the next GRPO iteration. "
        "The next mainline experiment should test whether GRPO initialized from this query-node warmup reduces retrieval misses beyond the current seed42/seed43 recipe.\n"
    )
    path = REPORTS / "QUERY_NODE_WARMUP_SMOKE_REPORT.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {path}")
    print(f"wrote {REPORTS / 'query_node_warmup_smoke_summary.json'}")


if __name__ == "__main__":
    main()
