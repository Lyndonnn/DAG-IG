#!/usr/bin/env python3
"""Rescore existing two-stage predictions with answer checker v4.

Input preference:
1. predictions already repaired/rescored by v3, if available;
2. original two-stage predictions otherwise.

This keeps parser repairs from v3 and changes only answer matching /
strict-success calculation under the current grpo_utils.answer_match_details.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import answer_match_details, tokenize

PREDS = ROOT / "two_stage_predictions"
PREDS_V3 = ROOT / "two_stage_predictions_rescored_v3"
METRICS = ROOT / "two_stage_metrics"
METRICS_V3 = ROOT / "two_stage_metrics_rescored_v3"
PREDS_V4 = ROOT / "two_stage_predictions_rescored_v4"
METRICS_V4 = ROOT / "two_stage_metrics_rescored_v4"
CHANGES = ROOT / "reports/parser_checker_v4_rescore_changes.json"
SUMMARY = ROOT / "reports/parser_checker_v4_rescore_summary.json"
REPORT = ROOT / "reports/CHECKER_V4_RESCORING_REPORT.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rate(rows: list[dict[str, Any]], key: str) -> float:
    denom = max(1, len(rows))
    return sum(1 for row in rows if row.get(key)) / denom


def summarize(rows: list[dict[str, Any]], source_prediction_path: Path, stem: str) -> dict[str, Any]:
    denom = max(1, len(rows))
    summary = {
        "n": len(rows),
        "stage1_format_parse_success": rate(rows, "stage1_format_parse_success"),
        "reader_format_parse_success": rate(rows, "reader_format_parse_success"),
        "format_parse_success": rate(rows, "format_parse_success"),
        "query_nonempty_rate": rate(rows, "query_nonempty"),
        "answer_in_query_rate": rate(rows, "answer_in_query"),
        "retrieval_top1_hit": rate(rows, "retrieval_top1_hit"),
        "retrieval_top3_hit": rate(rows, "retrieval_top3_hit"),
        "retrieval_top5_hit": rate(rows, "retrieval_top5_hit"),
        "answer_correct": rate(rows, "answer_correct"),
        "evidence_supported": rate(rows, "evidence_supported"),
        "strict_success": rate(rows, "strict_success"),
        "avg_query_len": sum(len(tokenize(row.get("search_query", ""))) for row in rows) / denom,
        "avg_reader_input_tokens": sum(int(row.get("reader_input_tokens") or 0) for row in rows) / denom,
        "avg_stage1_input_tokens": sum(int(row.get("stage1_input_tokens") or 0) for row in rows) / denom,
        "invalid_count": sum(1 for row in rows if row.get("stage1_error") or row.get("reader_error")),
        "breakdown": {
            "stage1_format_failure": sum(1 for row in rows if not row.get("stage1_format_parse_success")),
            "reader_format_failure": sum(1 for row in rows if not row.get("reader_format_parse_success")),
            "retrieval_miss": sum(1 for row in rows if not row.get("retrieval_top5_hit")),
            "retrieval_hit_answer_wrong": sum(
                1 for row in rows if row.get("retrieval_top5_hit") and not row.get("answer_correct")
            ),
        },
        "checker_version": "v4",
        "source_prediction_path": str(source_prediction_path),
        "model_tag": rows[0].get("reader_tag") if rows else stem,
        "split": rows[0].get("split") if rows else "",
        "top_k": len(rows[0].get("retrieved_docs") or []) if rows else 0,
    }
    # Preserve stable metadata from an existing metric file when possible.
    for metric_root in (METRICS_V3, METRICS):
        metric_path = metric_root / f"{stem}.json"
        if metric_path.exists():
            old_metric = json.loads(metric_path.read_text(encoding="utf-8"))
            for key in [
                "model_tag",
                "reader_tag",
                "split",
                "eval_file",
                "corpus_path",
                "adapter_path",
                "reader_adapter_path",
                "own_reader",
                "top_k",
                "stage1_prompt",
                "reader_prompt",
                "reader_prompt_version",
                "stage1_source",
                "reader_use_base",
            ]:
                if key in old_metric:
                    summary[key] = old_metric[key]
            break
    return summary


def stem_sources() -> dict[str, Path]:
    stems: dict[str, Path] = {}
    for path in sorted(PREDS.glob("*.jsonl")):
        stems[path.stem] = path
    for path in sorted(PREDS_V3.glob("*.jsonl")):
        stems[path.stem] = path
    return stems


def rescore_file(stem: str, source_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_rows = read_jsonl(source_path)
    output_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    old_answer_correct = 0
    old_strict = 0
    for row in input_rows:
        old_correct = bool(row.get("answer_correct"))
        old_strict_success = bool(row.get("strict_success"))
        old_answer_correct += int(old_correct)
        old_strict += int(old_strict_success)
        answer = str(row.get("final_answer") or "")
        gold = str(row.get("gold_answer") or "")
        aliases = row.get("gold_aliases")
        if not isinstance(aliases, list):
            aliases = None
        match = answer_match_details(answer, gold, aliases=aliases)
        new_correct = bool(match.get("answer_correct"))
        new_strict = bool(new_correct and row.get("evidence_supported"))
        new_row = dict(row)
        new_row["answer_match"] = match
        new_row["answer_correct"] = new_correct
        new_row["strict_success"] = new_strict
        new_row["checker_version"] = "v4"
        output_rows.append(new_row)
        if old_correct != new_correct or old_strict_success != new_strict:
            changes.append(
                {
                    "sample_id": row.get("sample_id"),
                    "split": row.get("split"),
                    "gold_answer": gold,
                    "final_answer": answer,
                    "old_answer_correct": old_correct,
                    "new_answer_correct": new_correct,
                    "old_strict": old_strict_success,
                    "new_strict": new_strict,
                    "evidence_supported": bool(row.get("evidence_supported")),
                    "match_type": match.get("answer_match_type"),
                }
            )
    summary = summarize(output_rows, source_path, stem)
    summary["old_answer_correct_count"] = old_answer_correct
    summary["old_strict_count"] = old_strict
    summary["new_answer_correct_count"] = sum(1 for row in output_rows if row.get("answer_correct"))
    summary["new_strict_count"] = sum(1 for row in output_rows if row.get("strict_success"))
    summary["answer_correct_delta_count"] = summary["new_answer_correct_count"] - old_answer_correct
    summary["strict_delta_count"] = summary["new_strict_count"] - old_strict
    summary["changed_rows"] = len(changes)
    write_jsonl(PREDS_V4 / f"{stem}.jsonl", output_rows)
    write_json(METRICS_V4 / f"{stem}.json", summary)
    return summary, changes


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.1f}%"


def main() -> None:
    summaries: dict[str, dict[str, Any]] = {}
    all_changes: dict[str, list[dict[str, Any]]] = {}
    for stem, source_path in stem_sources().items():
        summary, changes = rescore_file(stem, source_path)
        summaries[stem] = summary
        if changes:
            all_changes[stem] = changes
        print(
            f"{stem}: n={summary['n']} strict={pct(summary['strict_success'])} "
            f"delta={summary['strict_delta_count']} source={source_path.parent.name}"
        )
    write_json(SUMMARY, summaries)
    write_json(CHANGES, all_changes)

    key_stems = [
        "format_sft_two_stage_own_full_dev",
        "format_sft_two_stage_own_full_test",
        "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev",
        "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test",
        "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev",
        "paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test",
        "paper_main_v1_goldfixed_scale60_s320_ckpt60_dev",
        "paper_main_v1_goldfixed_scale60_s320_ckpt60_test",
    ]
    lines: list[str] = []
    lines.append("# Checker v4 Rescoring Report\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "Existing two-stage predictions were rescored with answer checker v4. "
        "When a v3 parser-repaired prediction existed, v4 used that file as input; otherwise it used the original two-stage prediction. "
        "No model generation, retrieval, training data, or evidence pool was changed.\n\n"
    )
    lines.append("## Main Runs\n\n")
    lines.append("| stem | n | R@5 | answer correct | strict | strict delta | changed rows |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for stem in key_stems:
        if stem not in summaries:
            continue
        summary = summaries[stem]
        lines.append(
            f"| `{stem}` | {summary['n']} | {pct(summary.get('retrieval_top5_hit'))} | "
            f"{pct(summary.get('answer_correct'))} | {pct(summary.get('strict_success'))} | "
            f"{summary.get('strict_delta_count')} | {summary.get('changed_rows')} |\n"
        )
    lines.append("\n## Changed Rows\n\n")
    total_changed = sum(len(changes) for changes in all_changes.values())
    lines.append(f"- files rescored: `{len(summaries)}`\n")
    lines.append(f"- files with any checker change: `{len(all_changes)}`\n")
    lines.append(f"- changed rows total: `{total_changed}`\n")
    lines.append(f"- machine-readable changes: `{CHANGES}`\n")
    lines.append(f"- machine-readable summary: `{SUMMARY}`\n")
    REPORT.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
