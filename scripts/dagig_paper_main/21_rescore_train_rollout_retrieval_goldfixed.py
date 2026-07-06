#!/usr/bin/env python3
"""Rescore train rollout retrieval coverage with the gold-fixed train corpus."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import BM25Index, load_corpus, read_jsonl, support_rank, write_json, write_jsonl  # noqa: E402


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "reports/goldfixed_rollout_rescore"
ROLLOUTS = ROOT / "rollouts/train_rollouts_unified_scored.jsonl"
FIXED_CORPUS = ROOT / "derived_assets/bm25_train_corpus_goldfixed.jsonl"


def rollout_query(row: dict[str, Any]) -> str:
    rollout = row.get("rollout") or {}
    return str(rollout.get("search_query") or "").strip()


def old_hit(row: dict[str, Any]) -> bool:
    return bool((row.get("retrieval") or {}).get("hit5"))


def answer_correct(row: dict[str, Any]) -> bool:
    return bool((row.get("metrics") or {}).get("answer_correct"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(ROLLOUTS)
    bm25 = BM25Index.from_docs(load_corpus(FIXED_CORPUS))

    rescored = []
    sample_old_hits: dict[str, int] = defaultdict(int)
    sample_new_hits: dict[str, int] = defaultdict(int)
    source_old_hits = Counter()
    source_new_hits = Counter()
    source_counts = Counter()
    old_hit_rollouts = 0
    new_hit_rollouts = 0
    old_strict_rollouts = 0
    new_strict_rollouts = 0
    newly_hit_rollouts = 0

    for row in rows:
        sid = str(row.get("sample_id"))
        source = str(row.get("source_run") or "unknown")
        source_counts[source] += 1
        query = rollout_query(row)
        docs = bm25.search(query, top_k=10) if query else []
        rank5 = support_rank(docs, sid, 5)
        rank10 = support_rank(docs, sid, 10)
        old = old_hit(row)
        new = rank5 is not None
        old_strict = bool((row.get("metrics") or {}).get("strict_success"))
        new_strict = bool(new and answer_correct(row))
        if old:
            old_hit_rollouts += 1
            sample_old_hits[sid] += 1
            source_old_hits[source] += 1
        if new:
            new_hit_rollouts += 1
            sample_new_hits[sid] += 1
            source_new_hits[source] += 1
        if old_strict:
            old_strict_rollouts += 1
        if new_strict:
            new_strict_rollouts += 1
        if new and not old:
            newly_hit_rollouts += 1
        rescored.append(
            {
                "sample_id": sid,
                "source_run": source,
                "micro_step": row.get("micro_step"),
                "generation_index": row.get("generation_index"),
                "query": query,
                "old_hit5": old,
                "new_hit5": new,
                "old_support_rank5": (row.get("retrieval") or {}).get("support_rank5"),
                "new_support_rank5": rank5,
                "new_support_rank10": rank10,
                "old_strict": old_strict,
                "new_strict_approx": new_strict,
                "answer_correct": answer_correct(row),
            }
        )

    sample_ids = {str(row.get("sample_id")) for row in rows}
    old_hit_samples = set(sample_old_hits)
    new_hit_samples = set(sample_new_hits)
    newly_hit_samples = sorted(new_hit_samples - old_hit_samples)
    still_nohit_samples = sorted(sample_ids - new_hit_samples)
    summary = {
        "rollouts": len(rows),
        "samples": len(sample_ids),
        "old_hit_rollouts": old_hit_rollouts,
        "new_hit_rollouts": new_hit_rollouts,
        "newly_hit_rollouts": newly_hit_rollouts,
        "old_hit_samples": len(old_hit_samples),
        "new_hit_samples": len(new_hit_samples),
        "newly_hit_samples": len(newly_hit_samples),
        "old_nohit_samples": len(sample_ids - old_hit_samples),
        "new_nohit_samples": len(still_nohit_samples),
        "old_strict_rollouts": old_strict_rollouts,
        "new_strict_rollouts_approx": new_strict_rollouts,
        "source_rollouts": dict(source_counts),
        "source_old_hit_rollouts": dict(source_old_hits),
        "source_new_hit_rollouts": dict(source_new_hits),
        "newly_hit_sample_ids": newly_hit_samples,
        "still_nohit_sample_ids": still_nohit_samples,
    }
    write_jsonl(OUT / "train_rollouts_goldfixed_retrieval_rescore.jsonl", rescored)
    write_json(OUT / "goldfixed_rollout_rescore_summary.json", summary)

    lines: list[str] = []
    lines.append("# Gold-Fixed Train Rollout Retrieval Rescore\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This recomputes train rollout retrieval hit labels using the fixed train BM25 corpus. "
        "Rollout text and model predictions are unchanged; only corpus `is_gold` labels are fixed.\n\n"
    )
    lines.append("## Summary\n\n")
    lines.append(f"- rollouts: `{len(rows)}`\n")
    lines.append(f"- samples: `{len(sample_ids)}`\n")
    lines.append(f"- old hit rollouts: `{old_hit_rollouts}`\n")
    lines.append(f"- new hit rollouts: `{new_hit_rollouts}`\n")
    lines.append(f"- newly hit rollouts: `{newly_hit_rollouts}`\n")
    lines.append(f"- old hit samples: `{len(old_hit_samples)}`\n")
    lines.append(f"- new hit samples: `{len(new_hit_samples)}`\n")
    lines.append(f"- newly hit samples: `{len(newly_hit_samples)}`\n")
    lines.append(f"- old no-hit samples: `{len(sample_ids - old_hit_samples)}`\n")
    lines.append(f"- new no-hit samples: `{len(still_nohit_samples)}`\n")
    lines.append(f"- old strict rollouts: `{old_strict_rollouts}`\n")
    lines.append(f"- new strict rollouts approx: `{new_strict_rollouts}`\n\n")
    lines.append("## Source Hit Counts\n\n")
    lines.append("| source | rollouts | old hit | new hit |\n|---|---:|---:|---:|\n")
    for source in sorted(source_counts):
        lines.append(f"| {source} | {source_counts[source]} | {source_old_hits[source]} | {source_new_hits[source]} |\n")
    lines.append("\n## Decision\n\n")
    lines.append(
        "Use the fixed corpus for future reward audits and GRPO. If the new no-hit sample count is materially lower, previous no-hit mining overestimated the model-side retrieval problem because some support docs were present but not marked gold.\n\n"
    )
    lines.append("## Artifacts\n\n")
    lines.append(f"- rescored rollouts: `{OUT / 'train_rollouts_goldfixed_retrieval_rescore.jsonl'}`\n")
    lines.append(f"- summary: `{OUT / 'goldfixed_rollout_rescore_summary.json'}`\n")
    (OUT / "GOLDFIXED_TRAIN_ROLLOUT_RETRIEVAL_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "GOLDFIXED_TRAIN_ROLLOUT_RETRIEVAL_REPORT.md")


if __name__ == "__main__":
    main()
