#!/usr/bin/env python3
"""Audit the frozen BM25 corpus used by the paper-main experiments.

The July 2026 review flagged that the corpus is not live web pages; it is a
small frozen pool of Pix2Fact evidence notes. This script makes that explicit
with machine-readable stats.
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
DERIVED = Path("outputs/dagig_grpo_main/derived_assets")
REPORT_DIR = ROOT / "reports"
OUT_JSON = REPORT_DIR / "corpus_reality_audit.json"
OUT_MD = REPORT_DIR / "CORPUS_REALITY_AUDIT.md"

CORPORA = {
    "train_original": DERIVED / "bm25_train_corpus.jsonl",
    "train_goldfixed": ROOT / "derived_assets/bm25_train_corpus_goldfixed.jsonl",
    "eval_devtest": DERIVED / "bm25_eval_corpus.jsonl",
}
SPLITS = {
    "train": DERIVED / "grpo_train.jsonl",
    "dev": DERIVED / "grpo_dev.jsonl",
    "test": DERIVED / "grpo_test.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_text(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(text))


def token_count(text: Any) -> int:
    return len(re.findall(r"\S+", str(text)))


def char_count(text: Any) -> int:
    return len(str(text))


def answer_in_text(answer: str, text: str) -> bool:
    if not answer:
        return False
    answer_norm = norm_text(answer)
    text_norm = norm_text(text)
    if len(answer_norm) >= 4 and answer_norm in text_norm:
        return True
    answer_compact = compact(answer)
    text_compact = compact(text)
    return len(answer_compact) >= 4 and answer_compact in text_compact


def sample_answers(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    answers: dict[str, list[str]] = {}
    for row in rows:
        vals: list[str] = []
        for key in ["gold_answer", "answer"]:
            if row.get(key):
                vals.append(str(row[key]))
        aliases = row.get("answer_aliases") or row.get("aliases") or []
        if isinstance(aliases, list):
            vals.extend(str(x) for x in aliases if x)
        answers[str(row["sample_id"])] = list(dict.fromkeys(vals))
    return answers


def summarize_lengths(docs: list[dict[str, Any]]) -> dict[str, float | int]:
    token_lengths = [token_count(doc.get("text", "")) for doc in docs]
    char_lengths = [char_count(doc.get("text", "")) for doc in docs]
    return {
        "token_mean": statistics.mean(token_lengths) if token_lengths else 0.0,
        "token_median": statistics.median(token_lengths) if token_lengths else 0.0,
        "token_min": min(token_lengths) if token_lengths else 0,
        "token_max": max(token_lengths) if token_lengths else 0,
        "char_mean": statistics.mean(char_lengths) if char_lengths else 0.0,
        "char_median": statistics.median(char_lengths) if char_lengths else 0.0,
        "char_min": min(char_lengths) if char_lengths else 0,
        "char_max": max(char_lengths) if char_lengths else 0,
    }


def summarize_corpus(name: str, docs: list[dict[str, Any]], split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gold_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        sid = str(doc.get("sample_id", ""))
        by_sample[sid].append(doc)
        if doc.get("is_gold"):
            gold_by_sample[sid].append(doc)

    split_counts = Counter(str(doc.get("split", "")) for doc in docs)
    domain_counts = Counter(str(doc.get("domain", "")) for doc in docs)
    source_counts = Counter(str(doc.get("source", "")) for doc in docs)
    gold_docs = [doc for doc in docs if doc.get("is_gold")]

    relevant_rows: list[dict[str, Any]] = []
    for split, rows in split_rows.items():
        if name.startswith("train") and split == "train":
            relevant_rows.extend(rows)
        elif name == "eval_devtest" and split in {"dev", "test"}:
            relevant_rows.extend(rows)

    answers = sample_answers(relevant_rows)
    samples = {str(row["sample_id"]) for row in relevant_rows}
    samples_with_any_doc = samples & set(by_sample)
    samples_with_gold_doc = samples & set(gold_by_sample)
    gold_answer_embedded = 0
    gold_doc_samples_checked = 0
    gold_doc_answer_checks: list[dict[str, Any]] = []
    for sid in sorted(samples_with_gold_doc):
        ans = answers.get(sid, [])
        if not ans:
            continue
        gold_doc_samples_checked += 1
        text = " ".join(str(doc.get("text", "")) for doc in gold_by_sample[sid])
        embedded = any(answer_in_text(a, text) for a in ans)
        gold_answer_embedded += int(embedded)
        gold_doc_answer_checks.append({"sample_id": sid, "embedded": embedded})

    sample_count = len(samples)
    return {
        "name": name,
        "docs": len(docs),
        "split_doc_counts": dict(split_counts),
        "sources": dict(source_counts),
        "unique_samples_in_corpus": len(by_sample),
        "expected_samples": sample_count,
        "samples_with_any_doc": len(samples_with_any_doc),
        "samples_with_gold_doc": len(samples_with_gold_doc),
        "sample_gold_doc_coverage": (len(samples_with_gold_doc) / sample_count) if sample_count else None,
        "gold_docs": len(gold_docs),
        "unique_domains": len([d for d in domain_counts if d]),
        "top_domains": domain_counts.most_common(10),
        "lengths": summarize_lengths(docs),
        "gold_doc_answer_embedded_samples": gold_answer_embedded,
        "gold_doc_answer_checked_samples": gold_doc_samples_checked,
        "gold_doc_answer_embedded_rate": (gold_answer_embedded / gold_doc_samples_checked)
        if gold_doc_samples_checked
        else None,
        "note": "Frozen Pix2Fact evidence-note corpus; not live web pages.",
    }


def write_report(summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Corpus Reality Audit")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This audit describes the frozen BM25 corpora used by the paper-main experiments. These corpora are Pix2Fact-derived evidence notes with URLs/domains, not live web pages and not a live web-search environment."
    )
    lines.append("")
    lines.append("## Corpus Summary")
    lines.append("")
    lines.append("| corpus | docs | expected samples | samples with gold doc | gold-doc coverage | median tokens | mean tokens | gold docs with answer text |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ["train_original", "train_goldfixed", "eval_devtest"]:
        c = summary["corpora"][name]
        lines.append(
            f"| {name} | {c['docs']} | {c['expected_samples']} | {c['samples_with_gold_doc']} | "
            f"{100*c['sample_gold_doc_coverage']:.1f}% | {c['lengths']['token_median']:.1f} | "
            f"{c['lengths']['token_mean']:.1f} | {100*c['gold_doc_answer_embedded_rate']:.1f}% |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The evaluation corpus is a small frozen dev/test pool, not a broad web index.")
    lines.append("- Evidence text is short annotation-style support text. The median whitespace token length is low, so the paper should not describe this as retrieval from noisy full web documents.")
    lines.append("- Gold support notes often contain the answer string directly. Strict success should therefore be interpreted as a controlled offline evidence-acquisition + extraction metric, not a live-web QA score.")
    lines.append("- The goldfixed train corpus fixes train-side gold labels only; dev/test corpora remain frozen.")
    lines.append("")
    lines.append("## Required Paper Wording")
    lines.append("")
    lines.append(
        "Use wording such as: `a frozen Pix2Fact evidence-note BM25 corpus with 201 dev/test documents`, not `live web search` or `noisy web documents`."
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    split_rows = {split: read_jsonl(path) for split, path in SPLITS.items()}
    corpora = {name: summarize_corpus(name, read_jsonl(path), split_rows) for name, path in CORPORA.items()}
    summary = {
        "corpora": corpora,
        "split_sizes": {split: len(rows) for split, rows in split_rows.items()},
        "claim_boundary": "offline frozen evidence-note BM25 corpus, not live web search",
    }
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
