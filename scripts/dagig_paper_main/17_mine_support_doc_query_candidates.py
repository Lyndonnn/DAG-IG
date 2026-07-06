#!/usr/bin/env python3
"""Mine stronger train-only queries from support-document lexical fields.

This is coverage analysis and candidate construction for train samples only.
It may use train support-document title/url/domain/text, but it does not use
dev/test labels, oracle trajectories, or final-answer tokens in generated query
candidates.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import (  # noqa: E402
    BM25Index,
    answer_leaks_in_query,
    extract_numeric_tokens,
    load_corpus,
    read_jsonl,
    support_rank,
    tokenize,
    write_json,
    write_jsonl,
)


ROOT = Path("outputs/dagig_paper_main_v1")
OUT = ROOT / "reports/support_doc_query_candidate_mining"
GROUPS = ROOT / "reports/hard_retrieval_mining/train_hard_retrieval_groups.jsonl"
PREV_BEST = ROOT / "reports/nohit_query_candidate_mining/nohit_best_queries.jsonl"
TRAIN_FILE = Path("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl")
CORPUS = Path("outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl")


STOP = {
    "about", "after", "also", "answer", "based", "before", "being", "could",
    "current", "currently", "does", "from", "have", "image", "into", "know",
    "like", "looking", "many", "name", "need", "only", "photo", "picture",
    "please", "question", "show", "shown", "tell", "that", "this", "want",
    "what", "when", "where", "which", "with", "would", "your", "https",
    "http", "www", "com", "html", "php", "wiki", "url", "official",
}


def clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def semantic_anchor(row: dict[str, Any]) -> str:
    grounding = row.get("grounding")
    if isinstance(grounding, dict) and grounding.get("semantic_anchor"):
        return clean(grounding.get("semantic_anchor"))
    return clean(row.get("semantic_anchor"))


def visible_name(row: dict[str, Any]) -> str:
    grounding = row.get("grounding")
    if isinstance(grounding, dict):
        return clean(grounding.get("visible_text_or_name"))
    return ""


def answer_intent(question: str) -> str:
    q = question.lower()
    parts: list[str] = []
    if any(t in q for t in ("phone", "contact number", "call", "telephone", "hotline")):
        parts.append("contact phone number")
    if any(t in q for t in ("email", "e-mail")):
        parts.append("email contact")
    if any(t in q for t in ("address", "street", "located", "location", "closest", "boutique")):
        parts.append("address location")
    if any(t in q for t in ("opening", "hours", "closing", "open ", "close ", "time")):
        parts.append("opening hours")
    if any(t in q for t in ("price", "cost", "how much", "fee")):
        parts.append("price")
    if any(t in q for t in ("revenue", "sales", "market cap")):
        parts.append("revenue")
    if any(t in q for t in ("album", "song", "charity", "donated")):
        parts.append("album charity")
    if any(t in q for t in ("population", "gdp", "percentage", "percent", "how many", "number of", "count")):
        parts.append("number")
    years = sorted(set(re.findall(r"\b20\d{2}\b", q)))
    if years:
        parts.append(" ".join(years))
    return " ".join(dict.fromkeys(" ".join(parts).split()))


def url_tokens(url: str) -> list[str]:
    parsed = urlparse(url)
    text = " ".join([parsed.netloc.replace(".", " "), parsed.path.replace("/", " ").replace("-", " ").replace("_", " ")])
    return tokenize(text)


def answer_token_set(answer: str) -> set[str]:
    toks = set(tokenize(answer))
    toks.update(extract_numeric_tokens(answer))
    return {tok.lower() for tok in toks if tok}


def useful_tokens(text: str, answer_tokens: set[str], idf: dict[str, float], max_terms: int = 8) -> list[str]:
    counts = Counter(tok for tok in tokenize(text) if len(tok) >= 3 and tok not in STOP and tok not in answer_tokens)
    scored = sorted(
        counts.items(),
        key=lambda kv: (idf.get(kv[0], 0.0), kv[1], len(kv[0])),
        reverse=True,
    )
    out: list[str] = []
    for tok, _ in scored:
        if tok not in out:
            out.append(tok)
        if len(out) >= max_terms:
            break
    return out


def uniq(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    out = []
    for source, query in candidates:
        query = clean(query)
        key = " ".join(tokenize(query))
        if not query or key in seen:
            continue
        seen.add(key)
        out.append((source, query))
    return out


def candidates_for(row: dict[str, Any], gold_docs: list[dict[str, Any]], idf: dict[str, float]) -> list[tuple[str, str]]:
    anchor = semantic_anchor(row)
    visible = visible_name(row)
    hf = clean(row.get("hf_search_query"))
    intent = answer_intent(clean(row.get("question")))
    answer_tokens = answer_token_set(clean(row.get("gold_answer")))
    raw: list[tuple[str, str]] = []
    bases = [("semantic_anchor", anchor), ("visible_text", visible), ("hf_search_query", hf)]

    for doc in gold_docs:
        title = clean(doc.get("title"))
        domain = clean(doc.get("domain"))
        url = clean(doc.get("url"))
        text = clean(doc.get("text"))
        doc_terms = useful_tokens(" ".join([title, domain, url, text]), answer_tokens, idf, max_terms=10)
        title_terms = useful_tokens(title, answer_tokens, idf, max_terms=5)
        url_terms = [tok for tok in url_tokens(url) if len(tok) >= 3 and tok not in STOP and tok not in answer_tokens][:6]
        domain_terms = [tok for tok in tokenize(domain) if len(tok) >= 3 and tok not in STOP and tok not in answer_tokens][:4]
        doc_phrase = " ".join(doc_terms[:6])
        title_phrase = " ".join(title_terms[:4])
        url_phrase = " ".join(url_terms[:5])
        domain_phrase = " ".join(domain_terms[:3])
        for base_name, base in bases:
            if not base:
                continue
            if doc_phrase:
                raw.append((f"support_doc_terms+{base_name}", f"{base} {doc_phrase}"))
                if intent:
                    raw.append((f"support_doc_terms+{base_name}+intent", f"{base} {doc_phrase} {intent}"))
            if title_phrase:
                raw.append((f"support_title+{base_name}", f"{base} {title_phrase}"))
            if url_phrase:
                raw.append((f"support_url+{base_name}", f"{base} {url_phrase}"))
            if domain_phrase:
                raw.append((f"support_domain+{base_name}", f"{base} {domain_phrase} {intent}".strip()))
        if doc_phrase and intent:
            raw.append(("support_doc_terms+intent", f"{doc_phrase} {intent}"))
        if title_phrase and intent:
            raw.append(("support_title+intent", f"{title_phrase} {intent}"))
        if url_phrase and intent:
            raw.append(("support_url+intent", f"{url_phrase} {intent}"))
    return uniq(raw)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_rows = {str(row.get("sample_id")): row for row in read_jsonl(TRAIN_FILE)}
    corpus = load_corpus(CORPUS)
    bm25 = BM25Index.from_docs(corpus)
    docs_by_sample: dict[str, list[dict[str, Any]]] = {}
    for doc in corpus:
        if doc.get("is_gold"):
            docs_by_sample.setdefault(str(doc.get("sample_id")), []).append(doc)

    nohit_ids = [
        str(row.get("sample_id"))
        for row in read_jsonl(GROUPS)
        if row.get("status") == "candidate_insufficient_no_hit_rollout"
    ]
    previously_recovered = {str(row.get("sample_id")) for row in read_jsonl(PREV_BEST)}
    target_ids = [sid for sid in nohit_ids if sid not in previously_recovered]

    candidate_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    source_counts = Counter()
    hit_source_counts = Counter()
    leak_filtered = 0
    missing_gold_docs = []

    for sid in target_ids:
        row = train_rows.get(sid)
        gold_docs = docs_by_sample.get(sid, [])
        if not row or not gold_docs:
            missing_gold_docs.append(sid)
            continue
        best = None
        for source, query in candidates_for(row, gold_docs, bm25.idf):
            if answer_leaks_in_query(query, clean(row.get("gold_answer"))):
                leak_filtered += 1
                continue
            docs = bm25.search(query, top_k=5)
            rank = support_rank(docs, sid, 5)
            source_counts[source] += 1
            if rank is not None:
                hit_source_counts[source] += 1
            candidate = {
                "sample_id": sid,
                "source": source,
                "query": query,
                "support_rank5": rank,
                "hit5": rank is not None,
                "uses_train_support_doc_fields": True,
                "top_docs": [
                    {
                        "rank": d.get("rank"),
                        "score": d.get("score"),
                        "sample_id": d.get("sample_id"),
                        "is_gold": d.get("is_gold"),
                        "title": d.get("title"),
                        "url": d.get("url"),
                    }
                    for d in docs
                ],
            }
            candidate_rows.append(candidate)
            if rank is not None and (best is None or rank < best["support_rank5"]):
                best = candidate
        if best:
            best_rows.append(
                {
                    "sample_id": sid,
                    "question": row.get("question"),
                    "semantic_anchor": semantic_anchor(row),
                    "hf_search_query": row.get("hf_search_query"),
                    "chosen_query": best["query"],
                    "chosen_source": best["source"],
                    "support_rank5": best["support_rank5"],
                }
            )

    cumulative = len(previously_recovered) + len(best_rows)
    rank_counts = Counter(str(row["support_rank5"]) for row in best_rows)
    summary = {
        "scope": "train-only support-document lexical query mining",
        "nohit_samples": len(nohit_ids),
        "previous_recipe_recovered": len(previously_recovered),
        "remaining_target_samples": len(target_ids),
        "missing_gold_docs": missing_gold_docs,
        "candidate_rows": len(candidate_rows),
        "leak_filtered": leak_filtered,
        "newly_recovered_samples": len(best_rows),
        "new_recovery_rate_on_remaining": len(best_rows) / max(1, len(target_ids)),
        "cumulative_recovered_samples": cumulative,
        "cumulative_recovery_rate": cumulative / max(1, len(nohit_ids)),
        "support_rank_counts": dict(rank_counts),
        "candidate_source_counts": dict(source_counts),
        "hit_source_counts": dict(hit_source_counts),
        "outputs": {
            "candidates": str(OUT / "support_doc_query_candidates.jsonl"),
            "best": str(OUT / "support_doc_best_queries.jsonl"),
        },
    }
    write_jsonl(OUT / "support_doc_query_candidates.jsonl", candidate_rows)
    write_jsonl(OUT / "support_doc_best_queries.jsonl", best_rows)
    write_json(OUT / "support_doc_query_mining_summary.json", summary)

    lines: list[str] = []
    lines.append("# Support-Document Query Candidate Mining Report\n\n")
    lines.append("## Scope\n\n")
    lines.append(
        "This is train-only query mining for the no-hit samples that simple clean query recipes did not recover. "
        "It uses train support-document title/url/domain/text fields to build lexical query candidates, filters answer leakage, and evaluates only against the train BM25 corpus. "
        "It does not use dev/test labels or oracle trajectories.\n\n"
    )
    lines.append("## Results\n\n")
    lines.append(f"- original no-hit train samples: `{len(nohit_ids)}`\n")
    lines.append(f"- recovered by previous clean recipes: `{len(previously_recovered)}`\n")
    lines.append(f"- remaining target samples: `{len(target_ids)}`\n")
    lines.append(f"- generated candidate rows: `{len(candidate_rows)}`\n")
    lines.append(f"- answer-leak candidates filtered: `{leak_filtered}`\n")
    lines.append(f"- newly recovered samples: `{len(best_rows)}` (`{100.0 * summary['new_recovery_rate_on_remaining']:.1f}%` of remaining)\n")
    lines.append(f"- cumulative recovered no-hit samples: `{cumulative}` / `{len(nohit_ids)}` (`{100.0 * summary['cumulative_recovery_rate']:.1f}%`)\n")
    lines.append(f"- new support-rank counts: `{dict(rank_counts)}`\n\n")
    lines.append("## Candidate Source Counts\n\n")
    lines.append("| source | candidates | hit candidates |\n|---|---:|---:|\n")
    for source, count in sorted(source_counts.items(), key=lambda kv: (-hit_source_counts[kv[0]], kv[0])):
        lines.append(f"| {source} | {count} | {hit_source_counts[source]} |\n")
    lines.append("\n## Decision\n\n")
    if best_rows:
        lines.append(
            "A large part of the remaining no-hit gap is recoverable when train support-document lexical fields are allowed. "
            "This suggests the core bottleneck is candidate query generation rather than an unrecoverable BM25/corpus mismatch. "
            "Use this as supervised train-only evidence to design a cleaner query-candidate generator, but keep it labeled separately because it is support-doc-derived.\n"
        )
    else:
        lines.append(
            "Support-document lexical mining does not recover the remaining no-hit samples. "
            "The next issue is likely corpus/support mismatch or visual entity extraction, not query wording alone.\n"
        )
    lines.append("\n## Artifacts\n\n")
    lines.append(f"- candidates: `{OUT / 'support_doc_query_candidates.jsonl'}`\n")
    lines.append(f"- best queries: `{OUT / 'support_doc_best_queries.jsonl'}`\n")
    lines.append(f"- summary: `{OUT / 'support_doc_query_mining_summary.json'}`\n")
    (OUT / "SUPPORT_DOC_QUERY_CANDIDATE_MINING_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "SUPPORT_DOC_QUERY_CANDIDATE_MINING_REPORT.md")


if __name__ == "__main__":
    main()
