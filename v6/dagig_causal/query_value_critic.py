"""Leakage-safe semantic contract for the executed-query selector teacher."""

from __future__ import annotations

from typing import Any


FORBIDDEN_MODEL_FIELDS = {
    "action_id",
    "answer_correct",
    "child_expected_strict",
    "gold_answer",
    "positive_doc_ids",
    "qrels",
    "query_id",
    "query_strategy",
    "strict_proxy",
    "success_posterior_probability",
}


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def state_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Question: {clean(row['question'])}",
            f"Visual observation: {clean(row['visual_observation'])}",
            f"Requested answer type: {clean(row.get('inferred_answer_type')) or 'other'}",
        ]
    )


def action_text(row: dict[str, Any], max_chars_per_doc: int, max_docs: int) -> str:
    structured = [
        f"Entity quote: {clean(row.get('entity_quote'))}",
        f"Information need: {clean(row.get('information_need'))}",
        "Literal constraints: " + "; ".join(clean(value) for value in row.get("constraints") or []),
        f"Executed search query: {clean(row['search_query'])}",
        "Observed search results:",
    ]
    for doc in (row.get("retrieved_docs") or [])[:max_docs]:
        structured.append(
            "\n".join(
                [
                    f"Title: {clean(doc.get('title'))}",
                    f"Domain: {clean(doc.get('domain'))}",
                    f"Snippet: {clean(doc.get('snippet'))[:max_chars_per_doc]}",
                ]
            )
        )
    return "\n\n".join(structured)


def local_query_score(row: dict[str, Any]) -> float:
    docs = row.get("retrieved_docs") or []
    if not docs:
        return 0.0
    values = [
        0.65 * float(doc.get("normalized_bge_score", 0.0))
        + 0.20 * float(doc.get("question_keyword_overlap", 0.0))
        + 0.15 * float(doc.get("answer_type_pattern_match", 0.0))
        for doc in docs[:5]
    ]
    domains = {clean(doc.get("domain")).casefold() for doc in docs[:5] if clean(doc.get("domain"))}
    return sum(values) / len(values) + 0.10 * len(domains) / max(1, min(5, len(docs)))
