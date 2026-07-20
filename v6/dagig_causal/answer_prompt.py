"""Shared prompt contract for the executable final-answer node."""

from __future__ import annotations

import json
from typing import Any


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _evidence_text(docs: list[dict[str, Any]], snippet_chars: int) -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"[Evidence {index}]",
                f"Title: {_clean(doc.get('title'))}",
                f"URL: {_clean(doc.get('url'))}",
                f"Domain: {_clean(doc.get('domain'))}",
                f"Snippet: {_clean(doc.get('snippet'))[:snippet_chars]}",
            ]
        )
        for index, doc in enumerate(docs, 1)
    )


def build_answer_policy_prompt(row: dict[str, Any], snippet_chars: int = 1200) -> str:
    """Build a direct-answer prompt without gold values or action examples."""

    auxiliary = []
    if row.get("answer_box"):
        auxiliary.append("Search answer box: " + _clean(json.dumps(row["answer_box"], ensure_ascii=False))[:snippet_chars])
    if row.get("knowledge_graph"):
        auxiliary.append("Search knowledge graph: " + _clean(json.dumps(row["knowledge_graph"], ensure_ascii=False))[:snippet_chars])
    return "\n\n".join(
        [
            "You are the final-answer node of a multimodal web-search agent.",
            (
                "Use only the frozen visual observation and selected real-search evidence. "
                "Return one compact valid JSON object with exactly one string field named final_answer."
            ),
            (
                "Return the shortest supported value matching the requested answer type. "
                "If the evidence does not establish the exact entity, relation, time, location, or unit, return unknown."
            ),
            "Do not add reasoning, citations, candidate lists, or extra fields.",
            f"Question: {_clean(row['question'])}",
            f"Question-inferred answer type: {_clean(row.get('inferred_answer_type') or 'other')}",
            f"Frozen visual observation: {_clean(row['visual_observation'])}",
            f"Executed search query: {_clean(row['search_query'])}",
            "Selected real-search evidence:\n" + _evidence_text(row["selected_docs"], snippet_chars),
            "\n".join(auxiliary),
            "Final-answer JSON:",
        ]
    )


def answer_completion(value: Any) -> str:
    return json.dumps({"final_answer": _clean(value) or "unknown"}, ensure_ascii=False, separators=(",", ":"))
