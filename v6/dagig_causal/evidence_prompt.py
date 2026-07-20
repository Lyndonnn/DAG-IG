"""Shared prompt contract for the executable evidence-selection node."""

from __future__ import annotations

from typing import Any


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def build_evidence_selection_prompt(row: dict[str, Any]) -> str:
    """Build a schema-only prompt without demonstrating a concrete action.

    A concrete example such as ``[D1, D2, D3]`` acts as a strong default-action
    prior. That makes a format SFT initializer unsuitable for controlled credit
    comparisons, even when its targets are balanced across action generators.
    """

    docs = sorted(row["candidate_docs"], key=lambda doc: int(doc["rank"]))
    blocks = [
        (
            f"[D{index}] Title: {_clean(doc.get('title'))}\n"
            f"Domain: {_clean(doc.get('domain'))}\n"
            f"Snippet: {_clean(doc.get('snippet'))[:320]}"
        )
        for index, doc in enumerate(docs, 1)
    ]
    return "\n\n".join(
        [
            "You are the evidence-selection node of a multimodal web-search agent.",
            (
                "Return only one compact valid JSON object. It must have exactly "
                "one field named selected_evidence_ids, whose value is an array "
                "of exactly three distinct IDs copied from the retrieved candidate labels."
            ),
            (
                "Choose IDs from their content; do not use a fixed or default set. "
                "Prefer evidence that jointly supports the requested entity, relation, "
                "answer type, time, and location constraints."
            ),
            "Do not answer the question and do not add reasoning or extra fields.",
            f"Question: {_clean(row['question'])}",
            f"Frozen visual observation: {_clean(row['visual_observation'])}",
            f"Executed search query: {_clean(row['search_query'])}",
            "Retrieved candidates:\n" + "\n\n".join(blocks),
            "Evidence selection:",
        ]
    )
