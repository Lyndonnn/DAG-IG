"""Execution-level identities for formal DAG-IG policy actions.

Policy actions are equal only when the frozen executor receives the same
action.  In particular, case is preserved because it remains visible to
downstream prompts.  The evidence pointer is the exception: its action is an
unordered set of selected document IDs.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def compact_text(value: Any) -> str:
    """Match executor whitespace normalization without changing case."""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def execution_action_payload(node: str, action: Any) -> Any:
    """Return the exact semantic payload consumed by a node executor."""

    if node == "visual_action":
        value = action if isinstance(action, dict) else {}
        return {
            "grounding_expression": compact_text(value.get("grounding_expression")),
            "visual_anchor": compact_text(value.get("visual_anchor")),
        }
    if node == "search_query":
        return compact_text(action)
    if node == "evidence_action":
        value = action if isinstance(action, dict) else {}
        return {
            "selected_doc_ids": sorted(
                {str(doc_id) for doc_id in (value.get("selected_doc_ids") or [])}
            )
        }
    if node == "final_answer":
        return compact_text(action)
    raise KeyError(node)


def policy_action_identity(
    node: str,
    *,
    valid: bool,
    action: Any,
    raw_surface: str,
) -> dict[str, Any]:
    """Build an identity that distinguishes invalid raw policy responses."""

    return {
        "node": node,
        "valid": bool(valid),
        "action": (
            execution_action_payload(node, action)
            if valid
            else str(raw_surface or "").strip()
        ),
    }


def stable_identity_hash(value: Any, length: int | None = None) -> str:
    surface = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(surface.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def policy_action_id(
    node: str,
    *,
    valid: bool,
    action: Any,
    raw_surface: str,
    length: int = 24,
) -> str:
    return stable_identity_hash(
        policy_action_identity(
            node,
            valid=valid,
            action=action,
            raw_surface=raw_surface,
        ),
        length=length,
    )
