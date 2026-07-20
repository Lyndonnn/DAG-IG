"""Strict, execution-facing schemas for the four DAG-IG policy actions.

The policy may emit a single full JSON code fence as a reversible surface
normalization.  Substring extraction, trailing prose, type coercion, and
partial objects are intentionally rejected.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


_FULL_FENCE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.IGNORECASE | re.DOTALL)
_VISUAL_PLACEHOLDERS = {
    "",
    "unknown",
    "visible entity in the image",
    "concise visible text or object",
    "concise visible text/object",
}
_POINTER_LABELS = tuple(chr(ord("A") + index) for index in range(20))


@dataclass(frozen=True)
class SchemaResult:
    valid: bool
    stage: str
    reason: str
    normalization: str | None
    action: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_exact_json_object(raw: str) -> tuple[dict[str, Any] | None, str, str | None]:
    text = str(raw or "").strip()
    if not text:
        return None, "empty_surface", None
    normalization: str | None = None
    fence = _FULL_FENCE.fullmatch(text)
    if fence:
        text = fence.group(1).strip()
        normalization = "full_json_code_fence"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        reason = "trailing_or_extra_text" if exc.msg == "Extra data" else "invalid_json"
        return None, reason, normalization
    if not isinstance(value, dict):
        return None, "top_level_not_object", normalization
    return value, "ok", normalization


def _exact_keys(value: dict[str, Any], expected: set[str]) -> bool:
    return set(value) == expected


def validate_action_surface(stage: str, raw: str) -> SchemaResult:
    if stage == "evidence_action":
        expected_keys = {"selected_labels"}
    elif stage == "visual_action":
        expected_keys = {"grounding_expression", "visual_anchor"}
    elif stage == "search_query":
        expected_keys = {"search_query"}
    elif stage == "final_answer":
        expected_keys = {"final_answer"}
    else:
        return SchemaResult(False, stage, "unsupported_stage", None, None)

    value, reason, normalization = parse_exact_json_object(raw)
    if value is None:
        return SchemaResult(False, stage, reason, normalization, None)
    if not _exact_keys(value, expected_keys):
        return SchemaResult(False, stage, "wrong_key_set", normalization, value)

    if stage == "visual_action":
        expression = value["grounding_expression"]
        anchor = value["visual_anchor"]
        if not isinstance(expression, str) or not isinstance(anchor, str):
            return SchemaResult(False, stage, "non_string_visual_field", normalization, value)
        action = {
            "grounding_expression": _compact(expression),
            "visual_anchor": _compact(anchor),
        }
        if action["grounding_expression"].casefold() in _VISUAL_PLACEHOLDERS:
            return SchemaResult(False, stage, "invalid_grounding_expression", normalization, action)
        if action["visual_anchor"].casefold() in _VISUAL_PLACEHOLDERS:
            return SchemaResult(False, stage, "invalid_visual_anchor", normalization, action)
        return SchemaResult(True, stage, "ok", normalization, action)

    if stage in {"search_query", "final_answer"}:
        key = next(iter(expected_keys))
        item = value[key]
        if not isinstance(item, str):
            return SchemaResult(False, stage, "non_string_action_value", normalization, value)
        action = _compact(item)
        if not action:
            return SchemaResult(False, stage, "empty_action_value", normalization, action)
        return SchemaResult(True, stage, "ok", normalization, action)

    labels = value["selected_labels"]
    if not isinstance(labels, list):
        return SchemaResult(False, stage, "selected_labels_not_list", normalization, value)
    if len(labels) != 3:
        return SchemaResult(False, stage, "selected_labels_wrong_count", normalization, value)
    if not all(isinstance(label, str) and label in _POINTER_LABELS for label in labels):
        return SchemaResult(False, stage, "selected_labels_invalid", normalization, value)
    if len(set(labels)) != 3:
        return SchemaResult(False, stage, "selected_labels_not_unique", normalization, value)
    return SchemaResult(True, stage, "ok", normalization, {"selected_labels": labels})
