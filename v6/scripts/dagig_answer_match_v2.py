#!/usr/bin/env python3
"""Versioned conservative answer-equivalence amendment for DAG-IG v6.

The frozen baseline checker already accepts a numeric gold answer inside a
unit-bearing prediction (for example, ``30`` vs ``30 days``).  This module
adds only the missing symmetric case: a gold answer consisting of exactly one
number and one whitelisted simple unit may match a prediction consisting of
exactly the same number.  It intentionally does not accept free text, multiple
numbers, ranges, percentages, currencies, or unlisted units.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


PROTOCOL_VERSION = "dagig_answer_match_numeric_unit_symmetric_v2"

UNIT_PATTERNS: tuple[str, ...] = (
    "day",
    "days",
    "year",
    "years",
    "year old",
    "years old",
    "month",
    "months",
    "hour",
    "hours",
    "minute",
    "minutes",
    "second",
    "seconds",
    "location",
    "locations",
    "store",
    "stores",
    "branch",
    "branches",
    "variety",
    "varieties",
    "country",
    "countries",
    "person",
    "persons",
    "people",
)

_NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"
_UNIT = "|".join(sorted((re.escape(x) for x in UNIT_PATTERNS), key=len, reverse=True))
_NUMBER_ONLY_RE = re.compile(rf"^\s*({_NUMBER})\s*$", flags=re.IGNORECASE)
_NUMBER_UNIT_RE = re.compile(rf"^\s*({_NUMBER})\s+({_UNIT})\s*$", flags=re.IGNORECASE)


def _canonical_number(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).replace(",", "").strip()
    sign = ""
    if value.startswith(("+", "-")):
        sign, value = value[0], value[1:]
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    value = value.lstrip("0") or "0"
    return sign + value


def numeric_unit_symmetric_match(prediction: str, target: str) -> dict[str, Any] | None:
    """Return amendment details only for the exact symmetric unit case."""
    prediction = unicodedata.normalize("NFKC", str(prediction)).strip()
    target = unicodedata.normalize("NFKC", str(target)).strip()
    pred_match = _NUMBER_ONLY_RE.fullmatch(prediction)
    target_match = _NUMBER_UNIT_RE.fullmatch(target)
    if not pred_match or not target_match:
        return None
    pred_number = _canonical_number(pred_match.group(1))
    target_number = _canonical_number(target_match.group(1))
    if pred_number != target_number:
        return None
    return {
        "answer_correct": True,
        "answer_match_type": "numeric_unit_symmetric_exact",
        "normalized_gold": f"{target_number} {target_match.group(2).casefold()}",
        "normalized_pred": pred_number,
        "matched_alias": "",
        "matched_unit": target_match.group(2).casefold(),
        "protocol_version": PROTOCOL_VERSION,
    }


def answer_match_details(
    baseline_helper: Any,
    prediction: str,
    gold: str,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    """Apply the immutable baseline first, then the narrow amendment."""
    baseline = baseline_helper.answer_match_details(prediction, gold, aliases or [])
    if baseline.get("answer_correct"):
        return {**baseline, "protocol_version": "frozen_baseline"}
    for index, target in enumerate([gold, *(aliases or [])]):
        amended = numeric_unit_symmetric_match(prediction, str(target))
        if amended is not None:
            amended["matched_alias"] = "gold" if index == 0 else str(target)
            return amended
    return {**baseline, "protocol_version": PROTOCOL_VERSION}
