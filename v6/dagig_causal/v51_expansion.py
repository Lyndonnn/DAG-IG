"""Shared, leakage-aware utilities for the frozen v5.1 train-only expansion."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from PIL import Image, ImageOps


QUERY_STRATEGIES = (
    "entity_relation",
    "official_source",
    "domain_restricted",
    "exact_phrase",
    "alternate_name",
    "condition_specific",
    "answer_conditioned",
    "evidence_phrase",
    "multilingual",
    "document_type",
    "relation_paraphrase",
    "source_discovery",
)

QUERY_RESPONSE_SCHEMA = {
    "name": "dagig_v5_1_search_query_diversification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "queries": {
                "type": "array",
                "minItems": 12,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "strategy": {"type": "string", "enum": list(QUERY_STRATEGIES)},
                        "query": {"type": "string", "minLength": 3, "maxLength": 240},
                        "uses_target_answer": {"type": "boolean"},
                    },
                    "required": ["strategy", "query", "uses_target_answer"],
                },
            }
        },
        "required": ["queries"],
    },
}

QUERY_SYSTEM_PROMPT = (
    "You design diverse Google search queries for train-only web-evidence acquisition. "
    "The image, question, visual metadata, and private target answer describe one fact. "
    "Return exactly one concise query for each required strategy. Preserve entity, relation, "
    "location, date/time, comparison, and unit conditions. Prefer queries likely to find an "
    "independent page that explicitly states the requested fact. Do not invent entities. "
    "The private target answer may be used only by answer_conditioned or evidence_phrase queries. "
    "For all other strategies, do not include the target answer. Return schema-valid JSON only."
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact(value: Any, max_words: int | None = None) -> str:
    words = re.sub(r"\s+", " ", str(value or "")).strip().split()
    if max_words is not None:
        words = words[:max_words]
    return " ".join(words)


def normalize_query(value: Any) -> str:
    value = compact(value)
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    return compact(value)


def query_key(value: Any) -> str:
    return re.sub(r"[^\w]+", " ", normalize_query(value).casefold(), flags=re.UNICODE).strip()


def compact_alnum(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def query_id(sample_id: str, source: str, query: str) -> str:
    payload = f"{sample_id}\n{source}\n{query_key(query)}"
    return "v51q_" + text_sha256(payload)[:20]


def answer_in_query(answer: Any, query: Any) -> bool:
    answer_key = query_key(answer)
    query_norm = query_key(query)
    if len(answer_key) >= 3 and re.search(
        rf"(?<!\w){re.escape(answer_key)}(?!\w)", query_norm, flags=re.UNICODE
    ):
        return True
    answer_compact = compact_alnum(answer)
    return len(answer_compact) >= 4 and answer_compact in compact_alnum(query)


def canonical_url(value: Any) -> str:
    raw = compact(value)
    if not raw.startswith(("http://", "https://")):
        return ""
    parts = urlsplit(raw)
    host = parts.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+", "/", parts.path or "/").rstrip("/") or "/"
    blocked = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"}
    query = urlencode(
        sorted(
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=False)
            if not key.casefold().startswith("utm_") and key.casefold() not in blocked
        )
    )
    return urlunsplit((parts.scheme.casefold(), host, path, query, ""))


def image_data_uri(path: Path, max_side: int = 768) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=84, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def task_prompt(task: dict[str, Any]) -> str:
    payload = {
        "required_strategy_order": list(QUERY_STRATEGIES),
        "question": task["question"],
        "visual_anchor": task.get("visual_anchor", ""),
        "localized_entity": task.get("localized_entity", ""),
        "existing_policy_query": task.get("policy_or_teacher_query", ""),
        "information_or_evidence_hint": task.get("evidence_hint", ""),
        "known_source_domain": task.get("domain_hint", ""),
        "private_train_target_answer": task["gold_answer_train_acquisition_only"],
        "answer_type": task["answer_type"],
    }
    return (
        "Generate the 12 queries in required_strategy_order. Keep each under 24 words. "
        "Use site:domain only for domain_restricted. For multilingual, retain useful original-script "
        "entity text when available; otherwise use a plausible language variant without changing identity. "
        "Mark uses_target_answer accurately.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def deterministic_queries(target: dict[str, Any]) -> list[tuple[str, str, bool]]:
    entity = compact(target.get("localized_entity") or target.get("visual_anchor"), 12)
    anchor = compact(target.get("visual_anchor"), 10)
    policy = compact(target.get("policy_or_teacher_query"), 24)
    question = compact(target.get("question"), 22)
    hint = compact(target.get("evidence_hint"), 16)
    title = compact(target.get("positive_title_hint"), 12)
    domain = compact(target.get("domain_hint"), 3)
    answer = compact(target.get("gold_answer_train_acquisition_only"), 10)
    base = entity or anchor or policy
    site = f"site:{domain}" if domain else ""
    candidates = [
        ("det_official_relation", f"{base} {question} official", False),
        ("det_domain_relation", f"{site} {base} {question}", False),
        ("det_pdf_relation", f"{base} {question} filetype:pdf", False),
        ("det_policy_source", f"{policy} source", False),
        ("det_anchor_relation", f'"{anchor}" {question}', False),
        ("det_entity_facts", f'"{entity}" {question} facts', False),
        ("det_title_relation", f'"{title}" {base} {question}', False),
        ("det_hint_relation", f'"{hint}" {base}', False),
        ("det_answer_relation", f'"{answer}" {base} {question}', True),
        ("det_answer_official", f'{base} "{answer}" official', True),
        ("det_answer_domain", f'{site} {base} "{answer}"', True),
        ("det_answer_hint", f'{hint} "{answer}"', True),
        ("det_policy_answer", f'{policy} "{answer}"', True),
        ("det_entity_answer", f'{entity} "{answer}" details', True),
        ("det_anchor_answer", f'{anchor} "{answer}" source', True),
        ("det_question_answer", f'{question} "{answer}"', True),
        ("det_domain_policy", f"{site} {policy}", False),
        ("det_entity_reference", f"{base} reference documentation {question}", False),
        ("det_entity_history", f"{base} history facts {question}", False),
        ("det_exact_policy", f'"{policy}"', False),
    ]
    return [
        (source, normalize_query(query), uses_target)
        for source, query, uses_target in candidates
        if len(normalize_query(query).split()) >= 2
    ]
