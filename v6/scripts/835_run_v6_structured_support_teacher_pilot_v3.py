#!/usr/bin/env python3
"""Run one blinded role of the frozen structured support-label pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def evidence_documents(prompt: str) -> dict[int, str]:
    if "Selected evidence:\n" not in prompt:
        raise ValueError("Pilot prompt lacks selected evidence")
    evidence = prompt.split("Selected evidence:\n", 1)[1]
    matches = list(re.finditer(r"(?:^|\n\n)Document ([123])\n", evidence))
    documents: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        documents[int(match.group(1))] = evidence[match.end() : end].strip()
    if set(documents) != {1, 2, 3}:
        raise ValueError("Pilot prompt does not contain exactly three evidence documents")
    return documents


def citation_matches(span: str, document: str) -> bool:
    span_tokens = normalized(span).split()
    document_tokens = normalized(document).split()
    if not span_tokens:
        return False
    if " ".join(span_tokens) in " ".join(document_tokens):
        return True
    # Models sometimes copy two exact clauses separated by an ellipsis. Accept
    # that only when every substantive copied token is grounded in the cited doc.
    return len(span_tokens) >= 5 and sum(token in set(document_tokens) for token in span_tokens) / len(span_tokens) >= 0.95


def schema(batch_size: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_support_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array",
                        "minItems": batch_size,
                        "maxItems": batch_size,
                        "items": {
                            "type": "object",
                            "properties": {
                                "audit_id": {"type": "string"},
                                "supported": {"type": "boolean"},
                                "supporting_doc_indices": {"type": "array", "items": {"type": "integer"}},
                                "supporting_span": {"type": "string"},
                                "entailment_type": {"type": "string", "enum": ["exact", "normalized_equivalent", "rounding_or_conversion", "strong_entailment", "none"]},
                                "derivation": {"type": "string"},
                                "conflict_present": {"type": "boolean"},
                                "reason": {"type": "string"},
                            },
                            "required": ["audit_id", "supported", "supporting_doc_indices", "supporting_span", "entailment_type", "derivation", "conflict_present", "reason"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["decisions"],
                "additionalProperties": False,
            },
        },
    }


def validate_decisions(payload: dict[str, Any], batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    choices = payload.get("choices") or []
    if len(choices) != 1 or (choices[0].get("message") or {}).get("refusal"):
        raise ValueError("Missing or refused structured completion")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty structured completion")
    parsed = json.loads(content)
    values = parsed.get("decisions") if isinstance(parsed, dict) else None
    expected_ids = [row["audit_id"] for row in batch]
    if not isinstance(values, list) or len(values) != len(batch):
        raise ValueError("Wrong decision count")
    by_id = {row.get("audit_id"): row for row in values}
    if len(by_id) != len(values) or set(by_id) != set(expected_ids):
        raise ValueError("Missing, duplicate, or unexpected audit IDs")
    result = []
    for item in batch:
        row = by_id[item["audit_id"]]
        supported = row.get("supported")
        indices = row.get("supporting_doc_indices")
        span = str(row.get("supporting_span") or "").strip()
        entailment = row.get("entailment_type")
        if not isinstance(supported, bool) or not isinstance(indices, list) or any(type(value) is not int or value not in (1, 2, 3) for value in indices):
            raise ValueError(f"Invalid structured decision for {item['audit_id']}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate document indices for {item['audit_id']}")
        if supported:
            if not indices or not span or entailment == "none":
                raise ValueError(f"Positive decision lacks cited support for {item['audit_id']}")
            documents = evidence_documents(item["user_prompt_private"])
            citation_valid = any(citation_matches(span, documents[index]) for index in indices)
            if not citation_valid:
                raise ValueError(f"Supporting span is not copied from supplied evidence for {item['audit_id']}")
        else:
            if indices or span or entailment != "none":
                raise ValueError(f"Negative decision contains positive support fields for {item['audit_id']}")
            citation_valid = True
        reason = row.get("reason")
        if not isinstance(reason, str):
            raise ValueError(f"Invalid rationale type for {item['audit_id']}")
        result.append({**row, "supporting_span": span, "reason": reason.strip(), "citation_valid": citation_valid})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--role", choices=("teacher", "auditor"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse_dir", type=Path)
    parser.add_argument("--reuse_freeze", type=Path)
    parser.add_argument("--timeout_seconds", type=int, default=240)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_FROZEN":
        raise ValueError("Structured support pilot v3 is not frozen")
    if freeze["input_hashes"]["runner"] != sha256(Path(__file__).resolve()):
        raise ValueError("Structured support pilot runner changed after freeze")
    items_path = Path(freeze["output_paths"]["pilot_items"])
    if sha256(items_path) != freeze["output_hashes"]["pilot_items"]:
        raise ValueError("Frozen structured support pilot items changed")
    items = read_jsonl(items_path)
    if len(items) != 400 or any(set(row) != {"audit_id", "user_prompt_private"} for row in items):
        raise ValueError("Runner input is not the exact blinded 400-item universe")
    forbidden = ("local_label", "local_probability", "audit_category", "legacy_support_reason")
    if any(name in json.dumps(items, ensure_ascii=False) for name in forbidden):
        raise ValueError("Runner input contains evaluation-only metadata")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not visible")
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; use --resume for the same role: {output}")
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / f"{args.role}_structured_support_decisions_blinded.jsonl"
    request_path = output / "api_request_log.jsonl"
    usage_path = output / "api_usage.jsonl"
    manifest_path = output / "RUN_MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError(f"Run is already complete: {manifest_path}")
    reuse_record = None
    if args.reuse_dir or args.reuse_freeze:
        if not args.reuse_dir or not args.reuse_freeze:
            raise ValueError("--reuse_dir and --reuse_freeze must be supplied together")
        if decisions_path.exists():
            raise ValueError("Cannot import reusable decisions into a non-empty output")
        source_freeze_path = args.reuse_freeze.resolve()
        source_freeze = read_json(source_freeze_path)
        invariants = {
            "same_items": source_freeze["output_hashes"]["pilot_items"] == freeze["output_hashes"]["pilot_items"],
            "same_system_prompt": source_freeze["system_prompt"] == freeze["system_prompt"],
            "same_model": source_freeze["models"][args.role] == freeze["models"][args.role],
            "same_reasoning_effort": source_freeze["generation"]["reasoning_effort"][args.role] == freeze["generation"]["reasoning_effort"][args.role],
            "same_max_completion_tokens": source_freeze["generation"]["max_completion_tokens"] == freeze["generation"]["max_completion_tokens"],
        }
        if not all(invariants.values()):
            raise ValueError(f"Reusable role outputs violate frozen invariants: {invariants}")
        source_decisions_path = args.reuse_dir.resolve() / f"{args.role}_structured_support_decisions_blinded.jsonl"
        source_rows = read_jsonl(source_decisions_path)
        item_by_id = {row["audit_id"]: row for row in items}
        source_ids = [row["audit_id"] for row in source_rows]
        if len(source_ids) != len(set(source_ids)) or not set(source_ids).issubset(set(item_by_id)):
            raise ValueError("Reusable decision universe is inconsistent")
        for row in source_rows:
            item = item_by_id[row["audit_id"]]
            supported = row.get("supported")
            indices = row.get("supporting_doc_indices")
            span = str(row.get("supporting_span") or "").strip()
            entailment = row.get("entailment_type")
            if not isinstance(supported, bool) or not isinstance(indices, list) or any(type(value) is not int or value not in (1, 2, 3) for value in indices):
                raise ValueError(f"Invalid reusable decision: {row['audit_id']}")
            if supported:
                documents = evidence_documents(item["user_prompt_private"])
                if not indices or not span or entailment == "none" or not any(citation_matches(span, documents[index]) for index in indices):
                    raise ValueError(f"Reusable positive citation failed: {row['audit_id']}")
            elif indices or span or entailment != "none":
                raise ValueError(f"Reusable negative fields failed: {row['audit_id']}")
            append_jsonl(decisions_path, {**row, "reused_from_decisions_sha256": sha256(source_decisions_path)})
        reuse_record = {
            "source_dir": str(args.reuse_dir.resolve()),
            "source_freeze": str(source_freeze_path),
            "source_freeze_sha256": sha256(source_freeze_path),
            "source_decisions": str(source_decisions_path),
            "source_decisions_sha256": sha256(source_decisions_path),
            "reused_items": len(source_rows),
            "invariants": invariants,
        }
        (output / "REUSE_MANIFEST.json").write_text(json.dumps(reuse_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    existing = read_jsonl(decisions_path)
    complete_ids = {row["audit_id"] for row in existing}
    universe = {row["audit_id"] for row in items}
    if len(complete_ids) != len(existing) or not complete_ids.issubset(universe):
        raise ValueError("Existing role decisions are inconsistent")
    request_count = len(read_jsonl(request_path))
    usage_rows = read_jsonl(usage_path)
    input_tokens = sum(int(row.get("prompt_tokens", 0)) for row in usage_rows)
    output_tokens = sum(int(row.get("completion_tokens", 0)) for row in usage_rows)
    batch_size = int(freeze["generation"]["batch_size"])
    budget = freeze["budget"]
    remaining = [row for row in items if row["audit_id"] not in complete_ids]
    session = requests.Session()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    successful_batches = len(complete_ids) // batch_size
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        batch_index = successful_batches + start // batch_size
        body = {
            "model": freeze["models"][args.role],
            "messages": [
                {"role": "system", "content": freeze["system_prompt"]},
                {"role": "user", "content": json.dumps({"cases": batch}, ensure_ascii=False)},
            ],
            "max_completion_tokens": int(freeze["generation"]["max_completion_tokens"]),
            "reasoning_effort": freeze["generation"]["reasoning_effort"][args.role],
            "response_format": schema(len(batch)),
            "store": False,
        }
        for attempt in range(1, 6):
            if request_count >= int(budget["max_requests_per_role"]):
                raise RuntimeError("Frozen per-role request budget exhausted")
            request_count += 1
            started = time.time()
            status = None
            request_id = None
            try:
                response = session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=args.timeout_seconds)
                status = response.status_code
                request_id = response.headers.get("x-request-id")
                if status != 200:
                    try:
                        error_type = (response.json().get("error") or {}).get("type", "api_error")
                    except Exception:
                        error_type = "non_json_api_error"
                    raise RuntimeError(f"OpenAI API status {status} ({error_type})")
                response_json = response.json()
                decisions = validate_decisions(response_json, batch)
                usage = response_json.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                input_tokens += prompt_tokens
                output_tokens += completion_tokens
                if input_tokens > int(budget["max_input_tokens_per_role"]) or output_tokens > int(budget["max_output_tokens_per_role"]):
                    raise RuntimeError("Frozen token budget exceeded")
                append_jsonl(request_path, {"attempt": request_count, "batch_index": batch_index, "elapsed_seconds": round(time.time() - started, 3), "http_status": status, "n_items": len(batch), "request_id": request_id or response_json.get("id"), "result": "success"})
                append_jsonl(usage_path, {"batch_index": batch_index, "completion_tokens": completion_tokens, "n_items": len(batch), "prompt_tokens": prompt_tokens, "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens))})
                for row in decisions:
                    append_jsonl(decisions_path, {**row, "batch_index": batch_index, "model": response_json.get("model", freeze["models"][args.role]), "role": args.role})
                complete_ids.update(row["audit_id"] for row in decisions)
                print(json.dumps({"role": args.role, "completed": len(complete_ids), "total": len(items), "requests": request_count, "prompt_tokens": input_tokens, "completion_tokens": output_tokens}), flush=True)
                break
            except Exception as exc:
                append_jsonl(request_path, {"attempt": request_count, "batch_index": batch_index, "elapsed_seconds": round(time.time() - started, 3), "error": f"{type(exc).__name__}: {str(exc)[:300]}", "http_status": status, "n_items": len(batch), "request_id": request_id, "result": "failure"})
                if attempt == 5 or request_count >= int(budget["max_requests_per_role"]):
                    raise RuntimeError(f"{args.role} batch {batch_index} failed") from exc
                time.sleep(min(2 ** (attempt - 1), 16))
    rows = read_jsonl(decisions_path)
    ids = [row["audit_id"] for row in rows]
    if len(rows) != 400 or len(set(ids)) != 400 or set(ids) != universe:
        raise ValueError("Role output does not cover exact pilot universe")
    requests_rows = read_jsonl(request_path)
    manifest = {
        "decision": "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_ROLE_COMPLETE",
        "role": args.role,
        "model": freeze["models"][args.role],
        "freeze_path": str(freeze_path),
        "freeze_sha256": sha256(freeze_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "decisions_path": str(decisions_path),
        "decisions_sha256": sha256(decisions_path),
        "requests_path": str(request_path),
        "requests_sha256": sha256(request_path),
        "usage_path": str(usage_path),
        "usage_sha256": sha256(usage_path),
        "items": len(rows),
        "request_attempts": len(requests_rows),
        "successful_requests": sum(row["result"] == "success" for row in requests_rows),
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "reused_items": len(existing) if reuse_record else 0,
        "reuse_record": reuse_record,
        "teacher_outputs_visible": False if args.role == "auditor" else None,
        "private_key_visible": False,
        "api_key_stored": False,
        "serper_calls": 0,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
