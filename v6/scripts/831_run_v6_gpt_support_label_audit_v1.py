#!/usr/bin/env python3
"""Run the frozen blinded GPT support-label audit without local-label access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def response_schema(batch_size: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "support_audit_batch",
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
                                "reason": {"type": "string"},
                            },
                            "required": ["audit_id", "supported", "reason"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["decisions"],
                "additionalProperties": False,
            },
        },
    }


def parse_batch_response(payload: dict[str, Any], expected_ids: list[str]) -> list[dict[str, Any]]:
    choices = payload.get("choices") or []
    if len(choices) != 1:
        raise ValueError(f"Expected one completion choice, got {len(choices)}")
    message = choices[0].get("message") or {}
    if message.get("refusal"):
        raise ValueError("Auditor refused the batch")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Auditor returned empty content")
    parsed = json.loads(content)
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list) or len(decisions) != len(expected_ids):
        raise ValueError("Auditor returned the wrong decision count")
    ids = [row.get("audit_id") for row in decisions]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("Auditor returned missing, duplicate, or unexpected audit IDs")
    by_id = {row["audit_id"]: row for row in decisions}
    ordered = []
    for audit_id in expected_ids:
        row = by_id[audit_id]
        if not isinstance(row.get("supported"), bool):
            raise ValueError(f"Non-boolean support decision for {audit_id}")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Missing support rationale for {audit_id}")
        ordered.append({"audit_id": audit_id, "supported": row["supported"], "reason": reason.strip()})
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout_seconds", type=int, default=180)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_GPT_SUPPORT_LABEL_AUDIT_V1_FROZEN":
        raise ValueError("GPT support-label audit protocol is not frozen")
    if freeze["input_hashes"]["runner"] != sha256(Path(__file__).resolve()):
        raise ValueError("GPT audit runner changed after protocol freeze")
    items_path = Path(freeze["input_paths"]["blinded_items"])
    if sha256(items_path) != freeze["input_hashes"]["blinded_items"]:
        raise ValueError("Frozen blinded audit items changed")
    items = read_jsonl(items_path)
    if len(items) != 350:
        raise ValueError(f"Expected 350 blinded audit items, got {len(items)}")
    allowed_keys = {"audit_id", "system_prompt", "user_prompt_private"}
    if any(set(row) != allowed_keys for row in items):
        raise ValueError("Blinded runner input contains unexpected fields")
    serialized = json.dumps(items, ensure_ascii=False).casefold()
    forbidden = ("local_probability", "local_label", "audit_category", "legacy_support_reason")
    if any(name in serialized for name in forbidden):
        raise ValueError("Blinded runner input contains evaluation-only metadata")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not visible to the runner")

    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; use --resume only for the same frozen run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "gpt_support_label_decisions_blinded.jsonl"
    requests_path = output / "api_request_log.jsonl"
    usage_path = output / "api_usage.jsonl"
    manifest_path = output / "GPT_SUPPORT_LABEL_AUDIT_V1_RUN_MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError(f"A completed run manifest already exists: {manifest_path}")

    existing = read_jsonl(decisions_path)
    completed = {row["audit_id"] for row in existing}
    all_ids = [row["audit_id"] for row in items]
    if len(completed) != len(existing) or not completed.issubset(set(all_ids)):
        raise ValueError("Existing decision file is inconsistent with the frozen audit universe")
    request_log = read_jsonl(requests_path)
    request_count = len(request_log)
    usage_rows = read_jsonl(usage_path)
    input_tokens = sum(int(row.get("prompt_tokens", 0)) for row in usage_rows)
    output_tokens = sum(int(row.get("completion_tokens", 0)) for row in usage_rows)
    max_requests = int(freeze["budget"]["max_requests"])
    max_input_tokens = int(freeze["budget"]["max_total_input_tokens"])
    max_output_tokens = int(freeze["budget"]["max_total_output_tokens"])
    batch_size = int(freeze["generation"]["batch_size"])
    remaining = [row for row in items if row["audit_id"] not in completed]
    session = requests.Session()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        ids = [row["audit_id"] for row in batch]
        batch_number = (len(completed) + start) // batch_size
        user_payload = {
            "instruction": "Audit every case independently. Return supported=true only when at least one supplied document supports the private reference answer under all question constraints. Return exactly one decision for each audit_id.",
            "cases": [{"audit_id": row["audit_id"], "case_text": row["user_prompt_private"]} for row in batch],
        }
        request_body = {
            "model": freeze["model"],
            "messages": [
                {"role": "system", "content": freeze["auditor_system_prompt"]},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "max_completion_tokens": int(freeze["generation"]["max_completion_tokens"]),
            "response_format": response_schema(len(batch)),
            "store": False,
        }
        last_error = None
        for attempt in range(1, 6):
            if request_count >= max_requests:
                raise RuntimeError(f"Frozen request budget exhausted after {request_count} API attempts")
            request_count += 1
            started = time.time()
            status_code = None
            response_id = None
            try:
                response = session.post(
                    freeze["endpoint"], headers=headers, json=request_body, timeout=args.timeout_seconds
                )
                status_code = response.status_code
                response_id = response.headers.get("x-request-id")
                if response.status_code != 200:
                    try:
                        error_type = (response.json().get("error") or {}).get("type", "api_error")
                    except Exception:
                        error_type = "non_json_api_error"
                    raise RuntimeError(f"OpenAI API status {response.status_code} ({error_type})")
                response_json = response.json()
                decisions = parse_batch_response(response_json, ids)
                usage = response_json.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                input_tokens += prompt_tokens
                output_tokens += completion_tokens
                if input_tokens > max_input_tokens or output_tokens > max_output_tokens:
                    raise RuntimeError("Frozen token budget exceeded")
                append_jsonl(requests_path, {
                    "attempt": request_count,
                    "batch_index": batch_number,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "http_status": status_code,
                    "n_items": len(batch),
                    "request_id": response_id or response_json.get("id"),
                    "result": "success",
                })
                append_jsonl(usage_path, {
                    "batch_index": batch_number,
                    "completion_tokens": completion_tokens,
                    "n_items": len(batch),
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
                })
                for row in decisions:
                    append_jsonl(decisions_path, {
                        **row,
                        "audit_source": "independent_blinded_gpt5mini",
                        "batch_index": batch_number,
                        "model": response_json.get("model", freeze["model"]),
                    })
                completed.update(ids)
                print(json.dumps({
                    "completed": len(completed),
                    "total": len(items),
                    "requests_used": request_count,
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                }), flush=True)
                break
            except Exception as exc:
                last_error = exc
                append_jsonl(requests_path, {
                    "attempt": request_count,
                    "batch_index": batch_number,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "http_status": status_code,
                    "n_items": len(batch),
                    "request_id": response_id,
                    "result": "failure",
                })
                if attempt == 5 or request_count >= max_requests:
                    raise RuntimeError(f"Audit batch {batch_number} failed after {attempt} attempts") from exc
                time.sleep(min(2 ** (attempt - 1), 16))
        else:
            raise RuntimeError(f"Audit batch {batch_number} failed") from last_error

    final_rows = read_jsonl(decisions_path)
    final_ids = [row["audit_id"] for row in final_rows]
    if len(final_rows) != 350 or len(set(final_ids)) != 350 or set(final_ids) != set(all_ids):
        raise ValueError("Completed decision file does not cover the exact frozen audit universe")
    request_rows = read_jsonl(requests_path)
    success_requests = sum(row["result"] == "success" for row in request_rows)
    manifest = {
        "decision": "DAGIG_V6_GPT_SUPPORT_LABEL_AUDIT_V1_RUN_COMPLETE",
        "freeze_path": str(freeze_path),
        "freeze_sha256": sha256(freeze_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "decisions_path": str(decisions_path),
        "decisions_sha256": sha256(decisions_path),
        "requests_path": str(requests_path),
        "requests_sha256": sha256(requests_path),
        "usage_path": str(usage_path),
        "usage_sha256": sha256(usage_path),
        "audit_items": len(final_rows),
        "api_request_attempts": len(request_rows),
        "successful_api_requests": success_requests,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "model": freeze["model"],
        "local_labels_visible_to_runner": False,
        "private_audit_key_visible_to_runner": False,
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
