#!/usr/bin/env python3
"""Score one resumable shard of frozen full structured support states."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


def load_runner_helper() -> Any:
    path = Path(__file__).with_name("835_run_v6_structured_support_teacher_pilot_v3.py")
    spec = importlib.util.spec_from_file_location("dagig_structured_support_api_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amended_from_freeze", type=Path)
    parser.add_argument("--timeout_seconds", type=int, default=240)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_FROZEN":
        raise ValueError("Full structured support v3 protocol is not frozen")
    if freeze["input_hashes"]["scorer"] != sha256(Path(__file__).resolve()):
        raise ValueError("Full structured support scorer changed after freeze")
    helper_path = Path(freeze["input_paths"]["scorer_helper"])
    if freeze["input_hashes"]["scorer_helper"] != sha256(helper_path):
        raise ValueError("Structured support API helper changed after freeze")
    if args.num_shards != int(freeze["sharding"]["num_shards"]) or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Shard specification differs from freeze")
    prompt_path = Path(freeze["output_paths"]["remaining_prompts"])
    if sha256(prompt_path) != freeze["output_hashes"]["remaining_prompts"]:
        raise ValueError("Frozen remaining prompt universe changed")
    all_rows = read_jsonl(prompt_path)
    rows = [row for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index]
    expected = int(freeze["sharding"]["remaining_rows_per_shard"][args.shard_index])
    if len(rows) != expected:
        raise ValueError(f"Shard row count mismatch: {len(rows)} != {expected}")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not visible")
    helper = load_runner_helper()
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    partial_protocol_path = output / "PARTIAL_RUN_PROTOCOL.json"
    amendment_record = None
    if partial_protocol_path.exists():
        partial = read_json(partial_protocol_path)
        if partial.get("freeze_sha256") != sha256(freeze_path):
            if not args.amended_from_freeze:
                raise ValueError("Partial shard belongs to another protocol; provide --amended_from_freeze")
            old_freeze_path = args.amended_from_freeze.resolve()
            old = read_json(old_freeze_path)
            invariants = {
                "old_partial_hash_matches": partial.get("freeze_sha256") == sha256(old_freeze_path),
                "same_prompts": old["output_hashes"]["remaining_prompts"] == freeze["output_hashes"]["remaining_prompts"],
                "same_model": old["model"] == freeze["model"],
                "same_system_prompt": old["system_prompt"] == freeze["system_prompt"],
                "same_generation": old["generation"] == freeze["generation"],
                "same_sharding": old["sharding"] == freeze["sharding"],
                "same_token_budgets": {k: old["budget_per_shard"][k] for k in ("max_input_tokens", "max_output_tokens")} == {k: freeze["budget_per_shard"][k] for k in ("max_input_tokens", "max_output_tokens")},
                "request_cap_not_decreased": int(freeze["budget_per_shard"]["max_requests"]) >= int(old["budget_per_shard"]["max_requests"]),
            }
            if not all(invariants.values()):
                raise ValueError(f"Unsafe protocol amendment: {invariants}")
            amendment_record = {"old_freeze": str(old_freeze_path), "old_freeze_sha256": sha256(old_freeze_path), "new_freeze": str(freeze_path), "new_freeze_sha256": sha256(freeze_path), "invariants": invariants}
            (output / "PROTOCOL_AMENDMENT.json").write_text(json.dumps(amendment_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            partial_protocol_path.write_text(json.dumps({"freeze": str(freeze_path), "freeze_sha256": sha256(freeze_path)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        existing_artifacts = any((output / name).exists() for name in ("api_request_log.jsonl", "api_usage.jsonl")) or bool(list(output.glob("structured_support_v3_scores_*.jsonl")))
        if existing_artifacts:
            if not args.amended_from_freeze:
                raise ValueError("Legacy partial shard lacks protocol marker; provide --amended_from_freeze")
            old_freeze_path = args.amended_from_freeze.resolve()
            old = read_json(old_freeze_path)
            invariants = {
                "same_prompts": old["output_hashes"]["remaining_prompts"] == freeze["output_hashes"]["remaining_prompts"],
                "same_model": old["model"] == freeze["model"],
                "same_system_prompt": old["system_prompt"] == freeze["system_prompt"],
                "same_generation": old["generation"] == freeze["generation"],
                "same_sharding": old["sharding"] == freeze["sharding"],
                "same_token_budgets": {k: old["budget_per_shard"][k] for k in ("max_input_tokens", "max_output_tokens")} == {k: freeze["budget_per_shard"][k] for k in ("max_input_tokens", "max_output_tokens")},
                "request_cap_not_decreased": int(freeze["budget_per_shard"]["max_requests"]) >= int(old["budget_per_shard"]["max_requests"]),
            }
            if not all(invariants.values()):
                raise ValueError(f"Unsafe legacy protocol amendment: {invariants}")
            amendment_record = {"old_freeze": str(old_freeze_path), "old_freeze_sha256": sha256(old_freeze_path), "new_freeze": str(freeze_path), "new_freeze_sha256": sha256(freeze_path), "invariants": invariants, "legacy_partial_without_marker": True}
            (output / "PROTOCOL_AMENDMENT.json").write_text(json.dumps(amendment_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        partial_protocol_path.write_text(json.dumps({"freeze": str(freeze_path), "freeze_sha256": sha256(freeze_path)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    score_path = output / f"structured_support_v3_scores_shard{args.shard_index:02d}_of_{args.num_shards:02d}_private.jsonl"
    request_path = output / "api_request_log.jsonl"
    usage_path = output / "api_usage.jsonl"
    manifest_path = output / "SHARD_MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError(f"Shard already complete: {manifest_path}")
    existing = read_jsonl(score_path)
    complete_ids = {row["support_state_id"] for row in existing}
    universe = {row["audit_id"] for row in rows}
    if len(complete_ids) != len(existing) or not complete_ids.issubset(universe):
        raise ValueError("Existing shard output is inconsistent")
    request_count = len(read_jsonl(request_path))
    usage_rows = read_jsonl(usage_path)
    input_tokens = sum(int(row.get("prompt_tokens", 0)) for row in usage_rows)
    output_tokens = sum(int(row.get("completion_tokens", 0)) for row in usage_rows)
    remaining = [row for row in rows if row["audit_id"] not in complete_ids]
    batch_size = int(freeze["generation"]["batch_size"])
    budget = freeze["budget_per_shard"]
    session = requests.Session()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    pending = [remaining[start : start + batch_size] for start in range(0, len(remaining), batch_size)]
    logical_batch_index = len(existing) // batch_size
    split_events = 0
    while pending:
        batch = pending.pop(0)
        batch_index = logical_batch_index
        logical_batch_index += 1
        api_batch = [
            {"audit_id": f"case_{index}", "user_prompt_private": row["user_prompt_private"]}
            for index, row in enumerate(batch)
        ]
        local_to_state = {
            api_row["audit_id"]: state_row["audit_id"]
            for api_row, state_row in zip(api_batch, batch)
        }
        body = {
            "model": freeze["model"],
            "messages": [{"role": "system", "content": freeze["system_prompt"]}, {"role": "user", "content": json.dumps({"cases": api_batch}, ensure_ascii=False)}],
            "max_completion_tokens": int(freeze["generation"]["max_completion_tokens"]),
            "reasoning_effort": freeze["generation"]["reasoning_effort"],
            "response_format": helper.schema(len(batch)),
            "store": False,
        }
        batch_finished = False
        attempt = 0
        while not batch_finished:
            attempt += 1
            if request_count >= int(budget["max_requests"]):
                raise RuntimeError("Frozen shard request budget exhausted")
            request_count += 1
            started = time.time()
            status = None
            request_id = None
            usage_recorded = False
            try:
                response = session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=args.timeout_seconds)
                status = response.status_code
                request_id = response.headers.get("x-request-id")
                if status != 200:
                    try: error_type = (response.json().get("error") or {}).get("type", "api_error")
                    except Exception: error_type = "non_json_api_error"
                    raise RuntimeError(f"OpenAI API status {status} ({error_type})")
                response_json = response.json()
                usage = response_json.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0)); completion_tokens = int(usage.get("completion_tokens", 0))
                input_tokens += prompt_tokens; output_tokens += completion_tokens
                append_jsonl(usage_path, {"batch_index": batch_index, "completion_tokens": completion_tokens, "n_items": len(batch), "prompt_tokens": prompt_tokens, "total_tokens": int(usage.get("total_tokens", prompt_tokens+completion_tokens)), "accepted": None})
                usage_recorded = True
                if input_tokens > int(budget["max_input_tokens"]) or output_tokens > int(budget["max_output_tokens"]):
                    raise RuntimeError("Frozen shard token budget exceeded")
                decisions = helper.validate_decisions(response_json, api_batch)
                append_jsonl(request_path, {"attempt": request_count, "batch_index": batch_index, "elapsed_seconds": round(time.time()-started,3), "http_status": status, "n_items": len(batch), "request_id": request_id or response_json.get("id"), "result": "success"})
                for row in decisions:
                    action_id = local_to_state[row["audit_id"]]
                    clean = {key: value for key, value in row.items() if key != "audit_id"}
                    append_jsonl(score_path, {**clean, "support_state_id": action_id, "model": response_json.get("model", freeze["model"]), "shard_index": args.shard_index})
                complete_ids.update(row["audit_id"] for row in batch)
                if len(complete_ids) % 100 == 0 or len(complete_ids) == len(rows):
                    print(json.dumps({"shard": args.shard_index, "completed": len(complete_ids), "total": len(rows), "requests": request_count, "prompt_tokens": input_tokens, "completion_tokens": output_tokens}), flush=True)
                batch_finished = True
            except Exception as exc:
                append_jsonl(request_path, {"attempt": request_count, "batch_index": batch_index, "elapsed_seconds": round(time.time()-started,3), "error": f"{type(exc).__name__}: {str(exc)[:300]}", "http_status": status, "n_items": len(batch), "request_id": request_id, "response_tokens_recorded": usage_recorded, "result": "failure"})
                validation_failure = isinstance(exc, (ValueError, json.JSONDecodeError)) and status == 200
                if validation_failure and len(batch) > 1 and attempt >= 2:
                    midpoint = len(batch) // 2
                    pending.insert(0, batch[midpoint:])
                    pending.insert(0, batch[:midpoint])
                    split_events += 1
                    batch_finished = True
                    continue
                max_attempts = 8
                if attempt >= max_attempts or request_count >= int(budget["max_requests"]):
                    raise RuntimeError(f"Shard {args.shard_index} batch {batch_index} failed") from exc
                time.sleep(min(2 ** (attempt-1),16))
    final = read_jsonl(score_path)
    if len(final) != expected or {row["support_state_id"] for row in final} != universe:
        raise ValueError("Completed shard does not cover exact frozen universe")
    requests_rows = read_jsonl(request_path)
    manifest = {
        "decision": "DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_SHARD_COMPLETE",
        "freeze_path": str(freeze_path), "freeze_sha256": sha256(freeze_path),
        "shard_index": args.shard_index, "num_shards": args.num_shards, "rows": len(final),
        "score_path": str(score_path), "score_sha256": sha256(score_path),
        "request_path": str(request_path), "request_sha256": sha256(request_path),
        "usage_path": str(usage_path), "usage_sha256": sha256(usage_path),
        "request_attempts": len(requests_rows), "successful_requests": sum(row["result"]=="success" for row in requests_rows),
        "prompt_tokens": input_tokens, "completion_tokens": output_tokens,
        "protocol_amendment": amendment_record,
        "adaptive_batch_split_events": split_events,
        "model": freeze["model"], "api_key_stored": False, "serper_calls": 0, "dev_used": False, "test_used": False, "training_run": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
