#!/usr/bin/env python3
"""Freeze full deduplicated support labeling after the v3 pilot GO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pilot_helper() -> Any:
    path = Path(__file__).with_name("834_freeze_v6_structured_support_teacher_pilot_v3.py")
    spec = importlib.util.spec_from_file_location("dagig_v3_pilot_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def state_key(sample_id: str, doc_ids: list[str]) -> str:
    raw = json.dumps([sample_id, [str(value) for value in doc_ids]], ensure_ascii=False, separators=(",", ":"))
    return "support_state_v3_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_freeze", type=Path, required=True)
    parser.add_argument("--pilot_audit", type=Path, required=True)
    parser.add_argument("--pilot_teacher_dir", type=Path, required=True)
    parser.add_argument("--evidence_actions", type=Path, required=True)
    parser.add_argument("--private_labels", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: path.resolve() for name, path in {
        "pilot_freeze": args.pilot_freeze,
        "pilot_audit": args.pilot_audit,
        "pilot_teacher_dir": args.pilot_teacher_dir,
        "evidence_actions": args.evidence_actions,
        "private_labels": args.private_labels,
        "scorer": args.scorer,
        "builder": args.builder,
        "pilot_helper": Path(__file__).with_name("834_freeze_v6_structured_support_teacher_pilot_v3.py"),
        "scorer_helper": Path(__file__).with_name("835_run_v6_structured_support_teacher_pilot_v3.py"),
        "answer_type_helper": Path(__file__).with_name("826_build_audit_v6_gold_aware_support_labels_v1.py"),
    }.items()}
    for name, path in paths.items():
        if name != "pilot_teacher_dir" and not path.is_file():
            raise FileNotFoundError(path)
        if name == "pilot_teacher_dir" and not path.is_dir():
            raise FileNotFoundError(path)
    pilot_freeze = read_json(paths["pilot_freeze"])
    pilot_audit = read_json(paths["pilot_audit"])
    if pilot_freeze.get("decision") != "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_FROZEN" or pilot_audit.get("decision") != "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_GO":
        raise ValueError("Structured support v3 pilot has not passed")
    teacher_manifest_path = paths["pilot_teacher_dir"] / "RUN_MANIFEST.json"
    teacher_manifest = read_json(teacher_manifest_path)
    if teacher_manifest.get("decision") != "DAGIG_V6_STRUCTURED_SUPPORT_TEACHER_PILOT_V3_ROLE_COMPLETE" or teacher_manifest.get("role") != "teacher":
        raise ValueError("Pilot teacher run is incomplete")
    if teacher_manifest["freeze_sha256"] != sha256(paths["pilot_freeze"]):
        raise ValueError("Pilot teacher and pilot freeze mismatch")
    teacher_decision_path = Path(teacher_manifest["decisions_path"])
    if sha256(teacher_decision_path) != teacher_manifest["decisions_sha256"]:
        raise ValueError("Pilot teacher decisions changed")
    teacher_by_audit = {row["audit_id"]: row for row in read_jsonl(teacher_decision_path)}
    pilot_key_path = Path(pilot_freeze["output_paths"]["private_key"])
    pilot_key = read_jsonl(pilot_key_path)
    if len(pilot_key) != 400 or set(teacher_by_audit) != {row["audit_id"] for row in pilot_key}:
        raise ValueError("Pilot seed universe mismatch")

    helper = load_pilot_helper()
    actions = read_jsonl(paths["evidence_actions"])
    private = {row["sample_id"]: row for row in read_jsonl(paths["private_labels"])}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        grouped[state_key(action["sample_id"], action["selected_doc_ids"])].append(action)
    state_rows = []
    for support_state_id, members in sorted(grouped.items()):
        representative = sorted(members, key=lambda row: row["evidence_action_id"])[0]
        label = private[representative["sample_id"]]
        aliases = label.get("aliases") or []
        prompt = "\n\n".join([
            f"Question:\n{str(representative['question']).strip()}",
            f"Private reference answer:\n{str(label['gold_answer']).strip()}",
            f"Equivalent aliases:\n{'; '.join(str(value) for value in aliases) if aliases else 'none'}",
            f"Visual context:\n{str(representative['visual_observation']).strip()}",
            f"Executed search query:\n{str(representative['search_query']).strip()}",
            f"Selected evidence:\n{helper.evidence_text(representative['selected_docs'])}",
        ])
        state_rows.append({
            "support_state_id": support_state_id,
            "sample_id": representative["sample_id"],
            "partition": representative["partition"],
            "representative_evidence_action_id": representative["evidence_action_id"],
            "evidence_action_ids": sorted(row["evidence_action_id"] for row in members),
            "selected_doc_ids": representative["selected_doc_ids"],
            "user_prompt_private": prompt,
        })
    by_state = {row["support_state_id"]: row for row in state_rows}
    seed_rows = []
    for key_row in pilot_key:
        support_state_id = state_key(key_row["sample_id"], key_row["selected_doc_ids"])
        if support_state_id not in by_state:
            raise ValueError(f"Pilot state missing from full universe: {support_state_id}")
        decision = teacher_by_audit[key_row["audit_id"]]
        seed_rows.append({
            "support_state_id": support_state_id,
            "seed_audit_id": key_row["audit_id"],
            "seed_source": "structured_support_v3_pilot_teacher_go",
            **{key: decision[key] for key in ("supported", "supporting_doc_indices", "supporting_span", "entailment_type", "derivation", "conflict_present", "reason", "citation_valid", "model")},
        })
    if len({row["support_state_id"] for row in seed_rows}) != 400:
        raise ValueError("Pilot seeds are not 400 unique full states")
    seed_ids = {row["support_state_id"] for row in seed_rows}
    remaining = [row for row in state_rows if row["support_state_id"] not in seed_ids]
    partitions = Counter(row["partition"] for row in state_rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state_map_path = output / "v6_structured_support_v3_full_state_map_private.jsonl"
    prompt_path = output / "v6_structured_support_v3_remaining_prompts_private.jsonl"
    seed_path = output / "v6_structured_support_v3_pilot_seed_decisions_private.jsonl"
    with state_map_path.open("w", encoding="utf-8") as handle:
        for row in state_rows:
            public = {key: value for key, value in row.items() if key != "user_prompt_private"}
            handle.write(json.dumps(public, sort_keys=True) + "\n")
    with prompt_path.open("w", encoding="utf-8") as handle:
        for row in remaining:
            handle.write(json.dumps({"audit_id": row["support_state_id"], "user_prompt_private": row["user_prompt_private"]}, ensure_ascii=False, sort_keys=True) + "\n")
    with seed_path.open("w", encoding="utf-8") as handle:
        for row in sorted(seed_rows, key=lambda value: value["support_state_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    num_shards = 4
    shard_rows = [sum(index % num_shards == shard for index in range(len(remaining))) for shard in range(num_shards)]
    gates = {
        "exact_14770_actions": len(actions) == 14770,
        "deduplicated_state_universe_nonempty": len(state_rows) > 10000,
        "exact_400_pilot_go_seeds": len(seed_rows) == 400,
        "all_actions_mapped_once": sum(len(row["evidence_action_ids"]) for row in state_rows) == len(actions),
        "same_teacher_model_as_pilot": pilot_freeze["models"]["teacher"] == "gpt-5.4-mini-2026-03-17",
        "same_system_prompt_as_pilot": helper.SYSTEM_PROMPT == pilot_freeze["system_prompt"],
        "same_generation_contract_as_pilot": pilot_freeze["generation"]["reasoning_effort"]["teacher"] == "low",
        "private_prompts_not_runtime_features": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    protocol = {
        "decision": "DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_FROZEN" if all(gates.values()) else "DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_NO_GO",
        "protocol_version": "dagig_v6_structured_cited_support_labels_full_v3",
        "model": pilot_freeze["models"]["teacher"],
        "system_prompt": pilot_freeze["system_prompt"],
        "generation": {
            "batch_size": 10,
            "max_completion_tokens": pilot_freeze["generation"]["max_completion_tokens"],
            "reasoning_effort": pilot_freeze["generation"]["reasoning_effort"]["teacher"],
            "response_format": "strict_json_schema",
        },
        "sharding": {"num_shards": num_shards, "remaining_rows_per_shard": shard_rows},
        "budget_per_shard": {"max_requests": 700, "max_input_tokens": 2500000, "max_output_tokens": 750000},
        "counts": {"actions": len(actions), "deduplicated_states": len(state_rows), "pilot_seed_states": len(seed_rows), "remaining_api_states": len(remaining), "partitions": dict(sorted(partitions.items()))},
        "input_paths": {**{name: str(path) for name, path in paths.items() if name != "pilot_teacher_dir"}, "pilot_teacher_manifest": str(teacher_manifest_path), "pilot_teacher_decisions": str(teacher_decision_path)},
        "input_hashes": {**{name: sha256(path) for name, path in paths.items() if name != "pilot_teacher_dir"}, "pilot_teacher_manifest": sha256(teacher_manifest_path), "pilot_teacher_decisions": sha256(teacher_decision_path)},
        "output_paths": {"state_map": str(state_map_path), "remaining_prompts": str(prompt_path), "pilot_seed_decisions": str(seed_path)},
        "output_hashes": {"state_map": sha256(state_map_path), "remaining_prompts": sha256(prompt_path), "pilot_seed_decisions": sha256(seed_path)},
        "gates": gates,
        "labels_allowed_for_runtime_policy_features": False,
        "serper_calls": 0,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    freeze_path = output / "DAGIG_V6_STRUCTURED_SUPPORT_LABELS_FULL_V3_FREEZE.json"
    freeze_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": protocol["decision"], "counts": protocol["counts"], "sharding": protocol["sharding"], "gates": gates, "freeze": str(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
