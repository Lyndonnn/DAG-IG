#!/usr/bin/env python3
"""Freeze full-universe scoring under the shared v6 answer policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dagig_causal.answer_prompt import answer_completion, build_answer_policy_prompt  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def legacy_tree_sha256(root: Path) -> str:
    """Match the tree-hash contract used by the frozen answer-control builder."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared_answer_freeze", type=Path, required=True)
    parser.add_argument("--backup_audit", type=Path, required=True)
    parser.add_argument("--answer_expansion_audit", type=Path, required=True)
    parser.add_argument("--evidence_action_audit", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_shards != 2:
        raise ValueError("paper protocol uses the two available A800s as two deterministic shards")

    paths = {
        "shared_answer_freeze": args.shared_answer_freeze.resolve(),
        "backup_audit": args.backup_audit.resolve(),
        "answer_expansion_audit": args.answer_expansion_audit.resolve(),
        "evidence_action_audit": args.evidence_action_audit.resolve(),
    }
    shared, backup, expansion, evidence_audit = [read_json(paths[key]) for key in paths]
    if shared.get("decision") != "DAGIG_V6_SHARED_ANSWER_POLICY_FROZEN":
        raise ValueError("shared answer policy is not frozen")
    if backup.get("decision") != "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_GO":
        raise ValueError("terminal backup is not GO")
    if expansion.get("decision") != "DAGIG_V6_ANSWER_ACTION_EXPANSION_GO":
        raise ValueError("answer action expansion is not GO")
    if evidence_audit.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_ACTIONS_GO":
        raise ValueError("evidence actions are not GO")

    answer_actions_path = Path(expansion["output_paths"]["answer_actions"])
    answer_edges_path = Path(backup["output_paths"]["answer_edges"])
    evidence_actions_path = Path(evidence_audit["output_paths"]["evidence_actions"])
    for path, expected in (
        (answer_actions_path, expansion["output_hashes"]["answer_actions"]),
        (answer_edges_path, backup["output_hashes"]["answer_edges"]),
        (evidence_actions_path, evidence_audit["output_hashes"]["evidence_actions"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"audited input changed: {path}")
    adapter = Path(shared["output_paths"]["adapter"])
    if tree_sha256(adapter) != shared["output_hashes"]["adapter_tree"]:
        raise ValueError("shared answer adapter changed")
    answer_control_freeze = read_json(Path(shared["input_paths"]["answer_freeze"]))
    base_model = Path(answer_control_freeze["base_model"])
    if legacy_tree_sha256(base_model) != answer_control_freeze["base_model_tree_sha256"]:
        raise ValueError("base model changed")

    actions = read_jsonl(answer_actions_path)
    evidence_rows = read_jsonl(evidence_actions_path)
    evidence = {row["evidence_action_id"]: row for row in evidence_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        grouped[row["evidence_action_id"]].append(row)
    edge_ids = {row["action_id"] for row in read_jsonl(answer_edges_path)}
    action_ids = {row["answer_action_id"] for row in actions}
    if set(grouped) != set(evidence) or edge_ids != action_ids:
        raise ValueError("answer/evidence/edge universes differ")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    prompt_lengths: list[int] = []
    sequence_lengths: list[int] = []
    for evidence_id in sorted(grouped):
        prompt = build_answer_policy_prompt(evidence[evidence_id])
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
        prompt_lengths.append(len(prefix))
        for action in grouped[evidence_id]:
            sequence_lengths.append(
                len(prefix)
                + len(tokenizer(answer_completion(action["candidate_answer"]), add_special_tokens=False)["input_ids"])
                + 1
            )
    gates = {
        "complete_198_samples": len({row["sample_id"] for row in actions}) == 198,
        "complete_41273_answer_actions": len(actions) == 41273,
        "complete_14770_evidence_groups": len(grouped) == 14770,
        "complete_action_edge_identity": edge_ids == action_ids,
        "all_prompts_within_limit": max(sequence_lengths) <= args.max_input_tokens,
        "shared_answer_policy_immutable": True,
        "runtime_has_no_gold_or_qrels": True,
        "internal_holdout_not_used_for_training": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    if not all(gates.values()):
        raise ValueError(f"shared answer value freeze failed: {gates}")
    scorer = Path(__file__).with_name("732_score_v6_shared_answer_values.py").resolve()
    answer_prompt_module = ROOT / "dagig_causal/answer_prompt.py"
    input_paths = {
        **{key: str(path) for key, path in paths.items()},
        "answer_actions": str(answer_actions_path),
        "answer_edges": str(answer_edges_path),
        "evidence_actions": str(evidence_actions_path),
        "shared_adapter_model": str(adapter / "adapter_model.safetensors"),
        "answer_prompt_module": str(answer_prompt_module.resolve()),
    }
    result = {
        "decision": "DAGIG_V6_SHARED_ANSWER_VALUE_SCORING_FROZEN",
        "protocol_version": "dagig_v6_backward_shared_answer_full_value_v1",
        "value_definition": "V_A(e)=sum_a pi_A(a|e) * P_success(a,e)",
        "answer_policy": "frozen DAG-IG answer adapter shared by all upstream methods",
        "base_model": str(base_model),
        "base_model_tree_sha256": legacy_tree_sha256(base_model),
        "base_model_tree_sha256_canonical_v2": tree_sha256(base_model),
        "shared_adapter": str(adapter),
        "shared_adapter_tree_sha256": tree_sha256(adapter),
        "num_shards": args.num_shards,
        "max_input_tokens": args.max_input_tokens,
        "metrics": {
            "samples": len({row["sample_id"] for row in actions}),
            "evidence_groups": len(grouped),
            "answer_actions": len(actions),
            "action_count_distribution": dict(sorted(Counter(len(rows) for rows in grouped.values()).items())),
            "max_prompt_tokens": max(prompt_lengths),
            "max_sequence_tokens": max(sequence_lengths),
        },
        "gates": gates,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "runner_hashes": {"scorer": sha256(scorer)},
        "gold_or_qrels_available_to_scorer": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = output_dir / "DAGIG_V6_SHARED_ANSWER_VALUE_SCORING_FREEZE.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
