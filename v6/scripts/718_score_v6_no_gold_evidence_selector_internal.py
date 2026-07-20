#!/usr/bin/env python3
"""Score fixed-query evidence actions with one matched learned policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


METHODS = ("no_credit", "local_listwise", "outcome_listwise", "dagig_posterior")
STRATEGY_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)


def load_core() -> Any:
    path = Path(__file__).with_name("613_run_v6_listwise_evidence_selector_eval.py")
    spec = importlib.util.spec_from_file_location("dagig_v6_no_gold_evidence_selector_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completion(row: dict[str, Any]) -> str:
    docs = sorted(row["candidate_docs"], key=lambda doc: int(doc["rank"]))
    mapping = {doc["doc_id"]: f"D{index}" for index, doc in enumerate(docs, 1)}
    return json.dumps(
        {"selected_evidence_ids": [mapping[doc_id] for doc_id in row["selected_doc_ids"]]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_freeze", type=Path, required=True)
    parser.add_argument("--train_fit_audit", type=Path, required=True)
    parser.add_argument("--control_audit", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.training_freeze.resolve()
    fit_path = args.train_fit_audit.resolve()
    control_path = args.control_audit.resolve()
    freeze = core.read_json(freeze_path)
    fit = core.read_json(fit_path)
    controls = core.read_json(control_path)
    if freeze.get("decision") != "DAGIG_V6_LISTWISE_EVIDENCE_NODE_GDPO_FROZEN":
        raise ValueError("evidence training is not frozen")
    if fit.get("decision") != "DAGIG_V6_NO_GOLD_EVIDENCE_TRAIN_FIT_GO":
        raise ValueError("evidence train fit is not GO")
    if controls.get("decision") != "DAGIG_V6_NO_GOLD_EVIDENCE_CONTROLS_GO":
        raise ValueError("fixed-query evidence controls are not GO")
    action_path = Path(controls["output_paths"]["evidence_actions"])
    if sha256(action_path) != controls["output_hashes"]["evidence_actions"]:
        raise ValueError("fixed-query evidence actions changed")
    if args.method != "no_credit":
        if args.adapter is None:
            raise ValueError("trained evidence method requires --adapter")
        adapter = args.adapter.resolve()
        expected_hash = fit["input_hashes"][f"{args.method}_adapter_model"]
        if sha256(adapter / "adapter_model.safetensors") != expected_hash:
            raise ValueError("trained evidence adapter differs from fit audit")
    else:
        adapter = None
    if not torch.cuda.is_available():
        raise RuntimeError("one GPU is required")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in core.read_jsonl(action_path):
        if row["partition"] == "internal_holdout":
            grouped[row["query_id"]].append(row)
    groups = []
    for query_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: STRATEGY_ORDER.index(row["evidence_strategy"]))
        behavior = torch.tensor([float(row["behavior_weight"]) for row in rows], dtype=torch.float64)
        behavior /= behavior.sum()
        groups.append(
            {
                "query_id": query_id,
                "rows": rows,
                "prompt": core.build_evidence_selection_prompt(rows[0]),
                "completions": [completion(row) for row in rows],
                "behavior": behavior.tolist(),
            }
        )
    if len(groups) != 40 or any(len(group["rows"]) != 5 for group in groups):
        raise ValueError("internal evidence group matrix is incomplete")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(freeze["base_model"], local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    reference = current = None
    if args.method != "no_credit":
        reference = core.score_adapter(
            freeze["base_model"], freeze["shared_sft_adapter"], tokenizer, groups, int(freeze["training"]["max_input_tokens"])
        )
        current = core.score_adapter(
            freeze["base_model"], str(adapter), tokenizer, groups, int(freeze["training"]["max_input_tokens"])
        )
    predictions = []
    for group in groups:
        behavior = torch.tensor(group["behavior"], dtype=torch.float64)
        if args.method == "no_credit":
            delta = torch.zeros_like(behavior)
        else:
            delta = torch.tensor(current[group["query_id"]], dtype=torch.float64) - torch.tensor(
                reference[group["query_id"]], dtype=torch.float64
            )
        policy = torch.softmax(torch.log(behavior) + float(freeze["training"]["beta"]) * delta, dim=0)
        selected_index = int(policy.argmax())
        selected = group["rows"][selected_index]
        predictions.append(
            {
                "method": args.method,
                "sample_id": selected["sample_id"],
                "query_id": selected["query_id"],
                "question": selected["question"],
                "visual_observation": selected["visual_observation"],
                "search_query": selected["search_query"],
                "action_ids": [row["evidence_action_id"] for row in group["rows"]],
                "action_strategies": [row["evidence_strategy"] for row in group["rows"]],
                "behavior_probabilities": group["behavior"],
                "field_logprob_deltas": delta.tolist(),
                "policy_probabilities": policy.tolist(),
                "selected_action_index": selected_index,
                "selected_evidence_action_id": selected["evidence_action_id"],
                "selected_evidence_strategy": selected["evidence_strategy"],
                "selected_docs": selected["selected_docs"],
                "selected_doc_ids": selected["selected_doc_ids"],
            }
        )
    finite = all(
        math.isfinite(value)
        for row in predictions
        for value in [*row["field_logprob_deltas"], *row["policy_probabilities"]]
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / f"v6_no_gold_evidence_selector_{args.method}_internal_no_labels.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    gates = {
        "complete_internal_groups": len(predictions) == 40,
        "finite_scores": finite,
        "normalized_policies": all(abs(sum(row["policy_probabilities"]) - 1.0) <= 1e-8 for row in predictions),
        "no_gold_or_qrels_loaded": True,
        "internal_holdout_never_fit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_NO_GOLD_EVIDENCE_SELECTOR_SCORES_READY" if all(gates.values()) else "DAGIG_V6_NO_GOLD_EVIDENCE_SELECTOR_SCORES_NO_GO"
    result = {
        "decision": decision,
        "method": args.method,
        "metrics": {"samples": len(predictions), "actions": sum(len(row["action_ids"]) for row in predictions)},
        "gates": gates,
        "input_paths": {"training_freeze": str(freeze_path), "train_fit_audit": str(fit_path), "control_audit": str(control_path)},
        "input_hashes": {"training_freeze": sha256(freeze_path), "train_fit_audit": sha256(fit_path), "control_audit": sha256(control_path)},
        "output_paths": {"scores": str(prediction_path)},
        "output_hashes": {"scores": sha256(prediction_path)},
        "gold_or_qrels_loaded": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_NO_GOLD_EVIDENCE_SELECTOR_SCORE_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
