#!/usr/bin/env python3
"""Build provisional v2 labels and a blinded independent support audit pack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_helpers() -> Any:
    path = Path(__file__).with_name("826_build_audit_v6_gold_aware_support_labels_v1.py")
    spec = importlib.util.spec_from_file_location("dagig_support_label_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--score_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    helper = load_helpers()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V2_FROZEN":
        raise ValueError("Gold-aware support teacher v2 is not frozen")
    if freeze["input_hashes"]["auditor"] != sha256(Path(__file__).resolve()):
        raise ValueError("v2 label builder changed after freeze")
    helper_path = Path(freeze["input_paths"]["auditor_helper"])
    if sha256(helper_path) != freeze["input_hashes"]["auditor_helper"]:
        raise ValueError("v2 label-builder helper changed after freeze")
    prompt_path = Path(freeze["output_paths"]["private_prompts"])
    if sha256(prompt_path) != freeze["output_hashes"]["private_prompts"]:
        raise ValueError("Frozen v2 private prompts changed")
    prompts = {row["evidence_action_id"]: row for row in read_jsonl(prompt_path)}
    scores, manifests = {}, []
    for directory in args.score_dirs:
        manifest = read_json(directory.resolve() / "SHARD_MANIFEST.json")
        if manifest.get("decision") != "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V2_SHARD_COMPLETE":
            raise ValueError(f"Incomplete v2 shard: {directory}")
        if manifest["freeze_sha256"] != sha256(freeze_path):
            raise ValueError(f"v2 shard uses another freeze: {directory}")
        score_path = Path(manifest["score_path"])
        if sha256(score_path) != manifest["score_sha256"]:
            raise ValueError(f"v2 score file changed: {directory}")
        manifests.append(manifest)
        for row in read_jsonl(score_path):
            if row["evidence_action_id"] in scores:
                raise ValueError(f"Duplicate v2 score: {row['evidence_action_id']}")
            scores[row["evidence_action_id"]] = row
    if set(scores) != set(prompts) or sorted(row["shard_index"] for row in manifests) != list(range(manifests[0]["num_shards"])):
        raise ValueError("v2 score universe incomplete")
    actions = {row["evidence_action_id"]: row for row in read_jsonl(Path(freeze["input_paths"]["evidence_actions"]))}
    private = {row["sample_id"]: row for row in read_jsonl(Path(freeze["input_paths"]["private_labels"]))}
    corpus = {row["doc_id"]: row for row in read_jsonl(args.corpus.resolve())}
    threshold = float(freeze["teacher_contract"]["hard_label_threshold"])
    labels, categories = [], defaultdict(list)
    counts = Counter()
    for action_id in sorted(prompts):
        action, reference = actions[action_id], private[actions[action_id]["sample_id"]]
        accepted = [reference["gold_answer"], *(reference.get("aliases") or [])]
        positive_urls = {
            helper.canonical_url((corpus.get(doc_id) or {}).get("final_url") or (corpus.get(doc_id) or {}).get("url"))
            for doc_id in reference.get("positive_doc_ids") or []
        }
        selected_urls = {helper.canonical_url(doc.get("url")) for doc in action["selected_docs"]}
        text = " ".join(f"{doc.get('title', '')} {doc.get('snippet', '')}" for doc in action["selected_docs"])
        url_hit = bool(positive_urls & selected_urls)
        phrase_hit = any(helper.phrase_contains(text, answer) for answer in accepted)
        reason = "both" if url_hit and phrase_hit else "positive_url" if url_hit else "answer_phrase_only" if phrase_hit else "negative"
        probability = float(scores[action_id]["gold_aware_support_probability"])
        hard = probability >= threshold
        kind = helper.answer_type(reference["gold_answer"])
        row = {
            "evidence_action_id": action_id,
            "query_id": action["query_id"],
            "parent_visual_state_id": prompts[action_id]["parent_visual_state_id"],
            "sample_id": action["sample_id"],
            "partition": action["partition"],
            "gold_aware_support_probability": probability,
            "gold_aware_support_label": bool(hard),
            "gold_aware_support_logit": float(scores[action_id]["gold_aware_support_logit"]),
            "legacy_support_reason": reason,
            "answer_type": kind,
            "label_status": "provisional_pending_independent_audit",
        }
        labels.append(row)
        counts[(action["partition"], hard)] += 1
        counts[(kind, hard)] += 1
        if action["partition"] == "policy_train":
            if reason == "both": categories["legacy_both"].append(row)
            if reason == "positive_url": categories["legacy_url_only"].append(row)
            if reason == "answer_phrase_only" and kind == "short_numeric": categories["short_numeric_phrase_only"].append(row)
            if reason == "answer_phrase_only" and kind != "short_numeric": categories["other_phrase_only"].append(row)
            if reason == "negative" and hard: categories["semantic_repair_legacy_negative"].append(row)
            if reason != "negative" and not hard: categories["semantic_reject_legacy_positive"].append(row)
    rng = random.Random("dagig_v6_gold_aware_support_independent_audit_v2")
    selected, used_samples = [], set()
    targets = {"legacy_both": 50, "legacy_url_only": 50, "short_numeric_phrase_only": 60, "other_phrase_only": 50, "semantic_repair_legacy_negative": 70, "semantic_reject_legacy_positive": 70}
    for category, count in targets.items():
        for row in helper.sample_distinct(categories[category], count, rng, used_samples):
            selected.append({"audit_category": category, **row})
    if len(selected) < 300:
        raise ValueError(f"v2 independent audit pack too small: {len(selected)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    label_path = output / "v6_gold_aware_support_v2_labels_provisional_private.jsonl"
    item_path = output / "v6_gold_aware_support_v2_audit_items_blinded_private.jsonl"
    key_path = output / "v6_gold_aware_support_v2_audit_key_private.jsonl"
    with label_path.open("w", encoding="utf-8") as handle:
        for row in labels: handle.write(json.dumps(row, sort_keys=True) + "\n")
    with item_path.open("w", encoding="utf-8") as items, key_path.open("w", encoding="utf-8") as keys:
        for index, row in enumerate(selected):
            audit_id = f"support_v2_audit_{index:04d}"
            prompt = prompts[row["evidence_action_id"]]
            items.write(json.dumps({"audit_id": audit_id, "system_prompt": prompt["system_prompt"], "user_prompt_private": prompt["user_prompt_private"]}, ensure_ascii=False, sort_keys=True) + "\n")
            keys.write(json.dumps({
                "audit_id": audit_id,
                "audit_category": row["audit_category"],
                "evidence_action_id": row["evidence_action_id"],
                "sample_id": row["sample_id"],
                "local_probability": row["gold_aware_support_probability"],
                "local_label": row["gold_aware_support_label"],
                "answer_type": row["answer_type"],
                "legacy_support_reason": row["legacy_support_reason"],
            }, sort_keys=True) + "\n")
    metrics = {
        "actions": len(labels),
        "policy_train_actions": sum(row["partition"] == "policy_train" for row in labels),
        "internal_actions_scored_without_tuning": sum(row["partition"] == "internal_holdout" for row in labels),
        "independent_audit_items": len(selected),
        "policy_train_positive_rate": mean(float(row["gold_aware_support_label"]) for row in labels if row["partition"] == "policy_train"),
        "internal_positive_rate": mean(float(row["gold_aware_support_label"]) for row in labels if row["partition"] == "internal_holdout"),
        "audit_category_counts": dict(sorted(Counter(row["audit_category"] for row in selected).items())),
    }
    audit = {
        "decision": "DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_V2_PENDING_INDEPENDENT_AUDIT",
        "metrics": metrics,
        "input_paths": {"freeze": str(freeze_path), "score_dirs": [str(path.resolve()) for path in args.score_dirs], "corpus": str(args.corpus.resolve())},
        "input_hashes": {"freeze": sha256(freeze_path), "corpus": sha256(args.corpus.resolve())},
        "output_paths": {"provisional_labels": str(label_path), "blinded_audit_items": str(item_path), "audit_key": str(key_path)},
        "output_hashes": {"provisional_labels": sha256(label_path), "blinded_audit_items": sha256(item_path), "audit_key": sha256(key_path)},
        "provisional_labels_allowed_for_training_or_final_evaluation": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
    }
    audit_path = output / "DAGIG_V6_GOLD_AWARE_SUPPORT_LABEL_V2_BUILD_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": audit["decision"], "metrics": metrics, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
