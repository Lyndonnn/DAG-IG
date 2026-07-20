#!/usr/bin/env python3
"""Build provisional semantic-support labels and a blinded independent audit pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlsplit


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


def normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def phrase_contains(text: Any, phrase: Any) -> bool:
    source, target = normalized(text).split(), normalized(phrase).split()
    return bool(target and any(source[i : i + len(target)] == target for i in range(len(source) - len(target) + 1)))


def canonical_url(value: Any) -> str:
    parsed = urlsplit(str(value or ""))
    return parsed.netloc.casefold().removeprefix("www.") + re.sub(r"/+", "/", parsed.path).rstrip("/")


def answer_type(answer: str) -> str:
    value = normalized(answer)
    if re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", value):
        return "short_numeric"
    if "@" in answer:
        return "email"
    digits = re.sub(r"\D", "", answer)
    if len(digits) >= 7 and len(digits) / max(1, len(answer)) >= 0.45:
        return "phone_or_identifier"
    if re.search(r"\b(?:street|st|road|rd|avenue|ave|ward|ku|shi|chome|postal|zip)\b", value):
        return "address"
    if re.search(r"\b(?:am|pm|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", value):
        return "time"
    return "text_or_entity"


def sample_distinct(rows: list[dict[str, Any]], count: int, rng: random.Random, used_samples: set[str]) -> list[dict[str, Any]]:
    candidates = rows[:]
    rng.shuffle(candidates)
    selected = []
    for row in candidates:
        if row["sample_id"] in used_samples:
            continue
        selected.append(row)
        used_samples.add(row["sample_id"])
        if len(selected) == count:
            return selected
    for row in candidates:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--score_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V1_FROZEN":
        raise ValueError("Gold-aware support teacher is not frozen")
    if freeze["input_hashes"]["auditor"] != sha256(Path(__file__).resolve()):
        raise ValueError("Gold-aware support auditor changed after freeze")
    prompt_path = Path(freeze["output_paths"]["private_prompts"])
    if sha256(prompt_path) != freeze["output_hashes"]["private_prompts"]:
        raise ValueError("Private teacher prompts changed")
    prompts = {row["evidence_action_id"]: row for row in read_jsonl(prompt_path)}
    scores = {}
    manifests = []
    for directory in args.score_dirs:
        manifest = read_json(directory.resolve() / "SHARD_MANIFEST.json")
        if manifest.get("decision") != "DAGIG_V6_GOLD_AWARE_SUPPORT_TEACHER_V1_SHARD_COMPLETE":
            raise ValueError(f"Incomplete gold-aware score shard: {directory}")
        if manifest["freeze_sha256"] != sha256(freeze_path):
            raise ValueError(f"Gold-aware score shard uses another freeze: {directory}")
        score_path = Path(manifest["score_path"])
        if sha256(score_path) != manifest["score_sha256"]:
            raise ValueError(f"Gold-aware scores changed: {directory}")
        manifests.append(manifest)
        for row in read_jsonl(score_path):
            if row["evidence_action_id"] in scores:
                raise ValueError(f"Duplicate gold-aware score: {row['evidence_action_id']}")
            scores[row["evidence_action_id"]] = row
    if set(scores) != set(prompts) or sorted(item["shard_index"] for item in manifests) != list(range(manifests[0]["num_shards"])):
        raise ValueError("Gold-aware score universe is incomplete")

    evidence_actions = {row["evidence_action_id"]: row for row in read_jsonl(Path(freeze["input_paths"]["evidence_actions"]))}
    private_labels = {row["sample_id"]: row for row in read_jsonl(Path(freeze["input_paths"]["private_labels"]))}
    corpus = {row["doc_id"]: row for row in read_jsonl(args.corpus.resolve())}
    threshold = float(freeze["teacher_contract"]["hard_label_threshold"])
    labels = []
    categories = defaultdict(list)
    by_partition = Counter()
    by_answer_type = Counter()
    by_legacy_reason = Counter()
    for action_id in sorted(prompts):
        action = evidence_actions[action_id]
        private = private_labels[action["sample_id"]]
        accepted = [private["gold_answer"], *(private.get("aliases") or [])]
        positive_urls = {
            canonical_url((corpus.get(doc_id) or {}).get("final_url") or (corpus.get(doc_id) or {}).get("url"))
            for doc_id in private.get("positive_doc_ids") or []
        }
        selected_urls = {canonical_url(doc.get("url")) for doc in action["selected_docs"]}
        text = " ".join(f"{doc.get('title', '')} {doc.get('snippet', '')}" for doc in action["selected_docs"])
        url_hit = bool(positive_urls & selected_urls)
        phrase_hit = any(phrase_contains(text, answer) for answer in accepted)
        reason = "both" if url_hit and phrase_hit else "positive_url" if url_hit else "answer_phrase_only" if phrase_hit else "negative"
        probability = float(scores[action_id]["gold_aware_support_probability"])
        hard = probability >= threshold
        kind = answer_type(private["gold_answer"])
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
        by_partition[(action["partition"], hard)] += 1
        by_answer_type[(kind, hard)] += 1
        by_legacy_reason[(reason, hard)] += 1
        if action["partition"] == "policy_train":
            if reason == "both":
                categories["legacy_both"].append(row)
            if reason == "positive_url":
                categories["legacy_url_only"].append(row)
            if reason == "answer_phrase_only" and kind == "short_numeric":
                categories["short_numeric_phrase_only"].append(row)
            if reason == "answer_phrase_only" and kind != "short_numeric":
                categories["other_phrase_only"].append(row)
            if reason == "negative" and hard:
                categories["semantic_repair_legacy_negative"].append(row)
            if reason != "negative" and not hard:
                categories["semantic_reject_legacy_positive"].append(row)

    rng = random.Random("dagig_v6_gold_aware_support_independent_audit_v1")
    selected = []
    used_samples: set[str] = set()
    category_targets = {
        "legacy_both": 50,
        "legacy_url_only": 50,
        "short_numeric_phrase_only": 60,
        "other_phrase_only": 50,
        "semantic_repair_legacy_negative": 70,
        "semantic_reject_legacy_positive": 70,
    }
    for category, count in category_targets.items():
        for row in sample_distinct(categories[category], count, rng, used_samples):
            selected.append({"audit_category": category, **row})
    if len(selected) < 300:
        raise ValueError(f"Independent audit pack too small: {len(selected)}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    label_path = output / "v6_gold_aware_support_labels_provisional_private.jsonl"
    with label_path.open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit_items_path = output / "v6_gold_aware_support_independent_audit_items_blinded_private.jsonl"
    audit_key_path = output / "v6_gold_aware_support_independent_audit_key_private.jsonl"
    with audit_items_path.open("w", encoding="utf-8") as items, audit_key_path.open("w", encoding="utf-8") as key:
        for index, row in enumerate(selected):
            audit_id = f"support_audit_{index:04d}"
            prompt = prompts[row["evidence_action_id"]]
            items.write(json.dumps({
                "audit_id": audit_id,
                "system_prompt": prompt["system_prompt"],
                "user_prompt_private": prompt["user_prompt_private"],
            }, ensure_ascii=False, sort_keys=True) + "\n")
            key.write(json.dumps({
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
        "label_counts": {f"{partition}::{label}": count for (partition, label), count in sorted(by_partition.items())},
        "answer_type_label_counts": {f"{kind}::{label}": count for (kind, label), count in sorted(by_answer_type.items())},
        "legacy_reason_label_counts": {f"{reason}::{label}": count for (reason, label), count in sorted(by_legacy_reason.items())},
        "audit_category_counts": dict(sorted(Counter(row["audit_category"] for row in selected).items())),
    }
    audit = {
        "decision": "DAGIG_V6_GOLD_AWARE_SUPPORT_LABELS_PENDING_INDEPENDENT_AUDIT",
        "metrics": metrics,
        "input_paths": {"freeze": str(freeze_path), "score_dirs": [str(path.resolve()) for path in args.score_dirs], "corpus": str(args.corpus.resolve())},
        "input_hashes": {"freeze": sha256(freeze_path), "corpus": sha256(args.corpus.resolve())},
        "output_paths": {"provisional_labels": str(label_path), "blinded_audit_items": str(audit_items_path), "audit_key": str(audit_key_path)},
        "output_hashes": {"provisional_labels": sha256(label_path), "blinded_audit_items": sha256(audit_items_path), "audit_key": sha256(audit_key_path)},
        "provisional_labels_allowed_for_training_or_final_evaluation": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
    }
    audit_path = output / "DAGIG_V6_GOLD_AWARE_SUPPORT_LABEL_BUILD_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": audit["decision"], "metrics": metrics, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
