#!/usr/bin/env python3
"""Audit the legacy evidence-support label contract on policy-train only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
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
    source = normalized(text).split()
    target = normalized(phrase).split()
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


def compact_action(row: dict[str, Any], label: dict[str, Any], reason: str, semantic_logit: float | None) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "query_id": row["query_id"],
        "evidence_action_id": row["evidence_action_id"],
        "question": row["question"],
        "search_query": row["search_query"],
        "gold_answer": label["gold_answer"],
        "legacy_support_reason": reason,
        "semantic_support_logit": semantic_logit,
        "selected_docs": [
            {key: doc.get(key) for key in ("doc_id", "title", "domain", "date", "snippet", "url")}
            for doc in row["selected_docs"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_actions", type=Path, required=True)
    parser.add_argument("--legacy_support", type=Path, required=True)
    parser.add_argument("--private_labels", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--semantic_score_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "evidence_actions": args.evidence_actions.resolve(),
        "legacy_support": args.legacy_support.resolve(),
        "private_labels": args.private_labels.resolve(),
        "corpus": args.corpus.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    labels = {row["sample_id"]: row for row in read_jsonl(paths["private_labels"])}
    corpus = {row["doc_id"]: row for row in read_jsonl(paths["corpus"])}
    legacy = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(paths["legacy_support"])
        if row["partition"] == "policy_train"
    }
    semantic_scores = {}
    for directory in args.semantic_score_dirs:
        manifest = read_json(directory.resolve() / "SHARD_MANIFEST.json")
        for row in read_jsonl(Path(manifest["score_path"])):
            if row["partition"] == "policy_train":
                semantic_scores[row["query_action_id"]] = (
                    row["selected_evidence_action_id"],
                    float(row["semantic_support_logit"]),
                )

    reason_counts = Counter()
    type_counts = Counter()
    suspicious_short_numeric = []
    high_semantic_legacy_negative = []
    low_semantic_legacy_positive = []
    exact_reproduction = True
    actions = 0
    selected_actions_scored = 0
    for row in read_jsonl(paths["evidence_actions"]):
        if row["partition"] != "policy_train":
            continue
        actions += 1
        label = labels[row["sample_id"]]
        accepted = [label["gold_answer"], *(label.get("aliases") or [])]
        positive_urls = {
            canonical_url((corpus.get(doc_id) or {}).get("final_url") or (corpus.get(doc_id) or {}).get("url"))
            for doc_id in label.get("positive_doc_ids") or []
        }
        text = " ".join(f"{doc.get('title', '')} {doc.get('snippet', '')}" for doc in row["selected_docs"])
        selected_urls = {canonical_url(doc.get("url")) for doc in row["selected_docs"]}
        url_hit = bool(positive_urls & selected_urls)
        phrase_hit = any(phrase_contains(text, answer) for answer in accepted)
        legacy_value = bool(legacy[row["query_id"]][row["evidence_strategy"]])
        exact_reproduction &= legacy_value == (url_hit or phrase_hit)
        reason = "both" if url_hit and phrase_hit else "positive_url" if url_hit else "answer_phrase_only" if phrase_hit else "negative"
        reason_counts[reason] += 1
        kind = answer_type(label["gold_answer"])
        type_counts[(kind, reason)] += 1
        selected_semantic = semantic_scores.get(row["query_id"])
        semantic_logit = selected_semantic[1] if selected_semantic and selected_semantic[0] == row["evidence_action_id"] else None
        if kind == "short_numeric" and reason == "answer_phrase_only":
            suspicious_short_numeric.append(compact_action(row, label, reason, semantic_logit))
        if semantic_logit is not None:
            selected_actions_scored += 1
            score = semantic_logit
            if not legacy_value and score >= 0.75:
                high_semantic_legacy_negative.append(compact_action(row, label, reason, score))
            if legacy_value and score <= -4.0:
                low_semantic_legacy_positive.append(compact_action(row, label, reason, score))

    high_semantic_legacy_negative.sort(key=lambda row: -float(row["semantic_support_logit"]))
    low_semantic_legacy_positive.sort(key=lambda row: float(row["semantic_support_logit"]))
    suspicious_short_numeric.sort(key=lambda row: float(row["semantic_support_logit"] or 0.0))
    metrics = {
        "policy_train_evidence_actions": actions,
        "selected_query_actions_with_semantic_scores": selected_actions_scored,
        "legacy_rule_exactly_reproduced": bool(exact_reproduction),
        "legacy_support_reason_counts": dict(sorted(reason_counts.items())),
        "answer_type_by_reason": {f"{kind}::{reason}": count for (kind, reason), count in sorted(type_counts.items())},
        "short_numeric_answer_phrase_only_positives": len(suspicious_short_numeric),
        "high_semantic_legacy_negatives": len(high_semantic_legacy_negative),
        "low_semantic_legacy_positives": len(low_semantic_legacy_positive),
    }
    violations = {
        "answer_phrase_rule_has_no_entity_constraint_check": True,
        "answer_phrase_rule_has_no_time_location_constraint_check": True,
        "short_numeric_incidental_matches_possible": len(suspicious_short_numeric) > 0,
        "negative_rule_rejects_semantically_equivalent_address_translation_or_format": True,
        "legacy_label_is_not_semantic_entailment": True,
    }
    decision = "DAGIG_V6_SUPPORT_LABEL_CONTRACT_INVALID" if any(violations.values()) else "DAGIG_V6_SUPPORT_LABEL_CONTRACT_VALID"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    examples_path = output / "v6_support_label_conflicts_policy_train_private.jsonl"
    examples = []
    for category, rows in (
        ("short_numeric_answer_phrase_only", suspicious_short_numeric[:25]),
        ("high_semantic_legacy_negative", high_semantic_legacy_negative[:25]),
        ("low_semantic_legacy_positive", low_semantic_legacy_positive[:25]),
    ):
        for row in rows:
            examples.append({"category": category, **row})
    with examples_path.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "decision": decision,
        "metrics": metrics,
        "contract_violations": violations,
        "input_paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "output_paths": {"conflicts": str(examples_path)},
        "output_hashes": {"conflicts": sha256(examples_path)},
        "scope": "policy_train_only",
        "internal_used": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
    }
    audit_path = output / "DAGIG_V6_SUPPORT_LABEL_CONTRACT_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = output / "DAGIG_V6_SUPPORT_LABEL_CONTRACT_AUDIT_REPORT.md"
    lines = [
        "# DAG-IG v6 Support Label Contract Audit",
        "",
        f"Decision: `{decision}`",
        "",
        "## Scope",
        "",
        "Policy-train only. Internal holdout, dev, and test were not used.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in metrics.items())
    lines.extend([
        "",
        "## Finding",
        "",
        "The legacy support label is `positive URL match OR normalized answer phrase occurs anywhere in title/snippet`.",
        "The answer-phrase branch checks neither entity identity nor question conditions. Short numeric answers can therefore match incidental numbers in unrelated pages, while semantically equivalent addresses can be marked negative due to formatting or translation differences.",
        "This label is suitable as a loose retrieval-hit proxy, but not as semantic evidence support for calibrating `P_support` or for a paper-facing support metric.",
        "",
        "## Consequence",
        "",
        "Previous evidence/query support gates must be treated as provisional and recomputed after a frozen semantic-support label contract is established. Runtime policies remain gold-free; gold-aware judging is allowed only for private supervision/evaluation labels.",
        "",
        f"Conflict examples: `{examples_path}`",
        "",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "violations": violations, "audit": str(audit_path), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
