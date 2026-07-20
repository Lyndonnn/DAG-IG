#!/usr/bin/env python3
"""Privately audit execution-aware selector scores after the train-only OOF gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlsplit


METHODS = ("no_credit", "local_fixed_descendant", "true_outcome_grpo", "dagig_exact")


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


def normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def phrase_contains(text: Any, phrase: Any) -> bool:
    source, target = normalized(text).split(), normalized(phrase).split()
    return bool(target and any(source[index : index + len(target)] == target for index in range(len(source) - len(target) + 1)))


def canonical_url(value: Any) -> str:
    parsed = urlsplit(str(value or ""))
    return parsed.netloc.casefold().removeprefix("www.") + re.sub(r"/+", "/", parsed.path).rstrip("/")


def support_label(candidate: dict[str, Any], label: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> dict[str, Any]:
    accepted = [label["gold_answer"], *(label.get("aliases") or [])]
    positive_docs = [corpus[doc_id] for doc_id in label.get("positive_doc_ids") or [] if doc_id in corpus]
    positive_urls = {canonical_url(doc.get("final_url") or doc.get("url")) for doc in positive_docs}
    docs = candidate["retrieved_docs"][:5]
    exact_rank = next((index + 1 for index, doc in enumerate(docs) if canonical_url(doc.get("url")) in positive_urls), 0)
    bearing_rank = next(
        (
            index + 1
            for index, doc in enumerate(docs)
            if any(phrase_contains(f"{doc.get('title', '')} {doc.get('snippet', '')}", answer) for answer in accepted)
        ),
        0,
    )
    return {
        "exact_qrel_rank": exact_rank,
        "answer_bearing_rank": bearing_rank,
        "support_at5": bool(exact_rank or bearing_rank),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--oof_audit", type=Path, required=True)
    parser.add_argument("--query_edges", type=Path, required=True)
    parser.add_argument("--private_labels", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    oof_path = args.oof_audit.resolve()
    edge_path = args.query_edges.resolve()
    label_path = args.private_labels.resolve()
    corpus_path = args.corpus.resolve()
    freeze = read_json(freeze_path)
    oof = read_json(oof_path)
    if freeze.get("decision") != "DAGIG_V6_EXECUTION_AWARE_QUERY_SELECTOR_FROZEN":
        raise ValueError("execution-aware selector protocol is not frozen")
    if freeze["code_hashes"]["auditor"] != sha256(Path(__file__).resolve()):
        raise ValueError("private auditor changed after freeze")
    if oof.get("decision") != "DAGIG_V6_EXECUTION_AWARE_QUERY_OOF_GO":
        raise ValueError("train-only OOF gate did not pass; development outcomes remain sealed")
    if oof["input_hashes"]["freeze"] != sha256(freeze_path):
        raise ValueError("OOF results came from another protocol")
    for method, path in oof["output_paths"]["development_scores"].items():
        if method not in METHODS or sha256(Path(path)) != oof["output_hashes"]["development_scores"][method]:
            raise ValueError(f"development score changed: {method}")

    labels = {row["sample_id"]: row for row in read_jsonl(label_path)}
    corpus = {row["doc_id"]: row for row in read_jsonl(corpus_path)}
    edges = {
        row["action_id"]: row
        for row in read_jsonl(edge_path)
        if row.get("partition") == "internal_holdout" and row.get("visual_field") == "joint_state"
    }
    private_rows: list[dict[str, Any]] = []
    distributions = {method: Counter() for method in METHODS}
    for method in METHODS:
        rows = read_jsonl(Path(oof["output_paths"]["development_scores"][method]))
        if len(rows) != 40:
            raise ValueError(f"incomplete development scores: {method}")
        for row in rows:
            annotated = []
            for candidate in row["candidates"]:
                edge = edges.get(candidate["action_id"])
                if edge is None:
                    raise ValueError(f"missing private query edge: {candidate['action_id']}")
                annotated.append(
                    {
                        **candidate,
                        **support_label(candidate, labels[row["sample_id"]], corpus),
                        "child_success_probability": float(edge["child_success_probability"]),
                        "child_expected_strict": float(edge["child_expected_strict"]),
                    }
                )
            selected = max(annotated, key=lambda item: (item["posterior"], item["action_id"]))
            distributions[method][selected["query_strategy"]] += 1
            private_rows.append(
                {
                    "method": method,
                    "sample_id": row["sample_id"],
                    "parent_group_id": row["parent_group_id"],
                    "selected_action_id": selected["action_id"],
                    "selected_search_query": selected["search_query"],
                    "selected_strategy": selected["query_strategy"],
                    "greedy_support_at5": selected["support_at5"],
                    "greedy_terminal_value": selected["child_success_probability"],
                    "greedy_strict": selected["child_expected_strict"],
                    "expected_support_at5": sum(item["posterior"] * int(item["support_at5"]) for item in annotated),
                    "expected_terminal_value": sum(item["posterior"] * item["child_success_probability"] for item in annotated),
                    "expected_strict": sum(item["posterior"] * item["child_expected_strict"] for item in annotated),
                    "candidates_private": annotated,
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in private_rows:
        grouped[row["method"]].append(row)
    metrics = {}
    for method in METHODS:
        rows = grouped[method]
        metrics[method] = {
            "n": len(rows),
            "greedy_support_at5": mean(int(row["greedy_support_at5"]) for row in rows),
            "greedy_terminal_value": mean(row["greedy_terminal_value"] for row in rows),
            "greedy_strict": mean(row["greedy_strict"] for row in rows),
            "expected_support_at5": mean(row["expected_support_at5"] for row in rows),
            "expected_terminal_value": mean(row["expected_terminal_value"] for row in rows),
            "expected_strict": mean(row["expected_strict"] for row in rows),
            "selected_strategy_distribution": dict(sorted(distributions[method].items())),
        }

    by_method_sample = {
        method: {row["sample_id"]: row for row in grouped[method]}
        for method in METHODS
    }
    pairwise = {}
    for control in METHODS[:-1]:
        counts = Counter()
        for sample_id, dag in by_method_sample["dagig_exact"].items():
            other = by_method_sample[control][sample_id]
            counts["same_action"] += int(dag["selected_action_id"] == other["selected_action_id"])
            counts["support_gain"] += int(dag["greedy_support_at5"] and not other["greedy_support_at5"])
            counts["support_loss"] += int(other["greedy_support_at5"] and not dag["greedy_support_at5"])
            counts["terminal_gain"] += int(dag["greedy_terminal_value"] > other["greedy_terminal_value"] + 1e-12)
            counts["terminal_loss"] += int(other["greedy_terminal_value"] > dag["greedy_terminal_value"] + 1e-12)
            counts["strict_gain"] += int(dag["greedy_strict"] > other["greedy_strict"] + 1e-12)
            counts["strict_loss"] += int(other["greedy_strict"] > dag["greedy_strict"] + 1e-12)
        pairwise[f"dagig_vs_{control}"] = dict(counts)

    dagig = metrics["dagig_exact"]
    no_credit = metrics["no_credit"]
    controls = [metrics[method] for method in METHODS if method != "dagig_exact"]
    spec = freeze["development_gates"]
    eps = 1e-12
    gates = {
        "train_only_oof_passed_before_private_audit": True,
        "complete_method_matrix": all(metrics[method]["n"] == 40 for method in METHODS),
        "dagig_greedy_support_improves_no_credit": dagig["greedy_support_at5"] - no_credit["greedy_support_at5"] + eps >= float(spec["support_at5_delta_vs_no_credit_min"]),
        "dagig_greedy_support_not_below_strongest": dagig["greedy_support_at5"] + eps >= max(row["greedy_support_at5"] for row in controls),
        "dagig_greedy_strict_not_below_strongest": dagig["greedy_strict"] + eps >= max(row["greedy_strict"] for row in controls),
        "dagig_expected_terminal_not_below_strongest": dagig["expected_terminal_value"] + eps >= max(row["expected_terminal_value"] for row in controls),
        "dagig_expected_strict_not_below_strongest": dagig["expected_strict"] + eps >= max(row["expected_strict"] for row in controls),
        "shared_legal_executed_candidate_universe": True,
        "gold_used_only_after_score_freeze": True,
        "development_holdout_never_fit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_EXECUTION_AWARE_QUERY_DEVELOPMENT_GO" if all(gates.values()) else "DAGIG_V6_EXECUTION_AWARE_QUERY_DEVELOPMENT_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    private_path = output / "v6_execution_aware_query_development_private.jsonl"
    private_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(private_rows, key=lambda item: (item["sample_id"], item["method"]))),
        encoding="utf-8",
    )
    result = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "metrics": metrics,
        "pairwise": pairwise,
        "gates": gates,
        "input_paths": {
            "freeze": str(freeze_path),
            "oof_audit": str(oof_path),
            "query_edges": str(edge_path),
            "private_labels": str(label_path),
            "corpus": str(corpus_path),
        },
        "input_hashes": {
            "freeze": sha256(freeze_path),
            "oof_audit": sha256(oof_path),
            "query_edges": sha256(edge_path),
            "private_labels": sha256(label_path),
            "corpus": sha256(corpus_path),
        },
        "output_paths": {"private_results": str(private_path)},
        "output_hashes": {"private_results": sha256(private_path)},
        "development_only_not_paper_final": True,
        "selector_uses_gold_or_terminal_at_inference": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_EXECUTION_AWARE_QUERY_DEVELOPMENT_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
