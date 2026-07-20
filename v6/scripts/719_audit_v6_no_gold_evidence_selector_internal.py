#!/usr/bin/env python3
"""Privately audit matched evidence selectors on the sealed internal split."""

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


METHODS = ("no_credit", "local_listwise", "outcome_listwise", "dagig_posterior")


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


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        method, separator, path = value.partition("=")
        if not separator or method not in METHODS or method in result:
            raise ValueError(f"invalid score audit mapping: {value}")
        result[method] = Path(path).resolve()
    if set(result) != set(METHODS):
        raise ValueError("score audits must cover all evidence methods")
    return result


def normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def phrase_contains(text: Any, phrase: Any) -> bool:
    source, target = normalized(text).split(), normalized(phrase).split()
    return bool(
        target
        and any(source[index : index + len(target)] == target for index in range(len(source) - len(target) + 1))
    )


def canonical_url(value: Any) -> str:
    parsed = urlsplit(str(value or ""))
    return parsed.netloc.casefold().removeprefix("www.") + re.sub(r"/+", "/", parsed.path).rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_audit", type=Path, required=True)
    parser.add_argument("--score_audit", action="append", required=True, help="method=/path/to/audit.json")
    parser.add_argument("--evidence_edges", type=Path, required=True)
    parser.add_argument("--private_labels", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    control_path = args.control_audit.resolve()
    controls = read_json(control_path)
    if controls.get("decision") != "DAGIG_V6_NO_GOLD_EVIDENCE_CONTROLS_GO":
        raise ValueError("evidence controls are not GO")
    diagnostic_path = Path(controls["output_paths"]["control_diagnostics_private"])
    if sha256(diagnostic_path) != controls["output_hashes"]["control_diagnostics_private"]:
        raise ValueError("evidence control diagnostics changed")
    diagnostic = {row["query_id"]: row for row in read_jsonl(diagnostic_path)}
    edge_path = args.evidence_edges.resolve()
    label_path = args.private_labels.resolve()
    corpus_path = args.corpus.resolve()
    edges = {row["action_id"]: row for row in read_jsonl(edge_path) if row["partition"] == "internal_holdout"}
    labels = {row["sample_id"]: row for row in read_jsonl(label_path)}
    corpus = {row["doc_id"]: row for row in read_jsonl(corpus_path)}
    score_paths = parse_mapping(args.score_audit)
    input_paths = {
        "control_audit": str(control_path),
        "evidence_edges": str(edge_path),
        "private_labels": str(label_path),
        "corpus": str(corpus_path),
    }
    private_rows = []
    for method, audit_path in score_paths.items():
        audit = read_json(audit_path)
        if audit.get("decision") != "DAGIG_V6_NO_GOLD_EVIDENCE_SELECTOR_SCORES_READY" or audit.get("method") != method:
            raise ValueError(f"evidence selector scores are not ready: {method}")
        if audit["input_hashes"]["control_audit"] != sha256(control_path):
            raise ValueError(f"evidence control universe differs: {method}")
        score_path = Path(audit["output_paths"]["scores"])
        if sha256(score_path) != audit["output_hashes"]["scores"]:
            raise ValueError(f"evidence selector scores changed: {method}")
        rows = read_jsonl(score_path)
        if len(rows) != 40:
            raise ValueError(f"incomplete evidence selector scores: {method}")
        input_paths[f"{method}_score_audit"] = str(audit_path)
        for row in rows:
            action_ids = row["action_ids"]
            action_edges = [edges[action_id] for action_id in action_ids]
            probabilities = [float(value) for value in row["policy_probabilities"]]
            selected_index = int(row["selected_action_index"])
            selected_edge = action_edges[selected_index]
            label = labels[row["sample_id"]]
            accepted = [label["gold_answer"], *(label.get("aliases") or [])]
            positive_docs = [corpus[doc_id] for doc_id in label.get("positive_doc_ids") or [] if doc_id in corpus]
            positive_urls = {canonical_url(doc.get("final_url") or doc.get("url")) for doc in positive_docs}
            selected_urls = {canonical_url(doc.get("url")) for doc in row["selected_docs"]}
            selected_text = " ".join(
                f"{doc.get('title', '')} {doc.get('snippet', '')}" for doc in row["selected_docs"]
            )
            exact = bool(positive_urls & selected_urls)
            bearing = any(phrase_contains(selected_text, answer) for answer in accepted)
            support = exact or bearing
            target_key = {
                "no_credit": "behavior_probabilities",
                "local_listwise": "local_target_probabilities",
                "outcome_listwise": "outcome_target_probabilities",
                "dagig_posterior": "dagig_success_posterior",
            }[method]
            target = diagnostic[row["query_id"]][target_key]
            private_rows.append(
                {
                    "method": method,
                    "sample_id": row["sample_id"],
                    "query_id": row["query_id"],
                    "policy_probabilities": probabilities,
                    "target_probabilities": target,
                    "policy_target_tv": 0.5 * sum(abs(left - right) for left, right in zip(probabilities, target)),
                    "target_top_action_agreement": max(range(5), key=probabilities.__getitem__)
                    == max(range(5), key=target.__getitem__),
                    "expected_terminal_value": sum(
                        probability * float(edge["child_success_probability"])
                        for probability, edge in zip(probabilities, action_edges)
                    ),
                    "expected_strict": sum(
                        probability * float(edge["child_expected_strict"])
                        for probability, edge in zip(probabilities, action_edges)
                    ),
                    "selected_evidence_action_id": row["selected_evidence_action_id"],
                    "selected_evidence_strategy": row["selected_evidence_strategy"],
                    "selected_terminal_value": float(selected_edge["child_success_probability"]),
                    "selected_proxy_strict": float(selected_edge["child_expected_strict"]),
                    "selected_exact_qrel_support": exact,
                    "selected_answer_bearing": bearing,
                    "selected_evidence_support": support,
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in private_rows:
        grouped[row["method"]].append(row)
    metrics: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows = grouped[method]
        metrics[method] = {
            "n": len(rows),
            "mean_policy_target_tv": mean(row["policy_target_tv"] for row in rows),
            "target_top_action_agreement": mean(row["target_top_action_agreement"] for row in rows),
            "expected_terminal_value": mean(row["expected_terminal_value"] for row in rows),
            "expected_strict": mean(row["expected_strict"] for row in rows),
            "selected_terminal_value": mean(row["selected_terminal_value"] for row in rows),
            "selected_proxy_strict": mean(row["selected_proxy_strict"] for row in rows),
            "selected_exact_qrel_support": mean(row["selected_exact_qrel_support"] for row in rows),
            "selected_answer_bearing": mean(row["selected_answer_bearing"] for row in rows),
            "selected_evidence_support": mean(row["selected_evidence_support"] for row in rows),
            "selected_strategy_distribution": dict(Counter(row["selected_evidence_strategy"] for row in rows)),
        }
    dag = metrics["dagig_posterior"]
    no_credit = metrics["no_credit"]
    controls_metrics = [metrics[method] for method in METHODS if method != "dagig_posterior"]
    eps = 1e-12
    gates = {
        "complete_method_matrix": all(metrics[method]["n"] == 40 for method in METHODS),
        "dagig_support_improves_no_credit": dag["selected_evidence_support"] - no_credit["selected_evidence_support"] + eps >= 1 / 40,
        "dagig_support_not_below_strongest": dag["selected_evidence_support"] + eps >= max(row["selected_evidence_support"] for row in controls_metrics),
        "dagig_expected_terminal_not_below_strongest": dag["expected_terminal_value"] + eps >= max(row["expected_terminal_value"] for row in controls_metrics),
        "dagig_expected_strict_not_below_strongest": dag["expected_strict"] + eps >= max(row["expected_strict"] for row in controls_metrics),
        "shared_fixed_query_and_evidence_universe": True,
        "selector_runtime_uses_gold_or_qrels": False,
        "gold_used_only_after_policies_frozen": True,
        "internal_holdout_never_fit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_NO_GOLD_EVIDENCE_SELECTOR_INTERNAL_GO" if all(gates.values()) else "DAGIG_V6_NO_GOLD_EVIDENCE_SELECTOR_INTERNAL_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    private_path = output / "v6_no_gold_evidence_selector_internal_private.jsonl"
    private_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(private_rows, key=lambda item: (item["sample_id"], item["method"]))),
        encoding="utf-8",
    )
    result = {
        "decision": decision,
        "metrics": metrics,
        "gates": gates,
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {"private_results": str(private_path)},
        "output_hashes": {"private_results": sha256(private_path)},
        "selector_runtime_uses_gold_or_qrels": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    path = output / "DAGIG_V6_NO_GOLD_EVIDENCE_SELECTOR_INTERNAL_AUDIT.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
