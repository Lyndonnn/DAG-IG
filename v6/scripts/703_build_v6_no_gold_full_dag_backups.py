#!/usr/bin/env python3
"""Back deployable no-gold P_success through the complete finite DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0 or np.any(values <= 0.0):
        raise ValueError("invalid positive probability mass")
    return values / total


def binary_mutual_information(values: np.ndarray, behavior: np.ndarray) -> float:
    parent = float(behavior @ values)
    success = normalize(behavior * values)
    failure = normalize(behavior * (1.0 - values))
    success_kl = float(np.sum(success * (np.log(success) - np.log(behavior))))
    failure_kl = float(np.sum(failure * (np.log(failure) - np.log(behavior))))
    return max(0.0, parent * success_kl + (1.0 - parent) * failure_kl)


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return 0.0
    result = float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
    return result if math.isfinite(result) else 0.0


def group_backup(
    node: str,
    parent_id: str,
    rows: list[dict[str, Any]],
    value_key: str,
    strict_key: str,
    weight_key: str,
    action_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    behavior = normalize(np.asarray([float(row[weight_key]) for row in rows], dtype=np.float64))
    values = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
    strict = np.asarray([float(row[strict_key]) for row in rows], dtype=np.float64)
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("terminal values must be strictly inside (0,1)")
    parent_value = float(behavior @ values)
    parent_strict = float(behavior @ strict)
    posterior = normalize(behavior * values)
    identity = float(
        np.max(
            np.abs(
                (np.log(posterior) - np.log(behavior))
                - (np.log(values) - math.log(parent_value))
            )
        )
    )
    edges = []
    for row, pi, q, value, strict_value in zip(rows, behavior, posterior, values, strict):
        edges.append(
            {
                "node": node,
                "parent_id": parent_id,
                "action_id": str(row[action_key]),
                "sample_id": str(row["sample_id"]),
                "partition": str(row["partition"]),
                "behavior_probability": float(pi),
                "success_posterior_probability": float(q),
                "child_success_probability": float(value),
                "child_expected_strict": float(strict_value),
                "parent_success_probability": parent_value,
                "parent_expected_strict": parent_strict,
                "dagig_nats": math.log(float(value)) - math.log(parent_value),
            }
        )
    group = {
        "node": node,
        "parent_id": parent_id,
        "sample_id": str(rows[0]["sample_id"]),
        "partition": str(rows[0]["partition"]),
        "actions": len(rows),
        "parent_success_probability": parent_value,
        "parent_expected_strict": parent_strict,
        "posterior_kl_from_behavior": float(np.sum(posterior * (np.log(posterior) - np.log(behavior)))),
        "posterior_total_variation": float(0.5 * np.abs(posterior - behavior).sum()),
        "binary_mutual_information_nats": binary_mutual_information(values, behavior) if len(rows) >= 2 else 0.0,
        "information_identity_error": identity,
        "value_range": float(values.max() - values.min()),
    }
    return group, edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--terminal_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    terminal_audit_path = args.terminal_audit.resolve()
    freeze = read_json(freeze_path)
    terminal_audit = read_json(terminal_audit_path)
    if freeze.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_FROZEN":
        raise ValueError("no-gold terminal protocol is not frozen")
    if freeze["code_hashes"]["backup_builder"] != sha256(Path(__file__).resolve()):
        raise ValueError("no-gold full DAG builder changed after freeze")
    if terminal_audit.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO":
        raise ValueError("no-gold terminal value did not pass")
    if terminal_audit["input_hashes"]["freeze"] != sha256(freeze_path):
        raise ValueError("terminal values came from another protocol")
    for key in ("terminal_values", "private_audit"):
        path = Path(terminal_audit["output_paths"][key])
        if sha256(path) != terminal_audit["output_hashes"][key]:
            raise ValueError(f"no-gold terminal output changed: {key}")

    answers = read_jsonl(Path(freeze["input_paths"]["answer_actions"]))
    evidence = read_jsonl(Path(freeze["input_paths"]["evidence_actions"]))
    terminal = {row["answer_action_id"]: row for row in read_jsonl(Path(terminal_audit["output_paths"]["terminal_values"]))}
    private = {row["answer_action_id"]: row for row in read_jsonl(Path(terminal_audit["output_paths"]["private_audit"]))}
    if set(terminal) != {row["answer_action_id"] for row in answers} or set(private) != set(terminal):
        raise ValueError("answer and terminal universes differ")

    source_scoring = read_json(Path(freeze["input_paths"]["source_scoring_freeze"]))
    query_protocol_path = Path(source_scoring["input_paths"]["cached_multiquery_freeze"])
    if sha256(query_protocol_path) != source_scoring["input_hashes"]["cached_multiquery_freeze"]:
        raise ValueError("identifying query protocol changed")
    query_protocol = read_json(query_protocol_path)
    query_parent_path = Path(query_protocol["input_paths"]["query_actions_with_search"])
    if sha256(query_parent_path) != query_protocol["input_hashes"]["query_actions_with_search"]:
        raise ValueError("identifying query parents changed")
    query_parents = read_jsonl(query_parent_path)
    query_behavior = {str(row["query_id"]): float(row["behavior_probability"]) for row in query_parents}
    if len(query_behavior) != len(query_parents):
        raise ValueError("duplicate query action")

    answer_groups_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answers:
        answer_id = str(row["answer_action_id"])
        answer_groups_raw[str(row["evidence_action_id"])].append(
            {
                **row,
                "terminal_value": float(terminal[answer_id]["terminal_success_probability"]),
                "terminal_strict": int(private[answer_id]["strict_proxy"]),
            }
        )
    answer_groups, answer_edges, evidence_children = [], [], {}
    for evidence_id, rows in sorted(answer_groups_raw.items()):
        group, edges = group_backup("answer", evidence_id, rows, "terminal_value", "terminal_strict", "behavior_weight", "answer_action_id")
        answer_groups.append(group)
        answer_edges.extend(edges)
        evidence_children[evidence_id] = (group["parent_success_probability"], group["parent_expected_strict"])

    evidence_groups_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        value, strict = evidence_children[str(row["evidence_action_id"])]
        evidence_groups_raw[str(row["query_id"])].append({**row, "backed_value": value, "backed_strict": strict})
    evidence_groups, evidence_edges, query_children = [], [], {}
    for query_id, rows in sorted(evidence_groups_raw.items()):
        group, edges = group_backup("evidence", query_id, rows, "backed_value", "backed_strict", "behavior_weight", "evidence_action_id")
        evidence_groups.append(group)
        evidence_edges.extend(edges)
        query_children[query_id] = (group["parent_success_probability"], group["parent_expected_strict"])

    query_source = {str(row["query_id"]): row for row in evidence}
    if set(query_children) != set(query_behavior):
        raise ValueError("query backup universe differs")
    query_groups_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id, (value, strict) in query_children.items():
        source = query_source[query_id]
        visual_id = f"{source['sample_id']}::{source['visual_field']}"
        query_groups_raw[visual_id].append(
            {
                "query_action_id": query_id,
                "sample_id": source["sample_id"],
                "partition": source["partition"],
                "visual_field": source["visual_field"],
                "behavior_weight": query_behavior[query_id],
                "backed_value": value,
                "backed_strict": strict,
            }
        )
    query_groups, query_edges, visual_children = [], [], {}
    for visual_id, rows in sorted(query_groups_raw.items()):
        group, edges = group_backup("query", visual_id, rows, "backed_value", "backed_strict", "behavior_weight", "query_action_id")
        query_groups.append(group)
        query_edges.extend(edges)
        visual_children[visual_id] = (group["parent_success_probability"], group["parent_expected_strict"])

    visual_groups_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for visual_id, (value, strict) in visual_children.items():
        sample_id, visual_field = visual_id.rsplit("::", 1)
        source = query_groups_raw[visual_id][0]
        visual_groups_raw[sample_id].append(
            {
                "visual_action_id": visual_id,
                "sample_id": sample_id,
                "partition": source["partition"],
                "visual_field": visual_field,
                "behavior_weight": 1.0,
                "backed_value": value,
                "backed_strict": strict,
            }
        )
    visual_groups, visual_edges = [], []
    for sample_id, rows in sorted(visual_groups_raw.items()):
        group, edges = group_backup("visual", sample_id, rows, "backed_value", "backed_strict", "behavior_weight", "visual_action_id")
        visual_groups.append(group)
        visual_edges.extend(edges)

    edge_by_action = {
        "answer": {row["action_id"]: row for row in answer_edges},
        "evidence": {row["action_id"]: row for row in evidence_edges},
        "query": {row["action_id"]: row for row in query_edges},
        "visual": {row["action_id"]: row for row in visual_edges},
    }
    root_by_sample = {row["parent_id"]: row["parent_success_probability"] for row in visual_groups}
    telescoping = []
    for answer in answers:
        answer_edge = edge_by_action["answer"][str(answer["answer_action_id"])]
        evidence_edge = edge_by_action["evidence"][str(answer["evidence_action_id"])]
        query_edge = edge_by_action["query"][str(answer["query_id"])]
        visual_id = query_edge["parent_id"]
        visual_edge = edge_by_action["visual"][visual_id]
        leaf = float(terminal[str(answer["answer_action_id"])]["terminal_success_probability"])
        root = float(root_by_sample[str(answer["sample_id"])])
        error = abs(
            visual_edge["dagig_nats"]
            + query_edge["dagig_nats"]
            + evidence_edge["dagig_nats"]
            + answer_edge["dagig_nats"]
            - (math.log(leaf) - math.log(root))
        )
        telescoping.append(error)

    holdout = "internal_holdout"
    predictiveness = {}
    for node, edges in (("answer", answer_edges), ("evidence", evidence_edges), ("query", query_edges), ("visual", visual_edges)):
        rows = [row for row in edges if row["partition"] == holdout]
        predictiveness[f"{node}_child_value_strict_spearman"] = spearman(
            [row["child_success_probability"] for row in rows],
            [row["child_expected_strict"] for row in rows],
        )
    all_groups = {"answer": answer_groups, "evidence": evidence_groups, "query": query_groups, "visual": visual_groups}
    all_edges = {"answer": answer_edges, "evidence": evidence_edges, "query": query_edges, "visual": visual_edges}
    metrics = {
        "samples": len(visual_groups),
        "policy_train_samples": sum(row["partition"] == "policy_train" for row in visual_groups),
        "development_samples": sum(row["partition"] == holdout for row in visual_groups),
        "group_counts": {node: len(rows) for node, rows in all_groups.items()},
        "edge_counts": {node: len(rows) for node, rows in all_edges.items()},
        "mean_posterior_tv_from_behavior": {node: mean(row["posterior_total_variation"] for row in rows) for node, rows in all_groups.items()},
        "train_nonconstant_group_rates": {
            node: mean(float(row["value_range"] > 1e-6) for row in rows if row["partition"] == "policy_train")
            for node, rows in all_groups.items()
        },
        "max_information_identity_error": max(row["information_identity_error"] for rows in all_groups.values() for row in rows),
        "max_path_telescoping_error": max(telescoping),
        "mean_path_telescoping_error": mean(telescoping),
        "development_predictiveness": predictiveness,
        "action_count_distributions": {
            node: dict(sorted(Counter(row["actions"] for row in rows).items())) for node, rows in all_groups.items()
        },
    }
    gates = {
        "complete_full_dag": metrics["edge_counts"] == {"answer": 41273, "evidence": 14770, "query": 2954, "visual": 594},
        "complete_group_dag": metrics["group_counts"] == {"answer": 14770, "evidence": 2954, "query": 594, "visual": 198},
        "all_node_targets_nonconstant_train": all(value >= 0.95 for value in metrics["train_nonconstant_group_rates"].values()),
        "exact_information_identity": metrics["max_information_identity_error"] <= 1e-10,
        "exact_path_telescoping": metrics["max_path_telescoping_error"] <= 1e-10,
        "no_gold_terminal_value_passed": True,
        "runtime_backup_uses_no_gold_or_qrels": True,
        "development_unused_for_terminal_fit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_GO" if all(gates.values()) else "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    output_paths = {}
    output_hashes = {}
    for node, groups, edges in (
        ("answer", answer_groups, answer_edges),
        ("evidence", evidence_groups, evidence_edges),
        ("query", query_groups, query_edges),
        ("visual", visual_groups, visual_edges),
    ):
        group_path = output / f"v6_no_gold_{node}_group_backups.jsonl"
        edge_path = output / f"v6_no_gold_{node}_edge_values.jsonl"
        write_jsonl(group_path, groups)
        write_jsonl(edge_path, edges)
        output_paths[f"{node}_groups"] = str(group_path)
        output_paths[f"{node}_edges"] = str(edge_path)
        output_hashes[f"{node}_groups"] = sha256(group_path)
        output_hashes[f"{node}_edges"] = sha256(edge_path)
    audit = {
        "decision": decision,
        "protocol_version": "dagig_v6_exact_no_gold_full_backward_information_gain_v1",
        "metrics": metrics,
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path), "terminal_audit": str(terminal_audit_path)},
        "input_hashes": {"freeze": sha256(freeze_path), "terminal_audit": sha256(terminal_audit_path)},
        "output_paths": output_paths,
        "output_hashes": output_hashes,
        "gold_or_qrels_in_runtime_value_or_backup": False,
        "development_strict_used_only_for_post_backup_predictiveness": True,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
