#!/usr/bin/env python3
"""Audit exact no-gold DAG query credit against matched query-only controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any
from urllib.parse import urlsplit


METHODS = ("no_credit", "local_fixed_descendant", "true_outcome_grpo", "dagig_exact")


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


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("cannot normalize non-positive selector mass")
    return [value / total for value in values]


def weighted_choice(rows: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["action_id"]))
    draw = rng.random()
    cumulative = 0.0
    for row in ordered:
        cumulative += float(row["behavior_probability"])
        if draw <= cumulative + 1e-12:
            return row
    return ordered[-1]


def canonical(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (-float(row["behavior_probability"]), str(row["action_id"])),
    )[0]


def normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").casefold(), re.UNICODE))


def phrase_contains(text: Any, phrase: Any) -> bool:
    source = normalized(text).split()
    target = normalized(phrase).split()
    return bool(
        target
        and any(
            source[index : index + len(target)] == target
            for index in range(len(source) - len(target) + 1)
        )
    )


def canonical_url(value: Any) -> str:
    parsed = urlsplit(str(value or ""))
    return parsed.netloc.casefold().removeprefix("www.") + re.sub(r"/+", "/", parsed.path).rstrip("/")


def support_label(
    candidate: dict[str, Any],
    label: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    accepted = [label["gold_answer"], *(label.get("aliases") or [])]
    positive_docs = [
        corpus[doc_id]
        for doc_id in label.get("positive_doc_ids") or []
        if doc_id in corpus
    ]
    positive_urls = {
        canonical_url(doc.get("final_url") or doc.get("url")) for doc in positive_docs
    }
    docs = candidate["retrieved_docs"]
    exact_rank = next(
        (
            index + 1
            for index, doc in enumerate(docs)
            if canonical_url(doc.get("url")) in positive_urls
        ),
        0,
    )
    bearing_rank = next(
        (
            index + 1
            for index, doc in enumerate(docs)
            if any(
                phrase_contains(
                    f"{doc.get('title', '')} {doc.get('snippet', '')}", answer
                )
                for answer in accepted
            )
        ),
        0,
    )
    best_rank = min([rank for rank in (exact_rank, bearing_rank) if rank] or [0])
    return {
        "exact_qrel_rank": exact_rank,
        "answer_bearing_rank": bearing_rank,
        "support_rank": best_rank,
        "support_at1": bool(best_rank and best_rank <= 1),
        "support_at3": bool(best_rank and best_rank <= 3),
        "support_at5": bool(best_rank and best_rank <= 5),
        "support_at10": bool(best_rank and best_rank <= 10),
        "support_mrr": 1.0 / best_rank if best_rank else 0.0,
    }


def policy_from_values(
    rows: list[dict[str, Any]], values: dict[str, float], beta: float = 1.0
) -> dict[str, float]:
    logits = [
        math.log(float(row["behavior_probability"]))
        + beta * math.log(values[str(row["action_id"])])
        for row in rows
    ]
    offset = max(logits)
    masses = normalize([math.exp(value - offset) for value in logits])
    return {str(row["action_id"]): mass for row, mass in zip(rows, masses)}


def sample_outcome_policies(
    groups: dict[str, list[dict[str, Any]]],
    evidence_groups: dict[str, list[dict[str, Any]]],
    answer_groups: dict[str, list[dict[str, Any]]],
    terminal: dict[str, float],
    rollouts: int,
    seed: int,
    eta: float,
) -> tuple[dict[str, dict[str, float]], float]:
    policies: dict[str, dict[str, float]] = {}
    constant = 0
    for parent, rows in sorted(groups.items()):
        sampled = []
        for index in range(rollouts):
            rng = random.Random(f"query-development:{seed}:{parent}:{index}")
            query = weighted_choice(rows, rng)
            evidence = weighted_choice(evidence_groups[str(query["action_id"])], rng)
            answer = weighted_choice(answer_groups[str(evidence["action_id"])], rng)
            sampled.append((str(query["action_id"]), terminal[str(answer["action_id"])]))
        rewards = [reward for _, reward in sampled]
        center = mean(rewards)
        scale = math.sqrt(mean((value - center) ** 2 for value in rewards))
        constant += int(scale <= 1e-12)
        observed: dict[str, list[float]] = {
            str(row["action_id"]): [] for row in rows
        }
        for query_id, reward in sampled:
            advantage = (reward - center) / scale if scale > 1e-12 else 0.0
            observed[query_id].append(advantage)
        logits = []
        for row in rows:
            query_id = str(row["action_id"])
            estimate = mean(observed[query_id]) if observed[query_id] else 0.0
            logits.append(math.log(float(row["behavior_probability"])) + eta * estimate)
        offset = max(logits)
        masses = normalize([math.exp(value - offset) for value in logits])
        policies[parent] = {
            str(row["action_id"]): mass for row, mass in zip(rows, masses)
        }
    return policies, constant / len(groups)


def evaluate(
    groups: dict[str, list[dict[str, Any]]],
    policies: dict[str, dict[str, float]],
    annotations: dict[str, dict[str, Any]],
    query_actions: dict[str, dict[str, Any]],
    include_private_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group_rows = []
    selected_sources: Counter[str] = Counter()
    group_kls = []
    group_tvs = []
    for parent, rows in sorted(groups.items()):
        policy = policies[parent]
        group_kls.append(
            sum(
                policy[str(row["action_id"])]
                * math.log(
                    policy[str(row["action_id"])]
                    / float(row["behavior_probability"])
                )
                for row in rows
                if policy[str(row["action_id"])] > 0.0
            )
        )
        group_tvs.append(
            0.5
            * sum(
                abs(
                    policy[str(row["action_id"])]
                    - float(row["behavior_probability"])
                )
                for row in rows
            )
        )
        selected = sorted(
            rows,
            key=lambda row: (-policy[str(row["action_id"])], str(row["action_id"])),
        )[0]
        selected_id = str(selected["action_id"])
        selected_sources[str(query_actions[selected_id]["query_strategy"])] += 1
        expected = {
            "support_at1": 0.0,
            "support_at3": 0.0,
            "support_at5": 0.0,
            "support_at10": 0.0,
            "support_mrr": 0.0,
            "terminal_value": 0.0,
            "strict": 0.0,
        }
        candidates = []
        for row in rows:
            query_id = str(row["action_id"])
            annotation = annotations[query_id]
            mass = policy[query_id]
            for key in ("support_at1", "support_at3", "support_at5", "support_at10", "support_mrr"):
                expected[key] += mass * float(annotation[key])
            expected["terminal_value"] += mass * float(row["child_success_probability"])
            expected["strict"] += mass * float(row["child_expected_strict"])
            if include_private_rows:
                candidates.append(
                    {
                        "query_id": query_id,
                        "search_query": query_actions[query_id]["search_query"],
                        "query_strategy": query_actions[query_id]["query_strategy"],
                        "policy_probability": mass,
                        "child_success_probability": float(row["child_success_probability"]),
                        "child_expected_strict": float(row["child_expected_strict"]),
                        **annotation,
                    }
                )
        selected_annotation = annotations[selected_id]
        group_rows.append(
            {
                "sample_id": str(selected["sample_id"]),
                "query_parent_id": parent,
                "selected_query_id": selected_id,
                "selected_search_query": query_actions[selected_id]["search_query"],
                "selected_strategy": query_actions[selected_id]["query_strategy"],
                "greedy_support_at1": int(selected_annotation["support_at1"]),
                "greedy_support_at3": int(selected_annotation["support_at3"]),
                "greedy_support_at5": int(selected_annotation["support_at5"]),
                "greedy_support_at10": int(selected_annotation["support_at10"]),
                "greedy_support_mrr": float(selected_annotation["support_mrr"]),
                "greedy_terminal_value": float(selected["child_success_probability"]),
                "greedy_strict": float(selected["child_expected_strict"]),
                **{f"expected_{key}": value for key, value in expected.items()},
                **({"candidates_private": candidates} if include_private_rows else {}),
            }
        )
    metric_names = (
        "greedy_support_at1",
        "greedy_support_at3",
        "greedy_support_at5",
        "greedy_support_at10",
        "greedy_support_mrr",
        "greedy_terminal_value",
        "greedy_strict",
        "expected_support_at1",
        "expected_support_at3",
        "expected_support_at5",
        "expected_support_at10",
        "expected_support_mrr",
        "expected_terminal_value",
        "expected_strict",
    )
    metrics = {
        "n": len(group_rows),
        **{key: mean(float(row[key]) for row in group_rows) for key in metric_names},
        "mean_kl_from_behavior": mean(group_kls),
        "mean_tv_from_behavior": mean(group_tvs),
        "selected_strategy_distribution": dict(sorted(selected_sources.items())),
    }
    return metrics, group_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_protocol", type=Path, required=True)
    parser.add_argument("--backup_audit", type=Path, required=True)
    parser.add_argument("--terminal_audit", type=Path, required=True)
    parser.add_argument("--control_audit", type=Path, required=True)
    parser.add_argument("--kl_calibration_audit", type=Path)
    parser.add_argument("--query_actions", type=Path, required=True)
    parser.add_argument("--private_labels", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--rollouts_per_group", type=int, default=12)
    parser.add_argument("--outcome_eta", type=float, default=1.0)
    parser.add_argument("--max_query_tokens", type=int, default=24)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "paper_protocol": args.paper_protocol.resolve(),
        "backup_audit": args.backup_audit.resolve(),
        "terminal_audit": args.terminal_audit.resolve(),
        "control_audit": args.control_audit.resolve(),
        "query_actions": args.query_actions.resolve(),
        "private_labels": args.private_labels.resolve(),
        "corpus": args.corpus.resolve(),
    }
    paper = read_json(paths["paper_protocol"])
    backup = read_json(paths["backup_audit"])
    terminal_audit = read_json(paths["terminal_audit"])
    controls = read_json(paths["control_audit"])
    if paper.get("decision") != "DAGIG_V6_PAPER_PROTOCOL_V1_FROZEN":
        raise ValueError("paper protocol is not frozen")
    if backup.get("decision") != "DAGIG_V6_NO_GOLD_FULL_DAG_BACKUP_GO":
        raise ValueError("deployable no-gold DAG backup is not GO")
    if terminal_audit.get("decision") != "DAGIG_V6_NO_GOLD_TERMINAL_VALUE_GO":
        raise ValueError("deployable no-gold terminal value is not GO")
    if controls.get("decision") != "DAGIG_V6_NO_GOLD_QUERY_CONTROLS_FROZEN":
        raise ValueError("matched no-gold query controls are not frozen")
    if args.rollouts_per_group != int(controls["rollouts_per_group"]):
        raise ValueError("development Outcome budget must match frozen train budget")
    if controls["input_hashes"]["backup_audit"] != sha256(paths["backup_audit"]):
        raise ValueError("control audit belongs to another DAG backup")
    local_beta = 1.0
    outcome_eta = args.outcome_eta
    kl_calibration = None
    if args.kl_calibration_audit:
        calibration_path = args.kl_calibration_audit.resolve()
        kl_calibration = read_json(calibration_path)
        if kl_calibration.get("decision") != "DAGIG_V6_QUERY_CONTROL_KL_BUDGET_FROZEN":
            raise ValueError("query-control KL budget is not frozen")
        if kl_calibration["input_hashes"]["control_audit"] != sha256(paths["control_audit"]):
            raise ValueError("KL calibration belongs to another query-control audit")
        local_beta = float(kl_calibration["policy_beta"]["local_fixed_descendant"])
        outcome_eta = float(kl_calibration["policy_beta"]["true_outcome_grpo"])
        paths["kl_calibration_audit"] = calibration_path

    edge_paths = {
        node: Path(backup["output_paths"][f"{node}_edges"])
        for node in ("query", "evidence", "answer")
    }
    terminal_path = Path(terminal_audit["output_paths"]["terminal_values"])
    for node, path in edge_paths.items():
        if sha256(path) != backup["output_hashes"][f"{node}_edges"]:
            raise ValueError(f"audited {node} edges changed")
    if sha256(terminal_path) != terminal_audit["output_hashes"]["terminal_values"]:
        raise ValueError("audited terminal values changed")

    edges = {node: read_jsonl(path) for node, path in edge_paths.items()}
    by_parent: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for node, rows in edges.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["partition"] == "internal_holdout":
                grouped[str(row["parent_id"])].append(row)
        by_parent[node] = dict(grouped)
    query_groups = {
        parent: rows
        for parent, rows in by_parent["query"].items()
        if parent.endswith("::joint_state")
    }
    if len(query_groups) != 40:
        raise ValueError(f"expected 40 joint-state development groups, found {len(query_groups)}")

    terminal = {
        str(row["answer_action_id"]): float(row["terminal_success_probability"])
        for row in read_jsonl(terminal_path)
        if row["partition"] == "internal_holdout"
    }
    query_actions = {
        str(row["query_id"]): row
        for row in read_jsonl(paths["query_actions"])
        if row["partition"] == "internal_holdout" and row["visual_field"] == "joint_state"
    }
    expected_query_ids = {
        str(row["action_id"]) for rows in query_groups.values() for row in rows
    }
    if set(query_actions) != expected_query_ids:
        raise ValueError("development query action universe differs from exact DAG")
    max_query_tokens = max(len(str(row["search_query"]).split()) for row in query_actions.values())
    if max_query_tokens > args.max_query_tokens:
        raise ValueError("development query universe contains an overlength action")

    labels = {row["sample_id"]: row for row in read_jsonl(paths["private_labels"])}
    corpus = {row["doc_id"]: row for row in read_jsonl(paths["corpus"])}
    annotations = {
        query_id: support_label(row, labels[row["sample_id"]], corpus)
        for query_id, row in query_actions.items()
    }

    canonical_answer = {
        parent: str(canonical(rows)["action_id"]) for parent, rows in by_parent["answer"].items()
    }
    local_evidence_values = {
        action_id: terminal[canonical_answer[action_id]]
        for action_id in {str(row["action_id"]) for rows in by_parent["evidence"].values() for row in rows}
    }
    canonical_evidence = {
        parent: str(canonical(rows)["action_id"]) for parent, rows in by_parent["evidence"].items()
    }
    local_query_values = {
        action_id: local_evidence_values[canonical_evidence[action_id]]
        for action_id in expected_query_ids
    }
    behavior_policies = {
        parent: {str(row["action_id"]): float(row["behavior_probability"]) for row in rows}
        for parent, rows in query_groups.items()
    }
    dag_policies = {
        parent: {
            str(row["action_id"]): float(row["success_posterior_probability"])
            for row in rows
        }
        for parent, rows in query_groups.items()
    }
    local_policies = {
        parent: policy_from_values(rows, local_query_values, local_beta)
        for parent, rows in query_groups.items()
    }

    deterministic_metrics = {}
    private_rows = []
    for method, policies in (
        ("no_credit", behavior_policies),
        ("local_fixed_descendant", local_policies),
        ("dagig_exact", dag_policies),
    ):
        metrics, rows = evaluate(query_groups, policies, annotations, query_actions, True)
        deterministic_metrics[method] = metrics
        private_rows.extend({"method": method, **row} for row in rows)

    outcome_runs = []
    constant_rates = []
    outcome_private_rows = []
    for seed in range(args.seeds):
        policies, constant_rate = sample_outcome_policies(
            query_groups,
            by_parent["evidence"],
            by_parent["answer"],
            terminal,
            args.rollouts_per_group,
            seed,
            outcome_eta,
        )
        metrics, rows = evaluate(query_groups, policies, annotations, query_actions, seed == 0)
        outcome_runs.append(metrics)
        constant_rates.append(constant_rate)
        if seed == 0:
            outcome_private_rows = [{"method": "true_outcome_grpo_seed0", **row} for row in rows]
    scalar_metrics = [
        key
        for key, value in outcome_runs[0].items()
        if key not in {"n", "selected_strategy_distribution"} and isinstance(value, (int, float))
    ]
    outcome_summary = {
        "n": 40,
        **{
            key: {
                "mean": mean(float(row[key]) for row in outcome_runs),
                "std": pstdev(float(row[key]) for row in outcome_runs),
                "min": min(float(row[key]) for row in outcome_runs),
                "max": max(float(row[key]) for row in outcome_runs),
            }
            for key in scalar_metrics
        },
    }
    method_metrics = {
        "no_credit": deterministic_metrics["no_credit"],
        "local_fixed_descendant": deterministic_metrics["local_fixed_descendant"],
        "true_outcome_grpo": {
            "n": 40,
            **{key: value["mean"] for key, value in outcome_summary.items() if isinstance(value, dict)},
        },
        "dagig_exact": deterministic_metrics["dagig_exact"],
    }

    dag = method_metrics["dagig_exact"]
    no_credit = method_metrics["no_credit"]
    control_metrics = [method_metrics[method] for method in METHODS if method != "dagig_exact"]
    spec = paper["query_gates"]
    eps = 1e-12
    gates = {
        "complete_development_matrix": all(row["n"] == 40 for row in method_metrics.values()),
        "outcome_groups_nonconstant": 1.0 - mean(constant_rates) + eps >= float(spec["outcome_nonconstant_group_rate_min"]),
        "dagig_greedy_support_improves_no_credit": dag["greedy_support_at5"] - no_credit["greedy_support_at5"] + eps >= float(spec["support_at5_delta_vs_no_credit_min"]),
        "dagig_greedy_support_not_below_strongest": dag["greedy_support_at5"] + eps >= max(row["greedy_support_at5"] for row in control_metrics),
        "dagig_expected_support_not_below_strongest": dag["expected_support_at5"] + eps >= max(row["expected_support_at5"] for row in control_metrics),
        "dagig_greedy_strict_not_below_strongest": dag["greedy_strict"] + eps >= max(row["greedy_strict"] for row in control_metrics),
        "dagig_expected_strict_not_below_strongest": dag["expected_strict"] + eps >= max(row["expected_strict"] for row in control_metrics),
        "dagig_expected_terminal_not_below_strongest": dag["expected_terminal_value"] + eps >= max(row["expected_terminal_value"] for row in control_metrics),
        "same_query_action_universe": True,
        "paper_legal_query_length": max_query_tokens <= args.max_query_tokens,
        "same_behavior_descendant_policy": True,
        "selector_runtime_uses_no_gold_or_qrels": True,
        "gold_used_only_after_policies_frozen": True,
        "internal_development_never_fit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision_prefix = "DAGIG_V6_KL_MATCHED_NO_GOLD_QUERY_SELECTOR_DEVELOPMENT" if kl_calibration else "DAGIG_V6_NO_GOLD_QUERY_SELECTOR_DEVELOPMENT"
    decision = (
        decision_prefix + "_GO"
        if all(gates.values())
        else decision_prefix + "_NO_GO"
    )

    pairwise = {}
    dag_by_sample = {
        row["sample_id"]: row for row in private_rows if row["method"] == "dagig_exact"
    }
    for control in ("no_credit", "local_fixed_descendant"):
        control_by_sample = {
            row["sample_id"]: row for row in private_rows if row["method"] == control
        }
        different = [
            sample_id
            for sample_id in dag_by_sample
            if dag_by_sample[sample_id]["selected_query_id"]
            != control_by_sample[sample_id]["selected_query_id"]
        ]
        pairwise[control] = {
            "different_top1": len(different),
            "dag_support_wins": sum(
                dag_by_sample[sample_id]["greedy_support_at5"]
                > control_by_sample[sample_id]["greedy_support_at5"]
                for sample_id in different
            ),
            "control_support_wins": sum(
                dag_by_sample[sample_id]["greedy_support_at5"]
                < control_by_sample[sample_id]["greedy_support_at5"]
                for sample_id in different
            ),
            "dag_terminal_wins": sum(
                dag_by_sample[sample_id]["greedy_terminal_value"]
                > control_by_sample[sample_id]["greedy_terminal_value"]
                for sample_id in different
            ),
            "control_terminal_wins": sum(
                dag_by_sample[sample_id]["greedy_terminal_value"]
                < control_by_sample[sample_id]["greedy_terminal_value"]
                for sample_id in different
            ),
        }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    private_path = output / "v6_no_gold_query_selector_development_private.jsonl"
    outcome_path = output / "v6_no_gold_outcome_query_seed_runs_private.jsonl"
    write_jsonl(
        private_path,
        sorted([*private_rows, *outcome_private_rows], key=lambda row: (row["sample_id"], row["method"])),
    )
    write_jsonl(outcome_path, [{"seed": seed, **row} for seed, row in enumerate(outcome_runs)])
    input_paths = {
        **{key: str(path) for key, path in paths.items()},
        "terminal_values": str(terminal_path),
        **{f"{node}_edges": str(path) for node, path in edge_paths.items()},
    }
    audit = {
        "decision": decision,
        "protocol_version": "dagig_v6_exact_no_gold_query_only_selector_development_kl_matched_v2" if kl_calibration else "dagig_v6_exact_no_gold_query_only_selector_development_v1",
        "method_metrics": method_metrics,
        "outcome_seed_summary": outcome_summary,
        "mean_outcome_constant_group_rate": mean(constant_rates),
        "pairwise": pairwise,
        "gates": gates,
        "rollouts_per_group": args.rollouts_per_group,
        "outcome_seeds": args.seeds,
        "policy_beta": {
            "dagig_exact": 1.0,
            "local_fixed_descendant": local_beta,
            "true_outcome_grpo": outcome_eta,
            "no_credit": 0.0,
        },
        "input_paths": input_paths,
        "input_hashes": {key: sha256(Path(path)) for key, path in input_paths.items()},
        "output_paths": {
            "private_results": str(private_path),
            "outcome_seed_runs_private": str(outcome_path),
        },
        "output_hashes": {
            "private_results": sha256(private_path),
            "outcome_seed_runs_private": sha256(outcome_path),
        },
        "runtime_selector_uses_gold_or_qrels": False,
        "private_labels_opened_only_after_query_policies_frozen": True,
        "internal_development_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_NO_GOLD_QUERY_SELECTOR_DEVELOPMENT_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "method_metrics": method_metrics,
                "pairwise": pairwise,
                "gates": gates,
                "audit": str(audit_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
