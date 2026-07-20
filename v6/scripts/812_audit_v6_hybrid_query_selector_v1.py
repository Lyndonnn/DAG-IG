#!/usr/bin/env python3
"""Audit hybrid-backed query selector on the fixed development partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


METHODS = ("no_credit", "local_ig_m", "local_observable", "outcome", "dagig")


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
    parser.add_argument("--private_support", type=Path, required=True)
    parser.add_argument("--terminal_private", type=Path, required=True)
    parser.add_argument("--shared_answer_values", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_HYBRID_QUERY_VALUES_V1_FROZEN":
        raise ValueError("hybrid query values are not frozen")
    if freeze["auditor_hash"] != sha256(Path(__file__).resolve()):
        raise ValueError("hybrid query auditor changed after freeze")
    for key, raw_path in freeze["output_paths"].items():
        if sha256(Path(raw_path)) != freeze["output_hashes"][key]:
            raise ValueError(f"hybrid query output changed: {key}")
    public = read_jsonl(Path(freeze["output_paths"]["internal_targets"]))
    diagnostics = {row["parent_state_id"]: row for row in read_jsonl(Path(freeze["output_paths"]["diagnostics"])) if row["partition"] == "internal_holdout"}
    support_map = {row["query_id"]: row["strategy_support"] for row in read_jsonl(args.private_support.resolve()) if row["partition"] == "internal_holdout"}
    terminal = {row["answer_action_id"]: row for row in read_jsonl(args.terminal_private.resolve()) if row["partition"] == "internal_holdout"}
    shared = {row["evidence_action_id"]: row for row in read_jsonl(args.shared_answer_values.resolve())}
    if len(public) != 120 or len(diagnostics) != 120:
        raise ValueError("expected 120 internal visual states")
    rows = []
    for target in public:
        diagnostic = diagnostics[target["parent_state_id"]]
        actions = []
        for index, query_id in enumerate(diagnostic["query_action_ids"]):
            evidence_id = diagnostic["selected_evidence_action_ids"][index]
            value = shared[evidence_id]
            strategy = evidence_id.rsplit("::", 1)[-1]
            probabilities = [float(item) for item in value["answer_policy_probabilities"]]
            strict = [float(terminal[answer_id]["strict_proxy"]) for answer_id in value["answer_action_ids"]]
            mode = value["answer_action_ids"].index(value["mode_answer_action_id"])
            actions.append({
                "query_action_id": query_id,
                "query_strategy": query_id.rsplit("::", 1)[-1],
                "evidence_action_id": evidence_id,
                "support": float(support_map[query_id][strategy]),
                "expected_strict": sum(p * label for p, label in zip(probabilities, strict)),
                "mode_strict": strict[mode],
                "hybrid_value": diagnostic["hybrid_query_values"][index],
            })
        selected = {}
        for method in METHODS:
            posterior = target["target_distributions"][method]
            choice = max(range(len(posterior)), key=lambda index: (float(posterior[index]), -index))
            selected[method] = actions[choice]
        rows.append({"parent_state_id": target["parent_state_id"], "sample_id": target["parent_state_id"].split("::", 1)[0], "methods": selected})
    summary = {}
    for method in METHODS:
        chosen = [row["methods"][method] for row in rows]
        summary[method] = {
            "states": len(chosen),
            "samples": len({row["sample_id"] for row in rows}),
            "support": mean(row["support"] for row in chosen),
            "expected_strict": mean(row["expected_strict"] for row in chosen),
            "mode_strict": mean(row["mode_strict"] for row in chosen),
            "hybrid_value": mean(row["hybrid_value"] for row in chosen),
            "query_strategy_distribution": dict(sorted(Counter(row["query_strategy"] for row in chosen).items())),
        }
    disagreement = mean(row["methods"]["dagig"]["query_action_id"] != row["methods"]["outcome"]["query_action_id"] for row in rows)
    threshold = freeze["development_gates"]
    dag, no_credit, local, outcome = summary["dagig"], summary["no_credit"], summary["local_ig_m"], summary["outcome"]
    gates = {
        "complete_120_internal_visual_states": len(rows) == 120,
        "complete_40_internal_samples": len({row["sample_id"] for row in rows}) == 40,
        "support_not_below_no_credit": dag["support"] - no_credit["support"] >= threshold["support_delta_vs_no_credit_min"],
        "support_noninferior_local": dag["support"] >= local["support"] - threshold["support_noninferiority_vs_local_tolerance"],
        "support_noninferior_outcome": dag["support"] >= outcome["support"] - threshold["support_noninferiority_vs_outcome_tolerance"],
        "strict_noninferior_no_credit": dag["expected_strict"] >= no_credit["expected_strict"] - threshold["strict_noninferiority_vs_no_credit_tolerance"],
        "strict_noninferior_local": dag["expected_strict"] >= local["expected_strict"] - threshold["strict_noninferiority_vs_local_tolerance"],
        "strict_noninferior_outcome": dag["expected_strict"] >= outcome["expected_strict"] - threshold["strict_noninferiority_vs_outcome_tolerance"],
        "differs_from_outcome": disagreement >= threshold["top_action_disagreement_vs_outcome_min"],
        "query_strategy_diversity": len(dag["query_strategy_distribution"]) >= threshold["selected_query_strategies_min"],
        "development_result_not_paper_final": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    gates = {key: bool(value) for key, value in gates.items()}
    decision = "DAGIG_V6_HYBRID_QUERY_SELECTOR_DEVELOPMENT_GO" if all(gates.values()) else "DAGIG_V6_HYBRID_QUERY_SELECTOR_DEVELOPMENT_NO_GO"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "v6_hybrid_query_selector_internal_private.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "method_summary": summary,
        "dagig_outcome_top_disagreement": disagreement,
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": sha256(freeze_path)},
        "output_paths": {"private_rows": str(rows_path)},
        "output_hashes": {"private_rows": sha256(rows_path)},
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_HYBRID_QUERY_SELECTOR_DEVELOPMENT_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "method_summary": summary, "gates": gates, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
