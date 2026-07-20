#!/usr/bin/env python3
"""Run the one-shot private selector-only audit for cached multi-query evidence v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METHODS = ("no_credit", "local_ig", "outcome", "dagig")
STRATEGY_ORDER = (
    "serper_rank_top3",
    "bge_top3",
    "support_diverse_top3",
    "observable_low_support_top3",
    "entity_condition_mismatch_top3",
)


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


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} changed: expected {expected}, found {actual}: {path}")


def entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values() if count)


def argmax_fixed(probabilities: list[float]) -> tuple[int, bool]:
    maximum = max(probabilities)
    tied = [index for index, value in enumerate(probabilities) if abs(value - maximum) <= 1e-12]
    return tied[0], len(tied) > 1


def compare(left: float, right: float, tolerance: float = 1e-12) -> str:
    delta = left - right
    if delta > tolerance:
        return "gain"
    if delta < -tolerance:
        return "loss"
    return "tie"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    metric: str,
    baseline: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(
            float(row["methods"]["dagig"][metric]) - float(row["methods"][baseline][metric])
        )
    sample_ids = sorted(by_sample)
    rng = random.Random(f"{seed}:{metric}:{baseline}")
    draws: list[float] = []
    for _ in range(replicates):
        selected = [sample_ids[rng.randrange(len(sample_ids))] for _ in sample_ids]
        values = [value for sample_id in selected for value in by_sample[sample_id]]
        draws.append(mean(values))
    observed_values = [value for values in by_sample.values() for value in values]
    return {
        "observed_delta": mean(observed_values),
        "bootstrap_mean": mean(draws),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "replicates": replicates,
        "clusters": len(sample_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_freeze", type=Path, required=True)
    parser.add_argument("--target_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol_freeze.resolve()
    target_audit_path = args.target_audit.resolve()
    protocol = read_json(protocol_path)
    target_audit = read_json(target_audit_path)
    if protocol.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_FROZEN":
        raise ValueError("cached multi-query evidence v2 protocol is not frozen")
    if target_audit.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_EVIDENCE_V2_TARGETS_GO":
        raise ValueError("cached multi-query evidence v2 targets are not GO")
    assert_hash(protocol_path, target_audit["input_hashes"]["protocol_freeze"], "protocol freeze")
    for key, raw_path in target_audit["output_paths"].items():
        assert_hash(Path(raw_path), target_audit["output_hashes"][key], key)
    for key, raw_path in protocol["input_paths"].items():
        assert_hash(Path(raw_path), protocol["input_hashes"][key], key)

    internal_path = Path(target_audit["output_paths"]["internal_targets"])
    internal_rows = read_jsonl(internal_path)
    if len(internal_rows) != 238:
        raise ValueError(f"expected 238 frozen internal query states, found {len(internal_rows)}")

    values = {
        row["evidence_action_id"]: row
        for row in read_jsonl(Path(protocol["input_paths"]["shared_answer_values"]))
    }
    strict_by_answer = {
        row["answer_action_id"]: float(row["strict_proxy"])
        for row in read_jsonl(Path(protocol["input_paths"]["terminal_private_audit"]))
    }
    support_by_state = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(Path(protocol["input_paths"]["private_support"]))
    }

    private_rows: list[dict[str, Any]] = []
    missing_answer_labels = 0
    for public in internal_rows:
        state_id = public["parent_state_id"]
        sample_id = state_id.split("::", 1)[0]
        support = support_by_state.get(state_id)
        if support is None:
            raise ValueError(f"missing private support row for {state_id}")
        action_metrics: list[dict[str, Any]] = []
        for action in public["actions"]:
            strategy = action["strategy"]
            action_id = f"{state_id}::{strategy}"
            value = values.get(action_id)
            if value is None:
                raise ValueError(f"missing shared answer value for {action_id}")
            strict_values: list[float] = []
            for answer_id in value["answer_action_ids"]:
                if answer_id not in strict_by_answer:
                    missing_answer_labels += 1
                    strict_values.append(0.0)
                else:
                    strict_values.append(strict_by_answer[answer_id])
            expected_strict = sum(
                float(probability) * strict
                for probability, strict in zip(value["answer_policy_probabilities"], strict_values)
            )
            mode_index = value["answer_action_ids"].index(value["mode_answer_action_id"])
            action_metrics.append(
                {
                    "label": action["label"],
                    "strategy": strategy,
                    "evidence_action_id": action_id,
                    "expected_terminal_value": float(value["shared_answer_value"]),
                    "support": float(bool(support[strategy])),
                    "expected_strict": expected_strict,
                    "mode_strict": strict_values[mode_index],
                }
            )

        selected: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            posterior = public["target_distributions"][method]
            index, tied = argmax_fixed(posterior)
            selected[method] = {
                **action_metrics[index],
                "posterior_probability": float(posterior[index]),
                "posterior_tied_at_top": tied,
            }
        private_rows.append(
            {
                "parent_state_id": state_id,
                "sample_id": sample_id,
                "partition": "internal_holdout",
                "methods": selected,
            }
        )
    if missing_answer_labels:
        raise ValueError(f"missing private strict labels for {missing_answer_labels} answer actions")

    method_summary: dict[str, dict[str, Any]] = {}
    metric_keys = ("expected_terminal_value", "support", "expected_strict", "mode_strict")
    for method in METHODS:
        selected = [row["methods"][method] for row in private_rows]
        strategies = Counter(row["strategy"] for row in selected)
        method_summary[method] = {
            "query_states": len(selected),
            "samples": len({row["sample_id"] for row in private_rows}),
            "mean_expected_terminal_value": mean(row["expected_terminal_value"] for row in selected),
            "support_rate": mean(row["support"] for row in selected),
            "mean_expected_strict": mean(row["expected_strict"] for row in selected),
            "mode_strict_rate": mean(row["mode_strict"] for row in selected),
            "mean_selected_posterior_probability": mean(row["posterior_probability"] for row in selected),
            "top_tie_rate": mean(float(row["posterior_tied_at_top"]) for row in selected),
            "selected_strategy_distribution": dict(sorted(strategies.items())),
            "selected_strategy_count": len(strategies),
            "selected_strategy_entropy_nats": entropy_from_counts(strategies),
        }

    pairwise: dict[str, dict[str, Any]] = {}
    bootstrap_config = protocol["selector_only_evaluation"]["cluster_bootstrap"]
    for baseline in ("no_credit", "outcome", "local_ig"):
        comparison: dict[str, Any] = {
            "top_action_disagreement_rate": mean(
                row["methods"]["dagig"]["evidence_action_id"] != row["methods"][baseline]["evidence_action_id"]
                for row in private_rows
            ),
        }
        for metric in metric_keys:
            outcomes = Counter(
                compare(row["methods"]["dagig"][metric], row["methods"][baseline][metric])
                for row in private_rows
            )
            comparison[metric] = {
                "dagig_minus_baseline_mean": mean(
                    row["methods"]["dagig"][metric] - row["methods"][baseline][metric]
                    for row in private_rows
                ),
                "gain": outcomes["gain"],
                "loss": outcomes["loss"],
                "tie": outcomes["tie"],
                "cluster_bootstrap": cluster_bootstrap(
                    private_rows,
                    metric,
                    baseline,
                    replicates=int(bootstrap_config["replicates"]),
                    seed=int(bootstrap_config["seed"]),
                ),
            }
        pairwise[f"dagig_vs_{baseline}"] = comparison

    thresholds = protocol["selector_go_gates"]
    dag = method_summary["dagig"]
    no_credit = method_summary["no_credit"]
    outcome = method_summary["outcome"]
    gates = {
        "complete_238_internal_states": len(private_rows) == 238,
        "complete_40_internal_samples": len({row["sample_id"] for row in private_rows}) == 40,
        "direct_posterior_argmax_only": True,
        "dagig_terminal_gain_vs_no_credit": dag["mean_expected_terminal_value"] - no_credit["mean_expected_terminal_value"] >= thresholds["dagig_terminal_delta_vs_no_credit_min"],
        "dagig_terminal_noninferior_outcome": dag["mean_expected_terminal_value"] >= outcome["mean_expected_terminal_value"] - thresholds["dagig_terminal_noninferiority_vs_outcome_tolerance"],
        "dagig_support_not_below_no_credit": dag["support_rate"] - no_credit["support_rate"] >= thresholds["dagig_support_delta_vs_no_credit_min"],
        "dagig_support_noninferior_outcome": dag["support_rate"] >= outcome["support_rate"] - thresholds["dagig_support_noninferiority_vs_outcome_tolerance"],
        "dagig_expected_strict_noninferior_no_credit": dag["mean_expected_strict"] >= no_credit["mean_expected_strict"] - thresholds["dagig_expected_strict_noninferiority_tolerance"],
        "dagig_expected_strict_noninferior_outcome": dag["mean_expected_strict"] >= outcome["mean_expected_strict"] - thresholds["dagig_expected_strict_noninferiority_tolerance"],
        "dagig_mode_strict_noninferior_no_credit": dag["mode_strict_rate"] >= no_credit["mode_strict_rate"] - thresholds["dagig_mode_strict_noninferiority_tolerance"],
        "dagig_mode_strict_noninferior_outcome": dag["mode_strict_rate"] >= outcome["mode_strict_rate"] - thresholds["dagig_mode_strict_noninferiority_tolerance"],
        "dagig_differs_from_outcome": pairwise["dagig_vs_outcome"]["top_action_disagreement_rate"] >= thresholds["dagig_outcome_top_action_disagreement_min"],
        "dagig_action_diversity": dag["selected_strategy_count"] >= thresholds["dagig_selected_strategies_min"],
        "public_target_leakage_audit_passed": target_audit["gates"]["public_files_have_no_evaluation_fields"],
        "internal_not_used_for_fit_or_tuning": True,
        "new_search_calls_zero": True,
        "generator_training_not_run": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = (
        "DAGIG_V6_CACHED_MULTIQUERY_SELECTOR_ONLY_GO"
        if all(gates.values())
        else "DAGIG_V6_CACHED_MULTIQUERY_SELECTOR_ONLY_NO_GO"
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    private_path = output_dir / "v6_cached_multiquery_selector_only_internal_private.jsonl"
    private_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in private_rows),
        encoding="utf-8",
    )
    result = {
        "decision": decision,
        "protocol_version": protocol["protocol_version"],
        "selection_rule": "direct posterior argmax with frozen A-to-E tie break",
        "method_summary": method_summary,
        "paired_comparisons": pairwise,
        "gates": gates,
        "input_paths": {"protocol_freeze": str(protocol_path), "target_audit": str(target_audit_path), "internal_targets": str(internal_path)},
        "input_hashes": {"protocol_freeze": sha256(protocol_path), "target_audit": sha256(target_audit_path), "internal_targets": sha256(internal_path)},
        "output_paths": {"private_rows": str(private_path)},
        "output_hashes": {"private_rows": sha256(private_path)},
        "private_labels_used_only_for_this_selector_audit": True,
        "internal_holdout_used_for_training_or_tuning": False,
        "selector_audit_runs": 1,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_SELECTOR_ONLY_AUDIT.json"
    result["output_paths"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    report_lines = [
        "# Cached Multi-Query Evidence v2 Selector-Only Audit",
        "",
        "## Scope",
        "",
        "This is the one-shot internal selector-only evaluation frozen before private labels were opened. It uses cached real-search states only, performs no API calls, trains no generator, and does not open dev/test.",
        "",
        "## Methods",
        "",
        "| Method | Expected terminal | Support | Expected strict | Mode strict | Top tie | Strategies |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = method_summary[method]
        report_lines.append(
            f"| {method} | {row['mean_expected_terminal_value']:.6f} | {pct(row['support_rate'])} | {pct(row['mean_expected_strict'])} | {pct(row['mode_strict_rate'])} | {pct(row['top_tie_rate'])} | {row['selected_strategy_count']} |"
        )
    report_lines.extend(["", "## Paired DAG-IG Comparisons", ""])
    for baseline in ("no_credit", "outcome", "local_ig"):
        comparison = pairwise[f"dagig_vs_{baseline}"]
        report_lines.append(f"### DAG-IG vs {baseline}")
        report_lines.append("")
        report_lines.append(f"- top-action disagreement: {pct(comparison['top_action_disagreement_rate'])}")
        for metric in metric_keys:
            item = comparison[metric]
            ci = item["cluster_bootstrap"]
            report_lines.append(
                f"- {metric}: delta={item['dagig_minus_baseline_mean']:.6f}; gain/loss/tie={item['gain']}/{item['loss']}/{item['tie']}; sample-clustered 95% CI=[{ci['ci95_low']:.6f}, {ci['ci95_high']:.6f}]"
            )
        report_lines.append("")
    report_lines.extend(["## Action Diversity", ""])
    for method in METHODS:
        report_lines.append(f"- {method}: `{json.dumps(method_summary[method]['selected_strategy_distribution'], sort_keys=True)}`")
    report_lines.extend(["", "## Gates", ""])
    for key, passed in gates.items():
        report_lines.append(f"- {key}: `{passed}`")
    report_lines.extend(["", "## Decision", "", f"`{decision}`", ""])
    if decision.endswith("_GO"):
        report_lines.append("Proceed to a matched scalar evidence scorer/ranker using policy-train only. Primary objective: listwise KL; weighted pairwise cardinal ranking is an ablation. Do not train a categorical generator.")
    else:
        report_lines.append("Do not train a scorer/ranker. Diagnose cached query coverage, evidence intervention quality, frozen terminal value, and control-credit definitions without reopening this internal holdout for tuning.")
    report_path = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_SELECTOR_ONLY_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "method_summary": method_summary, "gates": gates, "audit": str(audit_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
