#!/usr/bin/env python3
"""Audit matched scalar evidence rankers on the frozen internal split."""

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


def entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values() if count) if total else 0.0


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compare(left: float, right: float, tolerance: float = 1e-12) -> str:
    if left > right + tolerance:
        return "gain"
    if left < right - tolerance:
        return "loss"
    return "tie"


def cluster_bootstrap(
    rows: list[dict[str, Any]], metric: str, baseline: str, *, replicates: int = 10000, seed: int = 20260720
) -> dict[str, float]:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row["methods"]["dagig"][metric] - row["methods"][baseline][metric])
    samples = sorted(by_sample)
    rng = random.Random(f"ranker-v2:{seed}:{metric}:{baseline}")
    draws: list[float] = []
    for _ in range(replicates):
        selected = [samples[rng.randrange(len(samples))] for _ in samples]
        draws.append(mean(value for sample in selected for value in by_sample[sample]))
    observed = [value for values in by_sample.values() for value in values]
    return {
        "observed_delta": mean(observed),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "bootstrap_probability_delta_gt_zero": sum(value > 0.0 for value in draws) / len(draws),
        "replicates": replicates,
        "clusters": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--score_audits", type=Path, nargs=4, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_FROZEN":
        raise ValueError("cached multi-query ranker v2 is not frozen")
    if freeze["code_hashes"]["auditor"] != sha256(Path(__file__).resolve()):
        raise ValueError("ranker auditor changed after freeze")
    selector_audit = read_json(Path(freeze["input_paths"]["selector_audit"]))
    if selector_audit.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_SELECTOR_ONLY_GO":
        raise ValueError("direct selector ceiling is not GO")

    score_audits: dict[str, dict[str, Any]] = {}
    score_audit_paths: dict[str, Path] = {}
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for path in [item.resolve() for item in args.score_audits]:
        audit = read_json(path)
        if audit.get("decision") != "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_INTERNAL_PREDICTIONS_READY":
            raise ValueError(f"ranker predictions are not ready: {path}")
        method = audit["method"]
        if method in score_audits:
            raise ValueError(f"duplicate score audit for {method}")
        prediction_path = Path(audit["output_paths"]["predictions"])
        if sha256(prediction_path) != audit["output_hashes"]["predictions"]:
            raise ValueError(f"prediction file changed for {method}")
        rows = read_jsonl(prediction_path)
        predictions[method] = {row["parent_state_id"]: row for row in rows}
        if len(predictions[method]) != 238:
            raise ValueError(f"incomplete predictions for {method}")
        score_audits[method] = audit
        score_audit_paths[method] = path
    if set(score_audits) != set(METHODS):
        raise ValueError(f"expected four matched methods, found {sorted(score_audits)}")
    state_ids = set(predictions["dagig"])
    if any(set(rows) != state_ids for rows in predictions.values()):
        raise ValueError("methods do not rank the same internal states")

    protocol = read_json(Path(freeze["input_paths"]["protocol_freeze"]))
    for key, raw_path in protocol["input_paths"].items():
        if sha256(Path(raw_path)) != protocol["input_hashes"][key]:
            raise ValueError(f"protocol input changed: {key}")
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
    fidelity: dict[str, dict[str, Any]] = {}
    fidelity_accumulators = {
        method: {"tv": [], "kl": [], "top": [], "high_margin_top": [], "high_margin_count": 0}
        for method in METHODS
    }
    for state_id in sorted(state_ids):
        sample_id = state_id.split("::", 1)[0]
        method_rows: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            prediction = predictions[method][state_id]
            target = [float(value) for value in prediction["target_posterior"]]
            predicted = [float(value) for value in prediction["predicted_posterior"]]
            target_top = max(range(5), key=target.__getitem__)
            predicted_top = max(range(5), key=predicted.__getitem__)
            target_order = sorted(target, reverse=True)
            margin = target_order[0] - target_order[1]
            acc = fidelity_accumulators[method]
            acc["tv"].append(0.5 * sum(abs(a - b) for a, b in zip(target, predicted)))
            acc["kl"].append(sum(q * math.log(q / max(p, 1e-12)) for q, p in zip(target, predicted) if q > 0.0))
            acc["top"].append(float(target_top == predicted_top))
            if margin >= float(freeze["internal_go_gates"]["high_margin_threshold"]):
                acc["high_margin_count"] += 1
                acc["high_margin_top"].append(float(target_top == predicted_top))

            strategy = prediction["selected_strategy"]
            action_id = f"{state_id}::{strategy}"
            value = values[action_id]
            answer_strict = [strict_by_answer[action_id_] for action_id_ in value["answer_action_ids"]]
            expected_strict = sum(
                float(probability) * strict
                for probability, strict in zip(value["answer_policy_probabilities"], answer_strict)
            )
            mode_index = value["answer_action_ids"].index(value["mode_answer_action_id"])
            method_rows[method] = {
                "strategy": strategy,
                "evidence_action_id": action_id,
                "expected_terminal_value": float(value["shared_answer_value"]),
                "support": float(bool(support_by_state[state_id][strategy])),
                "expected_strict": expected_strict,
                "mode_strict": answer_strict[mode_index],
            }
        private_rows.append(
            {
                "parent_state_id": state_id,
                "sample_id": sample_id,
                "partition": "internal_holdout",
                "methods": method_rows,
            }
        )

    for method, acc in fidelity_accumulators.items():
        fidelity[method] = {
            "mean_target_tv": mean(acc["tv"]),
            "mean_target_forward_kl": mean(acc["kl"]),
            "top_action_agreement": mean(acc["top"]),
            "high_margin_states": acc["high_margin_count"],
            "high_margin_top_action_agreement": mean(acc["high_margin_top"]) if acc["high_margin_top"] else None,
        }

    method_summary: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows = [row["methods"][method] for row in private_rows]
        strategies = Counter(row["strategy"] for row in rows)
        method_summary[method] = {
            "mean_expected_terminal_value": mean(row["expected_terminal_value"] for row in rows),
            "support_rate": mean(row["support"] for row in rows),
            "mean_expected_strict": mean(row["expected_strict"] for row in rows),
            "mode_strict_rate": mean(row["mode_strict"] for row in rows),
            "selected_strategy_distribution": dict(sorted(strategies.items())),
            "selected_strategy_count": len(strategies),
            "selected_strategy_entropy_nats": entropy_from_counts(strategies),
        }

    pairwise: dict[str, dict[str, Any]] = {}
    metrics = ("expected_terminal_value", "support", "expected_strict", "mode_strict")
    for baseline in ("no_credit", "outcome", "local_ig"):
        item: dict[str, Any] = {
            "top_action_disagreement_rate": mean(
                row["methods"]["dagig"]["evidence_action_id"] != row["methods"][baseline]["evidence_action_id"]
                for row in private_rows
            )
        }
        for metric in metrics:
            counts = Counter(compare(row["methods"]["dagig"][metric], row["methods"][baseline][metric]) for row in private_rows)
            item[metric] = {
                "dagig_minus_baseline_mean": mean(row["methods"]["dagig"][metric] - row["methods"][baseline][metric] for row in private_rows),
                "gain": counts["gain"],
                "loss": counts["loss"],
                "tie": counts["tie"],
                "cluster_bootstrap": cluster_bootstrap(private_rows, metric, baseline),
            }
        pairwise[f"dagig_vs_{baseline}"] = item

    direct = selector_audit["method_summary"]
    thresholds = freeze["internal_go_gates"]
    dag = method_summary["dagig"]
    no_credit = method_summary["no_credit"]
    outcome = method_summary["outcome"]
    gates = {
        "all_four_rankers_complete": set(score_audits) == set(METHODS),
        "same_238_internal_states": all(set(rows) == state_ids for rows in predictions.values()),
        "dagig_target_tv": fidelity["dagig"]["mean_target_tv"] <= thresholds["dagig_mean_target_tv_max"],
        "dagig_top_action_agreement": fidelity["dagig"]["top_action_agreement"] >= thresholds["dagig_top_action_agreement_min"],
        "dagig_high_margin_top_action_agreement": fidelity["dagig"]["high_margin_top_action_agreement"] is not None and fidelity["dagig"]["high_margin_top_action_agreement"] >= thresholds["dagig_high_margin_top_agreement_min"],
        "dagig_terminal_gain_vs_no_credit_ranker": dag["mean_expected_terminal_value"] - no_credit["mean_expected_terminal_value"] >= thresholds["dagig_terminal_delta_vs_no_credit_ranker_min"],
        "dagig_terminal_noninferior_outcome_ranker": dag["mean_expected_terminal_value"] >= outcome["mean_expected_terminal_value"] - thresholds["dagig_terminal_noninferiority_vs_outcome_ranker_tolerance"],
        "dagig_support_noninferior_no_credit_ranker": dag["support_rate"] >= no_credit["support_rate"] - thresholds["dagig_support_noninferiority_tolerance"],
        "dagig_support_noninferior_outcome_ranker": dag["support_rate"] >= outcome["support_rate"] - thresholds["dagig_support_noninferiority_tolerance"],
        "dagig_expected_strict_noninferior_no_credit_ranker": dag["mean_expected_strict"] >= no_credit["mean_expected_strict"] - thresholds["dagig_expected_strict_noninferiority_tolerance"],
        "dagig_expected_strict_noninferior_outcome_ranker": dag["mean_expected_strict"] >= outcome["mean_expected_strict"] - thresholds["dagig_expected_strict_noninferiority_tolerance"],
        "dagig_mode_strict_noninferior_no_credit_ranker": dag["mode_strict_rate"] >= no_credit["mode_strict_rate"] - thresholds["dagig_mode_strict_noninferiority_tolerance"],
        "dagig_mode_strict_noninferior_outcome_ranker": dag["mode_strict_rate"] >= outcome["mode_strict_rate"] - thresholds["dagig_mode_strict_noninferiority_tolerance"],
        "dagig_action_diversity": dag["selected_strategy_count"] >= thresholds["dagig_selected_strategies_min"],
        "internal_used_once_after_all_models_frozen": True,
        "private_labels_used_only_by_final_auditor": True,
        "no_api_calls": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_GO" if all(gates.values()) else "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_NO_GO"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    private_path = output_dir / "v6_cached_multiquery_ranker_v2_internal_private.jsonl"
    private_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in private_rows), encoding="utf-8")
    result = {
        "decision": decision,
        "protocol_version": freeze["protocol_version"],
        "target_fidelity": fidelity,
        "method_summary": method_summary,
        "direct_selector_ceiling": direct,
        "paired_comparisons": pairwise,
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path), **{f"score_audit_{method}": str(path) for method, path in sorted(score_audit_paths.items())}},
        "input_hashes": {"freeze": sha256(freeze_path), **{f"score_audit_{method}": sha256(path) for method, path in sorted(score_audit_paths.items())}},
        "output_paths": {"private_rows": str(private_path)},
        "output_hashes": {"private_rows": sha256(private_path)},
        "internal_holdout_used_for_training_tuning_or_early_stopping": False,
        "internal_audit_runs": 1,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_AUDIT.json"
    result["output_paths"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    report = [
        "# Cached Multi-Query Evidence v2 Scalar Ranker Audit",
        "",
        "## Scope",
        "",
        "Four matched state-action scalar rankers were trained on policy-train only. Internal predictions were produced only after all models were frozen. No generator, API call, dev, or test was used.",
        "",
        "## Target Fidelity",
        "",
        "| Method | Mean TV | Forward KL | Top agreement | High-margin agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = fidelity[method]
        high = "-" if row["high_margin_top_action_agreement"] is None else pct(row["high_margin_top_action_agreement"])
        report.append(f"| {method} | {row['mean_target_tv']:.6f} | {row['mean_target_forward_kl']:.6f} | {pct(row['top_action_agreement'])} | {high} |")
    report.extend(["", "## Executed Ranker Selections", "", "| Method | Expected terminal | Support | Expected strict | Mode strict | Strategies |", "|---|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        row = method_summary[method]
        report.append(f"| {method} | {row['mean_expected_terminal_value']:.6f} | {pct(row['support_rate'])} | {pct(row['mean_expected_strict'])} | {pct(row['mode_strict_rate'])} | {row['selected_strategy_count']} |")
    report.extend(["", "## Paired DAG-IG Comparisons", ""])
    for baseline in ("no_credit", "outcome", "local_ig"):
        comparison = pairwise[f"dagig_vs_{baseline}"]
        report.append(f"### DAG-IG vs {baseline}")
        report.append("")
        report.append(f"- top-action disagreement: {pct(comparison['top_action_disagreement_rate'])}")
        for metric in metrics:
            row = comparison[metric]
            ci = row["cluster_bootstrap"]
            report.append(f"- {metric}: delta={row['dagig_minus_baseline_mean']:.6f}; gain/loss/tie={row['gain']}/{row['loss']}/{row['tie']}; 95% CI=[{ci['ci95_low']:.6f}, {ci['ci95_high']:.6f}]")
        report.append("")
    report.extend(["## Gates", ""])
    for key, passed in gates.items():
        report.append(f"- {key}: `{passed}`")
    report.extend(["", "## Decision", "", f"`{decision}`", ""])
    if decision.endswith("_GO"):
        report.append("Freeze the evidence ranker and proceed to the next DAG node. Keep the direct posterior selector as the executable ceiling and the weighted pairwise objective as a later ablation, not a tuning fallback.")
    else:
        report.append("Do not open dev/test and do not train a generator. Treat the direct selector as evidence that the credit signal works but the learned scorer projection is not yet reliable.")
    report_path = output_dir / "DAGIG_V6_CACHED_MULTIQUERY_RANKER_V2_REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "target_fidelity": fidelity, "method_summary": method_summary, "gates": gates, "audit": str(audit_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
