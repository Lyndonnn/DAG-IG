#!/usr/bin/env python3
"""Run the preregistered visual-selector development audit."""

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


METHODS = ("no_credit", "local_ig_m", "outcome", "dagig")
METRICS = ("expected_terminal_value", "support", "expected_strict", "mode_strict")


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


def argmax_fixed(probabilities: list[float]) -> tuple[int, bool]:
    maximum = max(probabilities)
    tied = [index for index, value in enumerate(probabilities) if abs(value - maximum) <= 1e-12]
    return tied[0], len(tied) > 1


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values() if count) if total else 0.0


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap(
    rows: list[dict[str, Any]], metric: str, baseline: str, replicates: int, seed: int
) -> dict[str, float | int]:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(
            float(row["methods"]["dagig"][metric]) - float(row["methods"][baseline][metric])
        )
    sample_ids = sorted(by_sample)
    rng = random.Random(f"query-selector:{seed}:{baseline}:{metric}")
    draws = []
    for _ in range(replicates):
        sampled = [sample_ids[rng.randrange(len(sample_ids))] for _ in sample_ids]
        draws.append(mean(value for sample_id in sampled for value in by_sample[sample_id]))
    observed = [value for values in by_sample.values() for value in values]
    return {
        "observed_delta": mean(observed),
        "bootstrap_mean": mean(draws),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "replicates": replicates,
        "clusters": len(sample_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_freeze", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol_freeze.resolve()
    protocol = read_json(protocol_path)
    if protocol.get("decision") != "DAGIG_V6_VISUAL_SELECTOR_DEVELOPMENT_PROTOCOL_FROZEN":
        raise ValueError("visual development protocol is not frozen")
    if sha256(Path(protocol["runner_path"])) != protocol["runner_hash"]:
        raise ValueError("auditor changed after protocol freeze")
    for key, raw_path in protocol["input_paths"].items():
        assert_hash(Path(raw_path), protocol["input_hashes"][key], key)

    query_freeze = read_json(Path(protocol["input_paths"]["visual_value_freeze"]))
    if query_freeze.get("decision") != "DAGIG_V6_VISUAL_VALUES_V1_FROZEN":
        raise ValueError("visual values v1 are not frozen")
    internal_rows = read_jsonl(Path(protocol["input_paths"]["internal_targets"]))
    diagnostics = {
        row["parent_state_id"]: row
        for row in read_jsonl(Path(protocol["input_paths"]["diagnostics"]))
        if row["partition"] == "internal_holdout"
    }
    if len(internal_rows) != 40 or len(diagnostics) != 40:
        raise ValueError(f"expected 40 development roots, found {len(internal_rows)}/{len(diagnostics)}")

    shared_values = {
        row["evidence_action_id"]: row
        for row in read_jsonl(Path(protocol["input_paths"]["shared_answer_values"]))
    }
    strict_by_answer = {
        row["answer_action_id"]: float(row["strict_proxy"])
        for row in read_jsonl(Path(protocol["input_paths"]["terminal_private"]))
    }
    support_by_query = {
        row["query_id"]: row["strategy_support"]
        for row in read_jsonl(Path(protocol["input_paths"]["private_support"]))
    }

    private_rows: list[dict[str, Any]] = []
    for public in internal_rows:
        parent_id = public["parent_state_id"]
        diagnostic = diagnostics[parent_id]
        if len(public["actions"]) != 3 or len(diagnostic["visual_action_ids"]) != 3:
            raise ValueError(f"visual action count changed: {parent_id}")
        action_metrics = []
        for index, action in enumerate(public["actions"]):
            query_id = diagnostic["selected_query_action_ids"][index]
            evidence_id = diagnostic["selected_evidence_action_ids"][index]
            value = shared_values[evidence_id]
            evidence_strategy = evidence_id.rsplit("::", 1)[-1]
            strict_values = [strict_by_answer[answer_id] for answer_id in value["answer_action_ids"]]
            expected_strict = sum(
                float(probability) * strict
                for probability, strict in zip(value["answer_policy_probabilities"], strict_values)
            )
            mode_index = value["answer_action_ids"].index(value["mode_answer_action_id"])
            action_metrics.append(
                {
                    "label": action["label"],
                    "visual_strategy": action["strategy"],
                    "visual_action_id": diagnostic["visual_action_ids"][index],
                    "query_action_id": query_id,
                    "evidence_action_id": evidence_id,
                    "evidence_strategy": evidence_strategy,
                    "expected_terminal_value": float(value["shared_answer_value"]),
                    "support": float(bool(support_by_query[query_id][evidence_strategy])),
                    "expected_strict": expected_strict,
                    "mode_strict": strict_values[mode_index],
                }
            )
        selected = {}
        for method in METHODS:
            posterior = public["target_distributions"][method]
            selected_index, tied = argmax_fixed(posterior)
            selected[method] = {
                **action_metrics[selected_index],
                "posterior_probability": float(posterior[selected_index]),
                "posterior_tied_at_top": tied,
            }
        private_rows.append(
            {
                "parent_state_id": parent_id,
                "sample_id": parent_id.split("::", 1)[0],
                "partition": "internal_holdout",
                "methods": selected,
            }
        )

    summaries = {}
    for method in METHODS:
        selected = [row["methods"][method] for row in private_rows]
        query_strategies = Counter(row["visual_strategy"] for row in selected)
        evidence_strategies = Counter(row["evidence_strategy"] for row in selected)
        summaries[method] = {
            "visual_parent_states": len(selected),
            "samples": len({row["sample_id"] for row in private_rows}),
            "mean_expected_terminal_value": mean(row["expected_terminal_value"] for row in selected),
            "support_rate": mean(row["support"] for row in selected),
            "mean_expected_strict": mean(row["expected_strict"] for row in selected),
            "mode_strict_rate": mean(row["mode_strict"] for row in selected),
            "mean_selected_posterior_probability": mean(row["posterior_probability"] for row in selected),
            "top_tie_rate": mean(float(row["posterior_tied_at_top"]) for row in selected),
            "selected_visual_strategy_distribution": dict(sorted(query_strategies.items())),
            "selected_visual_strategy_count": len(query_strategies),
            "selected_visual_strategy_entropy_nats": entropy(query_strategies),
            "selected_evidence_strategy_distribution": dict(sorted(evidence_strategies.items())),
        }

    bootstrap = protocol["cluster_bootstrap"]
    pairwise = {}
    for baseline in ("no_credit", "outcome", "local_ig_m"):
        comparison = {
            "top_action_disagreement_rate": mean(
                row["methods"]["dagig"]["visual_action_id"] != row["methods"][baseline]["visual_action_id"]
                for row in private_rows
            )
        }
        for metric in METRICS:
            deltas = [
                row["methods"]["dagig"][metric] - row["methods"][baseline][metric]
                for row in private_rows
            ]
            comparison[metric] = {
                "dagig_minus_baseline_mean": mean(deltas),
                "gain": sum(delta > 1e-12 for delta in deltas),
                "loss": sum(delta < -1e-12 for delta in deltas),
                "tie": sum(abs(delta) <= 1e-12 for delta in deltas),
                "cluster_bootstrap": cluster_bootstrap(
                    private_rows,
                    metric,
                    baseline,
                    int(bootstrap["replicates"]),
                    int(bootstrap["seed"]),
                ),
            }
        pairwise[f"dagig_vs_{baseline}"] = comparison

    threshold = protocol["go_gates"]
    dag, no_credit, outcome, local = (
        summaries["dagig"], summaries["no_credit"], summaries["outcome"], summaries["local_ig_m"]
    )
    gates = {
        "complete_40_development_roots": len(private_rows) == 40,
        "complete_40_internal_samples": len({row["sample_id"] for row in private_rows}) == 40,
        "direct_visual_posterior_argmax_only": True,
        "dagig_terminal_gain_vs_no_credit": dag["mean_expected_terminal_value"] - no_credit["mean_expected_terminal_value"] >= threshold["dagig_terminal_delta_vs_no_credit_min"],
        "dagig_terminal_noninferior_outcome": dag["mean_expected_terminal_value"] >= outcome["mean_expected_terminal_value"] - threshold["dagig_terminal_noninferiority_tolerance"],
        "dagig_terminal_noninferior_local": dag["mean_expected_terminal_value"] >= local["mean_expected_terminal_value"] - threshold["dagig_terminal_noninferiority_vs_local_tolerance"],
        "dagig_support_not_below_no_credit": dag["support_rate"] - no_credit["support_rate"] >= threshold["dagig_support_delta_vs_no_credit_min"],
        "dagig_support_noninferior_outcome": dag["support_rate"] >= outcome["support_rate"] - threshold["dagig_support_noninferiority_tolerance"],
        "dagig_support_noninferior_local": dag["support_rate"] >= local["support_rate"] - threshold["dagig_support_noninferiority_vs_local_tolerance"],
        "dagig_expected_strict_noninferior_no_credit": dag["mean_expected_strict"] >= no_credit["mean_expected_strict"] - threshold["dagig_expected_strict_noninferiority_tolerance"],
        "dagig_expected_strict_noninferior_outcome": dag["mean_expected_strict"] >= outcome["mean_expected_strict"] - threshold["dagig_expected_strict_noninferiority_tolerance"],
        "dagig_expected_strict_noninferior_local": dag["mean_expected_strict"] >= local["mean_expected_strict"] - threshold["dagig_expected_strict_noninferiority_vs_local_tolerance"],
        "dagig_mode_strict_noninferior_no_credit": dag["mode_strict_rate"] >= no_credit["mode_strict_rate"] - threshold["dagig_mode_strict_noninferiority_tolerance"],
        "dagig_mode_strict_noninferior_outcome": dag["mode_strict_rate"] >= outcome["mode_strict_rate"] - threshold["dagig_mode_strict_noninferiority_tolerance"],
        "dagig_mode_strict_noninferior_local": dag["mode_strict_rate"] >= local["mode_strict_rate"] - threshold["dagig_mode_strict_noninferiority_vs_local_tolerance"],
        "dagig_differs_from_outcome": pairwise["dagig_vs_outcome"]["top_action_disagreement_rate"] >= threshold["dagig_control_top_action_disagreement_min"],
        "dagig_differs_from_local": pairwise["dagig_vs_local_ig_m"]["top_action_disagreement_rate"] >= threshold["dagig_local_top_action_disagreement_min"],
        "dagig_visual_action_diversity": dag["selected_visual_strategy_count"] >= threshold["dagig_selected_visual_strategies_min"],
        "full_query_targets_not_fit_to_private_development_labels": True,
        "development_result_not_paper_final": True,
        "new_search_calls_zero": True,
        "generator_training_not_run": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_VISUAL_SELECTOR_DEVELOPMENT_GO" if all(gates.values()) else "DAGIG_V6_VISUAL_SELECTOR_DEVELOPMENT_NO_GO"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    private_path = output_dir / "v6_visual_selector_development_private.jsonl"
    private_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in private_rows),
        encoding="utf-8",
    )
    result = {
        "decision": decision,
        "protocol_version": protocol["protocol_version"],
        "method_summary": summaries,
        "paired_comparisons": pairwise,
        "gates": gates,
        "input_paths": {"protocol_freeze": str(protocol_path)},
        "input_hashes": {"protocol_freeze": sha256(protocol_path)},
        "output_paths": {"private_rows": str(private_path)},
        "output_hashes": {"private_rows": sha256(private_path)},
        "private_labels_used_only_for_this_selector_audit": True,
        "development_samples_previously_consumed_by_query_diagnostics": True,
        "visual_targets_fit_to_private_development_labels": False,
        "selector_audit_runs": 1,
        "dev_used": False,
        "test_used": False,
        "api_calls": 0,
        "training_run": False,
    }
    audit_path = output_dir / "DAGIG_V6_VISUAL_SELECTOR_DEVELOPMENT_AUDIT.json"
    result["output_paths"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    lines = [
        "# Visual Selector Development Audit",
        "",
        "## Scope",
        "",
        "Development evaluation of the OCR/caption/joint visual interventions under frozen DAG query/evidence selectors and the shared answer policy. These 40 samples were previously used by query diagnostics, so this is not paper-final evidence.",
        "",
        "## Results",
        "",
        "| Method | Terminal value | Support | Expected strict | Mode strict | Query mix |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for method in METHODS:
        row = summaries[method]
        lines.append(
            f"| {method} | {row['mean_expected_terminal_value']:.6f} | {pct(row['support_rate'])} | {pct(row['mean_expected_strict'])} | {pct(row['mode_strict_rate'])} | `{json.dumps(row['selected_visual_strategy_distribution'], sort_keys=True)}` |"
        )
    lines.extend(["", "## Paired DAG-IG Comparisons", ""])
    for baseline in ("no_credit", "outcome", "local_ig_m"):
        comparison = pairwise[f"dagig_vs_{baseline}"]
        lines.extend([f"### DAG-IG vs {baseline}", "", f"- top-action disagreement: {pct(comparison['top_action_disagreement_rate'])}"])
        for metric in METRICS:
            item = comparison[metric]
            ci = item["cluster_bootstrap"]
            lines.append(
                f"- {metric}: delta={item['dagig_minus_baseline_mean']:.6f}; gain/loss/tie={item['gain']}/{item['loss']}/{item['tie']}; clustered 95% CI=[{ci['ci95_low']:.6f}, {ci['ci95_high']:.6f}]"
            )
        lines.append("")
    lines.extend(["## Gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    lines.extend(["", "## Decision", "", f"`{decision}`", ""])
    lines.append(
        "Proceed to the full-DAG direct controller audit; keep query/evidence as frozen posterior selectors."
        if decision == "DAGIG_V6_VISUAL_SELECTOR_DEVELOPMENT_GO"
        else "Do not train or open dev/test. Revise the query candidate/action protocol, not model capacity."
    )
    report_path = output_dir / "DAGIG_V6_VISUAL_SELECTOR_DEVELOPMENT_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "method_summary": summaries, "gates": gates, "audit": str(audit_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
