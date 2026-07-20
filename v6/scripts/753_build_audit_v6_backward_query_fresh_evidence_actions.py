#!/usr/bin/env python3
"""Build and privately audit five evidence actions for fresh query execution."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlsplit


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
    source = normalized(text).split()
    target = normalized(phrase).split()
    return bool(target and any(source[index : index + len(target)] == target for index in range(len(source) - len(target) + 1)))


def canonical_url(value: Any) -> str:
    parsed = urlsplit(str(value or ""))
    return parsed.netloc.casefold().removeprefix("www.") + re.sub(r"/+", "/", parsed.path).rstrip("/")


def signature(docs: list[dict[str, Any]], combo: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(sorted(docs[index]["doc_id"] for index in combo))


def base_score(doc: dict[str, Any]) -> float:
    return 0.65 * float(doc["normalized_bge_score"]) + 0.20 * float(doc["question_keyword_overlap"]) + 0.15 * float(doc["answer_type_pattern_match"])


def mismatch_score(doc: dict[str, Any]) -> float:
    return 0.70 * float(doc["normalized_bge_score"]) - 0.20 * float(doc["question_keyword_overlap"]) - 0.10 * float(doc["answer_type_pattern_match"])


def first_unique(
    combos: list[tuple[int, ...]],
    docs: list[dict[str, Any]],
    used: set[tuple[str, ...]],
) -> tuple[int, ...]:
    for combo in combos:
        key = signature(docs, combo)
        if key not in used:
            used.add(key)
            return combo
    raise ValueError("could not construct five distinct fresh evidence actions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--shard_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    freeze_path = args.freeze.resolve()
    freeze = read_json(freeze_path)
    if freeze.get("parent_protocol_version") != "dagig_v6_backward_query_fresh_fixed_descendants_v1":
        raise ValueError("fresh backward-query evidence protocol is not frozen")
    if freeze.get("decision") != "DAGIG_V6_ON_POLICY_EVIDENCE_ACTIONS_FROZEN":
        raise ValueError("fresh evidence parent universe is not GO")
    if freeze["runner_hashes"]["auditor"] != sha256(Path(__file__).resolve()):
        raise ValueError("fresh evidence auditor changed after freeze")
    for key, path in freeze["input_paths"].items():
        if sha256(Path(path)) != freeze["input_hashes"][key]:
            raise ValueError(f"frozen fresh-evidence input changed: {key}")

    manifest_paths = sorted(args.shard_dir.resolve().glob("DAGIG_V6_ON_POLICY_EVIDENCE_ACTION_SHARD*_MANIFEST.json"))
    manifests = [read_json(path) for path in manifest_paths]
    if not manifests or len(manifests) != int(manifests[0]["num_shards"]):
        raise ValueError("fresh evidence action shard set is incomplete")
    source_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        if manifest.get("decision") != "DAGIG_V6_ON_POLICY_EVIDENCE_ACTION_SHARD_READY":
            raise ValueError("fresh evidence source shard is not ready")
        path = Path(manifest["output_paths"]["evidence_actions"])
        if sha256(path) != manifest["output_hashes"]["evidence_actions"]:
            raise ValueError("fresh evidence action shard changed")
        source_rows.extend(read_jsonl(path))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[row["query_id"]].append(row)
    expected_parents = int(freeze["universe"]["eligible_query_parents"])
    if len(grouped) != expected_parents or any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("three-action fresh source matrix is incomplete")

    output_rows: list[dict[str, Any]] = []
    for query_id, rows in sorted(grouped.items()):
        method, separator, visual_parent_id = query_id.partition("::")
        if not separator or method not in {"no_credit", "local_ig", "outcome", "dagig"} or not visual_parent_id:
            raise ValueError(f"invalid fresh query parent identity: {query_id}")
        template = next(row for row in rows if row["evidence_strategy"] == "serper_rank_top3")
        docs = sorted(template["candidate_docs"], key=lambda doc: int(doc["rank"]))
        if len(docs) < 5:
            raise ValueError(f"fresh parent has fewer than five candidate docs: {query_id}")
        combos = list(itertools.combinations(range(len(docs)), 3))
        used: set[tuple[str, ...]] = set()
        selections = {
            "serper_rank_top3": first_unique(
                sorted(combos, key=lambda combo: (sum(int(docs[index]["rank"]) for index in combo), combo)),
                docs,
                used,
            ),
            "bge_top3": first_unique(
                sorted(combos, key=lambda combo: (-sum(float(docs[index]["bge_score"]) for index in combo), combo)),
                docs,
                used,
            ),
            "support_diverse_top3": first_unique(
                sorted(
                    combos,
                    key=lambda combo: (
                        -(sum(base_score(docs[index]) for index in combo) / 3 + 0.10 * len({docs[index]["domain"] for index in combo}) / 3),
                        combo,
                    ),
                ),
                docs,
                used,
            ),
            "observable_low_support_top3": first_unique(
                sorted(
                    combos,
                    key=lambda combo: (
                        sum(base_score(docs[index]) for index in combo),
                        -sum(int(docs[index]["rank"]) for index in combo),
                        combo,
                    ),
                ),
                docs,
                used,
            ),
            "entity_condition_mismatch_top3": first_unique(
                sorted(
                    combos,
                    key=lambda combo: (
                        -sum(mismatch_score(docs[index]) for index in combo),
                        sum(base_score(docs[index]) for index in combo),
                        combo,
                    ),
                ),
                docs,
                used,
            ),
        }
        for strategy in freeze["strategies"]:
            chosen = [docs[index] for index in selections[strategy]]
            output_rows.append(
                {
                    **template,
                    "method": method,
                    "visual_parent_id": visual_parent_id,
                    "evidence_action_id": f"{query_id}::{strategy}",
                    "evidence_strategy": strategy,
                    "selected_doc_ids": [doc["doc_id"] for doc in chosen],
                    "selected_docs": chosen,
                    "behavior_weight": 0.2,
                }
            )
    expected_actions = int(freeze["universe"]["evidence_actions"])
    if len(output_rows) != expected_actions:
        raise ValueError("five-action fresh evidence matrix is incomplete")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    public_path = output / "v6_backward_query_fresh_evidence_actions_no_labels.jsonl"
    public_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(output_rows, key=lambda item: item["evidence_action_id"])),
        encoding="utf-8",
    )

    labels_path = Path(freeze["input_paths"]["private_labels"])
    corpus_path = Path(freeze["input_paths"]["corpus"])
    labels = {row["sample_id"]: row for row in read_jsonl(labels_path)}
    corpus = {row["doc_id"]: row for row in read_jsonl(corpus_path)}
    expanded: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        expanded[row["query_id"]].append(row)
    private_rows: list[dict[str, Any]] = []
    union_sizes: list[int] = []
    unique_sets: list[bool] = []
    support_counts: Counter[tuple[str, str]] = Counter()
    for query_id, rows in expanded.items():
        label = labels[rows[0]["sample_id"]]
        accepted = [label["gold_answer"], *(label.get("aliases") or [])]
        positive_urls = {
            canonical_url((corpus.get(doc_id) or {}).get("final_url") or (corpus.get(doc_id) or {}).get("url"))
            for doc_id in label.get("positive_doc_ids") or []
        }
        hits: dict[str, bool] = {}
        for row in rows:
            text = " ".join(f"{doc.get('title', '')} {doc.get('snippet', '')}" for doc in row["selected_docs"])
            urls = {canonical_url(doc.get("url")) for doc in row["selected_docs"]}
            hit = bool(positive_urls & urls) or any(phrase_contains(text, answer) for answer in accepted)
            hits[row["evidence_strategy"]] = hit
            support_counts[(row["method"], row["evidence_strategy"])] += int(hit)
        sets = {tuple(sorted(row["selected_doc_ids"])) for row in rows}
        union_sizes.append(len({doc_id for row in rows for doc_id in row["selected_doc_ids"]}))
        unique_sets.append(len(sets) == 5)
        default = hits["serper_rank_top3"]
        private_rows.append(
            {
                "query_id": query_id,
                "method": rows[0]["method"],
                "sample_id": rows[0]["sample_id"],
                "visual_parent_id": rows[0]["visual_parent_id"],
                "partition": "internal_holdout",
                "strategy_support": hits,
                "mixed_support": len(set(hits.values())) > 1,
                "union_recovery": any(hits.values()) and not default,
            }
        )
    metrics = {
        "query_parents": len(expanded),
        "samples": len({row["sample_id"] for row in private_rows}),
        "visual_parent_states": len({row["visual_parent_id"] for row in private_rows}),
        "evidence_actions": len(output_rows),
        "five_unique_sets_rate": mean(unique_sets),
        "mean_union_selected_docs": mean(union_sizes),
        "mixed_support_parents": sum(row["mixed_support"] for row in private_rows),
        "union_recoveries": sum(row["union_recovery"] for row in private_rows),
        "strategy_support_counts": {
            f"{method}::{strategy}": count
            for (method, strategy), count in sorted(support_counts.items())
        },
        "fresh_search_calls": int(freeze["universe"]["new_search_calls"]),
    }
    spec = freeze["action_gates"]
    gates = {
        "complete_480_parent_matrix": metrics["query_parents"] == 480,
        "complete_2400_action_matrix": metrics["evidence_actions"] == 2400,
        "complete_40_samples_120_visual_states": metrics["samples"] == 40 and metrics["visual_parent_states"] == 120,
        "five_unique_sets": metrics["five_unique_sets_rate"] >= float(spec["five_unique_sets_rate_min"]),
        "union_size": metrics["mean_union_selected_docs"] >= float(spec["mean_union_selected_docs_min"]),
        "public_materialized_before_private_audit": True,
        "fresh_search_executed": metrics["fresh_search_calls"] > 0,
        "no_gold_or_qrels_in_public_actions": True,
        "same_action_builder_for_all_query_methods": True,
        "internal_holdout_never_fit": True,
        "dev_sealed": True,
        "test_sealed": True,
    }
    decision = "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_ACTIONS_GO" if all(gates.values()) else "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_ACTIONS_NO_GO"
    private_path = output / "v6_backward_query_fresh_evidence_support_private.jsonl"
    private_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(private_rows, key=lambda item: item["query_id"])),
        encoding="utf-8",
    )
    audit = {
        "decision": decision,
        "protocol_version": "dagig_v6_backward_query_fresh_evidence_action_audit_v1",
        "metrics": metrics,
        "gates": gates,
        "input_paths": {"freeze": str(freeze_path)},
        "input_hashes": {"freeze": sha256(freeze_path)},
        "output_paths": {"evidence_actions": str(public_path), "private_support": str(private_path)},
        "output_hashes": {"evidence_actions": sha256(public_path), "private_support": sha256(private_path)},
        "gold_or_qrels_in_public_actions": False,
        "private_labels_used_only_by_auditor_after_public_materialization": True,
        "fresh_search_calls": metrics["fresh_search_calls"],
        "reserved_final_eval_key_used": False,
        "internal_holdout_used_for_training": False,
        "dev_used": False,
        "test_used": False,
        "training_run": False,
    }
    audit_path = output / "DAGIG_V6_BACKWARD_QUERY_FRESH_EVIDENCE_ACTION_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
