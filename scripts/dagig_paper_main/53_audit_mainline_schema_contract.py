#!/usr/bin/env python3
"""Audit the paper-main DAG-IG data, rollout, credit, and eval schema contract."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("outputs/dagig_paper_main_v1")
ASSETS = ROOT / "paper_assets"
DERIVED = Path("outputs/dagig_grpo_main/derived_assets")
ROLL_OUT = ROOT / "rollouts/train_rollouts_unified_scored.jsonl"
NODE_SUMMARY = ROOT / "reports/node_credit_component_analysis/node_credit_component_summary.json"

OUT_JSON = ASSETS / "MAINLINE_SCHEMA_CONTRACT_AUDIT.json"
OUT_MD = ASSETS / "MAINLINE_SCHEMA_CONTRACT_AUDIT.md"

DATA_FILES = {
    "train": DERIVED / "grpo_train.jsonl",
    "dev": DERIVED / "grpo_dev.jsonl",
    "test": DERIVED / "grpo_test.jsonl",
}
EXPECTED_COUNTS = {"train": 458, "dev": 98, "test": 64}
CORPORA = {
    "train": DERIVED / "bm25_train_corpus.jsonl",
    "eval": DERIVED / "bm25_eval_corpus.jsonl",
}
EXPECTED_CORPUS_COUNTS = {"train": 610, "eval": 201}

PREDICTION_FILES = {
    "seed42_dev": ROOT / "two_stage_predictions/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_dev.jsonl",
    "seed42_test": ROOT / "two_stage_predictions/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_ckpt60_test.jsonl",
    "seed43_dev": ROOT / "two_stage_predictions/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_dev.jsonl",
    "seed43_test": ROOT / "two_stage_predictions/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43_ckpt60_test.jsonl",
}

TRAIN_REWARD_ROLLOUTS = {
    "seed42_main": ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320/reward_rollouts.jsonl",
    "seed43_confirm": ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_scale60_s320_seed43/reward_rollouts.jsonl",
    "goldfixed_control": ROOT / "checkpoints/paper_main_v1_two_stage_stage1loss_kl01_goldfixed_scale60_s320/reward_rollouts.jsonl",
}

DATA_REQUIRED_KEYS = {
    "sample_id",
    "split",
    "question",
    "gold_answer",
    "model_image_path",
    "reward_fields",
    "evidence_urls",
}
REWARD_FIELD_KEYS = {
    "use_for_visual_credit",
    "use_for_query_credit",
    "use_for_evidence_credit",
    "use_for_answer_credit",
}
ROLLOUT_REQUIRED_KEYS = {
    "sample_id",
    "split",
    "source_run",
    "question",
    "gold_answer",
    "image_path",
    "rollout",
    "retrieval",
    "metrics",
    "node_credits",
}
ROLLOUT_NODE_KEYS = {"visual_observation", "search_query", "final_answer", "raw", "parsed_json"}
RETRIEVAL_KEYS = {"top_k", "support_rank5", "support_rank10", "mrr10", "hit5", "top_docs"}
METRIC_KEYS = {
    "format_valid",
    "query_nonempty",
    "evidence_supported",
    "answer_correct",
    "strict_success",
    "answer_in_query",
    "path_token_penalty",
}
CREDIT_KEYS = {
    "format_credit",
    "visual_credit",
    "query_credit",
    "evidence_credit",
    "answer_credit",
    "leak_penalty",
    "path_penalty",
    "total_reward",
}
REWARD_COMPONENT_KEYS = {
    "format",
    "visual",
    "query",
    "evidence",
    "answer",
    "leakage_penalty",
    "path_penalty",
}
PREDICTION_REQUIRED_KEYS = {
    "sample_id",
    "split",
    "question",
    "gold_answer",
    "image_path",
    "visual_observation",
    "search_query",
    "retrieved_docs",
    "final_answer",
    "stage1_raw_generation",
    "reader_raw_generation",
    "stage1_format_parse_success",
    "reader_format_parse_success",
    "retrieval_top5_hit",
    "answer_correct",
    "evidence_supported",
    "strict_success",
    "answer_in_query",
}

FORBIDDEN_GENERATION_MARKERS = {
    "oracle_crop_query",
    "teacher_oracle",
    "ground_truth",
    "target_doc",
    "gold_answer",
    "support_label",
    "answer_correct",
    "strict_success",
}

EXPECTED_SOURCE_RUNS = {
    "outcome_grpo",
    "trajectory_grpo",
    "dagig_grpo_no_visual",
    "dagig_grpo_full",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.1f}%"


def reward_from_credit(credits: dict[str, Any]) -> float:
    return (
        0.10 * float(credits.get("format_credit", 0.0))
        + 0.15 * float(credits.get("visual_credit", 0.0))
        + 0.40 * float(credits.get("query_credit", 0.0))
        + 0.25 * float(credits.get("evidence_credit", 0.0))
        + 0.35 * float(credits.get("answer_credit", 0.0))
        - float(credits.get("leak_penalty", 0.0))
        - float(credits.get("path_penalty", 0.0))
    )


def reward_from_components(components: dict[str, Any]) -> float:
    return (
        0.10 * float(components.get("format", 0.0))
        + 0.15 * float(components.get("visual", 0.0))
        + 0.40 * float(components.get("query", 0.0))
        + 0.25 * float(components.get("evidence", 0.0))
        + 0.35 * float(components.get("answer", 0.0))
        + float(components.get("leakage_penalty", 0.0))
        + float(components.get("path_penalty", 0.0))
    )


def text_has_forbidden_marker(text: str) -> list[str]:
    low = text.lower()
    return sorted(marker for marker in FORBIDDEN_GENERATION_MARKERS if marker in low)


def check_data_split_contract() -> dict[str, Any]:
    manifest = read_json(DERIVED / "derived_manifest.json")
    asset_root = Path(manifest["asset_root"])
    problems: list[str] = []
    split_info: dict[str, Any] = {}
    ids_by_split: dict[str, set[str]] = {}

    for split, path in DATA_FILES.items():
        if not path.exists():
            problems.append(f"{split}: missing {path}")
            split_info[split] = {"exists": False}
            continue
        rows = read_jsonl(path)
        ids = {str(row.get("sample_id")) for row in rows}
        ids_by_split[split] = ids
        missing_keys = 0
        bad_split = 0
        missing_reward_fields = 0
        bad_reward_flags = 0
        missing_images = 0
        forbidden_rows = 0
        evidence_url_rows = 0
        for row in rows:
            if not DATA_REQUIRED_KEYS <= set(row):
                missing_keys += 1
            if row.get("split") != split:
                bad_split += 1
            reward_fields = row.get("reward_fields") or {}
            if not REWARD_FIELD_KEYS <= set(reward_fields):
                missing_reward_fields += 1
            if any(reward_fields.get(key) is not True for key in REWARD_FIELD_KEYS):
                bad_reward_flags += 1
            image_path = row.get("model_image_path") or row.get("image_path")
            if image_path:
                image = Path(image_path)
                if not image.is_absolute():
                    image = asset_root / image
                if not image.exists():
                    missing_images += 1
            generation_text = json.dumps(
                {
                    "question": row.get("question"),
                    "metadata": row.get("metadata"),
                    "reward_fields": row.get("reward_fields"),
                },
                ensure_ascii=False,
            )
            if text_has_forbidden_marker(generation_text):
                forbidden_rows += 1
            if row.get("evidence_urls"):
                evidence_url_rows += 1
        if len(rows) != EXPECTED_COUNTS[split]:
            problems.append(f"{split}: rows {len(rows)} != expected {EXPECTED_COUNTS[split]}")
        if len(ids) != len(rows):
            problems.append(f"{split}: duplicate sample ids detected")
        if missing_keys:
            problems.append(f"{split}: {missing_keys} rows missing required keys")
        if bad_split:
            problems.append(f"{split}: {bad_split} rows have wrong split")
        if missing_reward_fields:
            problems.append(f"{split}: {missing_reward_fields} rows missing reward field keys")
        if bad_reward_flags:
            problems.append(f"{split}: {bad_reward_flags} rows have non-true reward field flags")
        if missing_images:
            problems.append(f"{split}: {missing_images} image paths missing")
        if forbidden_rows:
            problems.append(f"{split}: {forbidden_rows} rows contain forbidden markers in non-label fields")
        split_info[split] = {
            "exists": True,
            "rows": len(rows),
            "unique_sample_ids": len(ids),
            "expected_rows": EXPECTED_COUNTS[split],
            "missing_required_keys": missing_keys,
            "bad_split_rows": bad_split,
            "missing_reward_fields": missing_reward_fields,
            "bad_reward_flags": bad_reward_flags,
            "missing_images": missing_images,
            "evidence_url_rows": evidence_url_rows,
            "forbidden_marker_rows": forbidden_rows,
        }

    overlaps: dict[str, int] = {}
    for a in ids_by_split:
        for b in ids_by_split:
            if a >= b:
                continue
            n = len(ids_by_split[a] & ids_by_split[b])
            overlaps[f"{a}_{b}"] = n
            if n:
                problems.append(f"sample id overlap {a}/{b}: {n}")
    return {
        "asset_root": str(asset_root),
        "split_info": split_info,
        "sample_id_overlaps": overlaps,
        "problems": problems,
        "passed": not problems,
    }


def check_corpus_contract() -> dict[str, Any]:
    problems: list[str] = []
    corpora: dict[str, Any] = {}
    docs_by_label: dict[str, list[dict[str, Any]]] = {}
    for label, path in CORPORA.items():
        if not path.exists():
            problems.append(f"{label}: missing {path}")
            corpora[label] = {"exists": False}
            continue
        rows = read_jsonl(path)
        docs_by_label[label] = rows
        doc_ids = [str(row.get("doc_id")) for row in rows]
        urls = [str(row.get("url")) for row in rows if row.get("url")]
        if len(rows) != EXPECTED_CORPUS_COUNTS[label]:
            problems.append(f"{label}: rows {len(rows)} != expected {EXPECTED_CORPUS_COUNTS[label]}")
        if len(set(doc_ids)) != len(doc_ids):
            problems.append(f"{label}: duplicate doc_id detected")
        corpora[label] = {
            "exists": True,
            "rows": len(rows),
            "expected_rows": EXPECTED_CORPUS_COUNTS[label],
            "unique_doc_ids": len(set(doc_ids)),
            "unique_urls": len(set(urls)),
        }
    train_docs = docs_by_label.get("train", [])
    eval_docs = docs_by_label.get("eval", [])
    train_ids = {str(row.get("doc_id")) for row in train_docs}
    eval_ids = {str(row.get("doc_id")) for row in eval_docs}
    train_urls = {str(row.get("url")) for row in train_docs if row.get("url")}
    eval_urls = {str(row.get("url")) for row in eval_docs if row.get("url")}
    doc_overlap = sorted(train_ids & eval_ids)
    url_overlap = sorted(train_urls & eval_urls)
    if doc_overlap:
        problems.append(f"train/eval corpus doc_id overlap: {len(doc_overlap)}")
    if url_overlap:
        problems.append(f"train/eval corpus url overlap: {len(url_overlap)}")
    return {
        "corpora": corpora,
        "train_eval_doc_id_overlap": len(doc_overlap),
        "train_eval_url_overlap": len(url_overlap),
        "doc_id_overlap_examples": doc_overlap[:10],
        "url_overlap_examples": url_overlap[:10],
        "problems": problems,
        "passed": not problems,
    }


def check_unified_rollout_contract() -> dict[str, Any]:
    problems: list[str] = []
    if not ROLL_OUT.exists():
        return {"path": str(ROLL_OUT), "problems": [f"missing {ROLL_OUT}"], "passed": False}
    rows = read_jsonl(ROLL_OUT)
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    top_k_counts: Counter[str] = Counter()
    missing_top = 0
    missing_nested = 0
    parsed_json = 0
    query_nonempty = 0
    answer_in_query = 0
    strict_success = 0
    reward_mismatch = 0
    forbidden_rows = 0
    top_doc_count_bad = 0
    numeric_credit_bad = 0
    examples: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        source_counts[str(row.get("source_run"))] += 1
        split_counts[str(row.get("split"))] += 1
        bad: list[str] = []
        if not ROLLOUT_REQUIRED_KEYS <= set(row):
            missing = sorted(ROLLOUT_REQUIRED_KEYS - set(row))
            bad.append(f"missing top keys {missing}")
            missing_top += 1
        rollout = row.get("rollout") if isinstance(row.get("rollout"), dict) else {}
        retrieval = row.get("retrieval") if isinstance(row.get("retrieval"), dict) else {}
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        credits = row.get("node_credits") if isinstance(row.get("node_credits"), dict) else {}
        for label, obj, required in [
            ("rollout", rollout, ROLLOUT_NODE_KEYS),
            ("retrieval", retrieval, RETRIEVAL_KEYS),
            ("metrics", metrics, METRIC_KEYS),
            ("node_credits", credits, CREDIT_KEYS),
        ]:
            missing = sorted(required - set(obj))
            if missing:
                bad.append(f"missing {label} keys {missing}")
                missing_nested += 1
        if rollout.get("parsed_json") is True:
            parsed_json += 1
        if metrics.get("query_nonempty"):
            query_nonempty += 1
        if metrics.get("answer_in_query"):
            answer_in_query += 1
        if metrics.get("strict_success"):
            strict_success += 1
        if retrieval.get("top_k") is not None:
            top_k_counts[str(retrieval.get("top_k"))] += 1
        if retrieval.get("top_k") != 5:
            top_doc_count_bad += 1
        if len(retrieval.get("top_docs") or []) > 5:
            top_doc_count_bad += 1
        try:
            for key in CREDIT_KEYS:
                float(credits.get(key))
            if abs(reward_from_credit(credits) - float(credits.get("total_reward", 0.0))) > 1e-8:
                reward_mismatch += 1
                bad.append("node credit total_reward does not match formula")
        except Exception:
            numeric_credit_bad += 1
            bad.append("node credit is non-numeric")
        generation_text = json.dumps(
            {
                "raw": rollout.get("raw"),
                "visual_observation": rollout.get("visual_observation"),
                "search_query": rollout.get("search_query"),
                "final_answer": rollout.get("final_answer"),
            },
            ensure_ascii=False,
        )
        hits = text_has_forbidden_marker(generation_text)
        if hits:
            forbidden_rows += 1
            bad.append(f"forbidden generation markers {hits}")
        if bad and len(examples) < 20:
            examples.append({"line": idx, "sample_id": row.get("sample_id"), "problems": bad})

    if len(rows) < 458 * 4:
        problems.append(f"unified rollout rows {len(rows)} < 1832")
    if set(source_counts) != EXPECTED_SOURCE_RUNS:
        problems.append(f"source runs {sorted(source_counts)} != expected {sorted(EXPECTED_SOURCE_RUNS)}")
    if split_counts != {"train": len(rows)}:
        problems.append(f"rollout split counts not train-only: {dict(split_counts)}")
    if top_k_counts != {"5": len(rows)}:
        problems.append(f"top_k counts are not all 5: {dict(top_k_counts)}")
    if missing_top:
        problems.append(f"{missing_top} rows missing top-level keys")
    if missing_nested:
        problems.append(f"{missing_nested} nested schema key failures")
    if parsed_json / max(1, len(rows)) < 0.95:
        problems.append("parsed_json rate below 95%")
    if query_nonempty / max(1, len(rows)) < 0.95:
        problems.append("query_nonempty rate below 95%")
    if answer_in_query / max(1, len(rows)) > 0.02:
        problems.append("answer_in_query rate above 2%")
    if reward_mismatch:
        problems.append(f"{reward_mismatch} rows have reward formula mismatches")
    if numeric_credit_bad:
        problems.append(f"{numeric_credit_bad} rows have non-numeric credits")
    if forbidden_rows:
        problems.append(f"{forbidden_rows} rollout rows contain forbidden generation markers")
    if top_doc_count_bad:
        problems.append(f"{top_doc_count_bad} rows violate top-k retrieval contract")

    return {
        "path": str(ROLL_OUT),
        "rows": len(rows),
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "top_k_counts": dict(top_k_counts),
        "parsed_json_rate": parsed_json / max(1, len(rows)),
        "query_nonempty_rate": query_nonempty / max(1, len(rows)),
        "answer_in_query_rate": answer_in_query / max(1, len(rows)),
        "strict_success_rate": strict_success / max(1, len(rows)),
        "reward_formula_mismatches": reward_mismatch,
        "non_numeric_credit_rows": numeric_credit_bad,
        "forbidden_generation_rows": forbidden_rows,
        "invalid_examples": examples,
        "problems": problems,
        "passed": not problems,
    }


def check_training_reward_rollout_contract() -> dict[str, Any]:
    problems: list[str] = []
    runs: dict[str, Any] = {}
    for name, path in TRAIN_REWARD_ROLLOUTS.items():
        if not path.exists():
            problems.append(f"{name}: missing {path}")
            runs[name] = {"exists": False}
            continue
        rows = read_jsonl(path)
        groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        component_missing = 0
        parsed_missing = 0
        reward_mismatch = 0
        forbidden_rows = 0
        parsed_json = 0
        stage1_parsed = 0
        reader_parsed = 0
        for row in rows:
            groups[(int(row.get("micro_step", 0)), str(row.get("sample_id", "")))].append(row)
            components = row.get("components") if isinstance(row.get("components"), dict) else {}
            parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
            if not REWARD_COMPONENT_KEYS <= set(components):
                component_missing += 1
            if not {"raw", "visual_observation", "search_query", "final_answer", "stage1_raw", "reader_raw"} <= set(parsed):
                parsed_missing += 1
            try:
                if abs(reward_from_components(components) - float(row.get("reward", 0.0))) > 1e-8:
                    reward_mismatch += 1
            except Exception:
                reward_mismatch += 1
            if parsed.get("parsed_json"):
                parsed_json += 1
            if parsed.get("stage1_parsed_json"):
                stage1_parsed += 1
            if parsed.get("reader_parsed_json"):
                reader_parsed += 1
            hits = text_has_forbidden_marker(json.dumps(parsed, ensure_ascii=False))
            if hits:
                forbidden_rows += 1
        group_sizes = Counter(len(items) for items in groups.values())
        constant_groups = sum(
            1
            for items in groups.values()
            if max(float(item.get("reward", 0.0)) for item in items)
            - min(float(item.get("reward", 0.0)) for item in items)
            < 1e-8
        )
        if len(rows) != 960:
            problems.append(f"{name}: reward rollout rows {len(rows)} != 960")
        if len(groups) != 240:
            problems.append(f"{name}: reward groups {len(groups)} != 240")
        if set(group_sizes) != {4}:
            problems.append(f"{name}: group sizes {dict(group_sizes)} != all 4")
        if constant_groups / max(1, len(groups)) > 0.02:
            problems.append(f"{name}: constant reward group rate > 2%")
        if component_missing:
            problems.append(f"{name}: {component_missing} rows missing reward components")
        if parsed_missing:
            problems.append(f"{name}: {parsed_missing} rows missing parsed stage fields")
        if reward_mismatch:
            problems.append(f"{name}: {reward_mismatch} rows have reward formula mismatch")
        if forbidden_rows:
            problems.append(f"{name}: {forbidden_rows} rows contain forbidden generation markers")
        runs[name] = {
            "exists": True,
            "rows": len(rows),
            "groups": len(groups),
            "group_sizes": dict(group_sizes),
            "constant_groups": constant_groups,
            "constant_group_rate": constant_groups / max(1, len(groups)),
            "parsed_json_rate": parsed_json / max(1, len(rows)),
            "stage1_parsed_json_rate": stage1_parsed / max(1, len(rows)),
            "reader_parsed_json_rate": reader_parsed / max(1, len(rows)),
            "component_missing_rows": component_missing,
            "parsed_missing_rows": parsed_missing,
            "reward_formula_mismatches": reward_mismatch,
            "forbidden_generation_rows": forbidden_rows,
        }
    return {"runs": runs, "problems": problems, "passed": not problems}


def check_node_credit_summary_contract() -> dict[str, Any]:
    problems: list[str] = []
    if not NODE_SUMMARY.exists():
        return {"path": str(NODE_SUMMARY), "problems": [f"missing {NODE_SUMMARY}"], "passed": False}
    summary = read_json(NODE_SUMMARY)
    runs: dict[str, Any] = {}
    for name in ["seed42_main", "seed43_confirm", "goldfixed_control"]:
        run = summary.get(name) or {}
        components = run.get("components") or {}
        groups = run.get("groups") or {}
        missing_components = sorted({"format", "visual", "query", "evidence", "answer", "leakage_penalty", "path_penalty"} - set(components))
        if missing_components:
            problems.append(f"{name}: missing components {missing_components}")
        hit_auc = run.get("reward_auc_retrieval_hit")
        strict_auc = run.get("reward_auc_strict_success")
        constant_rate = groups.get("constant_group_rate")
        if hit_auc is None or hit_auc < 0.95:
            problems.append(f"{name}: reward_auc_retrieval_hit {hit_auc} < 0.95")
        if strict_auc is None or strict_auc < 0.90:
            problems.append(f"{name}: reward_auc_strict_success {strict_auc} < 0.90")
        if constant_rate is None or constant_rate > 0.02:
            problems.append(f"{name}: constant_group_rate {constant_rate} > 0.02")
        for component, metric, threshold in [
            ("query", "auc_retrieval_hit", 0.95),
            ("evidence", "auc_retrieval_hit", 0.95),
            ("answer", "auc_strict_success", 0.95),
        ]:
            value = (components.get(component) or {}).get(metric)
            if value is None or value < threshold:
                problems.append(f"{name}: component {component}.{metric} {value} < {threshold}")
        runs[name] = {
            "reward_auc_retrieval_hit": hit_auc,
            "reward_auc_strict_success": strict_auc,
            "constant_group_rate": constant_rate,
            "query_auc_retrieval_hit": (components.get("query") or {}).get("auc_retrieval_hit"),
            "evidence_auc_retrieval_hit": (components.get("evidence") or {}).get("auc_retrieval_hit"),
            "answer_auc_strict_success": (components.get("answer") or {}).get("auc_strict_success"),
            "top_strict_success": groups.get("top_strict_success"),
            "bottom_strict_success": groups.get("bottom_strict_success"),
        }
    return {"path": str(NODE_SUMMARY), "runs": runs, "problems": problems, "passed": not problems}


def check_two_stage_prediction_contract() -> dict[str, Any]:
    problems: list[str] = []
    files: dict[str, Any] = {}
    for label, path in PREDICTION_FILES.items():
        if not path.exists():
            problems.append(f"{label}: missing {path}")
            files[label] = {"exists": False}
            continue
        rows = read_jsonl(path)
        split = "test" if label.endswith("_test") else "dev"
        expected = EXPECTED_COUNTS[split]
        missing_keys = 0
        stage1_fail = 0
        reader_fail = 0
        query_empty = 0
        answer_in_query = 0
        too_many_doc_count = 0
        short_doc_count = 0
        bad_retrieved_doc = 0
        bad_doc_rank = 0
        forbidden_rows = 0
        eval_file_bad = 0
        corpus_bad = 0
        sample_ids = set()
        for row in rows:
            sample_ids.add(str(row.get("sample_id")))
            if not PREDICTION_REQUIRED_KEYS <= set(row):
                missing_keys += 1
            if row.get("split") != split:
                eval_file_bad += 1
            if not row.get("stage1_format_parse_success"):
                stage1_fail += 1
            if not row.get("reader_format_parse_success"):
                reader_fail += 1
            if not str(row.get("search_query") or "").strip():
                query_empty += 1
            if row.get("answer_in_query"):
                answer_in_query += 1
            docs = row.get("retrieved_docs") or []
            if len(docs) > 5:
                too_many_doc_count += 1
            if len(docs) < 5:
                short_doc_count += 1
            for doc in docs:
                if not {"doc_id", "rank", "text", "url", "is_gold"} <= set(doc):
                    bad_retrieved_doc += 1
                    break
                try:
                    rank = int(doc.get("rank"))
                    if rank < 1 or rank > 5:
                        bad_doc_rank += 1
                        break
                except Exception:
                    bad_doc_rank += 1
                    break
            generation_text = json.dumps(
                {
                    "stage1_raw_generation": row.get("stage1_raw_generation"),
                    "reader_raw_generation": row.get("reader_raw_generation"),
                    "visual_observation": row.get("visual_observation"),
                    "search_query": row.get("search_query"),
                    "final_answer": row.get("final_answer"),
                },
                ensure_ascii=False,
            )
            if text_has_forbidden_marker(generation_text):
                forbidden_rows += 1
            if row.get("corpus_path") and Path(row["corpus_path"]).name != "bm25_eval_corpus.jsonl":
                corpus_bad += 1
        if len(rows) != expected:
            problems.append(f"{label}: rows {len(rows)} != expected {expected}")
        if len(sample_ids) != len(rows):
            problems.append(f"{label}: duplicate sample ids")
        if missing_keys:
            problems.append(f"{label}: {missing_keys} rows missing required keys")
        if stage1_fail:
            problems.append(f"{label}: {stage1_fail} stage1 parse failures")
        if reader_fail / max(1, len(rows)) > 0.05:
            problems.append(f"{label}: reader parse failure rate above 5%")
        if query_empty:
            problems.append(f"{label}: {query_empty} empty search queries")
        if answer_in_query / max(1, len(rows)) > 0.02:
            problems.append(f"{label}: answer-in-query rate above 2%")
        if too_many_doc_count:
            problems.append(f"{label}: {too_many_doc_count} rows have more than top-5 docs")
        if bad_retrieved_doc:
            problems.append(f"{label}: {bad_retrieved_doc} rows have malformed retrieved docs")
        if bad_doc_rank:
            problems.append(f"{label}: {bad_doc_rank} rows have retrieved docs outside rank 1..5")
        if forbidden_rows:
            problems.append(f"{label}: {forbidden_rows} rows contain forbidden generation markers")
        if eval_file_bad:
            problems.append(f"{label}: {eval_file_bad} rows have wrong split")
        if corpus_bad:
            problems.append(f"{label}: {corpus_bad} rows did not use eval corpus")
        files[label] = {
            "exists": True,
            "rows": len(rows),
            "expected_rows": expected,
            "unique_sample_ids": len(sample_ids),
            "missing_key_rows": missing_keys,
            "stage1_parse_failures": stage1_fail,
            "reader_parse_failures": reader_fail,
            "query_empty_rows": query_empty,
            "answer_in_query_rows": answer_in_query,
            "short_doc_count_rows": short_doc_count,
            "too_many_doc_count_rows": too_many_doc_count,
            "bad_retrieved_doc_rows": bad_retrieved_doc,
            "bad_doc_rank_rows": bad_doc_rank,
            "forbidden_generation_rows": forbidden_rows,
            "wrong_split_rows": eval_file_bad,
            "wrong_corpus_rows": corpus_bad,
            "strict_success_rate": sum(1 for row in rows if row.get("strict_success")) / max(1, len(rows)),
            "retrieval_top5_rate": sum(1 for row in rows if row.get("retrieval_top5_hit")) / max(1, len(rows)),
        }
    return {"files": files, "problems": problems, "passed": not problems}


def build_audit() -> dict[str, Any]:
    checks = {
        "data_split_contract": check_data_split_contract(),
        "corpus_contract": check_corpus_contract(),
        "unified_rollout_contract": check_unified_rollout_contract(),
        "training_reward_rollout_contract": check_training_reward_rollout_contract(),
        "node_credit_summary_contract": check_node_credit_summary_contract(),
        "two_stage_prediction_contract": check_two_stage_prediction_contract(),
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Machine-checkable contract for the DAG-IG paper mainline data, rollout schema, node credit, reward audit, and two-stage prediction files.",
        "overall_pass": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "contract": {
            "agent_rollout": "image+question -> visual_observation -> search_query -> BM25 top-5 evidence -> final_answer",
            "node_credits": ["visual", "query", "evidence", "answer"],
            "reward_formula": "0.10*format + 0.15*visual + 0.40*query + 0.25*evidence + 0.35*answer - leak_penalty - path_penalty",
            "train_corpus": str(CORPORA["train"]),
            "eval_corpus": str(CORPORA["eval"]),
            "forbidden_generation_markers": sorted(FORBIDDEN_GENERATION_MARKERS),
        },
    }


def build_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Mainline Schema Contract Audit\n\n")
    lines.append("This audit checks the paper-main DAG-IG contract directly from current files. It does not train models, edit predictions, or create new experimental results.\n\n")
    lines.append(f"- created_at_utc: `{audit['created_at_utc']}`\n")
    lines.append(f"- overall pass: `{audit['overall_pass']}`\n")
    lines.append(f"- rollout contract: `{audit['contract']['agent_rollout']}`\n")
    lines.append(f"- reward formula: `{audit['contract']['reward_formula']}`\n\n")

    data = audit["checks"]["data_split_contract"]
    lines.append("## 1. Data Split Contract\n\n")
    lines.append("| split | rows | expected | unique sample ids | image misses | reward field failures | evidence-url rows |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for split in ["train", "dev", "test"]:
        item = data["split_info"].get(split, {})
        reward_failures = int(item.get("missing_reward_fields", 0)) + int(item.get("bad_reward_flags", 0))
        lines.append(
            f"| {split} | {item.get('rows')} | {item.get('expected_rows')} | {item.get('unique_sample_ids')} | "
            f"{item.get('missing_images')} | {reward_failures} | {item.get('evidence_url_rows')} |\n"
        )
    lines.append(f"\n- sample id overlaps: `{data['sample_id_overlaps']}`\n")
    lines.append(f"- passed: `{data['passed']}`\n\n")

    corpus = audit["checks"]["corpus_contract"]
    lines.append("## 2. Corpus Contract\n\n")
    lines.append("| corpus | rows | expected | unique doc ids | unique URLs |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for label in ["train", "eval"]:
        item = corpus["corpora"].get(label, {})
        lines.append(f"| {label} | {item.get('rows')} | {item.get('expected_rows')} | {item.get('unique_doc_ids')} | {item.get('unique_urls')} |\n")
    lines.append(f"\n- train/eval doc_id overlap: `{corpus['train_eval_doc_id_overlap']}`\n")
    lines.append(f"- train/eval URL overlap: `{corpus['train_eval_url_overlap']}`\n")
    lines.append(f"- passed: `{corpus['passed']}`\n\n")

    rollout = audit["checks"]["unified_rollout_contract"]
    lines.append("## 3. Unified Rollout Contract\n\n")
    lines.append(f"- rows: `{rollout['rows']}`\n")
    lines.append(f"- source counts: `{rollout['source_counts']}`\n")
    lines.append(f"- split counts: `{rollout['split_counts']}`\n")
    lines.append(f"- top-k counts: `{rollout['top_k_counts']}`\n")
    lines.append(f"- parsed JSON rate: `{pct(rollout['parsed_json_rate'])}`\n")
    lines.append(f"- query nonempty rate: `{pct(rollout['query_nonempty_rate'])}`\n")
    lines.append(f"- answer-in-query rate: `{pct(rollout['answer_in_query_rate'])}`\n")
    lines.append(f"- reward formula mismatches: `{rollout['reward_formula_mismatches']}`\n")
    lines.append(f"- forbidden generation rows: `{rollout['forbidden_generation_rows']}`\n")
    lines.append(f"- passed: `{rollout['passed']}`\n\n")

    train = audit["checks"]["training_reward_rollout_contract"]
    lines.append("## 4. Training Reward Rollout Contract\n\n")
    lines.append("| run | rows | groups | group sizes | constant groups | stage1 parsed | reader parsed | reward mismatches |\n")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|\n")
    for name, item in train["runs"].items():
        lines.append(
            f"| {name} | {item.get('rows')} | {item.get('groups')} | `{item.get('group_sizes')}` | "
            f"{item.get('constant_groups')} ({pct(item.get('constant_group_rate'))}) | "
            f"{pct(item.get('stage1_parsed_json_rate'))} | {pct(item.get('reader_parsed_json_rate'))} | "
            f"{item.get('reward_formula_mismatches')} |\n"
        )
    lines.append(f"\n- passed: `{train['passed']}`\n\n")

    node = audit["checks"]["node_credit_summary_contract"]
    lines.append("## 5. Node Credit Summary Contract\n\n")
    lines.append("| run | reward AUC hit | reward AUC strict | constant groups | query AUC hit | evidence AUC hit | answer AUC strict |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for name, item in node["runs"].items():
        lines.append(
            f"| {name} | {item.get('reward_auc_retrieval_hit'):.3f} | {item.get('reward_auc_strict_success'):.3f} | "
            f"{pct(item.get('constant_group_rate'))} | {item.get('query_auc_retrieval_hit'):.3f} | "
            f"{item.get('evidence_auc_retrieval_hit'):.3f} | {item.get('answer_auc_strict_success'):.3f} |\n"
        )
    lines.append(f"\n- passed: `{node['passed']}`\n\n")

    preds = audit["checks"]["two_stage_prediction_contract"]
    lines.append("## 6. Two-Stage Prediction Contract\n\n")
    lines.append("| file | rows | top5 | strict | short retrieval rows | stage1 failures | reader failures | answer-in-query | malformed docs |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, item in preds["files"].items():
        lines.append(
            f"| {name} | {item.get('rows')} | {pct(item.get('retrieval_top5_rate'))} | {pct(item.get('strict_success_rate'))} | "
            f"{item.get('short_doc_count_rows')} | {item.get('stage1_parse_failures')} | {item.get('reader_parse_failures')} | {item.get('answer_in_query_rows')} | "
            f"{item.get('bad_retrieved_doc_rows')} |\n"
        )
    lines.append(f"\n- passed: `{preds['passed']}`\n\n")

    all_problems: list[str] = []
    for check_name, check in audit["checks"].items():
        for problem in check.get("problems", []):
            all_problems.append(f"{check_name}: {problem}")
    lines.append("## Problems\n\n")
    if all_problems:
        for problem in all_problems:
            lines.append(f"- {problem}\n")
    else:
        lines.append("- none\n")
    lines.append("\n## Boundary\n\n")
    lines.append("This audit supports only the main paper method: DAG-IG node-level GRPO for the two-stage Pix2Fact search agent. It does not make DAG-SFT, query reranking, evidence fusion, DPO, or answer repair part of the main method.\n")
    return "".join(lines)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_markdown(audit), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"overall_pass={audit['overall_pass']}")
    if not audit["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
