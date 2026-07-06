#!/usr/bin/env python3
"""Build auditable derived assets for the GRPO run.

The downloaded GRPO asset currently contains empty warmup/corpus files. This
script does not mutate the package. It derives missing files from package-local
source/grpo rows and writes them under outputs/dagig_grpo_main/derived_assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import (
    DEFAULT_ASSET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    has_forbidden_marker,
    read_jsonl,
    stable_doc_id,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def evidence_docs(rows: list[dict[str, Any]], split_group: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        for idx, ev in enumerate(row.get("evidences") or [], 1):
            if not isinstance(ev, dict):
                continue
            text = str(ev.get("text", "") or "")
            url = str(ev.get("url", "") or "")
            title = str(ev.get("title", "") or ev.get("domain", "") or "")
            if not (text or url or title):
                continue
            doc_id = stable_doc_id(sample_id, int(ev.get("rank", idx) or idx), url, text)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            docs.append(
                {
                    "doc_id": doc_id,
                    "sample_id": sample_id,
                    "split": row.get("split", split_group),
                    "title": title,
                    "text": text,
                    "url": url,
                    "domain": ev.get("domain", ""),
                    "rank": ev.get("rank", idx),
                    "source": f"{split_group}_source_evidence",
                    "is_gold": bool(ev.get("answer_supported")),
                }
            )
    return docs


def make_sft_row(asset_root: Path, grpo_row: dict[str, Any], source_row: dict[str, Any], prompt_text: str) -> dict[str, Any]:
    image_rel = grpo_row.get("model_image_path") or grpo_row.get("image_path")
    image_path = asset_root / str(image_rel)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image for {grpo_row.get('sample_id')}: {image_path}")
    visual = (
        source_row.get("ground_expression")
        or (source_row.get("grounding") or {}).get("ground_expression")
        or source_row.get("semantic_anchor")
        or source_row.get("image_description")
        or ""
    )
    query = source_row.get("hf_search_query") or source_row.get("semantic_anchor") or grpo_row.get("question", "")
    answer = grpo_row.get("gold_answer") or source_row.get("answer") or ""
    target = {
        "visual_observation": str(visual).strip()[:220],
        "search_query": str(query).strip()[:180],
        "final_answer": str(answer).strip(),
    }
    return {
        "sample_id": grpo_row.get("sample_id"),
        "image_path": str(image_path.resolve()),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve()), "max_pixels": 1003520},
                    {"type": "text", "text": f"{prompt_text.strip()}\n\nQuestion: {str(grpo_row.get('question', '')).strip()}"},
                ],
            },
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ],
        "setting": "format_sft_warmup_derived_from_package_source_fields",
    }


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.resolve()
    output_root = args.output_root.resolve()
    derived = output_root / "derived_assets"
    derived.mkdir(parents=True, exist_ok=True)

    prompt_path = asset_root / "prompts/grpo_policy_prompt.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    original_counts: dict[str, int] = {}
    forbidden_hits: list[str] = []
    for rel in [
        "data/grpo_train.jsonl",
        "data/grpo_dev.jsonl",
        "data/grpo_test.jsonl",
        "data/source_train.jsonl",
        "data/source_dev.jsonl",
        "data/source_test.jsonl",
        "data/reward_metadata.jsonl",
        "data/sft_warmup_train.jsonl",
        "data/bm25_train_corpus.jsonl",
        "data/bm25_eval_corpus.jsonl",
    ]:
        rows = read_jsonl(asset_root / rel)
        original_counts[rel] = len(rows)
        if any(has_forbidden_marker(row) for row in rows):
            forbidden_hits.append(rel)

    source_train = read_jsonl(asset_root / "data/source_train.jsonl")
    source_dev = read_jsonl(asset_root / "data/source_dev.jsonl")
    source_test = read_jsonl(asset_root / "data/source_test.jsonl")
    grpo_train = read_jsonl(asset_root / "data/grpo_train.jsonl")
    grpo_dev = read_jsonl(asset_root / "data/grpo_dev.jsonl")
    grpo_test = read_jsonl(asset_root / "data/grpo_test.jsonl")
    source_by_id = {
        **load_by_sample(asset_root / "data/source_train.jsonl"),
        **load_by_sample(asset_root / "data/source_dev.jsonl"),
        **load_by_sample(asset_root / "data/source_test.jsonl"),
    }

    train_docs = evidence_docs(source_train, "train")
    eval_docs = evidence_docs(source_dev + source_test, "eval")
    if not train_docs or not eval_docs:
        raise RuntimeError("Derived BM25 corpora would be empty; refusing to continue.")
    write_jsonl(derived / "bm25_train_corpus.jsonl", train_docs)
    write_jsonl(derived / "bm25_eval_corpus.jsonl", eval_docs)
    counts["derived_bm25_train_docs"] = len(train_docs)
    counts["derived_bm25_eval_docs"] = len(eval_docs)
    counts["derived_train_gold_docs"] = sum(1 for doc in train_docs if doc.get("is_gold"))
    counts["derived_eval_gold_docs"] = sum(1 for doc in eval_docs if doc.get("is_gold"))

    sft_rows = []
    for row in grpo_train:
        sid = str(row.get("sample_id", ""))
        if sid not in source_by_id:
            raise KeyError(f"Missing source row for warmup sample {sid}")
        sft_rows.append(make_sft_row(asset_root, row, source_by_id[sid], prompt_text))
    write_jsonl(derived / "sft_warmup_train.jsonl", sft_rows)
    counts["derived_sft_warmup_train"] = len(sft_rows)

    # Copy GRPO train/dev/test with absolute image paths for training/eval scripts.
    for split, rows in [("train", grpo_train), ("dev", grpo_dev), ("test", grpo_test)]:
        out_rows = []
        for row in rows:
            row = dict(row)
            image_rel = row.get("model_image_path") or row.get("image_path")
            image_path = asset_root / str(image_rel)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for {row.get('sample_id')}: {image_path}")
            row["image_abs_path"] = str(image_path.resolve())
            src = source_by_id.get(str(row.get("sample_id")), {})
            for key in ("ground_expression", "grounding", "semantic_anchor", "hf_search_query", "caption", "image_description"):
                if key in src:
                    row[key] = src[key]
            out_rows.append(row)
        write_jsonl(derived / f"grpo_{split}.jsonl", out_rows)
        counts[f"derived_grpo_{split}"] = len(out_rows)

    audit = {
        "asset_root": str(asset_root),
        "output_root": str(output_root),
        "derived_assets_root": str(derived),
        "original_counts": original_counts,
        "derived_counts": counts,
        "forbidden_marker_files": forbidden_hits,
        "repair_reason": (
            "Downloaded asset validates but has empty sft_warmup_train and empty bm25 corpus files. "
            "Derived files are built only from package-local source/grpo/evidence fields and do not mutate the asset."
        ),
        "image_field_used": "model_image_path (same value as image_path in sampled rows)",
        "hard_fail": bool(forbidden_hits),
    }
    write_json(output_root / "data_audit.json", audit)
    write_json(derived / "derived_manifest.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if forbidden_hits:
        raise RuntimeError(f"Forbidden markers found in package files: {forbidden_hits}")


if __name__ == "__main__":
    main()
