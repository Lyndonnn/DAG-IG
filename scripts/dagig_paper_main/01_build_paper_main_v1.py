#!/usr/bin/env python3
"""Build DAG-IG paper-main v1 unified rollout and reward audit artifacts.

This is not a training script. It freezes the current paper-facing protocol:

image + question -> visual_observation -> search_query -> retrieve top-k evidence -> final_answer

and rescoring existing sampled rollouts with dense node-level DAG-IG credits:
visual/query/evidence/answer. The output answers the one question that matters
before any new GRPO run: does the reward have enough non-constant signal?
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import (  # noqa: E402
    BM25Index,
    answer_leaks_in_query,
    answer_match_details,
    load_corpus,
    parse_policy_output,
    read_jsonl,
    tokenize,
    write_json,
    write_jsonl,
)


DEFAULT_DERIVED = PROJECT_ROOT / "outputs/dagig_grpo_main/derived_assets"
DEFAULT_OLD_GRPO = PROJECT_ROOT / "outputs/dagig_grpo_main/checkpoints"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/dagig_paper_main_v1"
ROLLOUT_RUNS = ("outcome_grpo", "trajectory_grpo", "dagig_grpo_no_visual", "dagig_grpo_full")
STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "with",
    "to",
    "for",
    "is",
    "are",
    "image",
    "picture",
    "photo",
    "shown",
    "visible",
    "question",
    "answer",
    "what",
    "which",
    "where",
    "when",
    "how",
    "please",
    "tell",
    "could",
}
PATH_TOKENS = {
    "http",
    "https",
    "www",
    "url",
    "url1",
    "url2",
    "url3",
    "wiki",
    "wikipedia",
    "yelp",
    "biz",
    "maps",
    "map",
    "google",
}


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def std(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def terms_from_texts(texts: list[str]) -> set[str]:
    out: set[str] = set()
    for text in texts:
        for tok in tokenize(text or ""):
            if len(tok) >= 3 and tok not in STOP and tok not in PATH_TOKENS:
                out.add(tok)
    return out


def visual_anchor_terms(row: dict[str, Any]) -> set[str]:
    texts = [
        str(row.get("ground_expression") or ""),
        str(row.get("semantic_anchor") or ""),
        str(row.get("image_description") or ""),
    ]
    grounding = row.get("grounding")
    if isinstance(grounding, dict):
        texts.extend(
            [
                str(grounding.get("ground_expression") or ""),
                str(grounding.get("semantic_anchor") or ""),
                str(grounding.get("visible_text_or_name") or ""),
                " ".join(str(x) for x in (grounding.get("visual_disambiguators") or [])),
            ]
        )
    return terms_from_texts(texts)


def query_quality_penalty(query: str) -> float:
    toks = set(tokenize(query or ""))
    penalty = 0.0
    if any(t in PATH_TOKENS or re.fullmatch(r"url\d*", t) for t in toks):
        penalty += 0.15
    if len(tokenize(query or "")) > 20:
        penalty += 0.05
    return penalty


def support_rank(docs: list[dict[str, Any]], sample_id: str, k: int = 10) -> int | None:
    for idx, doc in enumerate(docs[:k], 1):
        if str(doc.get("sample_id")) == str(sample_id) and bool(doc.get("is_gold")):
            return idx
    return None


def auc_binary(scores: list[float], labels: list[int]) -> float | None:
    pos = [(s, y) for s, y in zip(scores, labels) if y]
    neg = [(s, y) for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = ties = total = 0
    for ps, _ in pos:
        for ns, _ in neg:
            wins += int(ps > ns)
            ties += int(abs(ps - ns) < 1e-12)
            total += 1
    return (wins + 0.5 * ties) / total


def score_rollout(
    row: dict[str, Any],
    rollout: dict[str, Any],
    bm25: BM25Index,
    top_k: int,
) -> dict[str, Any]:
    parsed = parse_policy_output((rollout.get("parsed") or {}).get("raw") or "")
    if not parsed.get("raw"):
        parsed = rollout.get("parsed") or parsed
    visual = str(parsed.get("visual_observation") or "").strip()
    query = str(parsed.get("search_query") or "").strip()
    answer = str(parsed.get("final_answer") or "").strip()
    docs = bm25.search(query, top_k=max(10, top_k)) if query else []
    rank = support_rank(docs, row["sample_id"], k=10)
    rank5 = support_rank(docs, row["sample_id"], k=top_k)
    retrieval_mrr10 = 1.0 / rank if rank else 0.0
    retrieval_hit5 = rank5 is not None
    answer_match = answer_match_details(answer, str(row.get("gold_answer") or ""))
    answer_correct = bool(answer_match["answer_correct"])
    strict_success = bool(answer_correct and retrieval_hit5)

    fmt_credit = 0.0
    fmt_credit += 0.03 if parsed.get("parsed_json") else 0.0
    fmt_credit += 0.02 if visual else 0.0
    fmt_credit += 0.03 if query else 0.0
    fmt_credit += 0.02 if answer else 0.0
    fmt_credit = min(fmt_credit, 0.10)

    anchors = visual_anchor_terms(row)
    visual_tokens = set(tokenize(visual))
    visual_overlap = len(anchors & visual_tokens)
    visual_credit = min(1.0, visual_overlap / max(3, min(8, len(anchors)))) if visual else 0.0

    query_credit = retrieval_mrr10
    evidence_credit = 1.0 / rank5 if rank5 else 0.0
    answer_credit = 1.0 if strict_success else (0.35 if answer_correct else 0.0)
    leak_penalty = 0.25 if answer_leaks_in_query(query, str(row.get("gold_answer") or "")) else 0.0
    path_penalty = query_quality_penalty(query)
    total = (
        0.10 * fmt_credit
        + 0.15 * visual_credit
        + 0.40 * query_credit
        + 0.25 * evidence_credit
        + 0.35 * answer_credit
        - leak_penalty
        - path_penalty
    )
    total = max(-0.5, float(total))
    return {
        "sample_id": row["sample_id"],
        "split": row.get("split"),
        "source_run": rollout.get("source_run"),
        "epoch": rollout.get("epoch"),
        "micro_step": rollout.get("micro_step"),
        "generation_index": rollout.get("generation_index"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "image_path": row.get("image_path"),
        "rollout": {
            "visual_observation": visual,
            "search_query": query,
            "final_answer": answer,
            "raw": (rollout.get("parsed") or {}).get("raw", ""),
            "parsed_json": bool(parsed.get("parsed_json")),
        },
        "retrieval": {
            "top_k": top_k,
            "support_rank5": rank5,
            "support_rank10": rank,
            "mrr10": retrieval_mrr10,
            "hit5": retrieval_hit5,
            "top_docs": [
                {
                    "rank": idx,
                    "doc_id": doc.get("doc_id"),
                    "sample_id": doc.get("sample_id"),
                    "is_gold": bool(doc.get("is_gold")),
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "text": str(doc.get("text") or "")[:300],
                }
                for idx, doc in enumerate(docs[:top_k], 1)
            ],
        },
        "metrics": {
            "format_valid": bool(parsed.get("parsed_json") and query and answer),
            "query_nonempty": bool(query),
            "evidence_supported": retrieval_hit5,
            "answer_correct": answer_correct,
            "strict_success": strict_success,
            "answer_match_type": answer_match.get("answer_match_type"),
            "answer_in_query": bool(leak_penalty),
            "path_token_penalty": path_penalty,
        },
        "node_credits": {
            "format_credit": fmt_credit,
            "visual_credit": visual_credit,
            "query_credit": query_credit,
            "evidence_credit": evidence_credit,
            "answer_credit": answer_credit,
            "leak_penalty": leak_penalty,
            "path_penalty": path_penalty,
            "total_reward": total,
        },
    }


def group_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row.get("source_run")), int(row.get("micro_step") or -1), str(row.get("sample_id")))


def build_schema(out_dir: Path) -> None:
    schema = """# DAG-IG Paper Main v1 Schema

This is the only paper-main rollout schema for the next stage.

Pipeline:

```text
image + question
-> visual_observation
-> search_query
-> retrieve top-k evidence
-> final_answer
```

Each rollout row contains:

```json
{
  "sample_id": "...",
  "split": "train",
  "source_run": "dagig_grpo_full",
  "question": "...",
  "gold_answer": "...",
  "image_path": "...",
  "rollout": {
    "visual_observation": "...",
    "search_query": "...",
    "final_answer": "...",
    "raw": "...",
    "parsed_json": true
  },
  "retrieval": {
    "top_k": 5,
    "support_rank5": 1,
    "support_rank10": 1,
    "mrr10": 1.0,
    "hit5": true,
    "top_docs": []
  },
  "metrics": {
    "format_valid": true,
    "query_nonempty": true,
    "evidence_supported": true,
    "answer_correct": false,
    "strict_success": false,
    "answer_in_query": false
  },
  "node_credits": {
    "format_credit": 0.0,
    "visual_credit": 0.0,
    "query_credit": 0.0,
    "evidence_credit": 0.0,
    "answer_credit": 0.0,
    "leak_penalty": 0.0,
    "path_penalty": 0.0,
    "total_reward": 0.0
  }
}
```

Allowed reward-time information:

- generated visual_observation/search_query/final_answer;
- BM25 retrieval over the frozen train corpus during training;
- train support labels for train reward only;
- answer normalization against train gold answer for train reward only.

Disallowed:

- dev/test labels during training;
- teacher/oracle query as policy input;
- GPT/raw_pool/Qwen32B training data;
- URL/path-token shortcuts as positive query credit.
"""
    path = out_dir / "protocol/PAPER_MAIN_V1_SCHEMA.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema, encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_root
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    build_schema(out)
    train_rows = {str(r["sample_id"]): r for r in read_jsonl(args.derived_assets / "grpo_train.jsonl")}
    corpus = load_corpus(args.derived_assets / "bm25_train_corpus.jsonl")
    if not corpus:
        raise RuntimeError(f"Empty corpus: {args.derived_assets / 'bm25_train_corpus.jsonl'}")
    bm25 = BM25Index.from_docs(corpus)
    scored: list[dict[str, Any]] = []
    missing_rows = 0
    for run in args.rollout_runs:
        path = args.old_grpo_root / run / "reward_rollouts.jsonl"
        if not path.is_file():
            continue
        for rollout in read_jsonl(path):
            sid = str(rollout.get("sample_id"))
            row = train_rows.get(sid)
            if not row:
                missing_rows += 1
                continue
            rollout = dict(rollout)
            rollout["source_run"] = run
            scored.append(score_rollout(row, rollout, bm25, top_k=args.top_k))
    write_jsonl(out / "rollouts/train_rollouts_unified_scored.jsonl", scored)

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[group_key(row)].append(row)
    constant = 0
    group_stats = []
    for key, rows in groups.items():
        rewards = [float(r["node_credits"]["total_reward"]) for r in rows]
        rng = max(rewards) - min(rewards) if rewards else 0.0
        constant += int(rng < 1e-8)
        group_stats.append({"key": key, "n": len(rows), "mean": mean(rewards), "std": std(rewards), "range": rng})
    rewards = [float(r["node_credits"]["total_reward"]) for r in scored]
    labels_hit = [1 if r["metrics"]["evidence_supported"] else 0 for r in scored]
    labels_answer = [1 if r["metrics"]["answer_correct"] else 0 for r in scored]
    labels_strict = [1 if r["metrics"]["strict_success"] else 0 for r in scored]
    query_scores = [float(r["node_credits"]["query_credit"]) for r in scored]
    evidence_scores = [float(r["node_credits"]["evidence_credit"]) for r in scored]
    answer_scores = [float(r["node_credits"]["answer_credit"]) for r in scored]
    total_scores = [float(r["node_credits"]["total_reward"]) for r in scored]
    component_nonzero = {
        name: sum(1 for r in scored if abs(float(r["node_credits"].get(name, 0.0))) > 1e-12) / max(1, len(scored))
        for name in ["format_credit", "visual_credit", "query_credit", "evidence_credit", "answer_credit", "leak_penalty", "path_penalty"]
    }
    group_count = len(groups)
    audit = {
        "status": "go" if group_count and constant / group_count < args.max_constant_group_rate else "no_go",
        "input": {
            "derived_assets": str(args.derived_assets.resolve()),
            "old_grpo_root": str(args.old_grpo_root.resolve()),
            "rollout_runs": args.rollout_runs,
            "top_k": args.top_k,
        },
        "counts": {
            "rollouts": len(scored),
            "groups": group_count,
            "missing_train_rows": missing_rows,
            "constant_reward_groups": constant,
            "constant_reward_group_rate": constant / max(1, group_count),
        },
        "reward_distribution": {
            "mean": mean(rewards),
            "std": std(rewards),
            "min": min(rewards) if rewards else 0.0,
            "max": max(rewards) if rewards else 0.0,
        },
        "component_nonzero_rate": component_nonzero,
        "predictiveness": {
            "query_credit_auc_support_hit": auc_binary(query_scores, labels_hit),
            "evidence_credit_auc_support_hit": auc_binary(evidence_scores, labels_hit),
            "answer_credit_auc_answer_correct": auc_binary(answer_scores, labels_answer),
            "total_reward_auc_strict": auc_binary(total_scores, labels_strict),
        },
        "gates": {
            "max_constant_group_rate": args.max_constant_group_rate,
            "constant_group_gate": bool(group_count and constant / group_count < args.max_constant_group_rate),
            "min_query_auc": args.min_query_auc,
            "query_auc_gate": bool((auc_binary(query_scores, labels_hit) or 0.0) >= args.min_query_auc),
            "min_total_auc": args.min_total_auc,
            "total_auc_gate": bool((auc_binary(total_scores, labels_strict) or 0.0) >= args.min_total_auc),
        },
    }
    audit["status"] = "go" if all(audit["gates"][k] for k in ("constant_group_gate", "query_auc_gate", "total_auc_gate")) else "no_go"
    write_json(out / "reports/reward_audit.json", audit)

    lines = [
        "# DAG-IG Paper Main v1 Reward Audit",
        "",
        "## 1. Protocol",
        "",
        "- Frozen rollout schema: `protocol/PAPER_MAIN_V1_SCHEMA.md`",
        f"- Derived asset: `{audit['input']['derived_assets']}`",
        f"- Train corpus: `{args.derived_assets / 'bm25_train_corpus.jsonl'}`",
        "- This audit does not train. It rescored existing sampled train rollouts with dense node-level DAG-IG credits.",
        "",
        "## 2. Counts",
        "",
        f"- rollouts: `{audit['counts']['rollouts']}`",
        f"- groups: `{audit['counts']['groups']}`",
        f"- constant reward groups: `{audit['counts']['constant_reward_groups']}` = `{pct(audit['counts']['constant_reward_group_rate'])}`",
        f"- reward mean/std/min/max: `{audit['reward_distribution']['mean']:.4f}` / `{audit['reward_distribution']['std']:.4f}` / `{audit['reward_distribution']['min']:.4f}` / `{audit['reward_distribution']['max']:.4f}`",
        "",
        "## 3. Component Coverage",
        "",
    ]
    for name, value in component_nonzero.items():
        lines.append(f"- `{name}` nonzero: `{pct(value)}`")
    lines += [
        "",
        "## 4. Predictiveness",
        "",
    ]
    for name, value in audit["predictiveness"].items():
        lines.append(f"- `{name}`: `{value:.3f}`" if value is not None else f"- `{name}`: `null`")
    lines += [
        "",
        "## 5. Gates",
        "",
    ]
    for name, value in audit["gates"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines += [
        "",
        "## 6. Decision",
        "",
        f"- status: `{audit['status']}`",
    ]
    if audit["status"] == "go":
        lines.append("- Reward signal is sufficiently non-constant for a small GRPO smoke using this paper-main v1 reward.")
    else:
        lines.append("- Do not start GRPO. Fix reward/candidate rollout signal until gates pass.")
    (out / "reports/REWARD_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived_assets", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--old_grpo_root", type=Path, default=DEFAULT_OLD_GRPO)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rollout_runs", nargs="+", default=list(ROLLOUT_RUNS))
    parser.add_argument("--max_constant_group_rate", type=float, default=0.30)
    parser.add_argument("--min_query_auc", type=float, default=0.70)
    parser.add_argument("--min_total_auc", type=float, default=0.70)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
