#!/usr/bin/env python3
"""Lightweight answer verifier/selector for the paper-main two-stage system.

This is a non-VLM diagnostic. It trains a small logistic candidate scorer on
train rollouts only, then applies the frozen selector to dev/test predictions.
Gold answers are used only as train labels and evaluation labels. Inference
features never use gold answers, gold docs, support labels, or correctness.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import (  # noqa: E402
    BM25Index,
    answer_leaks_in_query,
    answer_match_details,
    compact_alnum,
    load_corpus,
    phone_digits,
    read_jsonl,
    support_rank,
    tokenize,
    write_json,
    write_jsonl,
)


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b|(?<!\d)\d{1,2}:\d{2}(?!\d)", re.I)
PRICE_RE = re.compile(r"(?:[$€£¥]\s*\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s*(?:usd|eur|gbp|jpy|rmb|cny|yen|dollars?|euros?))", re.I)
PERCENT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*%")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion|usd|eur|gbp|jpy|rmb|cny|locations?|days?|years?|people|vehicles))?", re.I)
ADDR_WORD_RE = re.compile(r"\b(address|located|location|store|branch|street|road|avenue|ave|st\.?|rd\.?|floor|ward|district|chome|building|suite)\b", re.I)


ANSWER_TYPES = ["phone", "email", "time", "price", "address", "numeric", "entity"]


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def infer_answer_type(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(e-?mail|email address)\b", q):
        return "email"
    if re.search(r"\b(phone|telephone|contact number|call|hotline|tel)\b", q):
        return "phone"
    if re.search(r"\b(opening time|opening hours|business hours|hours|check-?out|closing|what time)\b", q):
        return "time"
    if re.search(r"\b(price|cost|how much|pay|priced|fee)\b", q):
        return "price"
    if re.search(r"\b(address|located|location|where|moved to|mailing address|store in|branch in)\b", q):
        return "address"
    if re.search(r"\b(gdp|population|rate|ranking|rank|revenue|how many|number of|percentage|percent|accurate to|score|value)\b", q):
        return "numeric"
    return "entity"


def has_type(answer: str, answer_type: str) -> bool:
    text = str(answer or "")
    if answer_type == "email":
        return bool(EMAIL_RE.search(text))
    if answer_type == "phone":
        return len(phone_digits(text)) >= 7
    if answer_type == "time":
        return bool(TIME_RE.search(text))
    if answer_type == "price":
        return bool(PRICE_RE.search(text) or re.search(r"\d", text))
    if answer_type == "numeric":
        return bool(PERCENT_RE.search(text) or NUMBER_RE.search(text))
    if answer_type == "address":
        return bool(re.search(r"\d", text) and ("," in text or ADDR_WORD_RE.search(text) or re.search(r"[区市町号樓楼]", text)))
    return bool(normalize_space(text))


def weak_answer(answer: str, answer_type: str) -> bool:
    text = normalize_space(answer)
    if not text:
        return True
    low = text.lower()
    if any(marker in low for marker in ("don't have enough information", "cannot determine", "unknown", "not enough information", "无法确定")):
        return True
    if answer_type in {"phone", "email", "time", "price", "address", "numeric"}:
        return not has_type(text, answer_type)
    return False


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[。.!?])\s+|\n+", str(text or ""))
    return [normalize_space(chunk) for chunk in chunks if normalize_space(chunk)]


def address_candidates(text: str) -> list[str]:
    out: list[str] = []
    for sent in split_sentences(text):
        if re.search(r"\d", sent) and ("," in sent or ADDR_WORD_RE.search(sent) or re.search(r"[区市町号樓楼]", sent)):
            cleaned = re.sub(r"^(address|location|located at|mailing address)\s*[:：-]\s*", "", sent, flags=re.I)
            cleaned = normalize_space(cleaned).strip(" ,;:.")
            if 8 <= len(cleaned) <= 180:
                out.append(cleaned)
    return out


def extract_doc_spans(doc: dict[str, Any], answer_type: str) -> list[str]:
    text = " ".join(str(doc.get(k, "")) for k in ("title", "text", "url"))
    if answer_type == "email":
        spans = EMAIL_RE.findall(text)
    elif answer_type == "phone":
        spans = [m.group(0) for m in PHONE_RE.finditer(text) if len(phone_digits(m.group(0))) >= 7]
    elif answer_type == "time":
        spans = [m.group(0) for m in TIME_RE.finditer(text)]
    elif answer_type == "price":
        spans = [m.group(0) for m in PRICE_RE.finditer(text)]
    elif answer_type == "numeric":
        spans = [m.group(0) for m in PERCENT_RE.finditer(text)] or [m.group(0) for m in NUMBER_RE.finditer(text)]
    elif answer_type == "address":
        spans = address_candidates(text)
    else:
        # Entity/name questions are too open for reliable regex extraction.
        spans = []
    clean: list[str] = []
    seen: set[str] = set()
    for span in spans:
        span = normalize_space(span).strip(" ,;:.")
        key = compact_alnum(span) or span.lower()
        if not span or key in seen:
            continue
        seen.add(key)
        clean.append(span)
    return clean[:8]


def overlap(a: str, b: str) -> float:
    toks_a = {tok for tok in tokenize(a) if len(tok) >= 3}
    toks_b = {tok for tok in tokenize(b) if len(tok) >= 3}
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a)


def doc_text(doc: dict[str, Any] | None) -> str:
    if not doc:
        return ""
    return " ".join(str(doc.get(k, "")) for k in ("title", "text", "url", "domain"))


def candidate_features(row: dict[str, Any], cand: dict[str, Any]) -> dict[str, float]:
    answer = normalize_space(cand.get("answer", ""))
    question = str(row.get("question", ""))
    query = str(row.get("search_query", ""))
    answer_type = infer_answer_type(question)
    doc = cand.get("doc") or {}
    dtext = doc_text(doc)
    rank = int(doc.get("rank") or 99)
    score = float(doc.get("score") or 0.0)
    doc_has_answer = bool(answer and (answer.lower() in dtext.lower() or compact_alnum(answer) and compact_alnum(answer) in compact_alnum(dtext)))
    feats: dict[str, float] = {
        "bias": 1.0,
        "source_current_reader": 1.0 if cand.get("source") == "current_reader" else 0.0,
        "source_doc_span": 1.0 if cand.get("source") == "doc_span" else 0.0,
        "answer_len_chars": min(len(answer), 200) / 200.0,
        "answer_len_tokens": min(len(tokenize(answer)), 30) / 30.0,
        "answer_has_digit": 1.0 if re.search(r"\d", answer) else 0.0,
        "answer_has_phone": 1.0 if len(phone_digits(answer)) >= 7 else 0.0,
        "answer_has_email": 1.0 if EMAIL_RE.search(answer) else 0.0,
        "answer_has_time": 1.0 if TIME_RE.search(answer) else 0.0,
        "answer_has_price": 1.0 if PRICE_RE.search(answer) else 0.0,
        "answer_has_address": 1.0 if has_type(answer, "address") else 0.0,
        "answer_type_match": 1.0 if has_type(answer, answer_type) else 0.0,
        "current_answer_weak": 1.0 if weak_answer(str(row.get("final_answer", "")), answer_type) else 0.0,
        "answer_in_query": 1.0 if answer and compact_alnum(answer) and compact_alnum(answer) in compact_alnum(query) else 0.0,
        "answer_in_doc": 1.0 if doc_has_answer else 0.0,
        "doc_rank_inv": 0.0 if rank >= 99 else 1.0 / max(1, rank),
        "doc_rank_le1": 1.0 if rank <= 1 else 0.0,
        "doc_rank_le3": 1.0 if rank <= 3 else 0.0,
        "doc_score_log": math.log1p(max(0.0, score)) / 4.0,
        "question_doc_overlap": overlap(question, dtext),
        "query_doc_overlap": overlap(query, dtext),
        "answer_question_overlap": overlap(answer, question),
        "query_len": min(len(tokenize(query)), 20) / 20.0,
        "query_leaks_answer_like": 1.0 if answer_leaks_in_query(query, answer) else 0.0,
    }
    for atype in ANSWER_TYPES:
        feats[f"type_{atype}"] = 1.0 if answer_type == atype else 0.0
    return feats


def build_candidates(row: dict[str, Any], docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answer_type = infer_answer_type(str(row.get("question", "")))
    candidates: list[dict[str, Any]] = []
    current_answer = normalize_space(row.get("final_answer", ""))
    if current_answer:
        candidates.append({"source": "current_reader", "answer": current_answer, "doc": None})
    seen = {compact_alnum(current_answer)} if current_answer else set()
    for doc in docs:
        for span in extract_doc_spans(doc, answer_type):
            key = compact_alnum(span) or span.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append({"source": "doc_span", "answer": span, "doc": doc})
    return candidates


def label_candidate(answer: str, gold: str) -> int:
    return int(bool(answer_match_details(answer, gold).get("answer_correct")))


def make_train_groups(train_rollouts: list[dict[str, Any]], train_rows: dict[str, dict[str, Any]], bm25: BM25Index, top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for idx, rollout in enumerate(train_rollouts):
        sample_id = str(rollout.get("sample_id", ""))
        base = train_rows.get(sample_id)
        if not base:
            continue
        parsed = rollout.get("parsed") or {}
        query = normalize_space(parsed.get("search_query", ""))
        answer = normalize_space(parsed.get("final_answer", ""))
        docs = bm25.search(query, top_k=top_k) if query else []
        row = {
            "sample_id": sample_id,
            "question": base.get("question", ""),
            "gold_answer": base.get("gold_answer", ""),
            "search_query": query,
            "final_answer": answer,
        }
        evidence_supported = support_rank(docs, sample_id, top_k) is not None
        cands = build_candidates(row, docs)
        group_indices = []
        for cand in cands:
            feats = candidate_features(row, cand)
            label = label_candidate(cand["answer"], str(base.get("gold_answer", "")))
            candidate_rows.append(
                {
                    "group_id": f"train_{idx}",
                    "sample_id": sample_id,
                    "source": cand["source"],
                    "answer": cand["answer"],
                    "label_answer_correct": label,
                    "evidence_supported": evidence_supported,
                    "features": feats,
                }
            )
            group_indices.append(len(candidate_rows) - 1)
        if group_indices:
            groups.append(
                {
                    "group_id": f"train_{idx}",
                    "sample_id": sample_id,
                    "candidate_indices": group_indices,
                    "baseline_answer_correct": int(bool(rollout.get("answer_correct"))),
                    "baseline_strict": int(bool(rollout.get("strict_success"))),
                    "evidence_supported": evidence_supported,
                }
            )
    return candidate_rows, groups


class LogisticScorer:
    def __init__(self, feature_names: list[str], mean: np.ndarray, std: np.ndarray, weights: np.ndarray) -> None:
        self.feature_names = feature_names
        self.mean = mean
        self.std = std
        self.weights = weights

    def vectorize(self, feats: dict[str, float]) -> np.ndarray:
        return np.array([float(feats.get(name, 0.0)) for name in self.feature_names], dtype=np.float64)

    def score(self, feats: dict[str, float]) -> float:
        x = self.vectorize(feats)
        x = (x - self.mean) / self.std
        z = float(np.dot(x, self.weights))
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
        }


def train_logistic(candidate_rows: list[dict[str, Any]], epochs: int = 1200, lr: float = 0.05, l2: float = 0.01) -> LogisticScorer:
    all_features = sorted({name for row in candidate_rows for name in row["features"]})
    X = np.array([[float(row["features"].get(name, 0.0)) for name in all_features] for row in candidate_rows], dtype=np.float64)
    y = np.array([float(row["label_answer_correct"]) for row in candidate_rows], dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std
    w = np.zeros(Xn.shape[1], dtype=np.float64)
    pos = max(1.0, y.sum())
    neg = max(1.0, len(y) - y.sum())
    sample_w = np.where(y > 0.5, neg / pos, 1.0)
    for _ in range(epochs):
        z = np.clip(Xn @ w, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (Xn.T @ (sample_w * (p - y))) / len(y) + l2 * w
        w -= lr * grad
    return LogisticScorer(all_features, mean, std, w)


def score_candidate_rows(candidate_rows: list[dict[str, Any]], scorer: LogisticScorer) -> None:
    for row in candidate_rows:
        row["verifier_score"] = scorer.score(row["features"])


def candidate_is_eligible(row: dict[str, Any]) -> bool:
    feats = row.get("features") or {}
    if row.get("source") == "current_reader":
        return True
    answer_type = "entity"
    for atype in ANSWER_TYPES:
        if feats.get(f"type_{atype}"):
            answer_type = atype
            break
    if answer_type == "entity":
        return False
    return bool(feats.get("answer_type_match") or feats.get("current_answer_weak"))


def choose_for_group(group: dict[str, Any], candidates: list[dict[str, Any]], delta: float, min_alt_score: float) -> dict[str, Any]:
    group_cands = [candidates[i] for i in group["candidate_indices"]]
    current = next((c for c in group_cands if c.get("source") == "current_reader"), group_cands[0])
    alternatives = [c for c in group_cands if c is not current and candidate_is_eligible(c)]
    if not alternatives:
        return {"chosen": current, "replaced": False, "current": current, "best_alt": None}
    best_alt = max(alternatives, key=lambda c: float(c.get("verifier_score", 0.0)))
    replace = (float(best_alt.get("verifier_score", 0.0)) >= min_alt_score) and (
        float(best_alt.get("verifier_score", 0.0)) - float(current.get("verifier_score", 0.0)) >= delta
    )
    return {"chosen": best_alt if replace else current, "replaced": replace, "current": current, "best_alt": best_alt}


def tune_selector(groups: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    deltas = [round(x, 2) for x in np.arange(0.0, 0.81, 0.05)]
    min_scores = [round(x, 2) for x in np.arange(0.2, 0.91, 0.05)]
    for delta in deltas:
        for min_score in min_scores:
            rows = []
            for group in groups:
                decision = choose_for_group(group, candidates, delta=delta, min_alt_score=min_score)
                chosen = decision["chosen"]
                before = bool(group["baseline_strict"])
                after_answer = bool(chosen["label_answer_correct"])
                after = bool(after_answer and group["evidence_supported"])
                rows.append((before, after, decision["replaced"]))
            n = max(1, len(rows))
            before_rate = sum(1 for b, _, _ in rows if b) / n
            after_rate = sum(1 for _, a, _ in rows if a) / n
            recoveries = sum(1 for b, a, _ in rows if (not b) and a)
            harms = sum(1 for b, a, _ in rows if b and (not a))
            replace_rate = sum(1 for _, _, r in rows if r) / n
            objective = after_rate - before_rate - 2.0 * (harms / n) - 0.05 * replace_rate
            item = {
                "delta": delta,
                "min_alt_score": min_score,
                "train_before_strict": before_rate,
                "train_after_strict": after_rate,
                "train_recoveries": recoveries,
                "train_harms": harms,
                "train_replace_rate": replace_rate,
                "objective": objective,
            }
            if best is None or (item["objective"], item["delta"], item["min_alt_score"]) > (best["objective"], best["delta"], best["min_alt_score"]):
                best = item
    assert best is not None
    return best


def prediction_groups(pred_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for idx, row in enumerate(pred_rows):
        cands = build_candidates(row, row.get("retrieved_docs") or [])
        indices = []
        for cand in cands:
            feats = candidate_features(row, cand)
            label = label_candidate(cand["answer"], str(row.get("gold_answer", "")))
            candidates.append(
                {
                    "group_id": f"eval_{idx}",
                    "sample_id": row.get("sample_id"),
                    "source": cand["source"],
                    "answer": cand["answer"],
                    "label_answer_correct": label,
                    "features": feats,
                    "doc_id": (cand.get("doc") or {}).get("doc_id"),
                    "doc_rank": (cand.get("doc") or {}).get("rank"),
                }
            )
            indices.append(len(candidates) - 1)
        if indices:
            groups.append(
                {
                    "group_id": f"eval_{idx}",
                    "sample_id": row.get("sample_id"),
                    "candidate_indices": indices,
                    "baseline_answer_correct": int(bool(row.get("answer_correct"))),
                    "baseline_strict": int(bool(row.get("strict_success"))),
                    "evidence_supported": bool(row.get("evidence_supported")),
                    "row": row,
                }
            )
    return candidates, groups


def summarize_eval(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(before_rows))

    def rate(rows: list[dict[str, Any]], key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / n

    return {
        "n": len(before_rows),
        "before_answer_correct": rate(before_rows, "answer_correct"),
        "after_answer_correct": rate(after_rows, "answer_correct"),
        "before_strict_success": rate(before_rows, "strict_success"),
        "after_strict_success": rate(after_rows, "strict_success"),
        "retrieval_top5_hit": rate(before_rows, "retrieval_top5_hit"),
        "replacements": sum(1 for d in decisions if d["replaced"]),
        "strict_recoveries": sum(1 for d in decisions if (not d["old_strict"]) and d["new_strict"]),
        "strict_harms": sum(1 for d in decisions if d["old_strict"] and (not d["new_strict"])),
        "answer_recoveries": sum(1 for d in decisions if (not d["old_answer_correct"]) and d["new_answer_correct"]),
        "answer_harms": sum(1 for d in decisions if d["old_answer_correct"] and (not d["new_answer_correct"])),
        "hit_answer_wrong_before": sum(1 for row in before_rows if row.get("retrieval_top5_hit") and not row.get("answer_correct")),
        "hit_answer_wrong_after": sum(1 for row in after_rows if row.get("retrieval_top5_hit") and not row.get("answer_correct")),
    }


def apply_to_predictions(pred_path: Path, scorer: LogisticScorer, selector_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(pred_path)
    candidates, groups = prediction_groups(rows)
    score_candidate_rows(candidates, scorer)
    after_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    group_by_id = {group["group_id"]: group for group in groups}
    for group in groups:
        row = dict(group["row"])
        decision = choose_for_group(group, candidates, delta=float(selector_cfg["delta"]), min_alt_score=float(selector_cfg["min_alt_score"]))
        chosen = decision["chosen"]
        old_answer = row.get("final_answer", "")
        old_answer_correct = bool(row.get("answer_correct"))
        old_strict = bool(row.get("strict_success"))
        row["answer_verifier"] = {
            "selected_answer": chosen["answer"],
            "selected_source": chosen["source"],
            "selected_score": chosen.get("verifier_score"),
            "current_score": decision["current"].get("verifier_score"),
            "best_alt_answer": (decision["best_alt"] or {}).get("answer"),
            "best_alt_score": (decision["best_alt"] or {}).get("verifier_score"),
            "replaced": bool(decision["replaced"]),
            "delta": selector_cfg["delta"],
            "min_alt_score": selector_cfg["min_alt_score"],
        }
        if decision["replaced"]:
            row["original_final_answer"] = old_answer
            row["final_answer"] = chosen["answer"]
        match = answer_match_details(str(row.get("final_answer", "")), str(row.get("gold_answer", "")))
        row["answer_match"] = match
        row["answer_correct"] = bool(match["answer_correct"])
        row["strict_success"] = bool(row.get("answer_correct") and row.get("evidence_supported"))
        after_rows.append(row)
        decisions.append(
            {
                "sample_id": row.get("sample_id"),
                "old_answer": old_answer,
                "new_answer": row.get("final_answer"),
                "gold_answer": row.get("gold_answer"),
                "replaced": bool(decision["replaced"]),
                "selected_source": chosen["source"],
                "selected_score": chosen.get("verifier_score"),
                "current_score": decision["current"].get("verifier_score"),
                "best_alt_answer": (decision["best_alt"] or {}).get("answer"),
                "best_alt_score": (decision["best_alt"] or {}).get("verifier_score"),
                "old_answer_correct": old_answer_correct,
                "new_answer_correct": bool(row.get("answer_correct")),
                "old_strict": old_strict,
                "new_strict": bool(row.get("strict_success")),
                "retrieval_top5_hit": bool(row.get("retrieval_top5_hit")),
            }
        )
    # Preserve rows that somehow had no candidate group.
    done = {d["sample_id"] for d in decisions}
    for row in rows:
        if row.get("sample_id") not in done:
            after_rows.append(row)
    metrics = summarize_eval(rows, after_rows, decisions)
    stem = pred_path.stem
    write_jsonl(output_dir / f"{stem}_verifier_predictions.jsonl", after_rows)
    write_jsonl(output_dir / f"{stem}_verifier_decisions.jsonl", decisions)
    write_json(output_dir / f"{stem}_verifier_metrics.json", metrics)
    return {"input": str(pred_path), "metrics": metrics, "decisions_path": str(output_dir / f"{stem}_verifier_decisions.jsonl")}


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row.get("source", "") for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_rollouts", type=Path, default=Path("outputs/dagig_paper_main_v1/checkpoints/paper_main_v1_two_stage_stage1loss_kl01_medium30/reward_rollouts.jsonl"))
    parser.add_argument("--train_data", type=Path, default=Path("outputs/dagig_grpo_main/derived_assets/grpo_train.jsonl"))
    parser.add_argument("--train_corpus", type=Path, default=Path("outputs/dagig_grpo_main/derived_assets/bm25_train_corpus.jsonl"))
    parser.add_argument("--predictions", type=Path, nargs="+", default=[
        Path("outputs/dagig_paper_main_v1/two_stage_predictions_rescored_v3/paper_main_v1_two_stage_stage1loss_kl01_medium30_ckpt30_dev.jsonl"),
        Path("outputs/dagig_paper_main_v1/two_stage_predictions_rescored_v3/paper_main_v1_two_stage_stage1loss_kl01_medium30_ckpt30_test.jsonl"),
    ])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dagig_paper_main_v1/learned_answer_verifier"))
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = {str(row.get("sample_id")): row for row in read_jsonl(args.train_data)}
    train_rollouts = read_jsonl(args.train_rollouts)
    train_index = BM25Index.from_docs(load_corpus(args.train_corpus))
    train_candidates, train_groups = make_train_groups(train_rollouts, train_rows, train_index, top_k=args.top_k)
    if not train_candidates:
        raise RuntimeError("No train candidates were built.")
    positives = sum(row["label_answer_correct"] for row in train_candidates)
    if positives == 0:
        raise RuntimeError("No positive train answer candidates were found.")

    scorer = train_logistic(train_candidates)
    score_candidate_rows(train_candidates, scorer)
    selector_cfg = tune_selector(train_groups, train_candidates)

    write_jsonl(args.output_dir / "train_candidate_features.jsonl", train_candidates)
    write_json(args.output_dir / "verifier_model.json", scorer.to_json())
    write_json(args.output_dir / "selector_config.json", selector_cfg)

    runs = [apply_to_predictions(path, scorer, selector_cfg, args.output_dir) for path in args.predictions]
    summary = {
        "train": {
            "rollouts": len(train_rollouts),
            "groups": len(train_groups),
            "candidate_rows": len(train_candidates),
            "positive_candidates": positives,
            "positive_rate": positives / max(1, len(train_candidates)),
            "source_counts": source_counts(train_candidates),
            "selector_config": selector_cfg,
        },
        "runs": runs,
        "leakage_policy": {
            "actual_features_exclude_gold_answer": True,
            "actual_features_exclude_gold_doc_or_support_label": True,
            "actual_features_exclude_answer_correctness": True,
            "gold_used_only_for_train_labels_and_eval": True,
        },
    }
    write_json(args.output_dir / "learned_answer_verifier_summary.json", summary)

    lines = ["# Learned Answer Verifier Report\n\n"]
    lines.append("This diagnostic trains a lightweight answer-candidate scorer on train rollouts only. At inference it uses only question, search query, retrieved document text/rank/score, and candidate-answer text features. It does not use gold answer, gold doc labels, support labels, or correctness on dev/test.\n\n")
    lines.append("## Train Setup\n\n")
    lines.append(f"- train rollouts: `{len(train_rollouts)}`\n")
    lines.append(f"- train groups with candidates: `{len(train_groups)}`\n")
    lines.append(f"- train answer candidates: `{len(train_candidates)}`\n")
    lines.append(f"- positive candidate rate: `{100*positives/max(1, len(train_candidates)):.1f}%`\n")
    lines.append(f"- source counts: `{json.dumps(source_counts(train_candidates), ensure_ascii=False)}`\n")
    lines.append(f"- selector config: `{json.dumps(selector_cfg, ensure_ascii=False)}`\n\n")
    lines.append("## Evaluation\n\n")
    lines.append("| prediction file | before strict | after strict | before answer | after answer | replacements | recoveries | harms | hit-answer-wrong before | after |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for run in runs:
        m = run["metrics"]
        lines.append(
            f"| {Path(run['input']).name} | {100*m['before_strict_success']:.1f}% | {100*m['after_strict_success']:.1f}% | "
            f"{100*m['before_answer_correct']:.1f}% | {100*m['after_answer_correct']:.1f}% | {m['replacements']} | "
            f"{m['strict_recoveries']} | {m['strict_harms']} | {m['hit_answer_wrong_before']} | {m['hit_answer_wrong_after']} |\n"
        )
    lines.append("\n## Decision\n\n")
    dev_run = next((run for run in runs if "_dev" in Path(run["input"]).stem), runs[0] if runs else None)
    test_run = next((run for run in runs if "_test" in Path(run["input"]).stem), None)
    if dev_run and test_run:
        dev_gain = dev_run["metrics"]["after_strict_success"] - dev_run["metrics"]["before_strict_success"]
        test_gain = test_run["metrics"]["after_strict_success"] - test_run["metrics"]["before_strict_success"]
        if dev_gain > 0 and test_gain >= 0:
            lines.append("The lightweight verifier gives a clean non-negative dev/test signal. Next mainline step: train a real evidence-conditioned reader/verifier on train data, keeping the ckpt30 query policy fixed.\n")
        else:
            lines.append("The lightweight verifier does not produce a reliable clean gain. Do not add broad post-hoc repair to the paper system; the next mainline step should be a trained reader/verifier or more query/evidence supervision, not rules.\n")
    else:
        lines.append("Insufficient split coverage to make a dev/test decision.\n")
    (args.output_dir / "LEARNED_ANSWER_VERIFIER_REPORT.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
