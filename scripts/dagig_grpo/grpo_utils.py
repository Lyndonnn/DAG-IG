#!/usr/bin/env python3
"""Shared utilities for the DAG-IG GRPO main experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.environ.get("DAGIG_PROJECT_ROOT", Path.cwd())).resolve()
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "data/Pix2Fact_DAGIG_Clean_GRPO_ASSET"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/dagig_grpo_main"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LOCAL_3B_SNAPSHOT = (
    Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct")
    / "snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
)

TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} expected JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def assert_safe_student_model(model_name_or_path: str) -> None:
    lowered = str(model_name_or_path).lower()
    forbidden = ("32b", "qwen32", "qwen2.5-vl-32")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"Refusing to load apparent Qwen32B/teacher model: {model_name_or_path}")


def has_forbidden_marker(value: Any) -> bool:
    markers = ("oracle_crop_query", "raw_pool_week1", "gpt54_trajectory")
    if isinstance(value, str):
        text = value.lower()
        return any(marker in text for marker in markers)
    if isinstance(value, dict):
        return any(has_forbidden_marker(v) for v in value.values())
    if isinstance(value, list):
        return any(has_forbidden_marker(v) for v in value)
    return False


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return TOKEN_RE.findall(text)


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s+:/.-]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def relaxed_normalize_answer(text: str) -> str:
    return re.sub(r"[\s,;]+", " ", normalize_answer(text)).strip()


def compact_alnum(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", str(text)).lower())


def phone_digits(text: str) -> str:
    return re.sub(r"\D+", "", unicodedata.normalize("NFKC", str(text)))


def extract_numeric_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(text))
    tokens = re.findall(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])", text)
    normalized = []
    for token in tokens:
        value = token.replace(",", "")
        if value in {"", "+", "-"}:
            continue
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        sign = ""
        if value.startswith(("+", "-")):
            sign, value = value[0], value[1:]
        value = value.lstrip("0") or "0"
        normalized.append(sign + value)
    return normalized


def extract_number_word_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    words = re.findall(r"\b[a-z]+\b", normalized)
    return [NUMBER_WORDS[word] for word in words if word in NUMBER_WORDS]


def extract_time_tokens(text: str) -> list[str]:
    raw = unicodedata.normalize("NFKC", str(text)).lower()
    out: list[str] = []
    for match in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", raw):
        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        suffix = "am" if match.group(3).startswith("a") else "pm"
        out.append(f"{hour}:{minute:02d}{suffix}")
        if minute == 0:
            out.append(f"{hour}{suffix}")
    for match in re.finditer(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", raw):
        hour = int(match.group(1))
        minute = int(match.group(2))
        out.append(f"{hour:02d}:{minute:02d}")
        out.append(f"{hour}:{minute:02d}")
    return list(dict.fromkeys(out))


def answer_time_matches(target: str, prediction: str) -> bool:
    target_raw = unicodedata.normalize("NFKC", str(target)).lower()
    pred_raw = unicodedata.normalize("NFKC", str(prediction)).lower()
    suffixed_pattern = r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b"

    def suffixed(raw: str) -> list[str]:
        vals = []
        for match in re.finditer(suffixed_pattern, raw):
            hour = int(match.group(1))
            minute = int(match.group(2) or "0")
            suffix = "am" if match.group(3).startswith("a") else "pm"
            vals.append(f"{hour}:{minute:02d}{suffix}")
        return list(dict.fromkeys(vals))

    target_suffixed = suffixed(target_raw)
    pred_suffixed = suffixed(pred_raw)
    if target_suffixed:
        if len(target_suffixed) > 1:
            return set(target_suffixed).issubset(set(pred_suffixed))
        if target_suffixed[0] in pred_suffixed:
            return True
        target_bare = re.sub(r"(am|pm)$", "", target_suffixed[0])
        return target_bare in extract_time_tokens(prediction)
    target_times = extract_time_tokens(target)
    pred_times = extract_time_tokens(prediction)
    if not target_times or not pred_times:
        return False
    if len(target_times) > 1:
        return set(target_times).issubset(set(pred_times))
    return target_times[0] in pred_times


def is_numeric_answer(text: str) -> bool:
    nums = extract_numeric_tokens(text)
    if len(nums) != 1:
        return False
    raw = unicodedata.normalize("NFKC", str(text))
    residual = re.sub(r"[-+]?\d[\d,]*(?:\.\d+)?", "", raw)
    residual = re.sub(r"[\s,;:/.()+%$€£¥]+", "", residual)
    return residual == ""


def is_phone_like_answer(text: str) -> bool:
    raw = str(text)
    digits = phone_digits(raw)
    return len(digits) >= 7 and bool(re.search(r"[\s()+-]", raw))


def answer_match_details(prediction: str, gold: str, aliases: list[str] | None = None) -> dict[str, Any]:
    answers = [gold] + list(aliases or [])
    first: dict[str, Any] | None = None
    for idx, target in enumerate(answers):
        target = str(target or "")
        pred_norm = relaxed_normalize_answer(prediction)
        target_norm = relaxed_normalize_answer(target)
        result = {
            "answer_correct": False,
            "answer_match_type": "no_match",
            "normalized_gold": target_norm,
            "normalized_pred": pred_norm,
            "matched_alias": "gold" if idx == 0 else target,
        }
        if not target_norm:
            first = first or result
            continue
        if pred_norm == target_norm:
            result.update(answer_correct=True, answer_match_type="exact_normalized")
            return result
        target_phone = phone_digits(target)
        pred_phone = phone_digits(prediction)
        if is_phone_like_answer(target) and target_phone and target_phone in pred_phone:
            result.update(answer_correct=True, answer_match_type="phone_compact_contained")
            return result
        target_compact = compact_alnum(target)
        pred_compact = compact_alnum(prediction)
        if len(target_compact) >= 5 and target_compact and target_compact in pred_compact:
            result.update(answer_correct=True, answer_match_type="compact_alnum_contained")
            return result
        target_numbers = extract_numeric_tokens(target)
        pred_numbers = extract_numeric_tokens(prediction)
        if answer_time_matches(target, prediction):
            result.update(answer_correct=True, answer_match_type="time_normalized")
            return result
        if is_numeric_answer(target) and target_numbers:
            if target_numbers[0] in pred_numbers:
                result.update(answer_correct=True, answer_match_type="numeric_contained")
                return result
            if target_numbers[0] in extract_number_word_tokens(prediction):
                result.update(answer_correct=True, answer_match_type="number_word_contained")
                return result
        if target_numbers and len(target_numbers) == 1 and re.fullmatch(r"(?:no|number|num|#)\.?\s*\d+", target_norm):
            if target_numbers[0] in pred_numbers:
                result.update(answer_correct=True, answer_match_type="identifier_number_contained")
                return result
        if not is_numeric_answer(target) and len(target_norm) >= 4 and target_norm in pred_norm:
            result.update(answer_correct=True, answer_match_type="substring_normalized")
            return result
        first = first or result
    return first or {
        "answer_correct": False,
        "answer_match_type": "no_match",
        "normalized_gold": relaxed_normalize_answer(gold),
        "normalized_pred": relaxed_normalize_answer(prediction),
        "matched_alias": "",
    }


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def extract_loose_json_string_field(text: str, field_names: list[str]) -> str:
    raw = str(text or "").strip()
    for name in field_names:
        match = re.search(rf'"{re.escape(name)}"\s*:\s*"', raw, flags=re.IGNORECASE)
        if not match:
            continue
        tail = raw[match.end() :]
        close = re.search(r'"\s*}\s*$', tail, flags=re.DOTALL)
        if close:
            value = tail[: close.start()]
        else:
            value = re.sub(r"\s*}\s*$", "", tail, flags=re.DOTALL)
        return value.strip().strip("`").strip()
    return ""


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False).strip()


def parse_policy_output(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    obj = extract_json_object(raw)
    parsed_json = obj is not None
    visual = ""
    query = ""
    answer = ""
    if obj:
        visual = stringify(obj.get("visual_observation") or obj.get("visual_anchor") or obj.get("localized_entity"))
        query = stringify(obj.get("search_query") or obj.get("query") or obj.get("search_query_or_evidence_need"))
        answer = stringify(obj.get("final_answer") or obj.get("answer") or obj.get("pred_answer"))
    if not query:
        query = extract_loose_json_string_field(raw, ["search_query", "query", "search_query_or_evidence_need"])
    if not answer:
        answer = extract_loose_json_string_field(raw, ["final_answer", "answer", "pred_answer"])
    if not query:
        match = re.search(r"search[_\s-]*query\s*[:=]\s*(.+)", raw, flags=re.IGNORECASE)
        if match:
            query = match.group(1).strip().splitlines()[0].strip(" `\"'")
    if not answer:
        match = re.search(r"(?:final[_\s-]*answer|answer)\s*[:=]\s*(.+)", raw, flags=re.IGNORECASE)
        if match:
            answer = match.group(1).strip().splitlines()[0].strip(" `\"'")
    return {
        "raw": raw,
        "parsed_json": parsed_json,
        "visual_observation": visual,
        "search_query": query,
        "final_answer": answer,
    }


def stable_doc_id(sample_id: str, slot: int, url: str, text: str) -> str:
    key = "\t".join([sample_id, str(slot), url or "", (text or "")[:500]])
    return "doc_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class BM25Index:
    docs: list[dict[str, Any]]
    tokenized_docs: list[list[str]]
    idf: dict[str, float]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def from_docs(cls, docs: list[dict[str, Any]]) -> "BM25Index":
        tokenized_docs = [tokenize(" ".join(str(doc.get(k, "")) for k in ("title", "text", "url", "domain"))) for doc in docs]
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))
        n = max(1, len(docs))
        idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}
        avgdl = sum(len(tokens) for tokens in tokenized_docs) / max(1, len(tokenized_docs))
        return cls(docs=docs, tokenized_docs=tokenized_docs, idf=idf, avgdl=avgdl)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        terms = tokenize(query)
        if not terms:
            return []
        scores: defaultdict[int, float] = defaultdict(float)
        doc_tf = [Counter(tokens) for tokens in self.tokenized_docs]
        for term in Counter(terms):
            term_idf = self.idf.get(term)
            if term_idf is None:
                continue
            for idx, tf_counter in enumerate(doc_tf):
                tf = tf_counter.get(term, 0)
                if not tf:
                    continue
                dl = len(self.tokenized_docs[idx])
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                scores[idx] += term_idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = []
        for rank, (idx, score) in enumerate(ranked, 1):
            doc = dict(self.docs[idx])
            doc["rank"] = rank
            doc["score"] = float(score)
            out.append(doc)
        return out


def load_corpus(path: Path) -> list[dict[str, Any]]:
    docs = read_jsonl(path)
    for doc in docs:
        doc.setdefault("title", "")
        doc.setdefault("text", "")
        doc.setdefault("url", "")
        doc.setdefault("domain", "")
        doc.setdefault("is_gold", False)
    return docs


def anchor_terms(row: dict[str, Any]) -> set[str]:
    texts = [
        row.get("ground_expression", ""),
        row.get("semantic_anchor", ""),
        row.get("image_description", ""),
    ]
    grounding = row.get("grounding")
    if isinstance(grounding, dict):
        texts.extend(
            [
                grounding.get("ground_expression", ""),
                grounding.get("semantic_anchor", ""),
                grounding.get("visible_text_or_name", ""),
                " ".join(grounding.get("visual_disambiguators") or []),
            ]
        )
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "on", "with", "to", "for", "is", "are",
        "image", "picture", "photo", "shown", "visible", "question", "answer",
    }
    terms = {tok for text in texts for tok in tokenize(text) if len(tok) >= 3 and tok not in stop}
    return terms


def query_anchor_terms(row: dict[str, Any]) -> set[str]:
    grounding = row.get("grounding")
    texts = [
        row.get("semantic_anchor", ""),
        row.get("ground_expression", ""),
        row.get("hf_search_query", ""),
    ]
    if isinstance(grounding, dict):
        texts.extend(
            [
                grounding.get("semantic_anchor", ""),
                grounding.get("visible_text_or_name", ""),
                grounding.get("ground_expression", ""),
            ]
        )
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "on", "with", "to", "for", "is", "are",
        "image", "picture", "photo", "shown", "visible", "question", "answer", "search",
        "query", "what", "which", "where", "when", "how", "please", "tell", "could",
        "address", "phone", "number", "opening", "hours", "contact", "price",
    }
    return {tok for text in texts for tok in tokenize(text) if len(tok) >= 3 and tok not in stop and tok not in PATH_QUERY_TOKENS}


def query_anchor_coverage(row: dict[str, Any], query: str) -> tuple[float, int, int]:
    terms = query_anchor_terms(row)
    if not terms:
        return 0.0, 0, 0
    query_terms = set(tokenize(query or ""))
    overlap = len(terms & query_terms)
    denom = max(1, min(8, len(terms)))
    return min(1.0, overlap / denom), overlap, len(terms)


def answer_leaks_in_query(query: str, gold_answer: str) -> bool:
    gold = relaxed_normalize_answer(gold_answer)
    query_norm = relaxed_normalize_answer(query)
    if not gold or not query_norm:
        return False
    if is_numeric_answer(gold_answer):
        nums = extract_numeric_tokens(gold_answer)
        return bool(nums and nums[0] in extract_numeric_tokens(query))
    return len(gold) >= 4 and gold in query_norm


PATH_QUERY_TOKENS = {
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


def query_quality_penalty(query: str) -> float:
    toks = set(tokenize(query or ""))
    penalty = 0.0
    if any(tok in PATH_QUERY_TOKENS or re.fullmatch(r"url\d*", tok) for tok in toks):
        penalty += 0.15
    if len(tokenize(query or "")) > 20:
        penalty += 0.05
    return penalty


def support_rank(docs: list[dict[str, Any]], sample_id: str, k: int) -> int | None:
    for idx, doc in enumerate(docs[:k], 1):
        if str(doc.get("sample_id")) == str(sample_id) and bool(doc.get("is_gold")):
            return idx
    return None


def compute_reward(
    row: dict[str, Any],
    output_text: str,
    bm25: BM25Index,
    variant: str,
    top_k: int = 5,
) -> dict[str, Any]:
    parsed = parse_policy_output(output_text)
    query = parsed["search_query"].strip()
    answer = parsed["final_answer"].strip()
    visual = parsed["visual_observation"].strip()
    retrieved = bm25.search(query, top_k=max(10, top_k)) if query else []
    rank5 = support_rank(retrieved, str(row.get("sample_id")), top_k)
    rank10 = support_rank(retrieved, str(row.get("sample_id")), 10)
    evidence_supported = rank5 is not None
    retrieval_hit = evidence_supported
    answer_match = answer_match_details(answer, str(row.get("gold_answer") or row.get("answer") or ""))
    answer_correct = bool(answer_match["answer_correct"])
    strict_success = bool(answer_correct and evidence_supported)

    if variant in {"paper_main_v1", "paper_main_v2"}:
        format_credit = 0.0
        format_credit += 0.03 if parsed["parsed_json"] else 0.0
        format_credit += 0.02 if visual else 0.0
        format_credit += 0.03 if query else 0.0
        format_credit += 0.02 if answer else 0.0
        format_credit = min(format_credit, 0.10)

        visual_terms = anchor_terms(row)
        visual_tokens = set(tokenize(visual))
        visual_overlap = len(visual_terms & visual_tokens)
        visual_credit = min(1.0, visual_overlap / max(3, min(8, len(visual_terms)))) if visual else 0.0
        query_mrr_credit = 1.0 / rank10 if rank10 else 0.0
        query_anchor_credit, query_anchor_overlap, query_anchor_term_count = query_anchor_coverage(row, query)
        if variant == "paper_main_v2":
            query_credit = min(1.0, 0.85 * query_mrr_credit + 0.15 * query_anchor_credit)
        else:
            query_credit = query_mrr_credit
        evidence_credit = 1.0 / rank5 if rank5 else 0.0
        answer_credit = 1.0 if strict_success else (0.35 if answer_correct else 0.0)
        leakage_penalty = 0.25 if answer_leaks_in_query(query, str(row.get("gold_answer") or row.get("answer") or "")) else 0.0
        path_penalty = query_quality_penalty(query)
        missing_anchor_penalty = 0.0
        if variant == "paper_main_v2" and query and not rank10 and query_anchor_term_count and query_anchor_overlap == 0:
            missing_anchor_penalty = 0.08
        total = (
            0.10 * format_credit
            + 0.15 * visual_credit
            + 0.40 * query_credit
            + 0.25 * evidence_credit
            + 0.35 * answer_credit
            - leakage_penalty
            - path_penalty
            - missing_anchor_penalty
        )
        return {
            "reward": float(max(-0.5, total)),
            "components": {
                "format": float(format_credit),
                "visual": float(visual_credit),
                "query": float(query_credit),
                "query_mrr": float(query_mrr_credit),
                "query_anchor": float(query_anchor_credit),
                "evidence": float(evidence_credit),
                "answer": float(answer_credit),
                "leakage_penalty": float(-leakage_penalty),
                "path_penalty": float(-path_penalty),
                "missing_anchor_penalty": float(-missing_anchor_penalty),
            },
            "parsed": parsed,
            "retrieved_docs": retrieved[:top_k],
            "retrieval_hit": bool(retrieval_hit),
            "evidence_supported": bool(evidence_supported),
            "answer_correct": bool(answer_correct),
            "strict_success": bool(strict_success),
            "answer_match": answer_match,
            "visual_anchor_overlap": int(visual_overlap),
            "query_anchor_overlap": int(query_anchor_overlap),
            "query_anchor_term_count": int(query_anchor_term_count),
            "support_rank5": rank5,
            "support_rank10": rank10,
        }

    fmt = 0.0
    if parsed["parsed_json"]:
        fmt += 0.2
    if visual:
        fmt += 0.1
    if query:
        fmt += 0.15
    if answer:
        fmt += 0.15
    if len(output_text) > 1200:
        fmt -= 0.2
    if not parsed["parsed_json"] or not query or not answer:
        fmt -= 0.3

    visual_terms = anchor_terms(row)
    visual_tokens = set(tokenize(visual))
    visual_overlap = len(visual_terms & visual_tokens)
    visual_credit = min(0.35, 0.08 * visual_overlap) if visual else 0.0
    query_credit = 0.45 if retrieval_hit else 0.0
    evidence_credit = 0.25 if evidence_supported else 0.0
    answer_credit = 0.8 if strict_success else (0.15 if answer_correct else 0.0)
    leakage_penalty = -0.4 if answer_leaks_in_query(query, str(row.get("gold_answer") or row.get("answer") or "")) else 0.0

    if variant == "outcome_grpo":
        total = fmt + answer_credit + leakage_penalty
        visual_credit = 0.0
        query_credit = 0.0
        evidence_credit = 0.0
    elif variant == "trajectory_grpo":
        total = fmt + (0.5 if retrieval_hit else 0.0) + (0.9 if strict_success else 0.0) + leakage_penalty
        visual_credit = 0.0
        query_credit = 0.0
        evidence_credit = 0.0
    elif variant == "dagig_grpo_no_visual":
        total = fmt + query_credit + evidence_credit + answer_credit + leakage_penalty
        visual_credit = 0.0
    elif variant == "dagig_grpo_full":
        total = fmt + visual_credit + query_credit + evidence_credit + answer_credit + leakage_penalty
    elif variant == "dagig_grpo_no_query":
        total = fmt + visual_credit + evidence_credit + answer_credit + leakage_penalty
        query_credit = 0.0
    elif variant == "dagig_grpo_no_evidence":
        total = fmt + visual_credit + query_credit + answer_credit + leakage_penalty
        evidence_credit = 0.0
    else:
        raise ValueError(f"Unknown reward variant: {variant}")

    return {
        "reward": float(total),
        "components": {
            "format": float(fmt),
            "visual": float(visual_credit),
            "query": float(query_credit),
            "evidence": float(evidence_credit),
            "answer": float(answer_credit),
            "leakage_penalty": float(leakage_penalty),
        },
        "parsed": parsed,
        "retrieved_docs": retrieved[:top_k],
        "retrieval_hit": bool(retrieval_hit),
        "evidence_supported": bool(evidence_supported),
        "answer_correct": bool(answer_correct),
        "strict_success": bool(strict_success),
        "answer_match": answer_match,
        "visual_anchor_overlap": int(visual_overlap),
    }


def build_user_messages(image_path: str, question: str, prompt_text: str, max_pixels: int = 0) -> list[dict[str, Any]]:
    image_content: dict[str, Any] = {"type": "image", "image": str(image_path)}
    if max_pixels and max_pixels > 0:
        image_content["max_pixels"] = int(max_pixels)
    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": f"{prompt_text.strip()}\n\nQuestion: {question.strip()}"},
            ],
        }
    ]


def resolve_model_path(model_name_or_path: str) -> str:
    if model_name_or_path == DEFAULT_MODEL and LOCAL_3B_SNAPSHOT.exists():
        return str(LOCAL_3B_SNAPSHOT)
    return model_name_or_path
