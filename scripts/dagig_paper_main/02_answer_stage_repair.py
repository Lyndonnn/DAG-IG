#!/usr/bin/env python3
"""Conservative answer-stage repair for paper-main two-stage predictions.

This is a diagnostic, non-training verifier/repair pass. The actual repair
decision uses only question text, current answer text, and retrieved evidence
text. Gold answers and gold document labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dagig_grpo.grpo_utils import answer_match_details, phone_digits, write_json, write_jsonl


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
PRICE_RE = re.compile(r"(?:[$€£¥]\s*\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s*(?:usd|eur|gbp|jpy|rmb|cny|yen|dollars?|euros?))", re.I)
TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b|(?<!\d)\d{1,2}:\d{2}(?!\d)", re.I)
PERCENT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*%")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion|usd|eur|gbp|jpy|rmb|cny|locations?|days?|years?|people|vehicles))?", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def infer_answer_type(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(e-?mail|email address)\b", q):
        return "email"
    if re.search(r"\b(phone|telephone|contact number|call|hotline|tel)\b", q):
        return "phone"
    if re.search(r"\b(address|located|location|where|moved to|mailing address|store in|branch in)\b", q):
        return "address"
    if re.search(r"\b(opening time|opening hours|business hours|hours|check-?out|closing|what time)\b", q):
        return "time"
    if re.search(r"\b(price|cost|how much|pay|priced)\b", q):
        return "price"
    if re.search(r"\b(gdp|population|rate|ranking|rank|revenue|how many|number of|percentage|percent|accurate to)\b", q):
        return "numeric"
    return "entity"


def answer_has_type(answer: str, answer_type: str) -> bool:
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
        return bool(re.search(r"\d", text) and ("," in text or re.search(r"\b(st|street|road|rd|ave|avenue|ward|district|floor|no\\.?|chome|区|市|町|号)\b", text, re.I)))
    return bool(normalize_space(text))


def weak_current_answer(answer: str, answer_type: str) -> bool:
    a = normalize_space(answer)
    if not a or a in {"{", "...", "json"}:
        return True
    low = a.lower()
    if "don't have enough information" in low or "not applicable" in low or "unknown" in low:
        return True
    if answer_type in {"email", "phone", "time", "price", "numeric", "address"}:
        return not answer_has_type(a, answer_type)
    return False


def doc_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[。.!?])\s+|\n+", str(text or ""))
    out = []
    for chunk in chunks:
        chunk = normalize_space(chunk)
        if chunk:
            out.append(chunk)
    return out


def extract_address_candidates(text: str) -> list[str]:
    out: list[str] = []
    for sent in doc_sentences(text):
        low = sent.lower()
        has_addr_word = bool(re.search(r"\b(address|located|location|store|branch|mailing|street|road|avenue|ave|st\\.?|rd\\.?|floor|ward|district|chome)\b", low))
        has_cjk_addr = bool(re.search(r"[区市町号樓楼]", sent))
        if re.search(r"\d", sent) and ("," in sent or has_addr_word or has_cjk_addr):
            cleaned = re.sub(r"^(address|location|located at|mailing address)\s*[:：-]\s*", "", sent, flags=re.I)
            if 8 <= len(cleaned) <= 180:
                out.append(cleaned)
    return out


def extract_candidates(row: dict[str, Any], answer_type: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for doc in row.get("retrieved_docs") or []:
        rank = int(doc.get("rank") or 99)
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
            spans = extract_address_candidates(text)
        else:
            spans = []
        seen = set()
        for span in spans:
            span = normalize_space(span).strip(" ,;:.")
            if not span or span.lower() in seen:
                continue
            seen.add(span.lower())
            candidates.append({"answer": span, "rank": rank, "doc_id": doc.get("doc_id"), "doc_title": doc.get("title", "")})
    return candidates


def choose_repair(row: dict[str, Any]) -> dict[str, Any]:
    answer_type = infer_answer_type(str(row.get("question", "")))
    current = normalize_space(row.get("final_answer", ""))
    if not weak_current_answer(current, answer_type):
        return {"repaired": False, "answer_type": answer_type, "answer": current, "reason": "current_answer_has_required_type"}
    candidates = extract_candidates(row, answer_type)
    if not candidates:
        return {"repaired": False, "answer_type": answer_type, "answer": current, "reason": "no_candidate"}
    # Prefer the earliest retrieved strong-type span. This uses retrieval rank only,
    # not gold labels or answer correctness.
    best = sorted(candidates, key=lambda c: (c["rank"], len(c["answer"])))[0]
    return {
        "repaired": True,
        "answer_type": answer_type,
        "answer": best["answer"],
        "reason": "weak_current_answer_replaced_by_evidence_span",
        "candidate_rank": best["rank"],
        "candidate_doc_id": best.get("doc_id"),
    }


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    def avg(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / n if n else 0.0
    return {
        "n": n,
        "retrieval_top5_hit": avg("retrieval_top5_hit"),
        "answer_correct": avg("answer_correct"),
        "strict_success": avg("strict_success"),
        "reader_format_parse_success": avg("reader_format_parse_success"),
        "repairs_applied": sum(1 for row in rows if row.get("repair_applied")),
        "retrieval_hit_answer_wrong": sum(1 for row in rows if row.get("retrieval_top5_hit") and not row.get("answer_correct")),
    }


def repair_file(pred_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(pred_path)
    repaired_rows = []
    changes = []
    for row in rows:
        old_correct = bool(row.get("answer_correct"))
        old_strict = bool(row.get("strict_success"))
        decision = choose_repair(row)
        new_row = dict(row)
        new_row["repair_decision"] = decision
        new_row["repair_applied"] = bool(decision["repaired"])
        if decision["repaired"]:
            new_row["original_final_answer"] = row.get("final_answer", "")
            new_row["final_answer"] = decision["answer"]
        match = answer_match_details(str(new_row.get("final_answer", "")), str(new_row.get("gold_answer", "")))
        new_row["answer_match"] = match
        new_row["answer_correct"] = bool(match["answer_correct"])
        new_row["strict_success"] = bool(new_row["answer_correct"] and new_row.get("evidence_supported"))
        if old_correct != new_row["answer_correct"] or old_strict != new_row["strict_success"] or decision["repaired"]:
            changes.append({
                "sample_id": row.get("sample_id"),
                "answer_type": decision["answer_type"],
                "old_answer": row.get("final_answer", ""),
                "new_answer": new_row.get("final_answer", ""),
                "repair_applied": decision["repaired"],
                "reason": decision["reason"],
                "old_correct": old_correct,
                "new_correct": new_row["answer_correct"],
                "old_strict": old_strict,
                "new_strict": new_row["strict_success"],
                "gold_answer": row.get("gold_answer", ""),
            })
        repaired_rows.append(new_row)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pred_path.stem
    out_pred = output_dir / f"{stem}_repaired.jsonl"
    out_metrics = output_dir / f"{stem}_repair_metrics.json"
    out_changes = output_dir / f"{stem}_repair_changes.jsonl"
    before = evaluate_rows(rows)
    after = evaluate_rows(repaired_rows)
    metrics = {"input": str(pred_path), "predictions": str(out_pred), "changes": str(out_changes), "before": before, "after": after}
    write_jsonl(out_pred, repaired_rows)
    write_jsonl(out_changes, changes)
    write_json(out_metrics, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dagig_paper_main_v1/answer_repair"))
    args = parser.parse_args()
    all_metrics = [repair_file(path, args.output_dir) for path in args.predictions]
    write_json(args.output_dir / "answer_repair_summary.json", {"runs": all_metrics})
    lines = ["# Answer Stage Repair Report\n\n"]
    lines.append("Actual repair decisions use only question text, current answer text, and retrieved evidence text. Gold answers are used only for evaluation.\n\n")
    lines.append("| prediction file | before strict | after strict | before answer | after answer | repairs | hit-answer-wrong before | after |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for metrics in all_metrics:
        b, a = metrics["before"], metrics["after"]
        name = Path(metrics["input"]).name
        lines.append(
            f"| {name} | {100*b['strict_success']:.1f}% | {100*a['strict_success']:.1f}% | "
            f"{100*b['answer_correct']:.1f}% | {100*a['answer_correct']:.1f}% | "
            f"{a['repairs_applied']} | {b['retrieval_hit_answer_wrong']} | {a['retrieval_hit_answer_wrong']} |\n"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ANSWER_STAGE_REPAIR_REPORT.md").write_text("".join(lines))
    print(args.output_dir / "ANSWER_STAGE_REPAIR_REPORT.md")


if __name__ == "__main__":
    main()
