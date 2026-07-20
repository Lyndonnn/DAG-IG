"""Small in-memory frozen BM25 implementation for DAG-IG Causal v1."""

from __future__ import annotations

import json
import heapq
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PAPER_READY_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "i", "if", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "with", "you",
}


def tokenize(text: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def read_corpus(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("doc_id"):
                raise ValueError(f"Invalid corpus row at {path}:{line_no}")
            docs.append(row)
    return docs


class FrozenBM25:
    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()
        for index, doc in enumerate(docs):
            tokens = tokenize(" ".join(str(doc.get(key, "")) for key in ("title", "text", "url", "domain")))
            self.doc_lengths.append(len(tokens))
            tf = Counter(tokens)
            doc_freq.update(tf.keys())
            for term, count in tf.items():
                self.postings[term].append((index, count))
        n = max(1, len(docs))
        self.idf = {term: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}
        self.avgdl = sum(self.doc_lengths) / n

    @classmethod
    def from_jsonl(cls, path: Path) -> "FrozenBM25":
        return cls(read_corpus(path))

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        terms = [token for token in tokenize(query) if token not in STOP]
        if not terms:
            return []
        scores: defaultdict[int, float] = defaultdict(float)
        for term in Counter(terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_index, tf in self.postings.get(term, []):
                dl = self.doc_lengths[doc_index]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[doc_index] += idf * (tf * (self.k1 + 1.0)) / denom
        ranked = sorted(scores.items(), key=lambda item: (-item[1], str(self.docs[item[0]].get("doc_id"))))[:top_k]
        out: list[dict[str, Any]] = []
        for rank, (doc_index, score) in enumerate(ranked, 1):
            row = dict(self.docs[doc_index])
            row["rank"] = rank
            row["score"] = score
            out.append(row)
        return out


def paper_ready_tokenize(text: Any) -> list[str]:
    """Tokenization frozen by the 64k Pix2Fact-WebEvidence readiness audit."""
    return [token.lower() for token in PAPER_READY_TOKEN_RE.findall(str(text or "")) if len(token) > 1]


class PaperReadyBM25:
    """Exact BM25 protocol used by the 64k paper-ready corpus diagnostics."""

    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: list[int] = []
        for index, doc in enumerate(docs):
            counts = Counter(paper_ready_tokenize(" ".join([str(doc.get("title") or ""), str(doc.get("text") or "")])))
            self.doc_lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        n_docs = len(docs)
        self.avgdl = sum(self.doc_lengths) / max(n_docs, 1)
        self.idf = {
            term: math.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.postings.items()
        }
        self.doc_norm = [
            k1 * (1.0 - b + b * length / max(self.avgdl, 1e-9))
            for length in self.doc_lengths
        ]

    @classmethod
    def from_jsonl(cls, path: Path) -> "PaperReadyBM25":
        return cls(read_corpus(path))

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        query_terms = Counter(paper_ready_tokenize(query))
        scores: defaultdict[int, float] = defaultdict(float)
        for term, query_frequency in query_terms.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            term_idf = self.idf[term]
            for doc_index, term_frequency in postings:
                denominator = term_frequency + self.doc_norm[doc_index]
                scores[doc_index] += (
                    term_idf
                    * (term_frequency * (self.k1 + 1.0) / denominator)
                    * query_frequency
                )
        if not scores:
            return []
        ranked = heapq.nsmallest(
            top_k,
            scores.items(),
            key=lambda item: (-item[1], str(self.docs[item[0]]["doc_id"])),
        )
        results: list[dict[str, Any]] = []
        for rank, (doc_index, score) in enumerate(ranked, 1):
            row = dict(self.docs[doc_index])
            row["rank"] = rank
            row["score"] = round(float(score), 6)
            results.append(row)
        return results
