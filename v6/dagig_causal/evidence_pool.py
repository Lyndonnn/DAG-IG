"""Label-free evidence candidate normalization and sampling utilities."""

from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit


def compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def normalized_text_hash(value: Any) -> str:
    text = normalized_text(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def canonical_url(value: Any) -> str:
    raw = compact(value)
    if not raw:
        return ""
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
    query = [
        (key, val)
        for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in {"fbclid", "gclid", "ref", "source"}
    ]
    return f"{host}{path}" + (f"?{urlencode(sorted(query))}" if query else "")


def dedup_in_rank_order(
    docs: list[dict[str, Any]], *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Keep the first document for each canonical URL or exact text hash."""

    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_texts: set[str] = set()
    for doc in docs:
        url = canonical_url(doc.get("url"))
        text_hash = normalized_text_hash(doc.get("text"))
        if (url and url in seen_urls) or (text_hash and text_hash in seen_texts):
            continue
        selected.append(doc)
        if url:
            seen_urls.add(url)
        if text_hash:
            seen_texts.add(text_hash)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def plackett_luce_sample(
    scores: list[float], *, count: int, temperature: float, rng: random.Random
) -> tuple[int, ...]:
    if count < 1 or count > len(scores):
        raise ValueError("invalid evidence subset size")
    if temperature <= 0 or not all(math.isfinite(value) for value in scores):
        raise ValueError("invalid evidence policy scores or temperature")
    remaining = list(range(len(scores)))
    ordered: list[int] = []
    for _ in range(count):
        logits = [scores[index] / temperature for index in remaining]
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        threshold = rng.random() * sum(weights)
        cumulative = 0.0
        selected = remaining[-1]
        for index, weight in zip(remaining, weights):
            cumulative += weight
            if cumulative >= threshold:
                selected = index
                break
        ordered.append(selected)
        remaining.remove(selected)
    return tuple(ordered)


def plackett_luce_logprob(
    ordered_indices: tuple[int, ...], scores: list[float], *, temperature: float
) -> float:
    if len(set(ordered_indices)) != len(ordered_indices):
        raise ValueError("evidence action repeats a candidate")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    remaining = list(range(len(scores)))
    log_probability = 0.0
    for selected in ordered_indices:
        if selected not in remaining:
            raise ValueError("evidence action index is outside the candidate pool")
        logits = [scores[index] / temperature for index in remaining]
        maximum = max(logits)
        log_normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
        log_probability += scores[selected] / temperature - log_normalizer
        remaining.remove(selected)
    return log_probability
