"""Exact listwise evidence-pointer policy shared by rollout and training."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

import torch


LABELS = tuple("ABCDEFGHIJKLMNOPQRST")


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def pointer_prompt(
    question: str,
    visual_action: dict[str, Any],
    search_query: str,
    docs: list[dict[str, Any]],
    *,
    text_chars: int,
) -> str:
    candidates = docs[: len(LABELS)]
    if len(candidates) < 3:
        raise ValueError(f"Evidence pointer requires >=3 candidates, found {len(candidates)}")
    blocks: list[str] = []
    for label, doc in zip(LABELS, candidates):
        blocks.append(
            "\n".join(
                [
                    f"[Candidate {label}; retrieval_rank={doc.get('rank')}]",
                    f"Title: {compact(doc.get('title'))}",
                    f"Domain: {compact(doc.get('domain'))}",
                    f"Text: {compact(doc.get('text'))[:text_chars]}",
                ]
            )
        )
    return "\n\n".join(
        [
            "You are the constrained evidence pointer of a search agent.",
            f"Score candidates A through {LABELS[len(candidates) - 1]} by how well they support answering the question.",
            "The policy selects an unordered set of exactly three distinct labels.",
            "Do not answer the question. Candidate labels are local to this prompt.",
            f"Question: {question}",
            f"Visual action: {json.dumps(visual_action, ensure_ascii=False)}",
            f"Search query: {search_query}",
            "Frozen retriever candidates:",
            "\n\n".join(blocks),
            "The next tokens encode the selected local candidate labels.",
        ]
    )


def pointer_prefix(processor: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + '{"selected_labels":['


def candidate_token_ids(tokenizer: Any) -> torch.Tensor:
    token_ids: list[int] = []
    for label in LABELS:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != label:
            raise ValueError(f"Pointer label is not one exact token: {label} -> {encoded}")
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Pointer candidate token IDs are not unique")
    return torch.tensor(token_ids, dtype=torch.long)


def pointer_logits(
    model: Any,
    processor: Any,
    *,
    question: str,
    visual_action: dict[str, Any],
    search_query: str,
    docs: list[dict[str, Any]],
    text_chars: int,
    max_input_tokens: int,
) -> tuple[torch.Tensor, int]:
    prompt = pointer_prompt(
        question,
        visual_action,
        search_query,
        docs,
        text_chars=text_chars,
    )
    prefix = pointer_prefix(processor, prompt)
    batch = processor(text=[prefix], padding=True, return_tensors="pt")
    input_tokens = int(batch["input_ids"].shape[1])
    if input_tokens > max_input_tokens:
        raise ValueError(
            f"pointer_input_tokens_exceed_limit:{input_tokens}>{max_input_tokens}"
        )
    inputs = {
        key: value.to(model.device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    label_ids = candidate_token_ids(processor.tokenizer).to(logits.device)
    return logits[label_ids[: min(len(LABELS), len(docs))]], input_tokens


def sample_without_replacement(logits: torch.Tensor, seed: int, count: int = 3) -> tuple[int, ...]:
    if len(logits) < count:
        raise ValueError(f"Cannot sample {count} labels from {len(logits)} candidates")
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(int(seed))
    remaining = list(range(len(logits)))
    selected: list[int] = []
    for _ in range(count):
        local_logits = logits[torch.tensor(remaining, device=logits.device)]
        local_index = int(
            torch.multinomial(
                torch.softmax(local_logits, dim=0),
                num_samples=1,
                replacement=False,
                generator=generator,
            ).item()
        )
        selected.append(remaining.pop(local_index))
    # The policy action is an unordered set; canonical storage is index order.
    return tuple(sorted(selected))


def action_from_indices(indices: tuple[int, ...], docs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(indices) != 3 or len(set(indices)) != 3:
        raise ValueError(f"Expected three distinct evidence indices, got {indices}")
    candidates = docs[: len(LABELS)]
    if any(index < 0 or index >= len(candidates) for index in indices):
        raise ValueError(f"Evidence index outside candidate universe: {indices}")
    return {
        "selected_labels": [LABELS[index] for index in indices],
        "selected_ranks": [index + 1 for index in indices],
        "selected_doc_ids": [str(candidates[index]["doc_id"]) for index in indices],
        "surface": "listwise_pointer_labels",
    }


def sample_action(
    model: Any,
    processor: Any,
    *,
    question: str,
    visual_action: dict[str, Any],
    search_query: str,
    docs: list[dict[str, Any]],
    seed: int,
    sample: bool,
    text_chars: int,
    max_input_tokens: int,
) -> tuple[dict[str, Any], str, int]:
    with torch.no_grad():
        logits, input_tokens = pointer_logits(
            model,
            processor,
            question=question,
            visual_action=visual_action,
            search_query=search_query,
            docs=docs,
            text_chars=text_chars,
            max_input_tokens=max_input_tokens,
        )
    indices = (
        sample_without_replacement(logits, seed)
        if sample
        else tuple(sorted(int(value) for value in torch.topk(logits, k=3).indices.tolist()))
    )
    action = action_from_indices(indices, docs)
    raw = json.dumps(
        {"selected_labels": action["selected_labels"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return action, raw, input_tokens


def selected_indices(action: dict[str, Any], docs: list[dict[str, Any]]) -> tuple[int, int, int]:
    candidates = docs[: len(LABELS)]
    doc_index = {str(doc.get("doc_id")): index for index, doc in enumerate(candidates)}
    if action.get("selected_doc_ids"):
        ids = [str(value) for value in action["selected_doc_ids"]]
        if any(value not in doc_index for value in ids):
            raise ValueError(f"Evidence action contains an out-of-state doc id: {ids}")
        indices = [doc_index[value] for value in ids]
    elif action.get("selected_labels"):
        indices = [LABELS.index(str(value)) for value in action["selected_labels"]]
    else:
        indices = [int(value) - 1 for value in action.get("selected_ranks") or []]
    if len(indices) != 3 or len(set(indices)) != 3:
        raise ValueError(f"Invalid three-document evidence action: {indices}")
    if any(index < 0 or index >= len(candidates) for index in indices):
        raise ValueError(f"Evidence action outside pointer universe: {indices}")
    return tuple(sorted(indices))  # type: ignore[return-value]


def unordered_set_logprob(logits: torch.Tensor, selected: tuple[int, int, int]) -> torch.Tensor:
    """Exact Plackett--Luce probability marginalized over all 3! orders."""
    ordered: list[torch.Tensor] = []
    universe = set(range(len(logits)))
    for permutation in itertools.permutations(selected):
        remaining = set(universe)
        value = logits.new_zeros(())
        for index in permutation:
            active = torch.tensor(sorted(remaining), device=logits.device)
            value = value + logits[index] - torch.logsumexp(logits[active], dim=0)
            remaining.remove(index)
        ordered.append(value)
    return torch.logsumexp(torch.stack(ordered), dim=0)


def all_unordered_set_logprobs(
    logits: torch.Tensor,
) -> tuple[list[tuple[int, int, int]], torch.Tensor]:
    """Return the normalized distribution over every unordered top-3 set.

    The Plackett--Luce policy samples without replacement.  We marginalize the
    six orderings of each set and keep the computation differentiable so it can
    be used for an exact action-space KL during evidence-node optimization.
    """
    if logits.ndim != 1 or len(logits) < 3:
        raise ValueError("Evidence pointer requires a 1-D universe of at least 3 docs")
    combinations = list(itertools.combinations(range(len(logits)), 3))
    permutations = torch.tensor(
        [permutation for combo in combinations for permutation in itertools.permutations(combo)],
        dtype=torch.long,
        device=logits.device,
    ).view(len(combinations), 6, 3)
    scores = logits.float()
    first = permutations[:, :, 0]
    second = permutations[:, :, 1]
    third = permutations[:, :, 2]
    log_z = torch.logsumexp(scores, dim=0)
    n = len(scores)
    indices = torch.arange(n, device=scores.device)
    first_masked = scores.unsqueeze(0).expand(n, n).masked_fill(
        indices.unsqueeze(0) == indices.unsqueeze(1), float("-inf")
    )
    log_z_after_first = torch.logsumexp(first_masked, dim=1)
    candidate_index = indices.view(1, 1, n)
    first_index = indices.view(n, 1, 1)
    second_index = indices.view(1, n, 1)
    pair_mask = (candidate_index == first_index) | (
        candidate_index == second_index
    )
    pair_masked = scores.view(1, 1, n).expand(n, n, n).masked_fill(
        pair_mask, float("-inf")
    )
    log_z_after_pair = torch.logsumexp(pair_masked, dim=2)
    ordered_logprobs = (
        scores[first]
        - log_z
        + scores[second]
        - log_z_after_first[first]
        + scores[third]
        - log_z_after_pair[first, second]
    )
    set_logprobs = torch.logsumexp(ordered_logprobs, dim=1)
    # The six-order marginal is already normalized analytically. Renormalizing
    # removes only floating-point drift and makes downstream KL checks exact.
    set_logprobs = set_logprobs - torch.logsumexp(set_logprobs, dim=0)
    return combinations, set_logprobs
