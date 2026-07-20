"""Image-aware listwise evidence policy for the frozen Bayesian DAG-IG target.

The candidate order is randomized and all rank-like metadata is omitted from
the prompt.  Scores are mapped back to canonical document indices before the
reference-anchored policy is formed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch


LABELS = tuple("ABCDEFGHIJKLMNOPQRST")


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def candidate_token_ids(tokenizer: Any) -> torch.Tensor:
    token_ids: list[int] = []
    for label in LABELS:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != label:
            raise ValueError(f"Candidate label is not one exact token: {label} -> {encoded}")
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Candidate label token IDs are not unique")
    return torch.tensor(token_ids, dtype=torch.long)


def deterministic_permutation(
    group_id: str,
    *,
    seed: int,
    phase: str,
    epoch: int = 0,
    replicate: int = 0,
    count: int = 20,
) -> list[int]:
    payload = f"{seed}|{group_id}|{phase}|{epoch}|{replicate}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    values = list(range(count))
    random.Random(local_seed).shuffle(values)
    return values


def policy_prompt(
    context: dict[str, Any],
    permutation: list[int],
    *,
    candidate_text_chars: int,
) -> str:
    docs = list(context.get("retrieved_docs") or [])
    if len(docs) != len(LABELS) or sorted(permutation) != list(range(len(LABELS))):
        raise ValueError("Multimodal evidence policy requires one permutation of 20 documents")
    blocks: list[str] = []
    for position, canonical_index in enumerate(permutation):
        doc = docs[canonical_index]
        # Deliberately exclude doc_id, URL, source, retrieval rank, and BM25 score.
        blocks.append(
            "\n".join(
                [
                    f"[Candidate {LABELS[position]}]",
                    f"Title: {compact(doc.get('title'))}",
                    f"Domain: {compact(doc.get('domain'))}",
                    f"Text: {compact(doc.get('text'))[:candidate_text_chars]}",
                ]
            )
        )
    return "\n\n".join(
        [
            "You are the evidence-selection node of a multimodal search agent.",
            "Use the original image, question, upstream visual observation, and search query to select the single document that best supports answering the question.",
            "The candidate labels are randomly assigned for this presentation and carry no rank meaning.",
            "Do not answer the question and do not provide reasoning. Your next token must be exactly one candidate label A through T.",
            f"Question: {compact(context.get('question'))}",
            "Upstream visual node: " + json.dumps(context.get("visual_action") or {}, ensure_ascii=False, sort_keys=True),
            f"Upstream query node: {compact(context.get('search_query'))}",
            "Candidate evidence documents:",
            "\n\n".join(blocks),
        ]
    )


def resolve_image_path(image_path: str, project_root: Path) -> Path:
    path = Path(str(image_path))
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def pointer_logits(
    model: Any,
    processor: Any,
    *,
    context: dict[str, Any],
    image_path: Path,
    permutation: list[int],
    candidate_text_chars: int,
    max_pixels: int,
    max_input_tokens: int,
    label_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Return canonical-document logits from one randomly ordered image prompt."""
    from qwen_vl_utils import process_vision_info

    prompt = policy_prompt(
        context,
        permutation,
        candidate_text_chars=candidate_text_chars,
    )
    image: dict[str, Any] = {"type": "image", "image": str(image_path)}
    if max_pixels > 0:
        image["max_pixels"] = int(max_pixels)
    messages = [
        {
            "role": "user",
            "content": [image, {"type": "text", "text": prompt}],
        }
    ]
    prefix = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + "Best evidence candidate label: "
    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs:
        raise ValueError("Unexpected video input in multimodal evidence policy")
    if len(image_inputs) != 1:
        raise ValueError("Each evidence state must contain exactly one image")
    batch = processor(
        text=[prefix],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_tokens = int(batch["attention_mask"].sum().item())
    if input_tokens > max_input_tokens:
        raise ValueError(
            f"multimodal_pointer_input_tokens_exceed_limit:{input_tokens}>{max_input_tokens}"
        )
    inputs = batch.to(model.device)
    output = model(**inputs, use_cache=False)
    next_token_logits = output.logits[0, input_tokens - 1].float()
    ids = (
        label_token_ids
        if label_token_ids is not None
        else candidate_token_ids(processor.tokenizer)
    ).to(next_token_logits.device)
    position_logits = next_token_logits[ids]
    inverse = [0] * len(permutation)
    for position, canonical_index in enumerate(permutation):
        inverse[canonical_index] = position
    canonical_logits = torch.stack([position_logits[inverse[index]] for index in range(20)])
    return canonical_logits, input_tokens


def model_file_hash_inputs(model_dir: Path) -> list[Path]:
    names = {
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    }
    files = [path for path in model_dir.iterdir() if path.name in names]
    files.extend(sorted(model_dir.glob("model-*.safetensors")))
    if (model_dir / "model.safetensors").is_file():
        files.append(model_dir / "model.safetensors")
    unique = sorted({path.resolve() for path in files})
    if not any(path.suffix == ".safetensors" for path in unique):
        raise FileNotFoundError(f"No model weights under {model_dir}")
    return unique


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
