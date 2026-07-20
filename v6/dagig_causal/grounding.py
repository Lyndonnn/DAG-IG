"""Frozen GroundingDINO transition for executable visual actions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision.ops import box_convert, nms


class FrozenGrounder:
    def __init__(
        self,
        repo: Path,
        config: Path,
        checkpoint: Path,
        *,
        box_threshold: float = 0.1,
        text_threshold: float = 0.1,
        nms_threshold: float = 0.5,
        max_proposals: int = 5,
        device: str = "cuda",
    ) -> None:
        for path in (repo, config, checkpoint):
            if not path.exists():
                raise FileNotFoundError(path)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from groundingdino.util.inference import load_image, load_model, predict

        self._load_image = load_image
        self._predict = predict
        self.model = load_model(str(config), str(checkpoint), device=device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.max_proposals = max_proposals
        self.device = device

    def propose(self, image_path: str, expression: str) -> list[dict[str, Any]]:
        if not str(expression).strip():
            return []
        _image_source, image = self._load_image(str(image_path))
        boxes, scores, phrases = self._predict(
            model=self.model,
            image=image,
            caption=str(expression),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        if len(boxes) == 0:
            return []
        xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").clamp(0.0, 1.0)
        keep = nms(xyxy, scores, self.nms_threshold)[: self.max_proposals]
        proposals: list[dict[str, Any]] = []
        for rank, index_tensor in enumerate(keep, 1):
            index = int(index_tensor.item())
            proposals.append({
                "rank": rank,
                "bbox_0_1000_xyxy": [float(value * 1000.0) for value in xyxy[index].tolist()],
                "score": float(scores[index].item()),
                "phrase": str(phrases[index]),
            })
        return proposals


def crop_from_bbox(image_path: str, bbox_0_1000_xyxy: list[float]) -> tuple[Image.Image, list[int]]:
    if not isinstance(bbox_0_1000_xyxy, list) or len(bbox_0_1000_xyxy) != 4:
        raise ValueError(f"invalid_crop_bbox:{bbox_0_1000_xyxy}")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in bbox_0_1000_xyxy]
    pixel_bbox = [
        max(0, min(width - 1, round(x1 * width / 1000))),
        max(0, min(height - 1, round(y1 * height / 1000))),
        max(1, min(width, round(x2 * width / 1000))),
        max(1, min(height, round(y2 * height / 1000))),
    ]
    if pixel_bbox[2] <= pixel_bbox[0]:
        pixel_bbox[2] = min(width, pixel_bbox[0] + 1)
    if pixel_bbox[3] <= pixel_bbox[1]:
        pixel_bbox[3] = min(height, pixel_bbox[1] + 1)
    crop = image.crop(tuple(pixel_bbox))
    if crop.width < 1 or crop.height < 1:
        raise ValueError(f"empty_crop:{pixel_bbox}")
    return crop, pixel_bbox


def proposal_crops(image_path: str, proposals: list[dict[str, Any]]) -> tuple[list[Image.Image], list[list[int]]]:
    crops: list[Image.Image] = []
    pixel_boxes: list[list[int]] = []
    for proposal in proposals:
        crop, pixel_bbox = crop_from_bbox(image_path, proposal["bbox_0_1000_xyxy"])
        crops.append(crop)
        pixel_boxes.append(pixel_bbox)
    return crops, pixel_boxes


def full_image_proposal() -> list[dict[str, Any]]:
    return [{
        "rank": 1,
        "bbox_0_1000_xyxy": [0.0, 0.0, 1000.0, 1000.0],
        "score": 0.0,
        "phrase": "full image control",
    }]
