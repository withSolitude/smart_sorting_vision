from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from vision.color.color_classifier import ColorClassifier
from vision.shape.shape_classifier import ShapeClassifier


@dataclass(frozen=True)
class TopAttributeResult:
    color: str
    shape: str
    color_confidence: float
    shape_confidence: float
    reason: str


class TopAttributeRecognizer:
    """识别顶部实例的颜色和形状。

    顶部 YOLO-seg 只负责给出实例 mask。颜色分类器直接读取 mask 内的
    HSV/Lab 像素，形状分类器读取同一 mask 裁剪出的单物体图像。
    """

    def __init__(
        self,
        *,
        color_classifier: ColorClassifier,
        shape_classifier: ShapeClassifier,
        padding_ratio: float = 0.12,
        shape_background_bgr: tuple[int, int, int] = (0, 0, 0),
        min_mask_pixels: int = 150,
    ) -> None:
        self.color_classifier = color_classifier
        self.shape_classifier = shape_classifier
        self.padding_ratio = max(0.0, float(padding_ratio))
        self.shape_background_bgr = tuple(int(v) for v in shape_background_bgr)
        self.min_mask_pixels = max(1, int(min_mask_pixels))

    @staticmethod
    def _polygon_mask(
        polygon: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        try:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        except (TypeError, ValueError):
            return None
        if len(points) < 3:
            return None
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
        return mask

    def _crop_instance(
        self,
        frame: np.ndarray,
        polygon: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        mask = self._polygon_mask(polygon, frame.shape)
        if mask is None:
            return None
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        if not rows.any() or not cols.any():
            return None

        y_indices = np.flatnonzero(rows)
        x_indices = np.flatnonzero(cols)
        y1, y2 = int(y_indices[0]), int(y_indices[-1]) + 1
        x1, x2 = int(x_indices[0]), int(x_indices[-1]) + 1
        dx = int(round((x2 - x1) * self.padding_ratio))
        dy = int(round((y2 - y1) * self.padding_ratio))
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
        x2, y2 = min(width, x2 + dx), min(height, y2 + dy)

        crop = frame[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        if int(np.count_nonzero(crop_mask)) < self.min_mask_pixels:
            return None
        return crop, crop_mask

    def classify(self, frame: np.ndarray, polygon: np.ndarray) -> TopAttributeResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return TopAttributeResult("unknown", "unknown", 0.0, 0.0, "invalid_frame")

        cropped = self._crop_instance(frame, polygon)
        if cropped is None:
            return TopAttributeResult("unknown", "unknown", 0.0, 0.0, "invalid_mask")

        crop, crop_mask = cropped
        try:
            color_result: dict[str, Any] = self.color_classifier.classify(crop, crop_mask)
        except Exception as exc:
            color_result = {"color": "unknown", "confidence": 0.0, "reason": str(exc)}

        # 形状模型只看当前实例，mask 外的像素填固定背景，避免邻物进入分类器。
        shape_image = np.empty_like(crop)
        shape_image[...] = np.asarray(self.shape_background_bgr, dtype=np.uint8)
        shape_image[crop_mask > 0] = crop[crop_mask > 0]
        try:
            shape_result = self.shape_classifier.classify(shape_image)
        except Exception as exc:
            shape_result = {"shape": "unknown", "confidence": 0.0, "reason": str(exc)}

        shape = str(shape_result.get("shape", "unknown"))
        if not bool(shape_result.get("object_present", True)):
            shape = "unknown"
        color = str(color_result.get("color", "unknown"))
        return TopAttributeResult(
            color=color,
            shape=shape,
            color_confidence=float(color_result.get("confidence", 0.0)),
            shape_confidence=float(shape_result.get("confidence", 0.0)),
            reason="ready" if color != "unknown" and shape != "unknown" else "unknown_attribute",
        )

    def enrich_objects(self, frame: np.ndarray, objects: list[dict[str, Any]]) -> None:
        for obj in objects:
            result = self.classify(frame, obj.get("polygon"))
            obj["color"] = result.color
            obj["shape"] = result.shape
            obj["color_confidence"] = result.color_confidence
            obj["shape_confidence"] = result.shape_confidence
            obj["attribute_reason"] = result.reason
