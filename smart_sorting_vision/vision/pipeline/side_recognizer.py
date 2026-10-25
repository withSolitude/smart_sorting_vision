from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vision.color.color_classifier import ColorClassifier
from vision.code_reader.qr_detector import QRDetector
from vision.shape.shape_classifier import ShapeClassifier


class SideRecognizer:
    """
    侧面识别统一接口。

    当前顺序：
        1. Color
        2. QR
        3. Shape

    QR 存在时会额外做一次“二维码区域抑制后的 Shape 推理”。
    目的不是删除真实物体轮廓，而是降低高对比二维码贴纸对分类模型的域偏移。
    正常无二维码物体仍只跑一次 Shape，不增加常态计算量。
    """

    def __init__(
        self,
        *,
        color_classifier: ColorClassifier,
        qr_detector: QRDetector,
        shape_classifier: ShapeClassifier | None,
        enable_color: bool = True,
        enable_qr: bool = True,
        enable_shape: bool = True,
    ) -> None:

        self.color_classifier = color_classifier
        self.qr_detector = qr_detector
        self.shape_classifier = shape_classifier
        self.enable_color = bool(enable_color)
        self.enable_qr = bool(enable_qr)
        self.enable_shape = bool(enable_shape)

    @staticmethod
    def _validate(
        crop: np.ndarray,
        crop_mask: np.ndarray,
    ) -> None:

        if crop is None:
            raise ValueError("SideRecognizer: crop 为空")

        if crop_mask is None:
            raise ValueError("SideRecognizer: crop_mask 为空")

        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("SideRecognizer: crop 必须是 BGR 三通道图像")

        if crop_mask.ndim != 2:
            raise ValueError("SideRecognizer: crop_mask 必须是单通道图像")

        if crop.shape[:2] != crop_mask.shape[:2]:
            raise ValueError("SideRecognizer: crop 与 crop_mask 尺寸不一致")

    @staticmethod
    def _qr_suppressed_shape_image(
        crop: np.ndarray,
        crop_mask: np.ndarray,
        qr_result: dict[str, Any],
    ) -> np.ndarray | None:
        """
        用 OpenCV inpaint 抑制二维码贴纸的高对比纹理。

        - 仅在 QR 已通过 QRDetector 的 Finder 验证后调用；
        - 只修改二维码四边形附近；
        - 物体外轮廓保持不变。
        """

        points = qr_result.get("points")
        if points is None:
            return None

        try:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None

        if pts.shape != (4, 2):
            return None

        center = np.mean(pts, axis=0, keepdims=True)

        # 稍微覆盖二维码白边，避免模型仍被高对比外框吸引。
        expanded = center + (pts - center) * 1.10
        expanded[:, 0] = np.clip(expanded[:, 0], 0, crop.shape[1] - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, crop.shape[0] - 1)

        qr_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(
            qr_mask,
            np.round(expanded).astype(np.int32),
            255,
        )

        # 若 object mask 在二维码处有小孔，不应因此保留二维码；
        # 只要求二维码中心大部分仍位于 object crop 内。
        object_region = crop_mask > 0
        overlap = np.count_nonzero((qr_mask > 0) & object_region)
        qr_area = max(1, np.count_nonzero(qr_mask))

        if overlap / qr_area < 0.45:
            return None

        # TELEA 对平面 3D 打印件表面足够稳定，并且只在 QR 物体上执行。
        return cv2.inpaint(
            crop,
            qr_mask,
            7.0,
            cv2.INPAINT_TELEA,
        )

    @staticmethod
    def _shape_confidence(result: dict[str, Any] | None) -> float:
        if not result:
            return 0.0
        return float(result.get("confidence", 0.0))

    def recognize(
        self,
        crop: np.ndarray,
        crop_mask: np.ndarray,
    ) -> dict[str, Any]:

        self._validate(crop, crop_mask)

        # ========================================================
        # 1. Color
        # ========================================================

        if self.enable_color:
            color_result = self.color_classifier.classify(
                crop,
                crop_mask,
            )
        else:
            color_result = {"color": "disabled", "confidence": 0.0}

        # ========================================================
        # 2. QR
        # ========================================================

        if self.enable_qr:
            qr_result = self.qr_detector.detect(crop)
        else:
            qr_result = {"found": False, "disabled": True}

        # ========================================================
        # 3. Shape
        # ========================================================

        if self.enable_shape:
            shape_raw = self.shape_classifier.classify(crop)
        else:
            shape_raw = {"shape": "disabled", "confidence": 0.0}
        shape_result = shape_raw

        shape_meta: dict[str, Any] = {
            "qr_suppression_applied": False,
            "selected_mode": "raw",
            "raw_shape": shape_raw.get("raw_shape", shape_raw.get("shape", "unknown")),
            "raw_confidence": self._shape_confidence(shape_raw),
            "suppressed_shape": None,
            "suppressed_confidence": None,
        }

        if bool(qr_result.get("found", False)):
            suppressed = self._qr_suppressed_shape_image(
                crop,
                crop_mask,
                qr_result,
            )

            if suppressed is not None:
                shape_meta["qr_suppression_applied"] = True
                shape_clean = self.shape_classifier.classify(suppressed)

                clean_conf = self._shape_confidence(shape_clean)
                raw_conf = self._shape_confidence(shape_raw)

                shape_meta["suppressed_shape"] = shape_clean.get(
                    "raw_shape",
                    shape_clean.get("shape", "unknown"),
                )
                shape_meta["suppressed_confidence"] = clean_conf

                # 只在抑制 QR 后明显更有把握时替换，避免正常样本被无谓改变。
                if clean_conf >= raw_conf + 0.03:
                    shape_result = shape_clean
                    shape_meta["selected_mode"] = "qr_suppressed"

        return {
            "object_found": True,
            "color": color_result.get("color", "unknown"),
            "color_confidence": float(color_result.get("confidence", 0.0)),
            "qr_present": bool(qr_result.get("found", False)),
            "shape": shape_result.get("shape", "unknown"),
            "shape_confidence": float(shape_result.get("confidence", 0.0)),
            "details": {
                "color": color_result,
                "qr": qr_result,
                "shape": shape_result,
                "shape_selection": shape_meta,
            },
        }
