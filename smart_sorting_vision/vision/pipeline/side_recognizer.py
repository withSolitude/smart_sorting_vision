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

    正式流水线的顺序：
        1. 完整侧面 ROI 先做 Shape / no_object 判断；
        2. 确认有物体后再做 Color 和 QR；
        3. 独立调用本类时，若没有传入预计算形状结果，仍兼容原来的
           crop -> Color -> QR -> Shape 调用方式。

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
        qr_check_content: bool = False,
    ) -> None:

        self.color_classifier = color_classifier
        self.qr_detector = qr_detector
        self.shape_classifier = shape_classifier
        self.enable_color = bool(enable_color)
        self.enable_qr = bool(enable_qr)
        self.enable_shape = bool(enable_shape)
        self.qr_check_content = bool(qr_check_content)

    def _normalize_qr_content(self, value: Any) -> str:
        """去掉解码结果首尾空白，供多帧一致性比较。"""
        return "" if value is None else str(value).strip()

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
        shape_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        self._validate(crop, crop_mask)
        # 记录调用方是否已经在完整 ROI 上完成形状分类。变量不能在下面
        # 复用为字典，否则二维码抑制复核的分支条件会永远失效。
        shape_was_precomputed = shape_result is not None

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

        qr_content = None
        if bool(qr_result.get("found", False)):
            normalized = self._normalize_qr_content(qr_result.get("text", ""))
            qr_content = normalized or None

        # ========================================================
        # 3. Shape
        # ========================================================

        if self.enable_shape:
            # 正式流程先在完整侧面 ROI 上做一次形状/空背景分类。
            # 传入预计算结果时不再对裁剪图重复推理，避免两次输入尺度不同
            # 导致“先判断有物体、后又判断成另一种形状”。
            shape_raw = (
                shape_result
                if shape_result is not None
                else self.shape_classifier.classify(crop)
            )
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

        if (
            bool(qr_result.get("found", False))
            and self.shape_classifier is not None
            and not shape_was_precomputed
        ):
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
            # 即使关闭内容共识，也保留解码内容供调试显示；
            # 是否参与 READY 判断由 SideRecognitionConsensus 决定。
            "qr_content": qr_content,
            "shape": shape_result.get("shape", "unknown"),
            "shape_confidence": float(shape_result.get("confidence", 0.0)),
            "details": {
                "color": color_result,
                "qr": qr_result,
                "shape": shape_result,
                "shape_selection": shape_meta,
            },
        }
