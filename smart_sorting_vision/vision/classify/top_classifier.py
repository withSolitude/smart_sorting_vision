"""顶部轻量分类：只给“颜色 + 形状族”，供决赛的候选筛选使用。

设计约束（现场得出）：

- 顶部是俯视：圆柱 / 圆台 / 圆锥 / 球的轮廓都是圆，靠 mask 几何**无法**
  区分，因此这里只输出形状“族”（tri / quad / pent / hex / round），
  具体形状由侧面 seg 决定；决赛的筛选也按族匹配。
- 颜色直接复用侧面同一套 ``ColorClassifier`` 规则，保证顶部和侧面的
  颜色名称一致（箱子里的键是侧面写进去的）。
- 初赛（``finals.enabled=false``）不会实例化本模块，现有流程零开销。

所有方法都不抛异常给上层：识别失败返回 unknown，由调用方决定是否跳过。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision.color.color_classifier import ColorClassifier


# 侧面 seg / 分类输出的 12 个形状（不含“空托盘”）-> 俯视轮廓族。
# 俯视无法区分同一族内部的具体形状，匹配时按族匹配。
SHAPE_FAMILY = {
    "三棱柱": "tri",
    "三棱锥": "tri",
    "正方体": "quad",
    "长方体": "quad",
    "四棱锥": "quad",
    "四棱台": "quad",
    "五棱柱": "pent",
    "六棱柱": "hex",
    "圆柱": "round",
    "圆台": "round",
    "圆锥": "round",
    "球": "round",
}


def shape_family(shape: str) -> str:
    """把侧面形状名映射成俯视族；未知返回 unknown。"""
    return SHAPE_FAMILY.get(str(shape).strip(), "unknown")


@dataclass(frozen=True)
class TopClassResult:
    color: str
    shape_family: str
    confidence: float
    color_confidence: float
    shape_confidence: float
    reason: str = ""

    def key(self) -> str:
        return f"{self.color}|{self.shape_family}"


class TopObjectClassifier:
    """在顶部 mask 上做颜色 + 形状族分类。"""

    def __init__(self, color_classifier: ColorClassifier | None = None) -> None:
        self.color_classifier = color_classifier or ColorClassifier()

    def classify(
        self,
        frame: np.ndarray,
        polygon: np.ndarray,
        *,
        rectangularity: float,
        circularity: float,
        hull_vertex_count: int,
        aspect_ratio: float,
    ) -> TopClassResult:
        family, shape_confidence = self._shape_family(
            rectangularity=float(rectangularity),
            circularity=float(circularity),
            hull_vertex_count=int(hull_vertex_count),
            aspect_ratio=float(aspect_ratio),
        )
        color, color_confidence, reason = self._color(frame, polygon)
        return TopClassResult(
            color=color,
            shape_family=family,
            confidence=float(min(color_confidence, shape_confidence)),
            color_confidence=float(color_confidence),
            shape_confidence=float(shape_confidence),
            reason=reason,
        )

    # ------------------------------------------------------------------
    # 形状族
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_family(
        *,
        rectangularity: float,
        circularity: float,
        hull_vertex_count: int,
        aspect_ratio: float,
    ) -> tuple[str, float]:
        # 圆：圆度足够高且凸包顶点多。
        if circularity >= 0.88 and hull_vertex_count >= 8:
            return "round", 0.85
        if hull_vertex_count == 3:
            return "tri", 0.80
        if hull_vertex_count == 4:
            # 正方体/长方体/四棱锥/四棱台俯视都是四边形，只能给族。
            return "quad", 0.80 if rectangularity >= 0.85 else 0.65
        if hull_vertex_count == 5:
            return "pent", 0.75
        if hull_vertex_count == 6:
            return "hex", 0.75
        return "unknown", 0.20

    # ------------------------------------------------------------------
    # 颜色
    # ------------------------------------------------------------------

    def _color(
        self,
        frame: np.ndarray,
        polygon: np.ndarray,
    ) -> tuple[str, float, str]:
        crop, mask = self._crop_mask(frame, polygon)
        if crop is None or mask is None:
            return "unknown", 0.0, "crop_failed"
        try:
            result = self.color_classifier.classify(crop, mask)
        except Exception as exc:  # 颜色识别失败不应影响决赛流程
            return "unknown", 0.0, f"color_error:{exc}"
        if not bool(result.get("found", False)):
            return "unknown", 0.0, str(result.get("reason", "color_not_found"))
        return (
            str(result.get("color", "unknown")),
            float(result.get("confidence", 0.0)),
            str(result.get("reason", "")),
        )

    @staticmethod
    def _crop_mask(
        frame: np.ndarray,
        polygon: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3 or not np.all(np.isfinite(pts)):
            return None, None
        height, width = frame.shape[:2]
        x0 = max(0, int(np.floor(float(pts[:, 0].min()))) - 2)
        y0 = max(0, int(np.floor(float(pts[:, 1].min()))) - 2)
        x1 = min(width, int(np.ceil(float(pts[:, 0].max()))) + 3)
        y1 = min(height, int(np.ceil(float(pts[:, 1].max()))) + 3)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, None
        crop = frame[y0:y1, x0:x1]
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        shifted = np.round(pts - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [shifted], 255)
        if cv2.countNonZero(mask) < 50:
            return None, None
        return crop, mask
