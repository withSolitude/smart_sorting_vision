"""顶部相机内参标定板检测工具。

当前工程使用大凡 HC7 系列 7×7 对称圆点板，不再使用棋盘格角点。
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def read_pattern_config(config: dict[str, Any]) -> tuple[str, tuple[int, int], float]:
    """读取并检查 YAML 中的标定板类型、点阵尺寸和圆心间距。"""
    pattern_type = str(config.get("pattern_type", "symmetric_circles")).strip().lower()
    pattern_size_value = config.get("pattern_size")
    if (
        not isinstance(pattern_size_value, (list, tuple))
        or len(pattern_size_value) != 2
    ):
        raise ValueError("calibration.top_camera.pattern_size 必须是 [列数, 行数]")
    pattern_size = (int(pattern_size_value[0]), int(pattern_size_value[1]))
    if min(pattern_size) < 3:
        raise ValueError("圆点阵列每个方向至少需要 3 个点")

    spacing = config.get("circle_spacing_mm")
    if spacing is None or float(spacing) <= 0:
        raise ValueError(
            "请在 calibration.top_camera.circle_spacing_mm 填写相邻圆心的实际距离（mm）"
        )
    return pattern_type, pattern_size, float(spacing)


def create_dark_circle_detector() -> cv2.SimpleBlobDetector:
    """创建适合白底黑圆点板的 Blob 检测器。"""
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 240
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 100000
    params.filterByCircularity = True
    params.minCircularity = 0.55
    params.filterByConvexity = True
    params.minConvexity = 0.70
    params.filterByInertia = True
    params.minInertiaRatio = 0.35
    params.filterByColor = True
    params.blobColor = 0
    return cv2.SimpleBlobDetector_create(params)


def find_pattern(
    gray: np.ndarray,
    pattern_type: str,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """检测当前图像中的圆点阵列并返回圆心。"""
    if pattern_type not in {"symmetric_circles", "circles", "circle_grid"}:
        raise ValueError(f"不支持的标定板类型: {pattern_type}")
    flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
    found, centers = cv2.findCirclesGrid(
        gray,
        pattern_size,
        flags=flags,
        blobDetector=create_dark_circle_detector(),
    )
    return bool(found), centers if found else None


def make_object_points(
    pattern_size: tuple[int, int],
    spacing_mm: float,
) -> np.ndarray:
    """按圆点中心距生成标定板平面上的三维点。"""
    columns, rows = pattern_size
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32)
        * float(spacing_mm)
    )
    return points
