from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TargetEngagement:
    """目标物体与某个候选夹爪姿态之间的基础几何关系。"""

    width_mm: float
    longitudinal_overlap_mm: float
    left_gap_mm: float
    right_gap_mm: float
    min_side_gap_mm: float
    fits_between_fingers: bool
    has_longitudinal_overlap: bool

    @property
    def valid(self) -> bool:
        return bool(self.fits_between_fingers and self.has_longitudinal_overlap)


def normalize_axial_deg(angle_deg: float) -> float:
    """把180°周期的夹爪闭合轴归一化到 [-90, 90)。"""
    value = (float(angle_deg) + 90.0) % 180.0 - 90.0
    # 避免浮点导致 +90 恰好出现。
    if value >= 90.0:
        value -= 180.0
    return float(value)


def axis_vectors(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """返回闭合轴和指爪纵向轴在机器人XY平面中的单位向量。"""
    theta = math.radians(float(yaw_deg))
    closing = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
    longitudinal = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    return closing, longitudinal


def evaluate_target_engagement(
    polygon_robot_mm: np.ndarray,
    center_robot_mm: tuple[float, float] | np.ndarray,
    yaw_deg: float,
    opening_mm: float,
    *,
    finger_length_mm: float,
    finger_longitudinal_offset_mm: float = 0.0,
    min_side_gap_mm: float = 0.25,
    min_longitudinal_overlap_mm: float = 1.5,
) -> TargetEngagement:
    """
    检查目标物体是否真正位于两指之间，并且两根指爪的纵向覆盖仍能碰到目标。

    这一步是多姿态搜索中非常重要的保护：仅做“与其他物体不碰撞”还不够，
    抓取中心平移过多时，夹爪可能已经从目标旁边掠过。这里在进入障碍碰撞检查前
    先把这种候选剔除。
    """
    poly = np.asarray(polygon_robot_mm, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3 or not np.all(np.isfinite(poly)):
        return TargetEngagement(0.0, 0.0, -math.inf, -math.inf, -math.inf, False, False)

    center = np.asarray(center_robot_mm, dtype=np.float64).reshape(2)
    closing, longitudinal = axis_vectors(yaw_deg)
    rel = poly - center

    close_proj = rel @ closing
    long_proj = rel @ longitudinal

    close_min = float(np.min(close_proj))
    close_max = float(np.max(close_proj))
    width = max(0.0, close_max - close_min)

    half_open = max(0.0, float(opening_mm)) * 0.5
    left_gap = close_min + half_open
    right_gap = half_open - close_max
    min_gap = min(left_gap, right_gap)

    finger_half = max(0.0, float(finger_length_mm)) * 0.5
    finger_center = float(finger_longitudinal_offset_mm)
    finger_min = finger_center - finger_half
    finger_max = finger_center + finger_half
    target_min = float(np.min(long_proj))
    target_max = float(np.max(long_proj))
    overlap = max(0.0, min(finger_max, target_max) - max(finger_min, target_min))

    fits = bool(min_gap >= max(0.0, float(min_side_gap_mm)))
    has_overlap = bool(overlap >= max(0.0, float(min_longitudinal_overlap_mm)))

    return TargetEngagement(
        width_mm=float(width),
        longitudinal_overlap_mm=float(overlap),
        left_gap_mm=float(left_gap),
        right_gap_mm=float(right_gap),
        min_side_gap_mm=float(min_gap),
        fits_between_fingers=fits,
        has_longitudinal_overlap=has_overlap,
    )


def unique_float_sequence(values: Iterable[float] | None, fallback: tuple[float, ...]) -> tuple[float, ...]:
    source = fallback if values is None else tuple(values)
    result: list[float] = []
    for value in source:
        f = float(value)
        if not math.isfinite(f):
            continue
        if not any(abs(f - old) < 1e-9 for old in result):
            result.append(f)
    if not result:
        result = [0.0]
    if not any(abs(v) < 1e-9 for v in result):
        result.insert(0, 0.0)
    return tuple(result)


def build_search_offsets(
    yaw_offsets_deg: Iterable[float] | None,
    longitudinal_offsets_mm: Iterable[float] | None,
    closing_offsets_mm: Iterable[float] | None,
    *,
    max_trials: int,
) -> list[tuple[float, float, float]]:
    """按配置顺序生成 (yaw_offset, longitudinal_offset, closing_offset)。"""
    yaws = unique_float_sequence(yaw_offsets_deg, (0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0))
    longs = unique_float_sequence(longitudinal_offsets_mm, (0.0, -2.0, 2.0, -4.0, 4.0))
    closes = unique_float_sequence(closing_offsets_mm, (0.0,))

    limit = max(1, int(max_trials))
    out: list[tuple[float, float, float]] = []
    for yaw in yaws:
        for longitudinal in longs:
            for closing in closes:
                out.append((float(yaw), float(longitudinal), float(closing)))
                if len(out) >= limit:
                    return out
    return out
