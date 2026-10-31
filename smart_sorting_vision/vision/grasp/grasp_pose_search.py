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
    # 局部接触带两侧支撑中点与夹爪中线的距离；0表示居中，越大越偏。
    # 放在末尾并提供默认值，兼容旧的7个位置参数构造。
    contact_balance_mm: float = 0.0

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


def _clip_longitudinal_strip(
    local_polygon: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """在夹爪局部坐标中裁取纵向接触带，保留边与带边界的交点。

    不能只筛选原顶点：圆锥/三角形的接触带往往不包含任何原顶点。
    输入保持完整实例的凹轮廓，不能先取凸包填平真实凹口。
    """
    points = np.asarray(local_polygon, dtype=np.float64).reshape(-1, 2)
    for boundary, sign in ((float(minimum), 1.0), (float(maximum), -1.0)):
        if len(points) == 0:
            break
        clipped: list[np.ndarray] = []
        previous = points[-1]
        previous_inside = sign * (previous[1] - boundary) >= 0.0
        for current in points:
            current_inside = sign * (current[1] - boundary) >= 0.0
            if current_inside != previous_inside:
                alpha = (boundary - previous[1]) / (current[1] - previous[1])
                clipped.append(previous + alpha * (current - previous))
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        points = np.asarray(clipped, dtype=np.float64).reshape(-1, 2)
    return points


def _closing_line_crosses_target(local_polygon: np.ndarray, longitudinal: float) -> bool:
    """确认实际指爪中心处的闭合线穿过中线两侧的同一段目标实体。

    对原始凹轮廓做扫描线奇偶配对。不能用凸包或整条扫描线的最左/最右点，
    否则U形目标中间的空凹口会被误当成可夹物体。
    """
    crossings: list[float] = []
    previous = local_polygon[-1]
    for current in local_polygon:
        y0, y1 = float(previous[1]), float(current[1])
        # 半开区间避免一条线经过顶点时把该顶点重复计数；水平边不参与配对。
        if (y0 <= longitudinal < y1) or (y1 <= longitudinal < y0):
            alpha = (longitudinal - y0) / (y1 - y0)
            crossings.append(float(previous[0] + alpha * (current[0] - previous[0])))
        previous = current
    crossings.sort()
    return any(
        left < -1.0e-9 and right > 1.0e-9
        for left, right in zip(crossings[0::2], crossings[1::2])
    )


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
    local_contact: bool = False,
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

    if local_contact:
        # 估宽只读取实际指爪覆盖的截面，避免锥体另一端的宽底面把开口带大。
        # 这里只估局部接触几何，完整目标与指爪包络的侵入检查仍由流水线完成。
        local_polygon = np.column_stack((close_proj, long_proj))
        finger_center = float(finger_longitudinal_offset_mm)
        finger_half = max(0.0, float(finger_length_mm)) * 0.5
        if (
            finger_half <= 0.0
            or not np.all(np.isfinite(local_polygon))
            or not _closing_line_crosses_target(local_polygon, finger_center)
        ):
            return TargetEngagement(
                0.0, 0.0, -math.inf, -math.inf, -math.inf, False, False,
                contact_balance_mm=math.inf,
            )
        local_polygon = _clip_longitudinal_strip(
            local_polygon, finger_center - finger_half, finger_center + finger_half,
        )
        if len(local_polygon) < 3:
            return TargetEngagement(
                0.0, 0.0, -math.inf, -math.inf, -math.inf, False, False,
                contact_balance_mm=math.inf,
            )
        close_proj = local_polygon[:, 0]
        long_proj = local_polygon[:, 1]

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
        contact_balance_mm=(abs((close_min + close_max) * 0.5) if local_contact else 0.0),
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
