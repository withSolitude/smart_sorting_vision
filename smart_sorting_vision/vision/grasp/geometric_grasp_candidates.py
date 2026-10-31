"""完整实例轮廓的二维抓取候选点，不把收缩区域当作真实顶面。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class GraspPointSeed:
    pixel: tuple[int, int]
    source: str
    clearance_px: float
    quality: float = 0.0


def _ellipse(polygon: np.ndarray):
    """只有轮廓确实接近椭圆时才接受拟合，避免把方块拟合成圆。"""
    if len(polygon) < 8:
        return None
    ellipse = cv2.fitEllipse(polygon.astype(np.float32).reshape(-1, 1, 2))
    center, diameters, angle = ellipse
    a, b = float(diameters[0]) * 0.5, float(diameters[1]) * 0.5
    if min(a, b) <= 1.0:
        return None
    theta = math.radians(angle)
    delta = polygon - np.asarray(center)
    local_x = delta[:, 0] * math.cos(theta) + delta[:, 1] * math.sin(theta)
    local_y = -delta[:, 0] * math.sin(theta) + delta[:, 1] * math.cos(theta)
    residual = np.abs(np.sqrt((local_x / a) ** 2 + (local_y / b) ** 2) - 1.0)
    if float(np.median(residual)) > 0.08 or float(np.quantile(residual, 0.9)) > 0.16:
        return None
    return ellipse


def describe_geometry(polygon: np.ndarray) -> str:
    """只描述二维轮廓，不用顶点数断言物体的真实三维类别。"""
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return "general"
    contour = pts.reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    approx = cv2.approxPolyDP(hull, max(0.75, 0.025 * perimeter), True).reshape(-1, 2)
    if len(approx) == 3:
        return "tapered"
    if len(approx) == 4:
        edges = np.roll(approx, -1, axis=0) - approx
        lengths = np.linalg.norm(edges, axis=1)
        if np.min(lengths) > 1.0:
            directions = edges / lengths[:, None]
            # 两组对边的夹角都小于约12度，归为平行边几何；不假定它一定是长方体。
            solidity = abs(float(cv2.contourArea(contour))) / max(abs(float(cv2.contourArea(hull))), 1.0)
            if solidity > 0.9 and min(abs(directions[0] @ directions[2]), abs(directions[1] @ directions[3])) > 0.978:
                return "parallel"
    if _ellipse(pts) is not None:
        return "round"
    return "general"


def build_grasp_point_seeds(
    polygon: np.ndarray,
    safe_mask: np.ndarray,
    *,
    config: dict,
) -> list[GraspPointSeed]:
    """产生少量抓取点，最终由流水线联合比较方向、开口和夹指空间。

    配置：max_points、min_spacing_px、min_radius_ratio、min_clearance_px、
    axis_sample_fractions。截面采样比例相对于完整主轴长度，不是物体宽度。
    最大内接圆依据完整轮廓计算；safe_mask仅限制候选落点及安全间隙。
    """
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    allowed = np.asarray(safe_mask)
    if len(pts) < 3 or allowed.ndim != 2 or not np.any(allowed):
        return []
    max_points = max(1, int(config.get("max_points", 6)))
    spacing = max(1.0, float(config.get("min_spacing_px", 12.0)))
    radius_ratio = float(config.get("min_radius_ratio", 0.45))
    min_clearance = max(0.0, float(config.get("min_clearance_px", 3.0)))
    if not 0.0 <= radius_ratio <= 1.0:
        raise ValueError("min_radius_ratio 必须位于[0,1]")

    # 只在目标的局部外接矩形上做距离变换；一圈黑边让贴图像边缘的距离仍正确。
    height, width = allowed.shape
    bx, by, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2))
    x0, y0 = max(0, bx - 1), max(0, by - 1)
    x1, y1 = min(width, bx + bw + 1), min(height, by + bh + 1)
    if x1 <= x0 or y1 <= y0:
        return []
    origin = np.asarray([x0 - 1, y0 - 1], dtype=np.float64)
    local = np.zeros((y1 - y0 + 2, x1 - x0 + 2), dtype=np.uint8)
    cv2.fillPoly(local, [np.round(pts - origin).astype(np.int32)], 255)
    # 若轮廓超出图像，把图外部分清除，安全距离不向图外延伸。
    safe = np.zeros_like(local)
    safe[1:-1, 1:-1] = (allowed[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    local[0, :] = local[-1, :] = 0
    local[:, 0] = local[:, -1] = 0
    safe = cv2.bitwise_and(local, safe)
    thickness = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    clearance = cv2.distanceTransform(safe, cv2.DIST_L2, 5)
    maximum_radius = float(thickness.max())
    if maximum_radius <= 0.0:
        return []
    eligible = (safe > 0) & (clearance >= min_clearance) & (
        thickness >= max(min_clearance, maximum_radius * radius_ratio)
    )
    if not np.any(eligible):
        return []

    field = np.where(eligible, thickness, 0.0).astype(np.float32)
    peaks = ((field >= cv2.dilate(field, np.ones((3, 3), np.uint8)) - 1.0e-5) & eligible)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), 8)
    peak_values = np.zeros(count, dtype=np.float32)
    np.maximum.at(peak_values, labels.ravel(), field.ravel())
    peak_order = sorted(range(1, count), key=lambda i: (-peak_values[i], -stats[i, cv2.CC_STAT_AREA]))
    peak_points: list[np.ndarray] = []
    for label in peak_order:
        px, py, pw, ph = stats[label, :4]
        yy, xx = np.nonzero(labels[py:py + ph, px:px + pw] == label)
        locations = np.column_stack((xx + px, yy + py)).astype(np.float64)
        # 长方形的最大值是一整段平台：选平台中央，不选minMaxLoc遇到的第一个像素。
        best = locations[int(np.argmin(np.sum((locations - centroids[label]) ** 2, axis=1)))]
        if all(float(np.linalg.norm(best - old)) >= spacing for old in peak_points):
            peak_points.append(best)
        if len(peak_points) >= max_points:
            break

    moments = cv2.moments(pts.reshape(-1, 1, 2))
    center = (
        np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        if abs(moments["m00"]) > 1.0e-9 else pts.mean(axis=0)
    ) - origin
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    box_center = np.asarray(rect[0], dtype=np.float64) - origin
    box = cv2.boxPoints(rect).astype(np.float64) - origin
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    longest = int(np.argmax(lengths))
    axis = edges[longest] / max(float(lengths[longest]), 1.0e-9)

    geometry = describe_geometry(pts)
    proposals: list[tuple[np.ndarray, str]] = []
    if geometry == "round":
        ellipse = _ellipse(pts)
        proposals.append((np.asarray(ellipse[0]) - origin, "ellipse_center"))
    if peak_points:
        proposals.append((peak_points[0], "inscribed_peak"))
    proposals.append((center, "polygon_center"))
    if geometry != "tapered":
        proposals.append((box_center, "box_center"))
    # 多个粗主体/局部峰值要在主轴采样前参与，防止被固定采样点挤掉。
    proposals.extend((point, "inscribed_peak") for point in peak_points[1:])
    if geometry != "round" and (geometry == "tapered" or max(lengths) > 1.15 * max(min(lengths), 1.0)):
        axis_center = box_center
        axis_length = float(lengths[longest])
        if geometry == "tapered":
            # 三角轮廓的外接矩形可能斜跨尖角；优先沿尖端到对边中点的中线采样。
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            triangle = cv2.approxPolyDP(hull, max(0.75, 0.025 * cv2.arcLength(hull, True)), True).reshape(-1, 2)
            if len(triangle) == 3:
                angles = []
                for i in range(3):
                    a = triangle[(i + 1) % 3] - triangle[i]
                    b = triangle[(i + 2) % 3] - triangle[i]
                    angles.append(float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-9))
                tip_i = int(np.argmax(angles))
                tip = triangle[tip_i].astype(np.float64) - origin
                base = (triangle[(tip_i + 1) % 3] + triangle[(tip_i + 2) % 3]) * 0.5 - origin
                axis_length = float(np.linalg.norm(base - tip))
                axis = (base - tip) / max(axis_length, 1.0e-9)
                axis_center = (tip + base) * 0.5
        for fraction in config.get("axis_sample_fractions", [-0.3, 0.0, 0.3]):
            proposals.append((axis_center + float(fraction) * axis_length * axis, "axis_section"))

    # 中心/轴线候选允许在很小范围内投影到合法厚截面，禁止跨越凹口或跳到另一主体。
    snap_limit = max(2.0, min(spacing, maximum_radius * 0.35))
    seeds: list[GraspPointSeed] = []
    for point, source in proposals:
        ix, iy = int(round(float(point[0]))), int(round(float(point[1])))
        if not (0 <= iy < eligible.shape[0] and 0 <= ix < eligible.shape[1] and eligible[iy, ix]):
            sx0, sy0 = max(0, ix - math.ceil(snap_limit)), max(0, iy - math.ceil(snap_limit))
            sx1, sy1 = min(eligible.shape[1], ix + math.ceil(snap_limit) + 1), min(eligible.shape[0], iy + math.ceil(snap_limit) + 1)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            yy, xx = np.nonzero(eligible[sy0:sy1, sx0:sx1])
            if len(xx) == 0:
                continue
            distances = (xx + sx0 - point[0]) ** 2 + (yy + sy0 - point[1]) ** 2
            best = int(np.argmin(distances))
            if float(distances[best]) > snap_limit ** 2:
                continue
            ix, iy = int(xx[best] + sx0), int(yy[best] + sy0)
        pixel = (int(ix + origin[0]), int(iy + origin[1]))
        if any(math.hypot(pixel[0] - old.pixel[0], pixel[1] - old.pixel[1]) < spacing for old in seeds):
            continue
        quality = 0.7 * float(thickness[iy, ix]) / maximum_radius + 0.3 * min(
            1.0, float(clearance[iy, ix]) / max(float(thickness[iy, ix]), 1.0)
        )
        seeds.append(GraspPointSeed(pixel, source, float(clearance[iy, ix]), quality))
        if len(seeds) >= max_points:
            break
    return seeds
