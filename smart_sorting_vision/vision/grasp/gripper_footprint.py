from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np

from vision.calibration.pixel_to_robot import PixelToRobotTransform


@dataclass
class GripperFootprintCheck:
    """一次实际夹爪平面包络碰撞检查结果。"""

    safe: bool
    reason: str

    # 名义夹爪轮廓（不含安全余量），用于调试显示。
    footprint_polygons_px: list[np.ndarray]

    # 名义轮廓到最近“其他物体/固定禁区”的距离。
    # 没有任何障碍物时给一个较大的有限值，便于评分。
    clearance_px: float
    clearance_mm: float

    object_collision_pixels: int = 0
    wall_collision_pixels: int = 0
    border_violation: bool = False
    allowed_region_violation: bool = False


@dataclass
class GripperCollisionContext:
    """同一目标的一组多姿态搜索共享的障碍缓存，避免每个候选重复做整幅图 distanceTransform。"""

    image_shape: tuple[int, ...]
    other_mask: np.ndarray
    wall_mask: np.ndarray
    distance_to_obstacle: np.ndarray | None



class GripperFootprintModel:
    """
    顶视二维真实夹爪包络模型。

    坐标约定：
        local +X : 夹爪闭合轴（yaw_deg）
        local +Y : 与闭合轴垂直

    两个指爪按矩形包络建模。相比直接把完整 CAD 网格带进实时循环，
    这种做法更保守、速度更快，而且所有关键尺寸都可以在 YAML 中实测微调。

    重要：
        1. preopen_mm 是下降前两指内侧的“有效净开口”。
        2. 其他物体来自顶部 segmentation polygon。
        3. 固定挡板/托盘边缘使用 mechanics.forbidden_zones_mm，推荐四点 polygon。
        4. target_polygon 本身不作为障碍物，因为它就是要抓的目标。
    """

    def __init__(
        self,
        *,
        transform: PixelToRobotTransform,
        enabled: bool = True,
        finger_length_mm: float = 20.0,
        finger_thickness_mm: float = 9.0,
        finger_longitudinal_offset_mm: float = 0.0,
        opening_uncertainty_mm: float = 1.5,
        object_margin_mm: float = 1.0,
        wall_margin_mm: float = 2.0,
        fallback_opening_mm: float = 38.3,
        forbidden_zones_mm: Iterable | None = None,
        allowed_region_mm: Iterable[float] | None = None,
        allowed_polygon_mm: Iterable | None = None,
        object_overlap_tolerance_px: int = 4,
        enforce_image_border: bool = True,
        image_border_margin_px: int = 2,
        body_enabled: bool = False,
        body_width_mm: float = 50.0,
        body_length_mm: float = 20.0,
        body_longitudinal_offset_mm: float = -25.0,
    ) -> None:
        self.transform = transform
        self.enabled = bool(enabled)

        self.finger_length_mm = float(finger_length_mm)
        self.finger_thickness_mm = float(finger_thickness_mm)
        self.finger_longitudinal_offset_mm = float(finger_longitudinal_offset_mm)
        self.opening_uncertainty_mm = float(opening_uncertainty_mm)
        self.object_margin_mm = float(object_margin_mm)
        self.wall_margin_mm = float(wall_margin_mm)
        self.fallback_opening_mm = float(fallback_opening_mm)

        self.enforce_image_border = bool(enforce_image_border)
        self.image_border_margin_px = int(image_border_margin_px)

        self.body_enabled = bool(body_enabled)
        self.body_width_mm = float(body_width_mm)
        self.body_length_mm = float(body_length_mm)
        self.body_longitudinal_offset_mm = float(body_longitudinal_offset_mm)

        if self.finger_length_mm <= 0.0:
            raise ValueError("finger_length_mm must be > 0")
        if self.finger_thickness_mm <= 0.0:
            raise ValueError("finger_thickness_mm must be > 0")
        if self.opening_uncertainty_mm < 0.0:
            raise ValueError("opening_uncertainty_mm must be >= 0")
        if self.object_margin_mm < 0.0 or self.wall_margin_mm < 0.0:
            raise ValueError("collision margins must be >= 0")
        if self.fallback_opening_mm <= 0.0:
            raise ValueError("fallback_opening_mm must be > 0")
        if self.image_border_margin_px < 0:
            raise ValueError("image_border_margin_px must be >= 0")
        if self.body_enabled and (self.body_width_mm <= 0.0 or self.body_length_mm <= 0.0):
            raise ValueError("body dimensions must be > 0 when body_enabled")

        self.forbidden_zones_mm = self._parse_forbidden_zones(forbidden_zones_mm or [])
        self.allowed_region_mm = self._parse_allowed_region(allowed_region_mm)
        self.allowed_polygon_mm = self._parse_allowed_polygon(allowed_polygon_mm)
        self.object_overlap_tolerance_px = max(0, int(object_overlap_tolerance_px))

    @staticmethod
    def _parse_allowed_region(value) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        vals = list(value)
        if len(vals) != 4:
            raise ValueError("allowed_region_mm must be [xmin, ymin, xmax, ymax]")
        xmin, ymin, xmax, ymax = map(float, vals)
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        return xmin, ymin, xmax, ymax

    @staticmethod
    def _parse_allowed_polygon(value) -> np.ndarray | None:
        """解析托盘可进入区域四边形/多边形，坐标单位 mm。"""
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64).reshape(-1, 2)
        if len(arr) < 3:
            raise ValueError("allowed_polygon_mm must have at least 3 points")
        if not np.all(np.isfinite(arr)):
            raise ValueError("allowed_polygon_mm contains invalid values")
        return arr.copy()

    @staticmethod
    def _parse_forbidden_zones(values) -> list[tuple[str, np.ndarray]]:
        """
        解析固定挡板/禁入区。

        推荐格式（四点多边形，适合透视视角和斜放挡板）：
            {
              "name": "wall_1",
              "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            }

        兼容旧格式：
            {"name": "wall_1", "rect": [xmin, ymin, xmax, ymax]}

        所有点均使用机械/机器人平面坐标 mm。
        """
        parsed: list[tuple[str, np.ndarray]] = []
        for i, item in enumerate(values):
            name = f"zone_{i + 1}"

            if isinstance(item, dict):
                name = str(item.get("name", name))
                poly = item.get("polygon", item.get("points", item.get("vertices")))
                rect = item.get("rect", item.get("rectangle", item.get("xyxy")))
            else:
                poly = None
                rect = item

            if poly is not None:
                arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
                if len(arr) < 3:
                    raise ValueError(
                        f"forbidden zone {name} polygon must have at least 3 points"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"forbidden zone {name} polygon contains invalid values")
                parsed.append((name, arr.copy()))
                continue

            if rect is None:
                raise ValueError(
                    f"forbidden zone {name} missing polygon/rect"
                )

            vals = list(rect)
            if len(vals) != 4:
                raise ValueError(f"forbidden zone {name} rect must have 4 values")
            xmin, ymin, xmax, ymax = map(float, vals)
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            arr = np.asarray(
                [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ],
                dtype=np.float64,
            )
            parsed.append((name, arr))

        return parsed

    @staticmethod
    def _rect_local(
        x1: float,
        x2: float,
        y1: float,
        y2: float,
    ) -> np.ndarray:
        return np.asarray(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotate_translate(
        local_points: np.ndarray,
        center_robot_mm: tuple[float, float],
        yaw_deg: float,
    ) -> np.ndarray:
        theta = math.radians(float(yaw_deg))
        c = math.cos(theta)
        s = math.sin(theta)
        R = np.asarray([[c, -s], [s, c]], dtype=np.float64)
        center = np.asarray(center_robot_mm, dtype=np.float64).reshape(1, 2)
        return np.asarray(local_points, dtype=np.float64) @ R.T + center

    def _robot_to_pixels(self, points_robot_mm: np.ndarray) -> np.ndarray:
        # 包络与禁区采用不含 TCP 偏置的机械坐标，返回当前识别 ROI 的局部像素。
        pts = np.asarray(points_robot_mm, dtype=np.float64).reshape(-1, 2)
        return self.transform.robot_to_pixels(pts, apply_tcp_offset=False)

    def _make_footprint_robot(
        self,
        *,
        center_robot_mm: tuple[float, float],
        yaw_deg: float,
        opening_mm: float,
        extra_margin_mm: float = 0.0,
        include_opening_uncertainty: bool = False,
    ) -> list[np.ndarray]:
        """
        生成两个指爪（以及可选主体）在机械平面上的矩形包络。

        extra_margin_mm 会在四周扩大指爪碰撞包络。
        opening_uncertainty_mm 会额外向“内/外”扩大 X 方向，覆盖动态预夹
        标定偏差、角度补偿以及硅胶变形带来的开口误差。
        """
        opening = max(0.0, float(opening_mm))
        margin = max(0.0, float(extra_margin_mm))
        opening_unc = self.opening_uncertainty_mm if include_opening_uncertainty else 0.0

        half_open = opening * 0.5
        half_len = self.finger_length_mm * 0.5
        y_center = self.finger_longitudinal_offset_mm

        # 误差包络：每个指爪沿闭合轴两侧各扩 opening_unc/2。
        ux = opening_unc * 0.5

        left = self._rect_local(
            -half_open - self.finger_thickness_mm - margin - ux,
            -half_open + margin + ux,
            y_center - half_len - margin,
            y_center + half_len + margin,
        )
        right = self._rect_local(
            half_open - margin - ux,
            half_open + self.finger_thickness_mm + margin + ux,
            y_center - half_len - margin,
            y_center + half_len + margin,
        )

        result = [
            self._rotate_translate(left, center_robot_mm, yaw_deg),
            self._rotate_translate(right, center_robot_mm, yaw_deg),
        ]

        if self.body_enabled:
            body = self._rect_local(
                -self.body_width_mm * 0.5 - margin,
                self.body_width_mm * 0.5 + margin,
                self.body_longitudinal_offset_mm - self.body_length_mm * 0.5 - margin,
                self.body_longitudinal_offset_mm + self.body_length_mm * 0.5 + margin,
            )
            result.append(self._rotate_translate(body, center_robot_mm, yaw_deg))

        return result

    def _robot_polygons_to_pixels(self, polygons_robot: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for poly in polygons_robot:
            px = self._robot_to_pixels(poly)
            if not np.all(np.isfinite(px)):
                continue
            out.append(px.astype(np.float32))
        return out

    @staticmethod
    def _polygons_to_mask(polygons_px: list[np.ndarray], image_shape) -> np.ndarray:
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = []
        for poly in polygons_px:
            p = np.asarray(poly, dtype=np.float32)
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
                continue
            pts.append(np.round(p).astype(np.int32).reshape(-1, 1, 2))
        # 一次传入多个轮廓使用 even-odd 填充，重叠区可能被当成孔洞。
        # 指爪与主体必须取实体并集，不能相互抵消。
        for polygon in pts:
            cv2.fillPoly(mask, [polygon], 255)
        return mask

    @staticmethod
    def _other_objects_mask(other_polygons: list[np.ndarray], image_shape) -> np.ndarray:
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = []
        for poly in other_polygons:
            p = np.asarray(poly, dtype=np.float32)
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
                continue
            pts.append(np.round(p).astype(np.int32).reshape(-1, 1, 2))
        # 不同物体的分割轮廓可能局部交叠，障碍应取并集而非奇偶填充。
        for polygon in pts:
            cv2.fillPoly(mask, [polygon], 255)
        return mask

    def _forbidden_zone_mask(self, image_shape, expand_mm: float = 0.0) -> np.ndarray:
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        polys_px: list[np.ndarray] = []
        e = max(0.0, float(expand_mm))

        for _, robot_poly in self.forbidden_zones_mm:
            robot = np.asarray(robot_poly, dtype=np.float64).reshape(-1, 2)

            # 当前碰撞策略主要通过 wall_safe_robot 对夹爪自身做 wall_margin 扩张，
            # 因此生产路径通常传 expand_mm=0。这里仍保留 expand_mm 兼容能力：
            # 使用 OpenCV 对多边形做近似外扩，而不是退回轴对齐矩形。
            if e > 0.0 and len(robot) >= 3:
                # 先转到临时高分辨率平面做 morphology，再取最大外轮廓。
                # 这条分支不在默认实时路径中，仅用于未来扩展。
                min_xy = robot.min(axis=0) - e - 2.0
                max_xy = robot.max(axis=0) + e + 2.0
                scale = 10.0
                size = np.maximum(8, np.ceil((max_xy - min_xy) * scale).astype(int) + 4)
                tmp = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
                pts = np.round((robot - min_xy) * scale).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(tmp, [pts], 255)
                k = max(1, int(round(e * scale)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                tmp = cv2.dilate(tmp, kernel)
                contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
                    robot = c / scale + min_xy

            px = self._robot_to_pixels(robot)
            if np.all(np.isfinite(px)):
                polys_px.append(px.astype(np.float32))

        if polys_px:
            mask = self._polygons_to_mask(polys_px, image_shape)
        return mask

    def _inside_allowed_polygon(self, polygons_robot: list[np.ndarray], margin_mm: float) -> bool:
        """
        检查夹爪是否完整位于托盘允许四边形内。

        与“把托盘四边形填成 forbidden zone”不同，这里把四边形内部视为可用区，
        外部才是墙/边界。pointPolygonTest 的有符号距离直接以机器人 mm 计算，
        因此 wall_margin_mm 可以作为真实机械安全距离。
        """
        if self.allowed_polygon_mm is None:
            return True
        contour = np.asarray(self.allowed_polygon_mm, dtype=np.float32).reshape(-1, 1, 2)
        required = max(0.0, float(margin_mm))
        for poly in polygons_robot:
            p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            for x, y in p:
                d = float(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
                if d < required:
                    return False
        return True

    def allowed_polygon_pixels(self) -> np.ndarray | None:
        if self.allowed_polygon_mm is None:
            return None
        px = self._robot_to_pixels(self.allowed_polygon_mm)
        return px.astype(np.float32) if np.all(np.isfinite(px)) else None

    def forbidden_zone_polygons_pixels(self) -> list[tuple[str, np.ndarray]]:
        out: list[tuple[str, np.ndarray]] = []
        for name, robot_poly in self.forbidden_zones_mm:
            px = self._robot_to_pixels(np.asarray(robot_poly, dtype=np.float64))
            if np.all(np.isfinite(px)):
                out.append((name, px.astype(np.float32)))
        return out

    def _inside_allowed_region(self, polygons_robot: list[np.ndarray], extra_margin_mm: float) -> bool:
        if self.allowed_region_mm is None:
            return True
        xmin, ymin, xmax, ymax = self.allowed_region_mm
        m = max(0.0, float(extra_margin_mm))
        xmin += m
        ymin += m
        xmax -= m
        ymax -= m
        if xmin > xmax or ymin > ymax:
            return False
        for poly in polygons_robot:
            p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            if np.any(p[:, 0] < xmin) or np.any(p[:, 0] > xmax):
                return False
            if np.any(p[:, 1] < ymin) or np.any(p[:, 1] > ymax):
                return False
        return True

    def _inside_image(self, polygons_px: list[np.ndarray], image_shape) -> bool:
        if not self.enforce_image_border:
            return True
        h, w = image_shape[:2]
        m = float(self.image_border_margin_px)
        for poly in polygons_px:
            p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            if np.any(p[:, 0] < m) or np.any(p[:, 0] > (w - 1 - m)):
                return False
            if np.any(p[:, 1] < m) or np.any(p[:, 1] > (h - 1 - m)):
                return False
        return True

    def _mm_per_px_at(self, pixel: tuple[float, float]) -> float:
        """保守估计当前位置 1px 对应多少 mm。"""
        u, v = map(float, pixel)
        try:
            p0 = np.asarray(self.transform.pixel_to_robot(u, v, apply_tcp_offset=False))
            px = np.asarray(self.transform.pixel_to_robot(u + 1.0, v, apply_tcp_offset=False))
            py = np.asarray(self.transform.pixel_to_robot(u, v + 1.0, apply_tcp_offset=False))
            sx = float(np.linalg.norm(px - p0))
            sy = float(np.linalg.norm(py - p0))
            vals = [x for x in (sx, sy) if math.isfinite(x) and x > 1e-9]
            if vals:
                return min(vals)
        except Exception:
            pass
        return 1.0

    @staticmethod
    def _clearance_px(footprint_mask: np.ndarray, obstacle_mask: np.ndarray) -> float:
        if not np.any(obstacle_mask):
            h, w = footprint_mask.shape
            return float(np.hypot(w, h))
        free = cv2.bitwise_not(obstacle_mask)
        distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        ys, xs = np.nonzero(footprint_mask)
        if len(xs) == 0:
            return 0.0
        return float(np.min(distance[ys, xs]))

    def prepare_context(
        self,
        *,
        other_polygons: list[np.ndarray],
        image_shape,
    ) -> GripperCollisionContext:
        """预计算同一目标所有候选姿态共用的障碍mask和距离场。"""
        other_mask = self._other_objects_mask(other_polygons, image_shape)
        wall_mask = self._forbidden_zone_mask(image_shape, expand_mm=0.0)
        combined = cv2.bitwise_or(other_mask, wall_mask)
        if np.any(combined):
            free = cv2.bitwise_not(combined)
            distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        else:
            distance = None
        return GripperCollisionContext(
            image_shape=tuple(image_shape),
            other_mask=other_mask,
            wall_mask=wall_mask,
            distance_to_obstacle=distance,
        )

    @staticmethod
    def _clearance_from_context(
        footprint_mask: np.ndarray,
        context: GripperCollisionContext,
    ) -> float:
        if context.distance_to_obstacle is None:
            h, w = footprint_mask.shape
            return float(np.hypot(w, h))
        ys, xs = np.nonzero(footprint_mask)
        if len(xs) == 0:
            return 0.0
        return float(np.min(context.distance_to_obstacle[ys, xs]))

    def check_nominal(
        self,
        *,
        grasp_pixel: tuple[float, float],
        yaw_deg: float,
        opening_mm: float | None,
        target_polygon_robot_mm: np.ndarray,
        other_polygons: list[np.ndarray],
        image_shape,
        context: GripperCollisionContext | None = None,
    ) -> GripperFootprintCheck:
        """检查实体预开口及闭合扫掠，不受 enabled 或任何膨胀余量影响。

        yaw_deg 为实际执行的机器人闭合轴角。目标以不含 TCP 偏置的机器人
        坐标传入；各指从预开口位置移动到目标在闭合轴上的支撑面即停止。
        返回的轮廓和 clearance 均对应整个扫掠，而非仅闭合前的静态位置。
        context 必须由相同的邻物、固定禁区和图像尺寸生成，可供多轴复用。
        缺失开口、无效几何或不完整投影均拒绝，避免把漏画的包络当作安全。
        """
        polygons_px: list[np.ndarray] = []

        def failed(reason: str, **kwargs) -> GripperFootprintCheck:
            return GripperFootprintCheck(
                safe=False,
                reason=reason,
                footprint_polygons_px=polygons_px,
                clearance_px=0.0,
                clearance_mm=0.0,
                **kwargs,
            )

        def valid_polygon(value) -> np.ndarray:
            polygon = np.asarray(value, dtype=np.float64)
            if (
                polygon.ndim != 2
                or polygon.shape[1] != 2
                or len(polygon) < 3
                or not np.all(np.isfinite(polygon))
                or np.any(np.abs(polygon) > np.finfo(np.float32).max)
            ):
                raise ValueError("invalid polygon")
            area = abs(cv2.contourArea(polygon.astype(np.float32)))
            if not math.isfinite(area) or area <= 1e-9:
                raise ValueError("degenerate polygon")
            return polygon

        def project(polygon: np.ndarray) -> np.ndarray:
            pixels = np.asarray(self._robot_to_pixels(polygon), dtype=np.float64)
            if pixels.shape != polygon.shape:
                raise ValueError("incomplete projection")
            return valid_polygon(pixels).astype(np.float32)

        try:
            h, w = image_shape[:2]
            if int(h) != h or int(w) != w or h < 1 or w < 1:
                raise ValueError("invalid image shape")
            target = valid_polygon(target_polygon_robot_mm)
            # prepare_context 的旧接口会略过坏轮廓；实体检查必须先显式验证。
            for polygon in other_polygons:
                valid_polygon(polygon)
            if self.allowed_polygon_mm is not None:
                valid_polygon(self.allowed_polygon_mm)
            if self.allowed_region_mm is not None and not np.all(np.isfinite(self.allowed_region_mm)):
                raise ValueError("invalid allowed region")
            opening = float(opening_mm)
            yaw = float(yaw_deg)
            geometry = [
                self.finger_length_mm,
                self.finger_thickness_mm,
                self.finger_longitudinal_offset_mm,
            ]
            if self.body_enabled:
                geometry += [
                    self.body_width_mm,
                    self.body_length_mm,
                    self.body_longitudinal_offset_mm,
                ]
            if (
                not math.isfinite(opening)
                or opening <= 0.0
                or not math.isfinite(yaw)
                or not np.all(np.isfinite(geometry))
                or self.finger_length_mm <= 0.0
                or self.finger_thickness_mm <= 0.0
                or (self.body_enabled and min(self.body_width_mm, self.body_length_mm) <= 0.0)
            ):
                raise ValueError("invalid gripper geometry")
        except (TypeError, ValueError, IndexError, OverflowError, cv2.error):
            return failed("invalid_nominal_geometry")

        try:
            center = np.asarray(
                self.transform.pixel_to_robot(*map(float, grasp_pixel), apply_tcp_offset=False),
                dtype=np.float64,
            )
            if center.shape != (2,) or not np.all(np.isfinite(center)):
                raise ValueError("invalid grasp transform")
            samples = [
                np.asarray(self.transform.pixel_to_robot(
                    float(grasp_pixel[0]) + du,
                    float(grasp_pixel[1]) + dv,
                    apply_tcp_offset=False,
                ), dtype=np.float64)
                for du, dv in ((1.0, 0.0), (0.0, 1.0))
            ]
            if any(p.shape != (2,) or not np.all(np.isfinite(p)) for p in samples):
                raise ValueError("invalid local scale")
            scales = [float(np.linalg.norm(p - center)) for p in samples]
            if min(scales) <= 1e-9:
                raise ValueError("degenerate local scale")
            mm_per_px = min(scales)
            theta = math.radians(yaw)
            closing_axis = np.asarray([math.cos(theta), math.sin(theta)])
            supports = (target - center) @ closing_axis
            support_min, support_max = float(supports.min()), float(supports.max())
            half_open = opening * 0.5
            if support_min < -half_open - 1e-6 or support_max > half_open + 1e-6:
                return failed("target_outside_preopen")

            half_length = self.finger_length_mm * 0.5
            y_min = self.finger_longitudinal_offset_mm - half_length
            y_max = self.finger_longitudinal_offset_mm + half_length
            # 左指内侧从 -half_open 到 support_min，右指从 +half_open 到
            # support_max。矩形并集精确覆盖平移过程，同时包含预开口位置。
            swept_local = [
                self._rect_local(
                    -half_open - self.finger_thickness_mm,
                    max(-half_open, support_min), y_min, y_max,
                ),
                self._rect_local(
                    min(half_open, support_max),
                    half_open + self.finger_thickness_mm, y_min, y_max,
                ),
            ]
            swept_robot = [self._rotate_translate(p, center, yaw) for p in swept_local]
            if self.body_enabled:
                swept_robot.append(self._make_footprint_robot(
                    center_robot_mm=center, yaw_deg=yaw, opening_mm=opening,
                    extra_margin_mm=0.0, include_opening_uncertainty=False,
                )[-1])
            polygons_px = [project(polygon) for polygon in swept_robot]
            # 固定禁区的投影也不能被旧 prepare_context 静默漏掉。
            for _, wall_polygon in self.forbidden_zones_mm:
                project(valid_polygon(wall_polygon))
        except Exception:
            return failed("nominal_projection_failed")

        allowed_polygon_clearance_mm = math.inf
        if self.allowed_polygon_mm is not None:
            contour = np.asarray(self.allowed_polygon_mm, dtype=np.float32).reshape(-1, 1, 2)

            def cross2(a, b) -> float:
                return float(a[0] * b[1] - a[1] * b[0])

            boundary = np.asarray(self.allowed_polygon_mm, dtype=np.float64)
            for polygon in swept_robot:
                # 仅检查顶点不足以覆盖凹允许区：矩形的一条边可能横穿凹口。
                distances = [float(cv2.pointPolygonTest(contour, tuple(map(float, p)), True)) for p in polygon]
                if min(distances) < 0.0:
                    return failed("allowed_polygon_collision", allowed_region_violation=True)
                allowed_polygon_clearance_mm = min(allowed_polygon_clearance_mm, *distances)
                polygon_contour = polygon.astype(np.float32).reshape(-1, 1, 2)
                allowed_polygon_clearance_mm = min(allowed_polygon_clearance_mm, min(
                    abs(float(cv2.pointPolygonTest(polygon_contour, tuple(map(float, p)), True)))
                    for p in boundary
                ))
                for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
                    direction = end - start
                    cuts = [0.0, 1.0]
                    for a, b in zip(boundary, np.roll(boundary, -1, axis=0)):
                        edge = b - a
                        denominator = cross2(direction, edge)
                        if abs(denominator) > 1e-10:
                            t = cross2(a - start, edge) / denominator
                            u = cross2(a - start, direction) / denominator
                            if 0.0 < t < 1.0 and -1e-9 <= u <= 1.0 + 1e-9:
                                cuts.append(t)
                        elif abs(cross2(a - start, direction)) <= 1e-10:
                            for point in (a, b):
                                t = float(np.dot(point - start, direction) / np.dot(direction, direction))
                                if 0.0 < t < 1.0:
                                    cuts.append(t)
                    cuts.sort()
                    for lo, hi in zip(cuts, cuts[1:]):
                        midpoint = start + (lo + hi) * 0.5 * direction
                        if cv2.pointPolygonTest(contour, tuple(map(float, midpoint)), False) < 0.0:
                            return failed("allowed_polygon_collision", allowed_region_violation=True)
        if not self._inside_allowed_region(swept_robot, 0.0):
            return failed("allowed_region_collision", allowed_region_violation=True)
        image_clearance_px = math.inf
        if self.enforce_image_border:
            # image_border_margin_px 也是额外余量，名义检查只使用真正的图像边界。
            image_clearance_px = min(
                float(np.min(np.concatenate((p[:, 0], w - 1 - p[:, 0], p[:, 1], h - 1 - p[:, 1]))))
                for p in polygons_px
            )
            if image_clearance_px < 0.0:
                return failed("image_border_collision", border_violation=True)

        try:
            if context is None or tuple(context.image_shape[:2]) != (h, w):
                context = self.prepare_context(other_polygons=other_polygons, image_shape=image_shape)
            for mask in (context.other_mask, context.wall_mask):
                if not isinstance(mask, np.ndarray) or mask.shape != (h, w) or mask.dtype != np.uint8:
                    raise ValueError("invalid obstacle mask")
            distance = context.distance_to_obstacle
            if distance is None:
                if np.any(context.other_mask) or np.any(context.wall_mask):
                    raise ValueError("missing obstacle distance")
            elif (
                not isinstance(distance, np.ndarray)
                or distance.shape != (h, w)
                or not np.issubdtype(distance.dtype, np.floating)
            ):
                raise ValueError("invalid obstacle distance")

            # 候选通常只占整图的一小块。先在全图坐标中取整，再平移整数轮廓，
            # 避免 round(x - roi_origin) 的 .5 舍入与整图版本产生一像素差异。
            rounded = [np.round(polygon) for polygon in polygons_px]
            limit = np.iinfo(np.int32).max
            if any(np.any(np.abs(polygon.astype(np.float64)) > limit) for polygon in rounded):
                return failed("nominal_projection_failed")
            contours = [polygon.astype(np.int32) for polygon in rounded]
            points = np.concatenate(contours, axis=0)
            x0, y0 = np.maximum(points.min(axis=0), 0).tolist()
            x1, y1 = np.minimum(points.max(axis=0).astype(np.int64) + 1, [w, h]).tolist()
            if x0 >= x1 or y0 >= y1:
                return failed("nominal_projection_failed")
            nominal_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            for polygon in contours:
                cv2.fillPoly(nominal_mask, [polygon], 255, offset=(-x0, -y0))
            if not np.any(nominal_mask):
                return failed("nominal_projection_failed")
            other_roi = context.other_mask[y0:y1, x0:x1]
            wall_roi = context.wall_mask[y0:y1, x0:x1]
            object_pixels = int(np.count_nonzero(cv2.bitwise_and(nominal_mask, other_roi)))
            if object_pixels > self.object_overlap_tolerance_px:
                return failed("other_object_collision", object_collision_pixels=object_pixels)
            wall_pixels = int(np.count_nonzero(cv2.bitwise_and(nominal_mask, wall_roi)))
            if wall_pixels:
                return failed("fixed_wall_collision", wall_collision_pixels=wall_pixels)
            if distance is None:
                # 无障碍时沿用原图对角线，不能随本次候选的 ROI 大小改变分数。
                obstacle_clearance_px = float(np.hypot(w, h))
            else:
                sampled_distance = distance[y0:y1, x0:x1][nominal_mask != 0]
                if not np.all(np.isfinite(sampled_distance)) or np.any(sampled_distance < 0.0):
                    raise ValueError("invalid queried obstacle distance")
                obstacle_clearance_px = float(np.min(sampled_distance))
            clearance_px = min(obstacle_clearance_px, image_clearance_px)
        except (AttributeError, TypeError, ValueError, IndexError, cv2.error):
            return failed("invalid_collision_context")

        clearance_mm = min(float(clearance_px * mm_per_px), allowed_polygon_clearance_mm)
        # 允许区也是真实边界，不能在没有邻物时误报一个无限大的自由间隙。
        for polygon in swept_robot:
            if self.allowed_region_mm is not None:
                xmin, ymin, xmax, ymax = self.allowed_region_mm
                clearance_mm = min(clearance_mm, float(np.min(np.concatenate((
                    polygon[:, 0] - xmin, xmax - polygon[:, 0],
                    polygon[:, 1] - ymin, ymax - polygon[:, 1],
                )))))
        return GripperFootprintCheck(
            safe=True,
            reason="ok",
            footprint_polygons_px=polygons_px,
            clearance_px=max(0.0, clearance_mm / mm_per_px),
            clearance_mm=max(0.0, clearance_mm),
            object_collision_pixels=object_pixels,
            wall_collision_pixels=wall_pixels,
        )

    def check(
        self,
        *,
        grasp_pixel: tuple[float, float],
        yaw_deg: float,
        opening_mm: float | None,
        other_polygons: list[np.ndarray],
        image_shape,
        context: GripperCollisionContext | None = None,
    ) -> GripperFootprintCheck:
        if not self.enabled:
            return GripperFootprintCheck(
                safe=True,
                reason="disabled",
                footprint_polygons_px=[],
                clearance_px=float(np.hypot(image_shape[1], image_shape[0])),
                clearance_mm=999.0,
            )

        if opening_mm is None or not math.isfinite(float(opening_mm)):
            opening = self.fallback_opening_mm
        else:
            opening = max(0.0, float(opening_mm))

        try:
            center_robot = self.transform.pixel_to_robot(
                float(grasp_pixel[0]),
                float(grasp_pixel[1]),
                apply_tcp_offset=False,
            )
        except Exception:
            return GripperFootprintCheck(
                safe=False,
                reason="grasp_pixel_transform_failed",
                footprint_polygons_px=[],
                clearance_px=0.0,
                clearance_mm=0.0,
            )

        # 名义几何：用于显示和真实 clearance 计算。
        nominal_robot = self._make_footprint_robot(
            center_robot_mm=center_robot,
            yaw_deg=yaw_deg,
            opening_mm=opening,
            extra_margin_mm=0.0,
            include_opening_uncertainty=False,
        )
        nominal_px = self._robot_polygons_to_pixels(nominal_robot)
        nominal_mask = self._polygons_to_mask(nominal_px, image_shape)

        # 对其他货物的安全包络：真实厚度 + 物体安全余量 + 开口标定误差。
        object_safe_robot = self._make_footprint_robot(
            center_robot_mm=center_robot,
            yaw_deg=yaw_deg,
            opening_mm=opening,
            extra_margin_mm=self.object_margin_mm,
            include_opening_uncertainty=True,
        )
        object_safe_px = self._robot_polygons_to_pixels(object_safe_robot)
        object_safe_mask = self._polygons_to_mask(object_safe_px, image_shape)

        # 对固定挡板/图像边界仍使用 wall margin 扩张包络。
        wall_safe_robot = self._make_footprint_robot(
            center_robot_mm=center_robot,
            yaw_deg=yaw_deg,
            opening_mm=opening,
            extra_margin_mm=self.wall_margin_mm,
            include_opening_uncertainty=True,
        )
        wall_safe_px = self._robot_polygons_to_pixels(wall_safe_robot)
        wall_safe_mask = self._polygons_to_mask(wall_safe_px, image_shape)

        # 托盘四点标定是“允许区域”，不能再当 forbidden zone 填满。
        # 这里用仅包含开口误差的实际指爪几何，再要求距离四边形边界 >= wall_margin_mm。
        allowed_check_robot = self._make_footprint_robot(
            center_robot_mm=center_robot,
            yaw_deg=yaw_deg,
            opening_mm=opening,
            extra_margin_mm=0.0,
            include_opening_uncertainty=True,
        )

        if not self._inside_allowed_polygon(allowed_check_robot, self.wall_margin_mm):
            return GripperFootprintCheck(
                safe=False,
                reason="allowed_polygon_collision",
                footprint_polygons_px=nominal_px,
                clearance_px=0.0,
                clearance_mm=0.0,
                allowed_region_violation=True,
            )

        if not self._inside_image(wall_safe_px, image_shape):
            return GripperFootprintCheck(
                safe=False,
                reason="image_border_collision",
                footprint_polygons_px=nominal_px,
                clearance_px=0.0,
                clearance_mm=0.0,
                border_violation=True,
            )

        if not self._inside_allowed_region(allowed_check_robot, self.wall_margin_mm):
            return GripperFootprintCheck(
                safe=False,
                reason="allowed_region_collision",
                footprint_polygons_px=nominal_px,
                clearance_px=0.0,
                clearance_mm=0.0,
                allowed_region_violation=True,
            )

        if context is None or tuple(context.image_shape[:2]) != tuple(image_shape[:2]):
            context = self.prepare_context(
                other_polygons=other_polygons,
                image_shape=image_shape,
            )

        other_mask = context.other_mask
        object_overlap = cv2.bitwise_and(object_safe_mask, other_mask)
        object_collision_pixels = int(np.count_nonzero(object_overlap))
        if object_collision_pixels > self.object_overlap_tolerance_px:
            return GripperFootprintCheck(
                safe=False,
                reason="other_object_collision",
                footprint_polygons_px=nominal_px,
                clearance_px=0.0,
                clearance_mm=0.0,
                object_collision_pixels=object_collision_pixels,
            )

        wall_mask = context.wall_mask
        wall_overlap = cv2.bitwise_and(wall_safe_mask, wall_mask)
        wall_collision_pixels = int(np.count_nonzero(wall_overlap))
        if wall_collision_pixels > 0:
            return GripperFootprintCheck(
                safe=False,
                reason="fixed_wall_collision",
                footprint_polygons_px=nominal_px,
                clearance_px=0.0,
                clearance_mm=0.0,
                wall_collision_pixels=wall_collision_pixels,
            )

        clearance_px = self._clearance_from_context(nominal_mask, context)
        mm_per_px = self._mm_per_px_at(grasp_pixel)
        clearance_mm = float(clearance_px * mm_per_px)

        return GripperFootprintCheck(
            safe=True,
            reason="ok",
            footprint_polygons_px=nominal_px,
            clearance_px=float(clearance_px),
            clearance_mm=float(clearance_mm),
            object_collision_pixels=0,
            wall_collision_pixels=0,
        )
