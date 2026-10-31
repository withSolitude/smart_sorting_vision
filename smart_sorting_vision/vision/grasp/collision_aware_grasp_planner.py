from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CollisionAwareGraspResult:
    valid: bool

    grasp_point: tuple[int, int] | None

    clearance: float

    self_clearance: float

    obstacle_clearance: float

    touches_edge: bool

    safe_area: int

    reason: str

    safe_mask: np.ndarray | None = None
    # 用于选取抓取点的首选区域；safe_mask 仍保留完整碰撞安全区域。
    preferred_mask: np.ndarray | None = None


class CollisionAwareGraspPlanner:
    """
    多物体防碰撞抓取规划器。

    输入：
        当前目标 polygon
        其他物体 polygons
        图像尺寸

    输出：
        当前目标内部一个尽量：
            1. 远离自身边缘
            2. 远离其他物体
            3. 远离ROI边缘
        的抓取点。

    当前阶段所有距离单位仍为 pixel。

    注意：
        这里使用的是“圆形安全区域”近似。

        等完成像素 -> 机械XY标定，并得到夹爪真实尺寸后，
        再升级成真正的夹爪 footprint 碰撞检查。
    """

    def __init__(
        self,
        obstacle_margin: int = 12,
        border_margin: int = 12,
        min_safe_area: int = 50,
        min_clearance: float = 3.0,
    ):
        if obstacle_margin < 0:
            raise ValueError(
                "obstacle_margin不能小于0"
            )

        if border_margin < 0:
            raise ValueError(
                "border_margin不能小于0"
            )

        if min_safe_area < 1:
            raise ValueError(
                "min_safe_area必须>=1"
            )

        if min_clearance < 0:
            raise ValueError(
                "min_clearance不能小于0"
            )

        self.obstacle_margin = int(
            obstacle_margin
        )

        self.border_margin = int(
            border_margin
        )

        self.min_safe_area = int(
            min_safe_area
        )

        self.min_clearance = float(
            min_clearance
        )

    # ========================================================
    # Polygon -> 二值Mask
    # ========================================================

    @staticmethod
    def polygon_to_mask(
        polygon: np.ndarray,
        image_shape,
    ) -> np.ndarray:

        height, width = (
            image_shape[:2]
        )

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or len(polygon) < 3
        ):
            return mask

        polygon_int = (
            np.round(
                polygon
            )
            .astype(
                np.int32
            )
            .reshape(
                -1,
                1,
                2,
            )
        )

        cv2.fillPoly(
            mask,
            [
                polygon_int
            ],
            255,
        )

        return mask

    # ========================================================
    # 所有障碍物合成一个Mask
    # ========================================================

    def build_obstacle_mask(
        self,
        other_polygons: list[np.ndarray],
        image_shape,
    ) -> np.ndarray:

        height, width = (
            image_shape[:2]
        )

        obstacle_mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        for polygon in other_polygons:

            mask = (
                self.polygon_to_mask(
                    polygon=polygon,
                    image_shape=image_shape,
                )
            )

            obstacle_mask = cv2.bitwise_or(
                obstacle_mask,
                mask,
            )

        return obstacle_mask

    # ========================================================
    # 障碍物膨胀
    #
    # 相当于给其他物体增加一圈安全距离。
    # ========================================================

    def inflate_obstacles(
        self,
        obstacle_mask: np.ndarray,
    ) -> np.ndarray:

        if self.obstacle_margin <= 0:

            return (
                obstacle_mask.copy()
            )

        kernel_size = (
            self.obstacle_margin
            * 2
            +
            1
        )

        kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    kernel_size,
                    kernel_size,
                ),
            )
        )

        inflated = cv2.dilate(
            obstacle_mask,
            kernel,
            iterations=1,
        )

        return inflated

    # ========================================================
    # ROI边界安全Mask
    #
    # border_margin范围内不允许作为抓取点。
    # ========================================================

    def build_border_safe_mask(
        self,
        image_shape,
    ) -> np.ndarray:

        height, width = (
            image_shape[:2]
        )

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        margin = (
            self.border_margin
        )

        if (
            width <= 2 * margin
            or
            height <= 2 * margin
        ):
            return mask

        mask[
            margin:
            height - margin,
            margin:
            width - margin
        ] = 255

        return mask

    # ========================================================
    # 当前目标是否贴ROI边缘
    # ========================================================

    def check_touches_edge(
        self,
        target_mask: np.ndarray,
    ) -> bool:

        height, width = (
            target_mask.shape
        )

        margin = (
            self.border_margin
        )

        if margin <= 0:
            return False

        top = target_mask[
            :margin,
            :
        ]

        bottom = target_mask[
            max(
                0,
                height - margin
            ):
            height,
            :
        ]

        left = target_mask[
            :,
            :margin
        ]

        right = target_mask[
            :,
            max(
                0,
                width - margin
            ):
            width
        ]

        return bool(
            np.any(top)
            or
            np.any(bottom)
            or
            np.any(left)
            or
            np.any(right)
        )

    # ========================================================
    # 到障碍物的距离
    #
    # 对没有障碍物的情况单独处理。
    # ========================================================

    @staticmethod
    def obstacle_distance_map(
        obstacle_mask: np.ndarray,
    ) -> np.ndarray:

        height, width = (
            obstacle_mask.shape
        )

        if not np.any(
            obstacle_mask
        ):
            # 没有其他物体：
            # 给一个足够大的统一值。
            maximum = float(
                np.hypot(
                    width,
                    height,
                )
            )

            return np.full(
                (
                    height,
                    width,
                ),
                maximum,
                dtype=np.float32,
            )

        free_space = cv2.bitwise_not(
            obstacle_mask
        )

        distance = cv2.distanceTransform(
            free_space,
            cv2.DIST_L2,
            5,
        )

        return distance

    # ========================================================
    # 正式规划一个目标
    # ========================================================

    def plan(
        self,
        target_polygon: np.ndarray,
        other_polygons: list[np.ndarray],
        image_shape,
        shape_name: str | None = None,
    ) -> CollisionAwareGraspResult:

        # ====================================================
        # 1. 当前目标Mask
        # ====================================================

        target_mask = (
            self.polygon_to_mask(
                polygon=target_polygon,
                image_shape=image_shape,
            )
        )

        if not np.any(
            target_mask
        ):
            return CollisionAwareGraspResult(
                valid=False,
                grasp_point=None,
                clearance=0.0,
                self_clearance=0.0,
                obstacle_clearance=0.0,
                touches_edge=False,
                safe_area=0,
                reason="invalid_target_mask",
                safe_mask=None,
            )

        # ====================================================
        # 2. 是否靠近SAFE ROI边缘
        # ====================================================

        touches_edge = (
            self.check_touches_edge(
                target_mask
            )
        )

        # ====================================================
        # 3. 其他物体Mask
        # ====================================================

        obstacle_mask = (
            self.build_obstacle_mask(
                other_polygons=
                    other_polygons,

                image_shape=
                    image_shape,
            )
        )

        # ====================================================
        # 4. 障碍物增加安全距离
        # ====================================================

        inflated_obstacles = (
            self.inflate_obstacles(
                obstacle_mask
            )
        )

        # ====================================================
        # 5. ROI内部允许区域
        # ====================================================

        border_safe_mask = (
            self.build_border_safe_mask(
                image_shape
            )
        )

        # ====================================================
        # 6. 当前目标内部
        #    并且避开其他物体膨胀区域
        # ====================================================

        safe_mask = (
            cv2.bitwise_and(
                target_mask,
                cv2.bitwise_not(
                    inflated_obstacles
                ),
            )
        )

        # ====================================================
        # 7. 再删除ROI边缘危险区域
        # ====================================================

        safe_mask = cv2.bitwise_and(
            safe_mask,
            border_safe_mask,
        )

        safe_area = int(
            np.count_nonzero(
                safe_mask
            )
        )

        if (
            safe_area
            <
            self.min_safe_area
        ):
            return CollisionAwareGraspResult(
                valid=False,
                grasp_point=None,
                clearance=0.0,
                self_clearance=0.0,
                obstacle_clearance=0.0,
                touches_edge=touches_edge,
                safe_area=safe_area,
                reason="safe_area_too_small",
                safe_mask=safe_mask,
            )

        # ====================================================
        # 8. Distance Transform
        #
        # 在safe_mask内部：
        # 值越大，代表距离危险边界越远。
        # ====================================================

        safe_distance = cv2.distanceTransform(
            safe_mask,
            cv2.DIST_L2,
            5,
        )

        (
            _,
            maximum_distance,
            _,
            _,
        ) = cv2.minMaxLoc(
            safe_distance
        )

        # 对长方形等目标，Distance Transform 的最大值经常不是唯一点，
        # 而是在中轴附近形成一段“等高平台”。直接使用 minMaxLoc 返回的
        # 第一个最大值会把抓取点随机偏到平台一端。
        #
        # 这里保留“安全优先”的原则：先取 clearance >= 90% 最大值的
        # 深安全区域，再在这些点中选择最接近目标几何质心的点。
        # 这样孤立规则物体会优先抓中心；若质心附近被障碍物/ROI边缘切掉，
        # 仍然会退到最近的深安全点，而不是强行抓不安全的质心。
        # 透视较大时，YOLO mask 的质心会被物体侧面向外的投影拉偏。
        # 用形状相关的“顶部核心区域”作为首选中心；完整 mask 仍用于
        # 接触宽度、碰撞和夹爪包络检查，避免把侧面误当成可抓区域。
        core_mask = self.build_top_core_mask(
            target_mask,
            shape_name=shape_name,
        )
        core_safe = cv2.bitwise_and(core_mask, safe_mask)

        preferred_mask = core_safe
        if int(np.count_nonzero(preferred_mask)) < max(12, self.min_safe_area // 5):
            # 遮挡严重或物体太小时，保留原安全区域作为退化方案，
            # 后续仍会经过夹爪完整包络检查。
            preferred_mask = safe_mask

        preferred_points = np.column_stack(np.nonzero(preferred_mask > 0))
        if len(preferred_points) > 0:
            center_y = float(np.mean(preferred_points[:, 0]))
            center_x = float(np.mean(preferred_points[:, 1]))
        else:
            center_x = center_y = 0.0

        moments = cv2.moments(
            np.asarray(target_polygon, dtype=np.float32)
        )
        if len(preferred_points) == 0 and abs(moments["m00"]) > 1e-9:
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
        elif len(preferred_points) == 0:
            pts = np.asarray(target_polygon, dtype=np.float32).reshape(-1, 2)
            center_x = float(np.mean(pts[:, 0]))
            center_y = float(np.mean(pts[:, 1]))

        plateau_threshold = max(
            float(self.min_clearance),
            float(maximum_distance) * 0.90,
        )
        ys, xs = np.nonzero(
            (safe_distance >= plateau_threshold)
            &
            (safe_mask > 0)
        )

        if len(xs) == 0:
            # 理论上只要 maximum_distance 有效就不会进入这里；保留兜底。
            _, _, _, maximum_location = cv2.minMaxLoc(safe_distance)
            grasp_x = int(maximum_location[0])
            grasp_y = int(maximum_location[1])
        else:
            d2 = (xs.astype(np.float64) - center_x) ** 2 + (ys.astype(np.float64) - center_y) ** 2
            best_i = int(np.argmin(d2))
            grasp_x = int(xs[best_i])
            grasp_y = int(ys[best_i])

        clearance = float(
            safe_distance[grasp_y, grasp_x]
        )

        # ====================================================
        # 9. 单独计算：
        #
        # 到自己边缘的距离
        # 到其他真实物体的距离
        # ====================================================

        self_distance_map = (
            cv2.distanceTransform(
                target_mask,
                cv2.DIST_L2,
                5,
            )
        )

        self_clearance = float(
            self_distance_map[
                grasp_y,
                grasp_x
            ]
        )

        obstacle_distance = (
            self.obstacle_distance_map(
                obstacle_mask
            )
        )

        obstacle_clearance = float(
            obstacle_distance[
                grasp_y,
                grasp_x
            ]
        )

        # ====================================================
        # 10. 最低安全深度
        # ====================================================

        if (
            clearance
            <
            self.min_clearance
        ):
            return CollisionAwareGraspResult(
                valid=False,
                grasp_point=(
                    grasp_x,
                    grasp_y,
                ),
                clearance=clearance,
                self_clearance=self_clearance,
                obstacle_clearance=(
                    obstacle_clearance
                ),
                touches_edge=touches_edge,
                safe_area=safe_area,
                reason="clearance_too_small",
                safe_mask=safe_mask,
                preferred_mask=preferred_mask,
            )

        return CollisionAwareGraspResult(
            valid=True,
            grasp_point=(
                grasp_x,
                grasp_y,
            ),
            clearance=clearance,
            self_clearance=self_clearance,
            obstacle_clearance=(
                obstacle_clearance
            ),
            touches_edge=touches_edge,
            safe_area=safe_area,
            reason="ok",
            safe_mask=safe_mask,
            preferred_mask=preferred_mask,
        )

    @staticmethod
    def build_top_core_mask(
        target_mask: np.ndarray,
        *,
        shape_name: str | None = None,
    ) -> np.ndarray:
        """从完整实例 mask 提取更接近顶部可抓区域的核心。

        侧面投影通常位于目标轮廓外圈，距离变换后会被优先剔除。
        不同形状使用不同保守比例：平面多边形收缩更强，圆柱/圆锥/球
        保留更大的核心以避免把旋转对称物体缩得过小。
        """
        mask = (np.asarray(target_mask) > 0).astype(np.uint8) * 255
        if not np.any(mask):
            return mask

        name = str(shape_name or "").strip().lower()
        # 顶部模型常常只有一个 object 类别；此时用 mask 的几何特征
        # 选择策略，避免依赖尚未训练好的形状标签。
        if not name or name in {"object", "unknown", "item"}:
            contour_list, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contour_list, key=cv2.contourArea) if contour_list else None
            if contour is not None:
                area = abs(float(cv2.contourArea(contour)))
                perimeter = float(cv2.arcLength(contour, True))
                circularity = (
                    4.0 * np.pi * area / (perimeter * perimeter)
                    if perimeter > 1.0e-6 else 0.0
                )
                hull = cv2.convexHull(contour).reshape(-1, 2)
                hull_perimeter = float(cv2.arcLength(hull.reshape(-1, 1, 2), True))
                approx = cv2.approxPolyDP(
                    hull.reshape(-1, 1, 2),
                    max(1.0, 0.025 * hull_perimeter),
                    True,
                )
                # 正方形的圆度约为0.785，不能把方块误判成圆形；
                # 只有明显接近圆/椭圆的轮廓才使用圆柱/球策略。
                if circularity >= 0.88:
                    name = "cylinder_or_sphere"
                elif len(approx) == 3:
                    # 三角轮廓更像圆锥/棱锥的顶部投影，避免套用长方体策略。
                    name = "pyramid_or_cone"
                elif len(approx) == 4:
                    name = "rectangular_prism"
                else:
                    name = "polygon_prism"

        if any(token in name for token in ("sphere", "ball", "球")):
            ratio = 0.38
        elif any(token in name for token in ("cylinder", "cone", "圆柱", "圆锥", "圆台")):
            ratio = 0.45
        elif any(token in name for token in ("pyramid", "棱锥", "prism", "棱柱")):
            ratio = 0.52
        else:
            # cube/rectangular prism/未知形状：保守地去掉外圈侧面投影。
            ratio = 0.58

        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        maximum = float(np.max(distance))
        if maximum <= 1.0:
            return mask

        threshold = max(1.5, maximum * ratio)
        core = ((distance >= threshold) & (mask > 0)).astype(np.uint8) * 255
        if not np.any(core):
            return mask

        # 只保留包含最大距离点的连通核心，避免遮挡/分割毛刺产生多个小核心。
        _, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
        if labels.max() <= 1:
            return core
        max_location = np.unravel_index(int(np.argmax(distance)), distance.shape)
        label = int(labels[max_location[0], max_location[1]])
        if label <= 0:
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return ((labels == label).astype(np.uint8) * 255)
