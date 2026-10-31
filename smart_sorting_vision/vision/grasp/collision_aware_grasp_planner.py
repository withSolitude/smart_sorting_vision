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
    safe_distance_map: np.ndarray | None = None


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

        # 基础入口保留深安全平台的中心；多点和方向联合选择由上层完成。
        moments = cv2.moments(np.asarray(target_polygon, dtype=np.float32))
        if abs(moments["m00"]) > 1e-9:
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
        else:
            center_x, center_y = np.mean(np.asarray(target_polygon), axis=0)
        ys, xs = np.nonzero(
            (safe_mask > 0) & (safe_distance >= max(self.min_clearance, maximum_distance * 0.90))
        )
        if len(xs):
            best_i = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
            grasp_x, grasp_y = int(xs[best_i]), int(ys[best_i])
        else:
            _, _, _, point = cv2.minMaxLoc(safe_distance)
            grasp_x, grasp_y = point

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
            safe_distance_map=safe_distance,
        )
