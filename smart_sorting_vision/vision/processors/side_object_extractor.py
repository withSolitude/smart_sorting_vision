from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class SideObjectResult:
    """
    SideObjectExtractor 输出。

    所有 bbox 坐标都位于 SideCamera.read() 返回的机械 ROI 坐标系。
    """

    found: bool
    reason: str

    mechanical_roi: np.ndarray

    crop: Optional[np.ndarray]
    object_mask: Optional[np.ndarray]
    crop_mask: Optional[np.ndarray]

    bbox: Optional[tuple[int, int, int, int]]
    crop_bbox: Optional[tuple[int, int, int, int]]

    area: int
    area_ratio: float

    # 调试信息
    difference_score: Optional[np.ndarray] = None
    core_mask: Optional[np.ndarray] = None


class SideObjectExtractor:
    """
    侧面识别工位统一物体提取模块 V5。

    处理链：

        SideCamera.read()
                ↓
        mechanical ROI
                ↓
        Signed-Lab 背景差分
                ↓
        候选前景
                ↓
        与强前景 core 重叠的连通域
                ↓
        object mask
                ↓
        bbox + padding
                ↓
        原始 BGR crop

    为什么 V3 不再使用上一版的“weak 太大就退回 core”：

        对白色方块时，core 通常已经覆盖完整主体；
        但对棕色、浅色、低对比度物体，core 可能只覆盖亮的顶面，
        直接退回 core 会导致只截到物体的一部分。

    V3 改为利用 Lab 的“方向信息”：

        1. 彩色物体：
           使用 a/b 色度差异检测；
           即使亮度与灰色托盘接近，也能找到主体。

        2. 白色/浅色物体：
           使用正向 L 差异：
               current_L - background_L > 0
           因此不会把“变暗的阴影”当成白色主体。

        3. 黑色/深色物体：
           使用较强的负向 L 差异；
           阈值比普通阴影更严格，降低阴影粘连。

    这比单纯 abs(L) 更适合当前固定侧面工位。
    """

    def __init__(
        self,

        # --------------------------------------------------------
        # 色度差阈值
        # OpenCV Lab 的 a/b 通道差。
        # 棕色、蓝色、紫色等主要靠这一条补全低对比度侧面。
        # --------------------------------------------------------
        chroma_threshold: int = 8,

        # --------------------------------------------------------
        # 亮度差阈值
        #
        # light:
        #   当前像素比空背景明显更亮，适合白/浅色物体。
        #
        # dark:
        #   当前像素比背景明显更暗，适合黑/深色物体。
        #   设置得比 light 更严格，是为了抑制普通阴影。
        # --------------------------------------------------------
        light_luma_threshold: int = 25,
        dark_luma_threshold: int = 50,

        # --------------------------------------------------------
        # 强前景 core
        # 只负责确定“真正物体种子”，不再直接拿它做最终 crop。
        # --------------------------------------------------------
        strong_chroma_threshold: int = 14,
        strong_light_luma_threshold: int = 45,
        strong_dark_luma_threshold: int = 75,

        # --------------------------------------------------------
        # 面积
        # --------------------------------------------------------
        min_object_area: int = 1200,
        min_object_area_ratio: float = 0.003,

        # --------------------------------------------------------
        # 空托盘误检抑制
        #
        # strong core 本身也必须达到最小面积，否则把它视作反光/噪声，
        # 不允许一个很小的强差分种子把大片弱前景“带起来”。
        #
        # 如果完全没有可靠 strong core，只能走 candidate fallback，
        # 则使用明显更严格的面积和几何门槛。这个分支主要保留给
        # 低对比度灰/浅色物体，而不是让空托盘的曝光波动直接通过。
        # --------------------------------------------------------
        min_core_area: int = 350,
        min_core_area_ratio: float = 0.0006,
        fallback_min_object_area: int = 4500,
        fallback_min_object_area_ratio: float = 0.010,
        fallback_min_bbox_width_ratio: float = 0.06,
        fallback_min_bbox_height_ratio: float = 0.10,

        # 如果前景占机械 ROI 太大，不静默接受，
        # 直接报告异常，避免又把整张机械 ROI 当物体。
        max_object_area_ratio: float = 0.80,

        # --------------------------------------------------------
        # Crop padding
        # --------------------------------------------------------
        padding_ratio: float = 0.15,
        min_padding_px: int = 12,

        # --------------------------------------------------------
        # morphology
        # --------------------------------------------------------
        open_kernel_size: int = 3,
        close_kernel_size: int = 5,
        close_iterations: int = 2,

        # --------------------------------------------------------
        # Gray / near-background texture assist
        #
        # 仅在检测到“低色度 + 中等亮度”的近灰色目标时启用。
        # 用当前帧与空背景的梯度差补全与灰色托盘颜色接近的区域。
        # --------------------------------------------------------
        enable_gray_texture_assist: bool = True,
        gray_texture_gradient_threshold: float = 15.0,
        gray_texture_luma_min: int = -12,
        gray_texture_luma_max: int = 25,
        gray_texture_support_x_ratio: float = 0.80,
        gray_texture_support_y_ratio: float = 0.15,
        gray_texture_close_kernel_size: int = 5,
        gray_texture_max_growth_ratio: float = 2.00,

        # --------------------------------------------------------
        # Achromatic safe crop
        #
        # 背景差分 Mask 只负责“定位种子”，最终 Crop 不要求 Mask
        # 像实例分割一样完整。
        #
        # 灰色/黑色/白色与灰色托盘对比度较低时，Mask 可能只覆盖
        # 物体的一部分。此时围绕检测中心强制保留一个最小安全窗口，
        # 保证 QR / Shape / Stain 等模块能看到完整物体。
        # --------------------------------------------------------
        enable_achromatic_safe_crop: bool = True,
        achromatic_saturation_threshold: int = 80,
        achromatic_chroma_delta_threshold: float = 10.0,
        achromatic_min_crop_width_ratio: float = 0.50,
        achromatic_min_crop_height_ratio: float = 0.93,
    ) -> None:

        self.chroma_threshold = int(
            chroma_threshold
        )

        self.light_luma_threshold = int(
            light_luma_threshold
        )

        self.dark_luma_threshold = int(
            dark_luma_threshold
        )

        self.strong_chroma_threshold = int(
            strong_chroma_threshold
        )

        self.strong_light_luma_threshold = int(
            strong_light_luma_threshold
        )

        self.strong_dark_luma_threshold = int(
            strong_dark_luma_threshold
        )

        self.min_object_area = int(
            min_object_area
        )

        self.min_object_area_ratio = float(
            min_object_area_ratio
        )

        self.min_core_area = int(
            min_core_area
        )

        self.min_core_area_ratio = float(
            min_core_area_ratio
        )

        self.fallback_min_object_area = int(
            fallback_min_object_area
        )

        self.fallback_min_object_area_ratio = float(
            fallback_min_object_area_ratio
        )

        self.fallback_min_bbox_width_ratio = float(
            fallback_min_bbox_width_ratio
        )

        self.fallback_min_bbox_height_ratio = float(
            fallback_min_bbox_height_ratio
        )

        self.max_object_area_ratio = float(
            max_object_area_ratio
        )

        self.padding_ratio = float(
            padding_ratio
        )

        self.min_padding_px = int(
            min_padding_px
        )

        self.open_kernel_size = self._odd(
            open_kernel_size
        )

        self.close_kernel_size = self._odd(
            close_kernel_size
        )

        self.close_iterations = int(
            close_iterations
        )

        self.enable_gray_texture_assist = bool(
            enable_gray_texture_assist
        )

        self.gray_texture_gradient_threshold = float(
            gray_texture_gradient_threshold
        )

        self.gray_texture_luma_min = int(
            gray_texture_luma_min
        )

        self.gray_texture_luma_max = int(
            gray_texture_luma_max
        )

        self.gray_texture_support_x_ratio = float(
            gray_texture_support_x_ratio
        )

        self.gray_texture_support_y_ratio = float(
            gray_texture_support_y_ratio
        )

        self.gray_texture_close_kernel_size = self._odd(
            gray_texture_close_kernel_size
        )

        self.gray_texture_max_growth_ratio = float(
            gray_texture_max_growth_ratio
        )

        self.enable_achromatic_safe_crop = bool(
            enable_achromatic_safe_crop
        )

        self.achromatic_saturation_threshold = int(
            achromatic_saturation_threshold
        )

        self.achromatic_chroma_delta_threshold = float(
            achromatic_chroma_delta_threshold
        )

        self.achromatic_min_crop_width_ratio = float(
            achromatic_min_crop_width_ratio
        )

        self.achromatic_min_crop_height_ratio = float(
            achromatic_min_crop_height_ratio
        )

        self.background_image: Optional[
            np.ndarray
        ] = None

    # ============================================================
    # Utilities
    # ============================================================

    @staticmethod
    def _odd(
        value: int,
    ) -> int:

        value = max(
            1,
            int(value),
        )

        if value % 2 == 0:
            value += 1

        return value

    # ============================================================
    # Background
    # ============================================================

    def set_background(
        self,
        image: np.ndarray,
    ) -> None:

        if (
            image is None
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError(
                "背景必须是有效的 BGR 三通道图像"
            )

        self.background_image = (
            image.copy()
        )

    def capture_background(
        self,
        camera: Any,
        frame_count: int = 30,
        settle_frames: int = 15,
    ) -> np.ndarray:
        """
        camera 只需要提供：
            camera.read() -> mechanical ROI
        """

        frame_count = max(
            1,
            int(frame_count),
        )

        settle_frames = max(
            0,
            int(settle_frames),
        )

        for _ in range(
            settle_frames
        ):
            camera.read()

        accumulator = None
        expected_shape = None

        for _ in range(
            frame_count
        ):

            frame = camera.read()

            if (
                frame is None
                or frame.ndim != 3
                or frame.shape[2] != 3
            ):
                raise RuntimeError(
                    "采集背景时读到非法图像"
                )

            if expected_shape is None:

                expected_shape = (
                    frame.shape
                )

            elif frame.shape != expected_shape:

                raise RuntimeError(
                    "采集背景过程中 ROI 尺寸发生变化"
                )

            if accumulator is None:

                accumulator = (
                    np.zeros_like(
                        frame,
                        dtype=np.float32,
                    )
                )

            accumulator += frame.astype(
                np.float32
            )

        if accumulator is None:

            raise RuntimeError(
                "背景采集失败"
            )

        background = (
            accumulator
            / float(
                frame_count
            )
        ).astype(
            np.uint8
        )

        self.set_background(
            background
        )

        return background

    def save_background(
        self,
        path: str | Path,
    ) -> Path:

        if self.background_image is None:

            raise RuntimeError(
                "尚未建立空背景"
            )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not cv2.imwrite(
            str(
                path
            ),
            self.background_image,
        ):

            raise RuntimeError(
                f"背景保存失败：{path}"
            )

        return path

    def load_background(
        self,
        path: str | Path,
    ) -> np.ndarray:

        path = Path(
            path
        )

        image = cv2.imread(
            str(
                path
            ),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise FileNotFoundError(
                f"无法读取背景：{path}"
            )

        self.set_background(
            image
        )

        return image

    # ============================================================
    # Signed Lab
    # ============================================================

    def _build_difference_masks(
        self,
        current: np.ndarray,
        background: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        返回：
            candidate_mask
            core_mask
            difference_score

        difference_score 仅用于调试显示。
        """

        current_lab = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2LAB,
        ).astype(
            np.int16
        )

        background_lab = cv2.cvtColor(
            background,
            cv2.COLOR_BGR2LAB,
        ).astype(
            np.int16
        )

        # --------------------------------------------------------
        # 有符号亮度差
        #
        # > 0 : 当前比背景亮
        # < 0 : 当前比背景暗
        # --------------------------------------------------------

        signed_luma = (
            current_lab[
                :,
                :,
                0,
            ]
            -
            background_lab[
                :,
                :,
                0,
            ]
        )

        delta_a = np.abs(
            current_lab[
                :,
                :,
                1,
            ]
            -
            background_lab[
                :,
                :,
                1,
            ]
        )

        delta_b = np.abs(
            current_lab[
                :,
                :,
                2,
            ]
            -
            background_lab[
                :,
                :,
                2,
            ]
        )

        chroma_delta = np.maximum(
            delta_a,
            delta_b,
        )

        # --------------------------------------------------------
        # 最终候选前景
        #
        # 1) 色度变化明显
        # 2) 比背景明显更亮
        # 3) 比背景“非常明显”更暗
        #
        # 普通阴影通常：
        #   chroma 很小
        #   signed_luma < 0
        #   但没深到 dark_luma_threshold
        # 因此会被抑制。
        # --------------------------------------------------------

        candidate = (
            (
                chroma_delta
                >=
                self.chroma_threshold
            )
            |
            (
                signed_luma
                >=
                self.light_luma_threshold
            )
            |
            (
                signed_luma
                <=
                -
                self.dark_luma_threshold
            )
        )

        # --------------------------------------------------------
        # 强 core
        #
        # core 只用于找到目标种子。
        # --------------------------------------------------------

        core = (
            (
                chroma_delta
                >=
                self.strong_chroma_threshold
            )
            |
            (
                signed_luma
                >=
                self.strong_light_luma_threshold
            )
            |
            (
                signed_luma
                <=
                -
                self.strong_dark_luma_threshold
            )
        )

        candidate_mask = (
            candidate.astype(
                np.uint8
            )
            * 255
        )

        core_mask = (
            core.astype(
                np.uint8
            )
            * 255
        )

        # 调试可视化分数，不参与最终判定。
        abs_luma = np.abs(
            signed_luma
        )

        difference_score = np.maximum(
            chroma_delta,
            abs_luma,
        )

        difference_score = np.clip(
            difference_score,
            0,
            255,
        ).astype(
            np.uint8
        )

        return (
            candidate_mask,
            core_mask,
            difference_score,
        )


    # ============================================================
    # Gray / near-background texture assist
    # ============================================================

    @staticmethod
    def _gradient_magnitude(
        image_gray: np.ndarray,
    ) -> np.ndarray:

        blurred = cv2.GaussianBlur(
            image_gray,
            (3, 3),
            0,
        )

        gx = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )

        gy = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        return cv2.magnitude(
            gx,
            gy,
        )

    def _gray_texture_assist(
        self,
        current: np.ndarray,
        background: np.ndarray,
        object_mask: np.ndarray,
        current_area: int,
    ) -> tuple[
        np.ndarray,
        int,
        bool,
    ]:
        """
        对“灰色/近灰色、且与灰色托盘颜色很接近”的目标做局部补全。

        关键点：
        - 不降低全局 Signed-Lab 阈值；
        - 只在当前已检测目标 bbox 附近搜索；
        - 只补“色度差很低、亮度也接近背景、但纹理/梯度变化明显”的像素；
        - 因此不会让普通彩色物体走这条分支；
        - 也比直接把阴影当目标稳健。

        该分支主要服务于定位 / crop 完整性。
        """

        if not self.enable_gray_texture_assist:
            return (
                object_mask,
                current_area,
                False,
            )

        mask_region = (
            object_mask > 0
        )

        if not np.any(
            mask_region
        ):
            return (
                object_mask,
                current_area,
                False,
            )

        hsv = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2HSV,
        )

        current_lab = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2LAB,
        ).astype(
            np.int16
        )

        background_lab = cv2.cvtColor(
            background,
            cv2.COLOR_BGR2LAB,
        ).astype(
            np.int16
        )

        signed_luma = (
            current_lab[:, :, 0]
            -
            background_lab[:, :, 0]
        )

        delta_a = np.abs(
            current_lab[:, :, 1]
            -
            background_lab[:, :, 1]
        )

        delta_b = np.abs(
            current_lab[:, :, 2]
            -
            background_lab[:, :, 2]
        )

        chroma_delta = np.maximum(
            delta_a,
            delta_b,
        )

        # --------------------------------------------------------
        # 判断当前目标是否像“近灰色中间调”
        #
        # 白色：V 通常更高；
        # 黑色：V 通常更低；
        # 彩色：chroma_delta 通常明显更高。
        # --------------------------------------------------------

        median_s = float(
            np.median(
                hsv[:, :, 1][
                    mask_region
                ]
            )
        )

        median_v = float(
            np.median(
                hsv[:, :, 2][
                    mask_region
                ]
            )
        )

        median_chroma_delta = float(
            np.median(
                chroma_delta[
                    mask_region
                ]
            )
        )

        is_gray_like = (
            median_s <= 65.0
            and
            100.0 <= median_v <= 225.0
            and
            median_chroma_delta <= 7.0
        )

        if not is_gray_like:
            return (
                object_mask,
                current_area,
                False,
            )

        bbox = self._bbox_from_mask(
            object_mask
        )

        if bbox is None:
            return (
                object_mask,
                current_area,
                False,
            )

        x1, y1, x2, y2 = bbox

        width = max(
            1,
            x2 - x1,
        )

        height = max(
            1,
            y2 - y1,
        )

        image_height, image_width = (
            current.shape[:2]
        )

        # 灰色目标有时只检测到一侧，所以 X 方向给更大的搜索余量。
        support_x = int(
            round(
                width
                *
                self.gray_texture_support_x_ratio
            )
        )

        support_y = int(
            round(
                height
                *
                self.gray_texture_support_y_ratio
            )
        )

        sx1 = max(
            0,
            x1 - support_x,
        )

        sx2 = min(
            image_width,
            x2 + support_x,
        )

        sy1 = max(
            0,
            y1 - support_y,
        )

        sy2 = min(
            image_height,
            y2 + support_y,
        )

        support_mask = np.zeros(
            current.shape[:2],
            dtype=np.uint8,
        )

        support_mask[
            sy1:sy2,
            sx1:sx2,
        ] = 255

        # --------------------------------------------------------
        # 梯度差
        # --------------------------------------------------------

        current_gray = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2GRAY,
        )

        background_gray = cv2.cvtColor(
            background,
            cv2.COLOR_BGR2GRAY,
        )

        current_gradient = (
            self._gradient_magnitude(
                current_gray
            )
        )

        background_gradient = (
            self._gradient_magnitude(
                background_gray
            )
        )

        gradient_difference = np.abs(
            current_gradient
            -
            background_gradient
        )

        # --------------------------------------------------------
        # 只补“近背景颜色，但纹理明显变化”的区域。
        #
        # signed_luma 下限用于排除大多数明显暗阴影；
        # 上限避免把本来已经由 light-luma 检出的区域重复扩张。
        # --------------------------------------------------------

        texture_candidate = (
            (
                support_mask > 0
            )
            &
            (
                chroma_delta
                <
                self.chroma_threshold
            )
            &
            (
                signed_luma
                >=
                self.gray_texture_luma_min
            )
            &
            (
                signed_luma
                <=
                self.gray_texture_luma_max
            )
            &
            (
                gradient_difference
                >=
                self.gray_texture_gradient_threshold
            )
        )

        assisted = object_mask.copy()

        assisted[
            texture_candidate
        ] = 255

        # 用较小的闭运算核连接局部 3D 打印纹理；避免把托盘纹理大片粘进来。
        close_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    self.gray_texture_close_kernel_size,
                    self.gray_texture_close_kernel_size,
                ),
            )
        )

        assisted = cv2.morphologyEx(
            assisted,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=2,
        )

        # 去掉孤立噪点。
        assisted = cv2.morphologyEx(
            assisted,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3),
            ),
            iterations=1,
        )

        # 只保留与原始目标重叠的候选连通域。
        assisted_component, assisted_area = (
            self._component_overlapping_core(
                assisted,
                object_mask,
            )
        )

        if assisted_component is None:
            return (
                object_mask,
                current_area,
                False,
            )

        growth_ratio = (
            assisted_area
            /
            max(
                current_area,
                1,
            )
        )

        # 如果突然膨胀太多，说明又可能吃进了背景，不接受。
        if (
            growth_ratio
            >
            self.gray_texture_max_growth_ratio
        ):
            return (
                object_mask,
                current_area,
                False,
            )

        if (
            assisted_area
            <=
            current_area
        ):
            return (
                object_mask,
                current_area,
                False,
            )

        return (
            assisted_component,
            assisted_area,
            True,
        )

    # ============================================================
    # Morphology
    # ============================================================

    def _clean_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:

        if self.open_kernel_size > 1:

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    self.open_kernel_size,
                    self.open_kernel_size,
                ),
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1,
            )

        if self.close_kernel_size > 1:

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    self.close_kernel_size,
                    self.close_kernel_size,
                ),
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self.close_iterations,
            )

        return mask

    # ============================================================
    # Components
    # ============================================================

    @staticmethod
    def _largest_component(
        mask: np.ndarray,
    ) -> tuple[
        Optional[np.ndarray],
        int,
    ]:

        (
            num_labels,
            labels,
            stats,
            _,
        ) = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        if num_labels <= 1:

            return (
                None,
                0,
            )

        areas = stats[
            1:,
            cv2.CC_STAT_AREA,
        ]

        index = int(
            np.argmax(
                areas
            )
        )

        label_id = (
            index + 1
        )

        area = int(
            areas[
                index
            ]
        )

        component = np.where(
            labels == label_id,
            255,
            0,
        ).astype(
            np.uint8
        )

        return (
            component,
            area,
        )

    @staticmethod
    def _component_overlapping_core(
        candidate_mask: np.ndarray,
        core_component: np.ndarray,
    ) -> tuple[
        Optional[np.ndarray],
        int,
    ]:
        """
        不直接取 candidate 最大连通域。

        而是选择：
            与“物体强 core”重叠最多的 candidate 连通域。

        这样可避免远处偶发背景变化抢走最大连通域。
        """

        (
            num_labels,
            labels,
            stats,
            _,
        ) = (
            cv2.connectedComponentsWithStats(
                candidate_mask,
                connectivity=8,
            )
        )

        if num_labels <= 1:

            return (
                None,
                0,
            )

        core_region = (
            core_component > 0
        )

        label_values = labels[
            core_region
        ]

        if label_values.size == 0:

            return SideObjectExtractor._largest_component(
                candidate_mask
            )

        overlaps = np.bincount(
            label_values.ravel(),
            minlength=num_labels,
        )

        overlaps[
            0
        ] = 0

        best_label = int(
            np.argmax(
                overlaps
            )
        )

        if (
            best_label <= 0
            or overlaps[
                best_label
            ] <= 0
        ):

            return SideObjectExtractor._largest_component(
                candidate_mask
            )

        component = np.where(
            labels == best_label,
            255,
            0,
        ).astype(
            np.uint8
        )

        area = int(
            stats[
                best_label,
                cv2.CC_STAT_AREA,
            ]
        )

        return (
            component,
            area,
        )

    # ============================================================
    # BBox
    # ============================================================

    @staticmethod
    def _bbox_from_mask(
        mask: np.ndarray,
    ) -> Optional[
        tuple[int, int, int, int]
    ]:

        points = cv2.findNonZero(
            mask
        )

        if points is None:

            return None

        x, y, w, h = cv2.boundingRect(
            points
        )

        return (
            int(
                x
            ),
            int(
                y
            ),
            int(
                x + w
            ),
            int(
                y + h
            ),
        )

    def _pad_bbox(
        self,
        bbox: tuple[
            int,
            int,
            int,
            int,
        ],
        image_shape,
    ) -> tuple[
        int,
        int,
        int,
        int,
    ]:

        x1, y1, x2, y2 = bbox

        width = (
            x2 - x1
        )

        height = (
            y2 - y1
        )

        pad_x = max(
            self.min_padding_px,
            int(
                round(
                    width
                    * self.padding_ratio
                )
            ),
        )

        pad_y = max(
            self.min_padding_px,
            int(
                round(
                    height
                    * self.padding_ratio
                )
            ),
        )

        image_height, image_width = (
            image_shape[
                :2
            ]
        )

        return (
            max(
                0,
                x1 - pad_x,
            ),
            max(
                0,
                y1 - pad_y,
            ),
            min(
                image_width,
                x2 + pad_x,
            ),
            min(
                image_height,
                y2 + pad_y,
            ),
        )


    # ============================================================
    # Safe crop helpers
    # ============================================================

    def _is_achromatic_like(
        self,
        current: np.ndarray,
        background: np.ndarray,
        object_mask: np.ndarray,
    ) -> bool:
        """
        只用于决定“最终 Crop 是否需要额外安全扩张”。

        它不是颜色分类器，因此阈值故意宽松。
        即使偶尔把低饱和棕色等也归到该分支，结果只是 Crop 稍大，
        不会把颜色判错。
        """

        region = (
            object_mask > 0
        )

        if not np.any(
            region
        ):
            return False

        hsv = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2HSV,
        )

        current_lab = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2LAB,
        ).astype(
            np.int16
        )

        background_lab = cv2.cvtColor(
            background,
            cv2.COLOR_BGR2LAB,
        ).astype(
            np.int16
        )

        delta_a = np.abs(
            current_lab[:, :, 1]
            -
            background_lab[:, :, 1]
        )

        delta_b = np.abs(
            current_lab[:, :, 2]
            -
            background_lab[:, :, 2]
        )

        chroma_delta = np.maximum(
            delta_a,
            delta_b,
        )

        median_saturation = float(
            np.median(
                hsv[:, :, 1][
                    region
                ]
            )
        )

        median_chroma_delta = float(
            np.median(
                chroma_delta[
                    region
                ]
            )
        )

        return (
            median_saturation
            <=
            self.achromatic_saturation_threshold
            or
            median_chroma_delta
            <=
            self.achromatic_chroma_delta_threshold
        )

    @staticmethod
    def _centered_interval(
        center: float,
        desired_length: int,
        limit: int,
    ) -> tuple[int, int]:
        """
        在 [0, limit] 内生成固定长度区间。
        如果靠近图像边缘，会整体平移而不是简单截短。
        """

        desired_length = max(
            1,
            min(
                int(
                    desired_length
                ),
                int(
                    limit
                ),
            ),
        )

        start = int(
            round(
                float(center)
                -
                desired_length
                / 2.0
            )
        )

        end = (
            start
            +
            desired_length
        )

        if start < 0:

            end -= start

            start = 0

        if end > limit:

            shift = (
                end
                -
                limit
            )

            start -= shift

            end = limit

        start = max(
            0,
            start,
        )

        end = min(
            limit,
            end,
        )

        return (
            int(
                start
            ),
            int(
                end
            ),
        )

    def _apply_achromatic_safe_crop(
        self,
        crop_bbox: tuple[
            int,
            int,
            int,
            int,
        ],
        object_bbox: tuple[
            int,
            int,
            int,
            int,
        ],
        image_shape,
    ) -> tuple[
        int,
        int,
        int,
        int,
    ]:
        """
        在已有 padding crop 基础上，只做“最小安全尺寸”保证。

        不会缩小原 crop，只可能扩大。
        """

        image_height, image_width = (
            image_shape[:2]
        )

        x1, y1, x2, y2 = (
            crop_bbox
        )

        current_width = (
            x2 - x1
        )

        current_height = (
            y2 - y1
        )

        minimum_width = int(
            round(
                image_width
                *
                self.achromatic_min_crop_width_ratio
            )
        )

        minimum_height = int(
            round(
                image_height
                *
                self.achromatic_min_crop_height_ratio
            )
        )

        desired_width = max(
            current_width,
            minimum_width,
        )

        desired_height = max(
            current_height,
            minimum_height,
        )

        bx1, by1, bx2, by2 = (
            object_bbox
        )

        center_x = (
            bx1 + bx2
        ) / 2.0

        center_y = (
            by1 + by2
        ) / 2.0

        safe_x1, safe_x2 = (
            self._centered_interval(
                center_x,
                desired_width,
                image_width,
            )
        )

        safe_y1, safe_y2 = (
            self._centered_interval(
                center_y,
                desired_height,
                image_height,
            )
        )

        # 必须包含原来的 padding crop。
        final_x1 = min(
            x1,
            safe_x1,
        )

        final_y1 = min(
            y1,
            safe_y1,
        )

        final_x2 = max(
            x2,
            safe_x2,
        )

        final_y2 = max(
            y2,
            safe_y2,
        )

        final_x1 = max(
            0,
            final_x1,
        )

        final_y1 = max(
            0,
            final_y1,
        )

        final_x2 = min(
            image_width,
            final_x2,
        )

        final_y2 = min(
            image_height,
            final_y2,
        )

        return (
            int(
                final_x1
            ),
            int(
                final_y1
            ),
            int(
                final_x2
            ),
            int(
                final_y2
            ),
        )

    # ============================================================
    # Main
    # ============================================================

    def extract(
        self,
        mechanical_roi: np.ndarray,
    ) -> SideObjectResult:

        if self.background_image is None:

            return SideObjectResult(
                found=False,
                reason="background_not_set",
                mechanical_roi=mechanical_roi,
                crop=None,
                object_mask=None,
                crop_mask=None,
                bbox=None,
                crop_bbox=None,
                area=0,
                area_ratio=0.0,
            )

        if (
            mechanical_roi is None
            or mechanical_roi.ndim != 3
            or mechanical_roi.shape[2] != 3
        ):

            raise ValueError(
                "mechanical_roi 必须为 BGR 三通道图像"
            )

        if (
            mechanical_roi.shape
            !=
            self.background_image.shape
        ):

            raise RuntimeError(
                "当前机械 ROI 与背景尺寸不一致：\n"
                f"current={mechanical_roi.shape}\n"
                f"background={self.background_image.shape}\n"
                "修改 SideCamera ROI 后必须重新采空背景。"
            )

        (
            candidate_mask,
            core_mask,
            difference_score,
        ) = self._build_difference_masks(
            mechanical_roi,
            self.background_image,
        )

        candidate_mask = self._clean_mask(
            candidate_mask
        )

        core_mask = self._clean_mask(
            core_mask
        )

        roi_area = (
            mechanical_roi.shape[0]
            *
            mechanical_roi.shape[1]
        )

        minimum_core_area = max(
            self.min_core_area,
            int(
                round(
                    roi_area
                    *
                    self.min_core_area_ratio
                )
            ),
        )

        core_component, core_area = (
            self._largest_component(
                core_mask
            )
        )

        # 很小的 strong core 更像局部反光/噪点，而不是可靠物体种子。
        # 将其降级到 candidate_without_core 分支，随后必须通过更严格门槛。
        if (
            core_component is not None
            and core_area < minimum_core_area
        ):
            core_component = None

        if core_component is None:

            # 没有强 core 时允许退回 candidate，
            # 但不会退回一个“只有顶面”的 core。
            object_mask, area = (
                self._largest_component(
                    candidate_mask
                )
            )

            selection_reason = (
                "candidate_without_core"
            )

        else:

            object_mask, area = (
                self._component_overlapping_core(
                    candidate_mask,
                    core_component,
                )
            )

            selection_reason = (
                "signed_lab"
            )

        if object_mask is None:

            return SideObjectResult(
                found=False,
                reason="object_not_found",
                mechanical_roi=mechanical_roi,
                crop=None,
                object_mask=None,
                crop_mask=None,
                bbox=None,
                crop_bbox=None,
                area=0,
                area_ratio=0.0,
                difference_score=difference_score,
                core_mask=core_component,
            )

        # --------------------------------------------------------
        # Gray / near-background fallback
        #
        # 只在低色度、中等亮度目标上尝试纹理补全。
        # 其它颜色仍保持 Signed-Lab 主链路，不受影响。
        # --------------------------------------------------------

        (
            assisted_mask,
            assisted_area,
            used_gray_texture_assist,
        ) = self._gray_texture_assist(
            mechanical_roi,
            self.background_image,
            object_mask,
            area,
        )

        if used_gray_texture_assist:

            object_mask = assisted_mask

            area = assisted_area

            if selection_reason == "candidate_without_core":
                selection_reason = (
                    "candidate_without_core_gray_texture"
                )
            else:
                selection_reason = (
                    "signed_lab_gray_texture"
                )

        area_ratio = (
            area
            /
            max(
                roi_area,
                1,
            )
        )

        if selection_reason.startswith("candidate_without_core"):
            minimum_area = max(
                self.fallback_min_object_area,
                int(
                    round(
                        roi_area
                        *
                        self.fallback_min_object_area_ratio
                    )
                ),
            )
        else:
            minimum_area = max(
                self.min_object_area,
                int(
                    round(
                        roi_area
                        *
                        self.min_object_area_ratio
                    )
                ),
            )

        if area < minimum_area:

            return SideObjectResult(
                found=False,
                reason="object_too_small",
                mechanical_roi=mechanical_roi,
                crop=None,
                object_mask=object_mask,
                crop_mask=None,
                bbox=self._bbox_from_mask(
                    object_mask
                ),
                crop_bbox=None,
                area=area,
                area_ratio=float(
                    area_ratio
                ),
                difference_score=difference_score,
                core_mask=core_component,
            )

        if (
            area_ratio
            >
            self.max_object_area_ratio
        ):

            return SideObjectResult(
                found=False,
                reason="foreground_too_large",
                mechanical_roi=mechanical_roi,
                crop=None,
                object_mask=object_mask,
                crop_mask=None,
                bbox=self._bbox_from_mask(
                    object_mask
                ),
                crop_bbox=None,
                area=area,
                area_ratio=float(
                    area_ratio
                ),
                difference_score=difference_score,
                core_mask=core_component,
            )

        bbox = self._bbox_from_mask(
            object_mask
        )

        if bbox is None:

            return SideObjectResult(
                found=False,
                reason="bbox_not_found",
                mechanical_roi=mechanical_roi,
                crop=None,
                object_mask=object_mask,
                crop_mask=None,
                bbox=None,
                crop_bbox=None,
                area=area,
                area_ratio=float(
                    area_ratio
                ),
                difference_score=difference_score,
                core_mask=core_component,
            )

        if selection_reason.startswith("candidate_without_core"):
            bx1, by1, bx2, by2 = bbox
            bbox_width_ratio = (
                (bx2 - bx1)
                /
                max(mechanical_roi.shape[1], 1)
            )
            bbox_height_ratio = (
                (by2 - by1)
                /
                max(mechanical_roi.shape[0], 1)
            )

            if (
                bbox_width_ratio < self.fallback_min_bbox_width_ratio
                or bbox_height_ratio < self.fallback_min_bbox_height_ratio
            ):
                return SideObjectResult(
                    found=False,
                    reason="candidate_without_core_weak_geometry",
                    mechanical_roi=mechanical_roi,
                    crop=None,
                    object_mask=object_mask,
                    crop_mask=None,
                    bbox=bbox,
                    crop_bbox=None,
                    area=area,
                    area_ratio=float(area_ratio),
                    difference_score=difference_score,
                    core_mask=core_component,
                )

        crop_bbox = self._pad_bbox(
            bbox,
            mechanical_roi.shape,
        )

        # --------------------------------------------------------
        # 重要：
        # Mask 和 Crop 的职责分开。
        #
        # object_mask 是 best-effort 前景区域；
        # crop 必须优先保证“完整看到物体”。
        #
        # 对灰/黑/白等低色度物体，如果 Mask 只覆盖部分主体，
        # 这里仍会围绕检测中心保留足够大的安全窗口。
        # --------------------------------------------------------

        if (
            self.enable_achromatic_safe_crop
            and
            self._is_achromatic_like(
                mechanical_roi,
                self.background_image,
                object_mask,
            )
        ):

            crop_bbox = (
                self._apply_achromatic_safe_crop(
                    crop_bbox,
                    bbox,
                    mechanical_roi.shape,
                )
            )

        x1, y1, x2, y2 = (
            crop_bbox
        )

        crop = mechanical_roi[
            y1:y2,
            x1:x2,
        ].copy()

        crop_mask = object_mask[
            y1:y2,
            x1:x2,
        ].copy()

        if crop.size == 0:

            return SideObjectResult(
                found=False,
                reason="empty_crop",
                mechanical_roi=mechanical_roi,
                crop=None,
                object_mask=object_mask,
                crop_mask=None,
                bbox=bbox,
                crop_bbox=crop_bbox,
                area=area,
                area_ratio=float(
                    area_ratio
                ),
                difference_score=difference_score,
                core_mask=core_component,
            )

        return SideObjectResult(
            found=True,
            reason=selection_reason,  
            mechanical_roi=mechanical_roi,
            crop=crop,
            object_mask=object_mask,
            crop_mask=crop_mask,
            bbox=bbox,
            crop_bbox=crop_bbox,
            area=area,
            area_ratio=float(
                area_ratio
            ),
            difference_score=difference_score,
            core_mask=core_component,
        )