from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class HueBand:
    name: str
    start: float
    end: float


class ColorClassifier:
    """
    通用颜色识别 V7 - 简化版

    目标：
        用尽量少、尽量可解释的规则完成通用颜色识别。

    输入：
        crop:
            SideObjectExtractor 输出的 BGR crop

        crop_mask:
            SideObjectExtractor 输出的 crop_mask

    输出：
        red / orange / yellow / green / cyan / blue /
        purple / pink / brown / black / white / gray / unknown

    ------------------------------------------------------------
    V7 刻意删除：
        - 颜色 Profile
        - 局部背景亮度补偿
        - 多层 white rescue
        - 大量互相叠加的白/灰特殊规则

    只保留四步：

        1. crop_mask 内缩，取可靠内部区域
        2. 先判断：彩色 vs 黑白灰
        3. 彩色：主 Hue 簇 + 少量必要的模糊区修正
        4. 黑白灰：用亮度分位数判断

    白色为什么不用 median 亮度：
        白色物体被遮光时，大部分区域可能很暗，
        median 会掉到 gray。

        但只要仍有一部分白色表面正常受光，
        L75 / L90 通常仍明显高于真正 gray。

        所以：
            black -> 看 median
            white -> 看较亮的上分位数
            其它 -> gray

    这是 V7 对 white/gray 的核心简化。
    """

    HUE_BANDS = (
        HueBand("orange", 8.0, 22.0),
        HueBand("yellow", 22.0, 38.0),
        HueBand("green", 38.0, 84.0),
        HueBand("cyan", 84.0, 104.0),
        HueBand("blue", 104.0, 130.0),
        HueBand("purple", 130.0, 157.0),
        HueBand("pink", 157.0, 173.0),
    )

    DEFAULT_COLORS = (
        "red",
        "orange",
        "yellow",
        "green",
        "cyan",
        "blue",
        "purple",
        "pink",
        # Competition colors do not include brown.
        # Keep brown logic in the classifier, but do not enable it by default;
        # this prevents dark orange faces from being intercepted as brown.
        "black",
        "white",
        "gray",
    )

    def __init__(
        self,

        # mask 内缩
        mask_erode_px: int = 3,

        # 彩色 / 无彩色门控
        achromatic_s_max: int = 80,
        achromatic_lab_chroma_max: float = 22.0,

        # 彩色可靠像素
        reliable_v_min: int = 35,
        reliable_v_max: int = 238,
        chromatic_s_min: int = 45,
        chromatic_lab_chroma_min: float = 10.0,

        # 主 Hue
        hue_hist_bins: int = 36,
        dominant_hue_half_width: int = 12,

        # black
        black_v_max: int = 45,
        black_lab_l_max: int = 75,

        # white
        #
        # 不再使用单一 median 阈值。
        # 需要 P75 和 P90 同时足够亮，降低 gray -> white。
        white_lab_l_p75_min: int = 195,
        white_lab_l_p90_min: int = 210,

        # brown
        #
        # brown = 较暗的 orange / yellow-orange
        brown_h_min: int = 6,
        brown_h_max: int = 25,
        brown_v_max: int = 115,
        brown_lab_l_max: int = 115,

        # blue / purple 模糊区
        purple_ambiguous_h_min: int = 115,
        purple_ambiguous_h_max: int = 133,
        purple_lab_a_min: int = 140,

        # 浅粉色可能落在 purple/pink 边缘。
        #
        # 当前侧面相机 + 4650K 白平衡下，实测粉色 3D 打印件：
        #   H≈146, S≈52, V≈163, Lab-L≈149
        # 而紫色样本：
        #   H≈118, S≈160, V≈144, Lab-L≈74
        #
        # 因此不能只靠 150~160 的 Hue 窄窗判断粉色；
        # 这里同时要求“低饱和 + 较亮”，避免把深紫误判为粉色。
        pastel_pink_h_min: int = 138,
        pastel_pink_h_max: int = 160,
        pastel_pink_s_max: int = 110,
        pastel_pink_v_min: int = 125,
        pastel_pink_lab_l_min: int = 120,

        # red / pink 低 Hue 模糊区
        pink_low_h_max: int = 7,
        pink_low_h_s_max: int = 175,
        pink_low_h_v_min: int = 150,

        min_pixels: int = 150,

        enabled_colors: Iterable[str] | None = None,
    ) -> None:

        self.mask_erode_px = int(mask_erode_px)

        self.achromatic_s_max = int(
            achromatic_s_max
        )

        self.achromatic_lab_chroma_max = float(
            achromatic_lab_chroma_max
        )

        self.reliable_v_min = int(reliable_v_min)
        self.reliable_v_max = int(reliable_v_max)
        self.chromatic_s_min = int(chromatic_s_min)
        self.chromatic_lab_chroma_min = float(
            chromatic_lab_chroma_min
        )

        self.hue_hist_bins = int(hue_hist_bins)
        self.dominant_hue_half_width = int(
            dominant_hue_half_width
        )

        self.black_v_max = int(black_v_max)
        self.black_lab_l_max = int(
            black_lab_l_max
        )

        self.white_lab_l_p75_min = int(
            white_lab_l_p75_min
        )

        self.white_lab_l_p90_min = int(
            white_lab_l_p90_min
        )

        self.brown_h_min = int(brown_h_min)
        self.brown_h_max = int(brown_h_max)
        self.brown_v_max = int(brown_v_max)
        self.brown_lab_l_max = int(
            brown_lab_l_max
        )

        self.purple_ambiguous_h_min = int(
            purple_ambiguous_h_min
        )

        self.purple_ambiguous_h_max = int(
            purple_ambiguous_h_max
        )

        self.purple_lab_a_min = int(
            purple_lab_a_min
        )

        self.pastel_pink_h_min = int(
            pastel_pink_h_min
        )

        self.pastel_pink_h_max = int(
            pastel_pink_h_max
        )

        self.pastel_pink_s_max = int(
            pastel_pink_s_max
        )

        self.pastel_pink_v_min = int(
            pastel_pink_v_min
        )

        self.pastel_pink_lab_l_min = int(
            pastel_pink_lab_l_min
        )

        self.pink_low_h_max = int(
            pink_low_h_max
        )

        self.pink_low_h_s_max = int(
            pink_low_h_s_max
        )

        self.pink_low_h_v_min = int(
            pink_low_h_v_min
        )

        self.min_pixels = int(min_pixels)

        if enabled_colors is None:

            self.enabled_colors = set(
                self.DEFAULT_COLORS
            )

        else:

            self.enabled_colors = {
                str(name).lower()
                for name in enabled_colors
            }

    # ============================================================
    # Mask
    # ============================================================

    @staticmethod
    def _validate(
        crop: np.ndarray,
        crop_mask: np.ndarray,
    ) -> None:

        if (
            crop is None
            or crop.ndim != 3
            or crop.shape[2] != 3
        ):
            raise ValueError(
                "crop 必须是 BGR 三通道图像"
            )

        if (
            crop_mask is None
            or crop_mask.ndim != 2
        ):
            raise ValueError(
                "crop_mask 必须是单通道 Mask"
            )

        if crop.shape[:2] != crop_mask.shape[:2]:
            raise ValueError(
                "crop 与 crop_mask 尺寸不一致"
            )

    def _stable_mask(
        self,
        crop_mask: np.ndarray,
    ) -> np.ndarray:

        mask = np.where(
            crop_mask > 0,
            255,
            0,
        ).astype(
            np.uint8
        )

        count = int(
            np.count_nonzero(mask)
        )

        if (
            count == 0
            or self.mask_erode_px <= 0
        ):
            return mask

        k = (
            self.mask_erode_px * 2 + 1
        )

        eroded = cv2.erode(
            mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (k, k),
            ),
            iterations=1,
        )

        eroded_count = int(
            np.count_nonzero(eroded)
        )

        # 小物体不能腐蚀掉太多
        if eroded_count < max(
            120,
            int(count * 0.55),
        ):
            return mask

        return eroded

    # ============================================================
    # Hue
    # ============================================================

    @staticmethod
    def _hue_distance(
        value,
        center: float,
    ):
        delta = np.abs(
            np.asarray(
                value,
                dtype=np.float32,
            )
            -
            float(center)
        )

        return np.minimum(
            delta,
            180.0 - delta,
        )

    @staticmethod
    def _circular_hue_mean(
        values: np.ndarray,
    ) -> float:

        angles = (
            values.astype(
                np.float64
            )
            *
            (
                2.0
                *
                math.pi
                /
                180.0
            )
        )

        angle = math.atan2(
            float(
                np.mean(
                    np.sin(angles)
                )
            ),
            float(
                np.mean(
                    np.cos(angles)
                )
            ),
        )

        if angle < 0:
            angle += 2.0 * math.pi

        return float(
            angle
            *
            180.0
            /
            (
                2.0
                *
                math.pi
            )
        )

    # ============================================================
    # Main feature extraction
    # ============================================================

    def _extract_arrays(
        self,
        crop: np.ndarray,
        crop_mask: np.ndarray,
    ) -> dict[str, Any]:

        stable_mask = self._stable_mask(
            crop_mask
        )

        region = (
            stable_mask > 0
        )

        count = int(
            np.count_nonzero(region)
        )

        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV,
        )

        lab = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2LAB,
        )

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        lab_l = lab[:, :, 0].astype(
            np.float32
        )

        lab_a = lab[:, :, 1].astype(
            np.float32
        )

        lab_b = lab[:, :, 2].astype(
            np.float32
        )

        lab_chroma = np.sqrt(
            np.square(
                lab_a - 128.0
            )
            +
            np.square(
                lab_b - 128.0
            )
        )

        return {
            "stable_mask":
                stable_mask,

            "region":
                region,

            "count":
                count,

            "h":
                h,

            "s":
                s,

            "v":
                v,

            "lab_l":
                lab_l,

            "lab_a":
                lab_a,

            "lab_b":
                lab_b,

            "lab_chroma":
                lab_chroma,
        }

    # ============================================================
    # Achromatic
    # ============================================================

    def _classify_achromatic(
        self,
        arrays: dict[str, Any],
    ) -> dict[str, Any]:

        region = arrays[
            "region"
        ]

        s_values = arrays[
            "s"
        ][region].astype(
            np.float32
        )

        v_values = arrays[
            "v"
        ][region].astype(
            np.float32
        )

        l_values = arrays[
            "lab_l"
        ][region].astype(
            np.float32
        )

        features = {
            "s_median":
                float(
                    np.median(
                        s_values
                    )
                ),

            "v_median":
                float(
                    np.median(
                        v_values
                    )
                ),

            "lab_l_median":
                float(
                    np.median(
                        l_values
                    )
                ),

            "lab_l_p75":
                float(
                    np.percentile(
                        l_values,
                        75,
                    )
                ),

            "lab_l_p90":
                float(
                    np.percentile(
                        l_values,
                        90,
                    )
                ),
        }

        # black：整体真的很暗
        if (
            features[
                "v_median"
            ]
            <=
            self.black_v_max
            and
            features[
                "lab_l_median"
            ]
            <=
            self.black_lab_l_max
        ):

            color = (
                "black"
                if "black"
                in self.enabled_colors
                else "unknown"
            )

            confidence = 0.90

        # white：
        # 不看 median，改看亮部的 P75 + P90。
        elif (
            features[
                "lab_l_p75"
            ]
            >=
            self.white_lab_l_p75_min
            and
            features[
                "lab_l_p90"
            ]
            >=
            self.white_lab_l_p90_min
        ):

            color = (
                "white"
                if "white"
                in self.enabled_colors
                else "unknown"
            )

            margin75 = (
                features[
                    "lab_l_p75"
                ]
                -
                self.white_lab_l_p75_min
            )

            margin90 = (
                features[
                    "lab_l_p90"
                ]
                -
                self.white_lab_l_p90_min
            )

            confidence = min(
                0.98,
                0.75
                +
                0.004
                *
                max(
                    0.0,
                    margin75
                    +
                    margin90,
                ),
            )

        else:

            color = (
                "gray"
                if "gray"
                in self.enabled_colors
                else "unknown"
            )

            confidence = 0.82

        return {
            "found":
                True,

            "color":
                color,

            "confidence":
                float(confidence),

            "kind":
                "achromatic",

            "reason":
                "simple_achromatic_percentile",

            "features":
                features,

            "stable_mask":
                arrays[
                    "stable_mask"
                ],

            "reliable_mask":
                (
                    arrays[
                        "region"
                    ].astype(
                        np.uint8
                    )
                    * 255
                ),
        }

    # ============================================================
    # Chromatic
    # ============================================================

    def _semantic_color(
        self,
        hue: float,
        saturation: float,
        value: float,
        lab_l: float,
        lab_a: float,
    ) -> str:

        h = float(hue)

        # 较暗的橙黄区域 -> brown
        if (
            self.brown_h_min
            <= h
            <= self.brown_h_max
            and
            value
            <=
            self.brown_v_max
            and
            lab_l
            <=
            self.brown_lab_l_max
            and
            "brown"
            in self.enabled_colors
        ):
            return "brown"

        # blue / purple 模糊区
        if (
            self.purple_ambiguous_h_min
            <= h
            <
            self.purple_ambiguous_h_max
            and
            lab_a
            >=
            self.purple_lab_a_min
            and
            "purple"
            in self.enabled_colors
        ):
            return "purple"

        # 浅粉色：
        # Hue 可以落在传统 purple 区，但通常饱和度较低、亮度较高。
        if (
            self.pastel_pink_h_min
            <= h
            <
            self.pastel_pink_h_max
            and
            saturation
            <=
            self.pastel_pink_s_max
            and
            value
            >=
            self.pastel_pink_v_min
            and
            lab_l
            >=
            self.pastel_pink_lab_l_min
            and
            "pink"
            in self.enabled_colors
        ):
            return "pink"

        # 靠近 red 的浅粉
        if (
            (
                h
                <=
                self.pink_low_h_max
                or
                h
                >=
                173.0
            )
            and
            saturation
            <=
            self.pink_low_h_s_max
            and
            value
            >=
            self.pink_low_h_v_min
            and
            "pink"
            in self.enabled_colors
        ):
            return "pink"

        # red 环绕 0/179
        if (
            h < 8.0
            or h >= 173.0
        ):

            return (
                "red"
                if "red"
                in self.enabled_colors
                else "unknown"
            )

        for band in self.HUE_BANDS:

            if (
                band.name
                in self.enabled_colors
                and
                band.start
                <= h
                <
                band.end
            ):
                return band.name

        return "unknown"

    def _classify_chromatic(
        self,
        arrays: dict[str, Any],
    ) -> dict[str, Any]:

        region = arrays[
            "region"
        ]

        h = arrays["h"]
        s = arrays["s"]
        v = arrays["v"]

        lab_l = arrays[
            "lab_l"
        ]

        lab_a = arrays[
            "lab_a"
        ]

        lab_chroma = arrays[
            "lab_chroma"
        ]

        reliable = (
            region
            &
            (
                v
                >=
                self.reliable_v_min
            )
            &
            (
                v
                <=
                self.reliable_v_max
            )
            &
            (
                (
                    s
                    >=
                    self.chromatic_s_min
                )
                |
                (
                    lab_chroma
                    >=
                    self.chromatic_lab_chroma_min
                )
            )
        )

        reliable_count = int(
            np.count_nonzero(
                reliable
            )
        )

        if reliable_count < self.min_pixels:

            return {
                "found":
                    True,

                "color":
                    "unknown",

                "confidence":
                    0.25,

                "kind":
                    "chromatic",

                "reason":
                    "too_few_reliable_color_pixels",

                "stable_mask":
                    arrays[
                        "stable_mask"
                    ],

                "reliable_mask":
                    (
                        reliable.astype(
                            np.uint8
                        )
                        * 255
                    ),
            }

        hue_values = h[
            reliable
        ]

        histogram, edges = np.histogram(
            hue_values,
            bins=self.hue_hist_bins,
            range=(0, 180),
        )

        best_bin = int(
            np.argmax(
                histogram
            )
        )

        center = (
            edges[
                best_bin
            ]
            +
            edges[
                best_bin + 1
            ]
        ) / 2.0

        dominant = (
            reliable
            &
            (
                self._hue_distance(
                    h,
                    center,
                )
                <=
                self.dominant_hue_half_width
            )
        )

        dominant_count = int(
            np.count_nonzero(
                dominant
            )
        )

        if dominant_count < self.min_pixels:

            return {
                "found":
                    True,

                "color":
                    "unknown",

                "confidence":
                    0.25,

                "kind":
                    "chromatic",

                "reason":
                    "dominant_color_too_small",

                "stable_mask":
                    arrays[
                        "stable_mask"
                    ],

                "reliable_mask":
                    (
                        reliable.astype(
                            np.uint8
                        )
                        * 255
                    ),
            }

        hue = self._circular_hue_mean(
            h[
                dominant
            ]
        )

        saturation = float(
            np.median(
                s[
                    dominant
                ]
            )
        )

        value = float(
            np.median(
                v[
                    dominant
                ]
            )
        )

        l_value = float(
            np.median(
                lab_l[
                    dominant
                ]
            )
        )

        a_value = float(
            np.median(
                lab_a[
                    dominant
                ]
            )
        )

        color = self._semantic_color(
            hue=hue,
            saturation=saturation,
            value=value,
            lab_l=l_value,
            lab_a=a_value,
        )

        purity = (
            dominant_count
            /
            max(
                reliable_count,
                1,
            )
        )

        confidence = float(
            min(
                0.98,
                0.65
                +
                0.30
                *
                purity,
            )
        )

        if color == "unknown":
            confidence = min(
                confidence,
                0.40,
            )

        return {
            "found":
                True,

            "color":
                color,

            "confidence":
                confidence,

            "kind":
                "chromatic",

            "reason":
                "simple_dominant_hue",

            "features": {
                "hue":
                    float(hue),

                "s":
                    saturation,

                "v":
                    value,

                "lab_l":
                    l_value,

                "lab_a":
                    a_value,

                "dominant_purity":
                    float(purity),
            },

            "stable_mask":
                arrays[
                    "stable_mask"
                ],

            "reliable_mask":
                (
                    reliable.astype(
                        np.uint8
                    )
                    * 255
                ),
        }

    # ============================================================
    # Public
    # ============================================================

    def classify(
        self,
        crop: np.ndarray,
        crop_mask: np.ndarray,
    ) -> dict[str, Any]:

        self._validate(
            crop,
            crop_mask,
        )

        arrays = self._extract_arrays(
            crop,
            crop_mask,
        )

        if arrays["count"] < self.min_pixels:

            return {
                "found":
                    False,

                "color":
                    "unknown",

                "confidence":
                    0.0,

                "reason":
                    "too_few_object_pixels",

                "stable_mask":
                    arrays[
                        "stable_mask"
                    ],
            }

        region = arrays[
            "region"
        ]

        median_s = float(
            np.median(
                arrays[
                    "s"
                ][region]
            )
        )

        median_lab_chroma = float(
            np.median(
                arrays[
                    "lab_chroma"
                ][region]
            )
        )

        # --------------------------------------------------------
        # 黑色先判：
        #
        # 黑色表面在低照度下 Hue / Saturation 本身没有意义，
        # 甚至会因为传感器噪声得到很高的 S。
        # 所以黑色不能依赖 achromatic gate，直接看整体亮度。
        # --------------------------------------------------------

        median_v = float(
            np.median(
                arrays[
                    "v"
                ][region]
            )
        )

        median_lab_l = float(
            np.median(
                arrays[
                    "lab_l"
                ][region]
            )
        )

        if (
            median_v
            <=
            self.black_v_max
            and
            median_lab_l
            <=
            self.black_lab_l_max
        ):

            return {
                "found":
                    True,

                "color":
                    (
                        "black"
                        if "black"
                        in self.enabled_colors
                        else "unknown"
                    ),

                "confidence":
                    0.90,

                "kind":
                    "achromatic",

                "reason":
                    "simple_black_brightness",

                "features": {
                    "v_median":
                        median_v,

                    "lab_l_median":
                        median_lab_l,
                },

                "stable_mask":
                    arrays[
                        "stable_mask"
                    ],

                "reliable_mask":
                    (
                        region.astype(
                            np.uint8
                        )
                        * 255
                    ),

                "global_s_median":
                    median_s,

                "global_lab_chroma_median":
                    median_lab_chroma,
            }

        # 非常简单的第二层：
        # 低饱和 + 低 Lab 色度 -> white / gray
        is_achromatic = (
            median_s
            <=
            self.achromatic_s_max
            and
            median_lab_chroma
            <=
            self.achromatic_lab_chroma_max
        )

        if is_achromatic:

            result = self._classify_achromatic(
                arrays
            )

        else:

            result = self._classify_chromatic(
                arrays
            )

        result[
            "global_s_median"
        ] = median_s

        result[
            "global_lab_chroma_median"
        ] = median_lab_chroma

        return result
