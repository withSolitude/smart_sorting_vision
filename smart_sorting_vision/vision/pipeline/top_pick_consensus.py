from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from vision.grasp.orientation_planner import axial_difference_deg, axial_mean_deg


@dataclass
class StableTopPick:
    ready: bool
    votes: int
    window_size: int
    grasp_pixel: tuple[int, int] | None
    x_mm: float | None
    y_mm: float | None
    yaw_deg: float | None
    grasp_width_mm: float | None
    preopen_mm: float | None
    confidence: float
    clearance_px: float
    class_name: str | None
    reason: str


class TopPickConsensus:
    """
    顶部抓取结果多帧稳定器。

    规则：
        - 5 帧窗口
        - 至少 4 帧匹配同一物体
        - 使用目标中心做跨帧关联，不依赖 YOLO index
        - 最终抓取点 / XY 取中位数
        - XY 离散过大时不输出 READY
    """

    def __init__(
        self,
        *,
        window_size: int = 5,
        required_hits: int = 4,
        max_track_center_shift_px: float = 55.0,
        max_grasp_spread_mm: float = 3.0,
        reset_after_misses: int = 3,
        max_yaw_spread_deg: float = 12.0,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size 必须 >= 1")

        if not (1 <= required_hits <= window_size):
            raise ValueError(
                "required_hits 必须满足 "
                "1 <= required_hits <= window_size"
            )

        self.window_size = int(window_size)
        self.required_hits = int(required_hits)
        self.max_track_center_shift_px = float(
            max_track_center_shift_px
        )
        self.max_grasp_spread_mm = float(
            max_grasp_spread_mm
        )
        self.reset_after_misses = int(
            reset_after_misses
        )
        self.max_yaw_spread_deg = float(max_yaw_spread_deg)
        if self.max_yaw_spread_deg <= 0.0:
            raise ValueError("max_yaw_spread_deg must be > 0")

        self.history: deque[
            dict[str, Any] | None
        ] = deque(maxlen=self.window_size)

        self.anchor_center: (
            tuple[float, float] | None
        ) = None

        self.consecutive_misses = 0

    def reset(self) -> None:
        self.history.clear()
        self.anchor_center = None
        self.consecutive_misses = 0

    @staticmethod
    def _best_initial_candidate(
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return max(
            candidates,
            key=lambda item: (
                float(item.get("score", 0.0)),
                float(item.get("clearance", 0.0)),
                float(item.get("confidence", 0.0)),
                -float(item["center_pixel"][1]),
                -float(item["center_pixel"][0]),
            ),
        )

    def _associate_candidate(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not candidates:
            return None

        if self.anchor_center is None:
            return self._best_initial_candidate(
                candidates
            )

        ax, ay = self.anchor_center

        matched: list[
            tuple[float, dict[str, Any]]
        ] = []

        for candidate in candidates:
            cx, cy = candidate["center_pixel"]

            distance = float(
                np.hypot(
                    float(cx) - ax,
                    float(cy) - ay,
                )
            )

            if (
                distance
                <=
                self.max_track_center_shift_px
            ):
                matched.append(
                    (distance, candidate)
                )

        if not matched:
            return None

        matched.sort(
            key=lambda item: (
                item[0],
                -float(
                    item[1].get(
                        "score",
                        0.0,
                    )
                ),
            )
        )

        return matched[0][1]

    def _update_anchor(self) -> None:
        valid = [
            item
            for item in self.history
            if item is not None
        ]

        if not valid:
            self.anchor_center = None
            return

        centers = np.asarray(
            [
                item["center_pixel"]
                for item in valid
            ],
            dtype=np.float64,
        )

        median_center = np.median(
            centers,
            axis=0,
        )

        self.anchor_center = (
            float(median_center[0]),
            float(median_center[1]),
        )

    def update(
        self,
        candidates: list[dict[str, Any]],
    ) -> StableTopPick:
        candidate = self._associate_candidate(
            candidates
        )

        if candidate is None:
            self.history.append(None)
            self.consecutive_misses += 1

            if (
                self.consecutive_misses
                >=
                self.reset_after_misses
            ):
                self.reset()

                if candidates:
                    candidate = (
                        self._best_initial_candidate(
                            candidates
                        )
                    )
                    self.history.append(candidate)
                    self.consecutive_misses = 0
        else:
            self.history.append(candidate)
            self.consecutive_misses = 0

        self._update_anchor()

        valid = [
            item
            for item in self.history
            if item is not None
        ]

        votes = len(valid)

        if votes < self.required_hits:
            return StableTopPick(
                ready=False,
                votes=votes,
                window_size=self.window_size,
                grasp_pixel=None,
                x_mm=None,
                y_mm=None,
                yaw_deg=None,
                grasp_width_mm=None,
                preopen_mm=None,
                confidence=0.0,
                clearance_px=0.0,
                class_name=None,
                reason="not_enough_votes",
            )

        robot_points = np.asarray(
            [
                [
                    float(item["x_mm"]),
                    float(item["y_mm"]),
                ]
                for item in valid
            ],
            dtype=np.float64,
        )

        median_xy = np.median(
            robot_points,
            axis=0,
        )

        deviations = np.linalg.norm(
            robot_points - median_xy,
            axis=1,
        )

        max_deviation = float(
            np.max(deviations)
        )

        if (
            max_deviation
            >
            self.max_grasp_spread_mm
        ):
            return StableTopPick(
                ready=False,
                votes=votes,
                window_size=self.window_size,
                grasp_pixel=None,
                x_mm=None,
                y_mm=None,
                yaw_deg=None,
                grasp_width_mm=None,
                preopen_mm=None,
                confidence=0.0,
                clearance_px=0.0,
                class_name=None,
                reason=(
                    "xy_spread_too_large:"
                    f"{max_deviation:.2f}mm"
                ),
            )

        # IMPORTANT: never average a real dynamic Yaw with a fallback Yaw.
        #
        # For elongated objects, orientation can occasionally toggle between
        # reliable/unreliable from frame to frame because the segmentation mask
        # slightly changes its aspect ratio.  The old code averaged e.g. -60°
        # (real short-axis direction) with 0° (fallback) and produced an
        # arbitrary tilted angle around -30°.  With Yaw-TCP compensation this
        # also generated a large, wrong XY correction.
        orientation_flags = [
            bool(item.get("orientation_reliable", False))
            for item in valid
        ]
        reliable_count = int(sum(orientation_flags))

        if 0 < reliable_count < votes:
            return StableTopPick(
                ready=False,
                votes=votes,
                window_size=self.window_size,
                grasp_pixel=None,
                x_mm=None,
                y_mm=None,
                yaw_deg=None,
                grasp_width_mm=None,
                preopen_mm=None,
                confidence=0.0,
                clearance_px=0.0,
                class_name=None,
                reason=(
                    "orientation_reliability_mixed:"
                    f"{reliable_count}/{votes}"
                ),
            )

        # If all frames are reliable, average the measured dynamic Yaw.
        # If none are reliable (square/circular/ambiguous object), every frame
        # carries the configured fallback Yaw; averaging those identical values
        # is intentional and deterministic.
        yaw_values = [float(item.get("yaw_deg", 0.0)) for item in valid]
        stable_yaw_deg = axial_mean_deg(yaw_values)
        yaw_deviations = np.asarray(
            [abs(axial_difference_deg(value, stable_yaw_deg)) for value in yaw_values],
            dtype=np.float64,
        )
        max_yaw_deviation = float(np.max(yaw_deviations))
        if max_yaw_deviation > self.max_yaw_spread_deg:
            return StableTopPick(
                ready=False,
                votes=votes,
                window_size=self.window_size,
                grasp_pixel=None,
                x_mm=None,
                y_mm=None,
                yaw_deg=None,
                grasp_width_mm=None,
                preopen_mm=None,
                confidence=0.0,
                clearance_px=0.0,
                class_name=None,
                reason=f"yaw_spread_too_large:{max_yaw_deviation:.1f}deg",
            )

        grasp_pixels = np.asarray(
            [
                [
                    float(item["grasp_x"]),
                    float(item["grasp_y"]),
                ]
                for item in valid
            ],
            dtype=np.float64,
        )

        median_pixel = np.median(
            grasp_pixels,
            axis=0,
        )

        confidence = float(
            np.median(
                [
                    float(
                        item.get(
                            "confidence",
                            0.0,
                        )
                    )
                    for item in valid
                ]
            )
        )

        clearance = float(
            np.median(
                [
                    float(
                        item.get(
                            "clearance",
                            0.0,
                        )
                    )
                    for item in valid
                ]
            )
        )

        width_values = [
            float(item["grasp_width_mm"])
            for item in valid
            if item.get("grasp_width_mm") is not None
            and np.isfinite(float(item["grasp_width_mm"]))
        ]
        stable_grasp_width_mm = (
            float(np.median(width_values)) if width_values else None
        )

        preopen_values = [
            float(item["preopen_mm"])
            for item in valid
            if item.get("preopen_mm") is not None
            and np.isfinite(float(item["preopen_mm"]))
        ]
        stable_preopen_mm = (
            float(np.median(preopen_values)) if preopen_values else None
        )

        class_names = [
            str(
                item.get(
                    "class_name",
                    "",
                )
            )
            for item in valid
        ]

        class_name = (
            Counter(class_names)
            .most_common(1)[0][0]
            if class_names
            else None
        )

        return StableTopPick(
            ready=True,
            votes=votes,
            window_size=self.window_size,
            grasp_pixel=(
                int(round(median_pixel[0])),
                int(round(median_pixel[1])),
            ),
            x_mm=float(median_xy[0]),
            y_mm=float(median_xy[1]),
            yaw_deg=float(stable_yaw_deg),
            grasp_width_mm=stable_grasp_width_mm,
            preopen_mm=stable_preopen_mm,
            confidence=confidence,
            clearance_px=clearance,
            class_name=class_name,
            reason="ready",
        )
