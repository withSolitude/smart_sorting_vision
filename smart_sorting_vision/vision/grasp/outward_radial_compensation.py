"""抓取点向外径向补偿：只调整位置，不修改轮廓或抓取方向。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OutwardRadialCompensator:
    enabled: bool = False
    gain: float = 0.0
    max_shift_px: float = 0.0
    center_roi_px: tuple[float, float] | None = None

    @staticmethod
    def validate_config(config: dict | None) -> tuple[bool, float, float, tuple | None]:
        if config is None:
            return False, 0.0, 0.0, None
        if not isinstance(config, dict):
            raise ValueError("outward_radial_compensation 必须是配置对象")
        enabled = config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("outward_radial_compensation.enabled 必须为 true/false")
        if enabled and ("gain" not in config or "max_shift_px" not in config):
            raise ValueError("启用向外补偿需要填写 gain 和 max_shift_px")
        gain = float(config.get("gain", 0.0))
        maximum = float(config.get("max_shift_px", 0.0))
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("outward_radial_compensation.gain 必须为非负有限数值，只允许向外补偿")
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("outward_radial_compensation.max_shift_px 必须为非负有限数值")
        center = config.get("center_full_px")
        if center is not None:
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError("center_full_px 必须为完整去畸变图的 [x, y]，或 null")
            center = tuple(float(v) for v in center)
            if not all(math.isfinite(v) for v in center):
                raise ValueError("center_full_px 必须是有限坐标")
        return enabled, gain, maximum, center

    @classmethod
    def from_config(
        cls,
        config: dict | None,
        *,
        camera_config: dict | None,
        intrinsics_path: str | Path | None,
    ) -> "OutwardRadialCompensator":
        enabled, gain, maximum, center = cls.validate_config(config)
        if not enabled:
            return cls(False, gain, maximum)
        if camera_config is None:
            raise ValueError("向外补偿需要同一份 camera.top 配置，以确定ROI坐标原点")
        x1, y1, x2, y2 = map(int, camera_config["safe_roi"])
        width, height = int(camera_config["width"]), int(camera_config["height"])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("向外补偿的 safe_roi 必须位于完整去畸变图内")
        if center is None:
            if intrinsics_path is None:
                raise ValueError("自动补偿中心需要 calibration.top_intrinsics 标定文件")
            # 当前TopCamera.read()已经去畸变，不能混用原始camera_matrix或ROI中心。
            with np.load(intrinsics_path, allow_pickle=False) as data:
                if "new_camera_matrix" not in data:
                    raise ValueError("内参文件缺少 new_camera_matrix，不能确定去畸变图的补偿中心")
                matrix = np.asarray(data["new_camera_matrix"], dtype=float)
                if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                    raise ValueError("new_camera_matrix 必须为有限的3×3矩阵")
                if "image_size" in data and tuple(np.asarray(data["image_size"]).ravel()) != (width, height):
                    raise ValueError("向外补偿内参分辨率与 camera.top 的完整图分辨率不一致")
                center = float(matrix[0, 2]), float(matrix[1, 2])
        # 中心始终保存于完整去畸变图；运行时减当前ROI原点，改ROI不需重填中心。
        return cls(True, gain, maximum, (center[0] - x1, center[1] - y1))

    def apply(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("待补偿抓取点必须为有限坐标")
        if not self.enabled:
            return x, y
        cx, cy = self.center_roi_px
        dx, dy = x - cx, y - cy
        distance = math.hypot(dx, dy)
        if distance == 0:
            return x, y
        # 所有比例均非负：限幅只缩短向外位移，绝不会生成向内位移。
        scale = min(self.gain, self.max_shift_px / distance)
        return x + scale * dx, y + scale * dy
