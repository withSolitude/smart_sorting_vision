"""单个 180° 抓取区间的 Yaw 标定；所有数值由运行 YAML 提供。"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class YawServoCalibration:
    reference_axis_deg: float
    reference_servo_deg: float
    servo_deg_per_axis_deg: float
    safe_command_min_deg: float
    safe_command_max_deg: float
    safe_command_tolerance_deg: float = 2.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(v)) for v in vars(self).values()):
            raise ValueError("Yaw 标定参数必须为有限数值")
        if abs(self.servo_deg_per_axis_deg) < 1e-6:
            raise ValueError("Yaw 标定比例不能为零")
        if self.safe_command_tolerance_deg < 0.0:
            raise ValueError("Yaw 安全角度容差不能为负数")
        if not -135.0 <= self.safe_command_min_deg < self.safe_command_max_deg <= 135.0:
            raise ValueError("Yaw 舵机安全区间必须位于 [-135, 135] 度内")

    @classmethod
    def from_config(cls, data: dict) -> "YawServoCalibration":
        values = {key: float(data[key]) for key in cls.__dataclass_fields__ if key in data}
        return cls(**values)

    @property
    def physical_min_deg(self) -> float:
        return self.reference_axis_deg - 90.0

    @property
    def physical_max_deg(self) -> float:
        return self.reference_axis_deg + 90.0

    def physical_axis_deg(self, axial_deg: float) -> float:
        """同一闭合轴只映射到参考轴附近唯一的 180° 区间，右端点不取。"""
        if not math.isfinite(float(axial_deg)):
            raise ValueError("抓取轴角度必须为有限数值")
        delta = (float(axial_deg) - self.reference_axis_deg + 90.0) % 180.0 - 90.0
        return self.reference_axis_deg + delta

    def command_deg(self, axial_deg: float) -> float:
        physical = self.physical_axis_deg(axial_deg)
        command = self.reference_servo_deg + self.servo_deg_per_axis_deg * (
            physical - self.reference_axis_deg
        )
        # 标定拟合可能略超边界；运行时始终压到配置的最大物理角，不能因此终止抓取。
        return min(self.safe_command_max_deg, max(self.safe_command_min_deg, command))

    def safe_wire_axis_deg(self, axial_deg: float) -> float:
        """将小拟合误差压回安全舵机角，再按协议1°量化并复核。"""
        physical = self.physical_axis_deg(axial_deg)
        raw_command = self.reference_servo_deg + self.servo_deg_per_axis_deg * (
            physical - self.reference_axis_deg
        )
        if raw_command < self.safe_command_min_deg:
            target_command = self.safe_command_min_deg
            target_physical = self.reference_axis_deg + (
                target_command - self.reference_servo_deg
            ) / self.servo_deg_per_axis_deg
        elif raw_command > self.safe_command_max_deg:
            target_command = self.safe_command_max_deg
            target_physical = self.reference_axis_deg + (
                target_command - self.reference_servo_deg
            ) / self.servo_deg_per_axis_deg
        else:
            target_physical = physical

        base = int(round(target_physical))
        direction = -1 if self.servo_deg_per_axis_deg > 0 else 1
        for offset in (0, direction, -direction, 2 * direction, -2 * direction):
            candidate = self.reference_axis_deg + 0.0 + (base + offset - self.reference_axis_deg)
            candidate = (candidate - self.reference_axis_deg + 90.0) % 180.0 - 90.0 + self.reference_axis_deg
            command = self.reference_servo_deg + self.servo_deg_per_axis_deg * (
                candidate - self.reference_axis_deg
            )
            if self.safe_command_min_deg <= command <= self.safe_command_max_deg:
                return float(candidate)
        raise ValueError("Yaw量化后仍无法落入安全舵机范围")
