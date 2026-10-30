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

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(v)) for v in vars(self).values()):
            raise ValueError("Yaw 标定参数必须为有限数值")
        if abs(self.servo_deg_per_axis_deg) < 1e-6:
            raise ValueError("Yaw 标定比例不能为零")
        if not -135.0 <= self.safe_command_min_deg < self.safe_command_max_deg <= 135.0:
            raise ValueError("Yaw 舵机安全区间必须位于 [-135, 135] 度内")

    @classmethod
    def from_config(cls, data: dict) -> "YawServoCalibration":
        return cls(**{key: float(data[key]) for key in cls.__dataclass_fields__})

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
        if not self.safe_command_min_deg <= command <= self.safe_command_max_deg:
            raise ValueError(f"Yaw 舵机目标 {command:.3f}° 超出已配置安全区间")
        return command
