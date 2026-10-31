"""靠近 X=0 一侧时，按真实舵机命令角筛选抓取姿态。"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

from vision.calibration.yaw_servo import YawServoCalibration


@dataclass(frozen=True)
class EdgeYawLimit:
    # 默认关闭兼容旧入口；启用后的参数全部从 YAML 传入。
    enabled: bool = False
    x_threshold_mm: float = -20.0
    servo_min_deg: float = 40.0
    servo_max_deg: float = 80.0
    preferred_servo_deg: float = 60.0
    sample_step_deg: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("edge_yaw_limit.enabled 必须为 true/false")
        for field in fields(self):
            if field.name != "enabled" and not math.isfinite(float(getattr(self, field.name))):
                raise ValueError(f"edge_yaw_limit.{field.name} 必须为有限数值")
        if not self.servo_min_deg <= self.preferred_servo_deg <= self.servo_max_deg:
            raise ValueError("边缘舵机角必须满足 servo_min_deg <= preferred_servo_deg <= servo_max_deg")
        if self.sample_step_deg <= 0:
            raise ValueError("edge_yaw_limit.sample_step_deg 必须大于0")

    @classmethod
    def from_config(cls, config: dict | None) -> "EdgeYawLimit":
        if config is None:
            return cls()
        if not isinstance(config, dict):
            raise ValueError("edge_yaw_limit 必须是配置对象")
        names = {field.name for field in fields(cls)}
        if set(config) - names:
            raise ValueError(f"edge_yaw_limit 存在未知参数：{', '.join(sorted(set(config) - names))}")
        enabled = config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("edge_yaw_limit.enabled 必须为 true/false")
        if enabled and names - set(config):
            raise ValueError(f"启用边缘Yaw限制需要填写：{', '.join(sorted(names - set(config)))}")
        values = {key: (value if key == "enabled" else float(value)) for key, value in config.items()}
        return cls(**values)

    def active(self, x_mm: float) -> bool:
        x = float(x_mm)
        if not math.isfinite(x):
            raise ValueError("用于边缘Yaw判断的最终X必须为有限数值")
        # x=-20 位于中转边界，严格按照用户指定的 x>-20 才限制角度。
        return self.enabled and x > self.x_threshold_mm

    def allows(self, x_mm: float, axis_yaw_deg: float, calibration: YawServoCalibration) -> bool:
        """使用最终X及量化后的协议角检查，绝不把协议角当成舵机角。"""
        try:
            active = self.active(x_mm)
            servo = calibration.command_deg(float(axis_yaw_deg))
        except (TypeError, ValueError):
            return False
        if not active:
            return True
        # 只容忍浮点反解的微小舍入误差，不扩大实际可用角度窗口。
        return self.servo_min_deg - 1e-7 <= servo <= self.servo_max_deg + 1e-7

    def candidate_axes(self, calibration: YawServoCalibration) -> list[float]:
        """把允许的舵机采样角反解为现有协议角，60°优先；不产生第二个180°分支。

        此处不做协议量化。流水线在1°量化后必须再次调用 allows，防止边界采样
        被舍入到40～80°之外；宽度、预开口和夹指接触检查仍由正式流水线执行。
        """
        if not self.enabled:
            return []
        count = math.floor((self.servo_max_deg - self.servo_min_deg) / self.sample_step_deg)
        servos = [self.servo_min_deg + index * self.sample_step_deg for index in range(count + 1)]
        servos.extend((self.servo_min_deg, self.servo_max_deg, self.preferred_servo_deg))
        servos = sorted(set(servos), key=lambda value: (abs(value - self.preferred_servo_deg), value))
        axes = []
        for requested in servos:
            physical = calibration.reference_axis_deg + (
                requested - calibration.reference_servo_deg
            ) / calibration.servo_deg_per_axis_deg
            axis = (physical + 90.0) % 180.0 - 90.0
            try:
                actual = calibration.command_deg(axis)
            except ValueError:
                continue
            # 反解跨过唯一执行区间时可能绕回另一端，不能当成同一舵机姿态。
            if math.isclose(actual, requested, abs_tol=1e-7, rel_tol=0.0):
                axes.append(float(axis))
        return axes
