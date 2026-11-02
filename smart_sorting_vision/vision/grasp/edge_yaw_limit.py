"""靠近电机一侧时，按真实舵机命令角筛选抓取姿态。

危险边界在哪条轴上由 ``boundary_axis`` 决定，必须与下位机
``Edge_Motion_Config::Boundary_Axis_Y`` 保持一致：
    "y" = 电机在 y=0 一侧（图像下边沿，现场实测）
    "x" = 电机在 x=0 一侧（旧设计）
"""
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
    workspace_clamp_mm: float = 5.0
    min_side_gap_mm: float = -1.0
    # 危险边界所在的轴：电机在 y=0 一侧（图像下边沿）时为 "y"。
    # 放在最后，保持既有按位置构造（EdgeYawLimit(True, -40, 80, ...)）不失效。
    boundary_axis: str = "x"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("edge_yaw_limit.enabled 必须为 true/false")
        if str(self.boundary_axis).lower() not in ("x", "y"):
            raise ValueError("edge_yaw_limit.boundary_axis 必须为 'x' 或 'y'")
        for field in fields(self):
            if field.name in ("enabled", "boundary_axis"):
                continue
            if not math.isfinite(float(getattr(self, field.name))):
                raise ValueError(f"edge_yaw_limit.{field.name} 必须为有限数值")
        if not self.servo_min_deg <= self.preferred_servo_deg <= self.servo_max_deg:
            raise ValueError("边缘舵机角必须满足 servo_min_deg <= preferred_servo_deg <= servo_max_deg")
        if self.sample_step_deg <= 0:
            raise ValueError("edge_yaw_limit.sample_step_deg 必须大于0")
        if self.workspace_clamp_mm < 0:
            raise ValueError("edge_yaw_limit.workspace_clamp_mm 必须非负")
        if not math.isfinite(self.min_side_gap_mm):
            raise ValueError("edge_yaw_limit.min_side_gap_mm 必须为有限数值")

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
        required_names = names - {
            "workspace_clamp_mm",
            "min_side_gap_mm",
            "boundary_axis",
        }
        if enabled and required_names - set(config):
            raise ValueError(f"启用边缘Yaw限制需要填写：{', '.join(sorted(required_names - set(config)))}")
        values = {
            key: (value if key in ("enabled", "boundary_axis") else float(value))
            for key, value in config.items()
        }
        return cls(**values)

    @property
    def boundary_is_y(self) -> bool:
        return str(self.boundary_axis).lower() == "y"

    def guarded(self, x_mm: float, y_mm: float) -> float:
        """返回"边界轴"上的坐标值。

        ``active`` / ``allows`` / ``clamp_workspace_x`` 都作用在这个值上，
        调用方不需要各自判断轴，避免上位机与下位机把边界记在不同轴上。
        """
        value = float(y_mm) if self.boundary_is_y else float(x_mm)
        if not math.isfinite(value):
            raise ValueError("用于边缘Yaw判断的边界轴坐标必须为有限数值")
        return value

    def active(self, boundary_mm: float) -> bool:
        """boundary_mm 是边界轴上的坐标（见 guarded）。"""
        x = float(boundary_mm)
        if not math.isfinite(x):
            raise ValueError("用于边缘Yaw判断的边界轴坐标必须为有限数值")
        # 机械坐标是负向的：边缘区明确表示 threshold < 边界轴坐标 <= 0，
        # 例如 threshold=-40 时就是 -40 < 坐标 <= 0；<= -40 不限角。
        return self.enabled and self.x_threshold_mm < x <= 0.0

    def clamp_workspace_x(self, x_mm: float, workspace_max_mm: float = 0.0) -> tuple[float, bool]:
        """只把靠近上边界的小标定越界压回边界，不吞掉明显错误坐标。"""
        x = float(x_mm)
        limit = float(workspace_max_mm)
        if (
            self.enabled
            and math.isfinite(x)
            and math.isfinite(limit)
            and x > limit
            and x <= limit + self.workspace_clamp_mm
        ):
            return limit, True
        return x, False

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
