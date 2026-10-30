"""夹爪预开口的统一线性公式。

PC 端只计算并发送目标净开口(mm)，STM32 端使用同一组参数换算舵机角度：

    angle = closed + (standby - closed) * opening / max_opening

这样调试时只需要修改闭合角、待机角和最大开口三个参数，不再维护 CAD 查表。
"""
from __future__ import annotations

import math


class ClawPreopenProfile:
    """根据夹爪开口公式计算预开口命令。"""

    def __init__(self, data: dict):
        self.kind = "linear"
        self.margin_mm = float(data["margin_mm"])
        self.min_command_mm = float(data["min_command_mm"])
        self.max_command_mm = float(data["max_command_mm"])
        self.closed_servo_deg = float(data["closed_servo_deg"])
        self.standby_servo_deg = float(data["standby_servo_deg"])
        self._validate()

    def _validate(self) -> None:
        values = (
            self.margin_mm,
            self.min_command_mm,
            self.max_command_mm,
            self.closed_servo_deg,
            self.standby_servo_deg,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("预开口公式参数必须是有限数值")
        if self.margin_mm <= 0.0:
            raise ValueError("margin_mm 必须大于0")
        if self.min_command_mm <= 0.0 or self.max_command_mm <= self.min_command_mm:
            raise ValueError("预开口范围无效")
        if self.max_command_mm > 120.0:
            raise ValueError("max_command_mm 不能大于120mm")
        if not -135.0 <= self.standby_servo_deg < self.closed_servo_deg <= 135.0:
            raise ValueError("舵机闭合角/待机角范围无效")
        if self.max_command_mm <= self.margin_mm:
            raise ValueError("margin_mm 过大，没有可用物体宽度")

    @classmethod
    def from_runtime_config(cls, data: dict) -> "ClawPreopenProfile":
        """从 runtime_config.yaml 创建公式配置。"""
        if data.get("configured") is False:
            raise ValueError("预开口公式尚未配置")
        try:
            return cls(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"预开口公式配置无效: {exc}") from exc

    def command_for_width(self, width_mm: float) -> float:
        """物体宽度加余量，输出协议使用的净开口(mm)。"""
        width = float(width_mm)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("物体宽度无效")
        command = max(width + self.margin_mm, self.min_command_mm)
        command = math.ceil(command * 10.0) / 10.0
        if command > self.max_command_mm:
            raise ValueError(
                f"目标开口 {command:.1f}mm 超过上限 {self.max_command_mm:.1f}mm"
            )
        return command

    def angle_for_gap(self, gap_mm: float) -> float:
        """使用与 STM32 一致的线性公式计算舵机角度。"""
        gap = float(gap_mm)
        if not math.isfinite(gap) or not self.min_command_mm <= gap <= self.max_command_mm:
            raise ValueError("开口超出配置范围")
        angle = self.closed_servo_deg + (
            self.standby_servo_deg - self.closed_servo_deg
        ) * gap / self.max_command_mm
        if not math.isfinite(angle) or not -135.0 <= angle <= 135.0:
            raise ValueError("公式计算出的舵机角度无效")
        return angle


def firmware_header(data: dict) -> str:
    """生成与 PC 参数一致的精简 STM32 公式头文件。"""
    profile = ClawPreopenProfile.from_runtime_config(data)
    return f'''// 由视觉端公式配置生成；不再使用开口查表。
#ifndef CLAW_PREOPEN_PROFILE_H
#define CLAW_PREOPEN_PROFILE_H
#include <cmath>
namespace ClawPreopen {{
static constexpr bool kConfigured = true;
static constexpr float kClosedAngleDeg = {profile.closed_servo_deg:.9f}f;
static constexpr float kStandbyAngleDeg = {profile.standby_servo_deg:.9f}f;
static constexpr float kMinGapMm = {profile.min_command_mm:.9f}f;
static constexpr float kMaxGapMm = {profile.max_command_mm:.9f}f;
inline bool ToAngle(float gap_mm, float &angle_deg) {{
    if (!kConfigured || !std::isfinite(gap_mm) ||
        gap_mm < kMinGapMm || gap_mm > kMaxGapMm ||
        kStandbyAngleDeg >= kClosedAngleDeg) return false;
    const float angle = kClosedAngleDeg +
        (kStandbyAngleDeg - kClosedAngleDeg) * gap_mm / kMaxGapMm;
    if (!std::isfinite(angle) || angle < kStandbyAngleDeg ||
        angle >= kClosedAngleDeg) return false;
    angle_deg = angle;
    return true;
}}
}} // namespace ClawPreopen
#endif
'''
