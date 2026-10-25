"""INNER gap profile from measurements or CAD with a known claw servo datum.

CAD samples are theoretical, not hardware-validated. Yaw/TCP are independent.
"""
from __future__ import annotations

import math


class ClawPreopenProfile:
    def __init__(self, data: dict):
        measured = data.get("schema_version") == 1 and data.get("calibrated") is True
        modeled = (data.get("schema_version") == 2 and data.get("profile_kind") == "cad_four_bar"
                   and data.get("configured") is True)
        if not (measured or modeled):
            raise ValueError("New claw preopen profile is not configured")
        self.kind = "cad_four_bar" if modeled else "measured"
        if modeled:
            closed, standby = float(data["closed_servo_deg"]), float(data["standby_servo_deg"])
            if not all(math.isfinite(v) for v in (closed, standby)) or not -135 <= standby < closed <= 135:
                raise ValueError("Invalid claw servo datum/standby range")
        self.margin_mm = float(data["margin_mm"])
        if not math.isfinite(self.margin_mm) or self.margin_mm <= 0:
            raise ValueError("margin_mm must be finite and positive")
        self.points = sorted(
            (float(p["gap_mm"]), float(p["servo_deg"])) for p in data["samples"]
        )
        if not 3 <= len(self.points) <= 512:
            raise ValueError("Need 3..512 preopen lookup positions")
        for gap, angle in self.points:
            if not math.isfinite(gap) or not 0 < gap <= 120:
                raise ValueError("Each inner gap must be finite and in (0,120] mm")
            if not math.isfinite(angle) or not -135 <= angle <= 135:
                raise ValueError("Each servo angle must be finite and in [-135,135] deg")
            if modeled and not standby <= angle < closed:
                raise ValueError("CAD preopen angle outside the configured operating interval")
        direction = self.points[-1][1] - self.points[0][1]
        if modeled and direction >= 0:
            raise ValueError("CAD claw opens as the servo command decreases")
        for (g0, a0), (g1, a1) in zip(self.points, self.points[1:]):
            if g1 - g0 < 1e-6 or (a1 - a0) * direction <= 0 or abs(a1 - a0) < 1e-6:
                raise ValueError("Gap and servo lookup positions must be strictly monotonic and distinct")
        # Outward-only rounding on the existing 0.1 mm wire protocol.
        self.min_command_mm = math.ceil(self.points[0][0] * 10) / 10
        self.max_command_mm = math.floor(self.points[-1][0] * 10) / 10
        if self.max_command_mm <= max(self.min_command_mm, self.margin_mm):
            raise ValueError("Configured preopen range is too small")

    @classmethod
    def from_runtime_config(cls, data: dict) -> "ClawPreopenProfile":
        """Build the production PC-side limits directly from runtime_config.yaml.

        The PC sends an inner opening in millimetres; only the MCU needs the
        dense opening-to-servo table compiled into claw_preopen_profile.h.
        """
        try:
            if data.get("configured") is not True:
                raise ValueError("configured must be true")
            kind = str(data.get("profile_kind", ""))
            if kind not in ("cad_four_bar", "measured"):
                raise ValueError("profile_kind must be cad_four_bar or measured")
            margin = float(data["margin_mm"])
            minimum = float(data["min_command_mm"])
            maximum = float(data["max_command_mm"])
            closed = float(data["closed_servo_deg"])
            standby = float(data["standby_servo_deg"])
            if not all(math.isfinite(v) for v in (margin, minimum, maximum, closed, standby)):
                raise ValueError("all numeric values must be finite")
            if margin <= 0:
                raise ValueError("margin_mm must be positive")
            if minimum <= 0 or maximum <= minimum or maximum > 120:
                raise ValueError("invalid command range")
            if maximum <= margin:
                raise ValueError("margin leaves no usable object width")
            if not -135 <= standby <= 135 or not -135 <= closed <= 135 or standby == closed:
                raise ValueError("invalid claw closed/standby angles")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid vision.top.grasp_orientation.dynamic_preopen: {exc}") from exc
        profile = cls.__new__(cls)
        profile.kind = kind
        profile.margin_mm = margin
        profile.min_command_mm = math.ceil(minimum * 10) / 10
        profile.max_command_mm = math.floor(maximum * 10) / 10
        profile.closed_servo_deg = closed
        profile.standby_servo_deg = standby
        profile.points = []
        return profile

    def command_for_width(self, width_mm: float) -> float:
        width = float(width_mm)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("invalid object width")
        required = max(width + self.margin_mm, self.min_command_mm)
        command = math.ceil(required * 10) / 10
        if command > self.max_command_mm:
            raise ValueError(f"required {command:.1f} mm exceeds configured opening {self.max_command_mm:.1f} mm")
        return command

    def angle_for_gap(self, gap_mm: float) -> float:
        if not self.points:
            raise ValueError("PC runtime has no servo lookup; conversion is compiled into the MCU header")
        gap = float(gap_mm)
        if not math.isfinite(gap) or not self.min_command_mm <= gap <= self.max_command_mm:
            raise ValueError("gap outside configured command range; extrapolation forbidden")
        for (g0, a0), (g1, a1) in zip(self.points, self.points[1:]):
            if g0 <= gap <= g1:
                return a0 + (a1 - a0) * (gap - g0) / (g1 - g0)
        raise ValueError("gap not bracketed by lookup positions")


def firmware_header(data: dict) -> str:
    """Emit the SAME lookup into standalone C++11 code for the MCU."""
    configured = data.get("calibrated") is True or data.get("configured") is True
    profile = ClawPreopenProfile(data) if configured else None
    pts = profile.points if profile else [(0.0, 0.0)]
    entries = "\n".join("    {%.9ff, %.9ff}," % (g, a) for g, a in pts)
    return '''// Generated claw INNER opening lookup. CAD data is theoretical, not measured.
// Not yaw/TCP parameters. Install together with the matching PC JSON.
#ifndef CLAW_PREOPEN_PROFILE_H
#define CLAW_PREOPEN_PROFILE_H
#include <cmath>
#include <cstddef>
namespace ClawPreopen {
struct Point { float gap_mm; float servo_deg; };
static constexpr bool kConfigured = %s;
static constexpr bool kCadModel = %s;
static constexpr float kClosedAngleDeg = %.9ff;
static constexpr float kStandbyAngleDeg = %.9ff;
static constexpr Point kPoints[] = {
%s
};
static constexpr std::size_t kCount = sizeof(kPoints) / sizeof(kPoints[0]);
static constexpr float kMinGapMm = %.9ff;
static constexpr float kMaxGapMm = %.9ff;
inline bool ToAngle(float gap_mm, float &angle_deg) {
    if (!kConfigured || kCount < 3 || !std::isfinite(gap_mm) ||
        gap_mm < kMinGapMm || gap_mm > kMaxGapMm) return false;
    const float direction = kPoints[kCount - 1].servo_deg - kPoints[0].servo_deg;
    for (std::size_t i = 0; i < kCount; ++i) {
        const Point &p = kPoints[i];
        if (!std::isfinite(p.gap_mm) || p.gap_mm <= 0 || p.gap_mm > 120 ||
            !std::isfinite(p.servo_deg) || p.servo_deg < -135 || p.servo_deg > 135)
            return false;
        if (kCadModel && (p.servo_deg < kStandbyAngleDeg ||
            p.servo_deg >= kClosedAngleDeg || direction >= 0)) return false;
        if (i && (p.gap_mm <= kPoints[i-1].gap_mm ||
            (p.servo_deg - kPoints[i-1].servo_deg) * direction <= 0)) return false;
    }
    for (std::size_t i = 1; i < kCount; ++i) {
        if (gap_mm <= kPoints[i].gap_mm) {
            const Point &a = kPoints[i-1], &b = kPoints[i];
            const float candidate = a.servo_deg + (b.servo_deg - a.servo_deg) *
                (gap_mm - a.gap_mm) / (b.gap_mm - a.gap_mm);
            if (!std::isfinite(candidate)) return false;
            angle_deg = candidate;
            return true;
        }
    }
    return false;
}
} // namespace ClawPreopen
#endif
''' % ("true" if configured else "false", "true" if data.get("profile_kind") == "cad_four_bar" else "false",
       float(data.get("closed_servo_deg", -15)),
       float(data.get("standby_servo_deg", -95)), entries,
       profile.min_command_mm if profile else 0, profile.max_command_mm if profile else 0)
