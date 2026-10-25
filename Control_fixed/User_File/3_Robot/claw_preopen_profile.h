// 夹爪预开口公式：开口越大，舵机角度越负。
// 只保留少量可标定参数，不再维护数百个 CAD 查表点。
#ifndef CLAW_PREOPEN_PROFILE_H
#define CLAW_PREOPEN_PROFILE_H

#include <cmath>

namespace ClawPreopen
{
// 机械实测后只需要修改这几个参数：
// angle = closed_angle + (standby_angle - closed_angle) * opening_mm / max_opening_mm
static constexpr bool kConfigured = true;
static constexpr float kClosedAngleDeg = -35.0f;
static constexpr float kStandbyAngleDeg = -95.0f;
static constexpr float kMinGapMm = 0.9f;
static constexpr float kMaxGapMm = 46.5f;

inline bool ToAngle(float gap_mm, float &angle_deg)
{
    if (!kConfigured || !std::isfinite(gap_mm) ||
        gap_mm < kMinGapMm || gap_mm > kMaxGapMm ||
        !std::isfinite(kMaxGapMm) || kMaxGapMm <= 0.0f ||
        !std::isfinite(kClosedAngleDeg) || !std::isfinite(kStandbyAngleDeg) ||
        kStandbyAngleDeg >= kClosedAngleDeg)
    {
        return false;
    }

    const float angle = kClosedAngleDeg +
        (kStandbyAngleDeg - kClosedAngleDeg) * gap_mm / kMaxGapMm;
    if (!std::isfinite(angle) || angle < kStandbyAngleDeg ||
        angle >= kClosedAngleDeg)
    {
        return false;
    }

    angle_deg = angle;
    return true;
}
}

#endif
