// 由视觉端公式配置生成；不再使用开口查表。
#ifndef CLAW_PREOPEN_PROFILE_H
#define CLAW_PREOPEN_PROFILE_H
#include <cmath>
namespace ClawPreopen {
static constexpr bool kConfigured = true;
static constexpr float kClosedAngleDeg = 0.710004610f;
static constexpr float kStandbyAngleDeg = -89.283079760f;
// static constexpr float kStandbyAngleDeg = -69.41f;
static constexpr float kMinGapMm = 13.000000000f;
static constexpr float kMaxGapMm = 77.000000000f;
inline bool ToAngle(float gap_mm, float &angle_deg) {
    if (!kConfigured || !std::isfinite(gap_mm) ||
        gap_mm < kMinGapMm || gap_mm > kMaxGapMm ||
        kStandbyAngleDeg >= kClosedAngleDeg) return false;
    const float angle = kClosedAngleDeg +
        (kStandbyAngleDeg - kClosedAngleDeg) * gap_mm / kMaxGapMm;
    if (!std::isfinite(angle) || angle < kStandbyAngleDeg ||
        angle >= kClosedAngleDeg) return false;
    angle_deg = angle;
    return true;
}
} // namespace ClawPreopen
#endif
