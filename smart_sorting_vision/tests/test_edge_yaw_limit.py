"""边缘抓取姿态限制回归：仅测标定和候选角，不加载模型或连接机械硬件。"""
from __future__ import annotations

import math
import unittest

from vision.calibration.yaw_servo import YawServoCalibration
from vision.grasp.edge_yaw_limit import EdgeYawLimit


def yaw_calibration(slope=1.04720792):
    return YawServoCalibration(
        reference_axis_deg=-63.280987,
        reference_servo_deg=0.232430,
        servo_deg_per_axis_deg=slope,
        safe_command_min_deg=-90.0,
        safe_command_max_deg=90.0,
    )


class EdgeYawLimitTests(unittest.TestCase):
    config = {
        "enabled": True, "x_threshold_mm": -20.0,
        "servo_min_deg": 40.0, "servo_max_deg": 80.0,
        "preferred_servo_deg": 60.0, "sample_step_deg": 5.0,
    }

    def setUp(self):
        self.limit = EdgeYawLimit.from_config(self.config)
        self.yaw = yaw_calibration()

    def axis_for_servo(self, servo, calibration=None):
        cal = calibration or self.yaw
        physical = cal.reference_axis_deg + (servo - cal.reference_servo_deg) / cal.servo_deg_per_axis_deg
        return (physical + 90.0) % 180.0 - 90.0

    def test_x_threshold_is_strict_and_safe_side_preserves_original_angles(self):
        self.assertFalse(self.limit.active(-20.0))
        self.assertFalse(self.limit.active(-20.01))
        self.assertTrue(self.limit.active(-19.99))
        axis = self.axis_for_servo(-30)
        self.assertTrue(self.limit.allows(-20.0, axis, self.yaw))
        self.assertTrue(self.limit.allows(-80.0, axis, self.yaw))
        self.assertFalse(self.limit.allows(-19.99, axis, self.yaw))

    def test_uses_real_servo_angle_and_accepts_both_endpoints(self):
        for servo in (40.0, 60.0, 80.0):
            with self.subTest(servo=servo):
                axis = self.axis_for_servo(servo)
                self.assertTrue(self.limit.allows(-10.0, axis, self.yaw))
                self.assertNotAlmostEqual(axis, servo)
        for servo in (39.0, 81.0):
            self.assertFalse(self.limit.allows(-10.0, self.axis_for_servo(servo), self.yaw))
        # 直接拿协议40°判定会把实际约-80.1°误放行。
        self.assertFalse(self.limit.allows(-10.0, 40.0, self.yaw))

    def test_candidates_include_endpoints_and_prefer_servo_60(self):
        axes = self.limit.candidate_axes(self.yaw)
        servos = [self.yaw.command_deg(axis) for axis in axes]
        self.assertEqual(len(servos), 9)
        self.assertAlmostEqual(servos[0], 60.0)
        self.assertEqual(sorted(round(value, 7) for value in servos), list(range(40, 81, 5)))
        self.assertTrue(all(-90.0 <= axis < 90.0 for axis in axes))
        self.assertTrue(all(self.limit.allows(-10.0, axis, self.yaw) for axis in axes))

    def test_quantized_axis_must_be_rechecked_at_servo_window_edges(self):
        # 协议按1°量化，边界采样反解后再舍入可能使舵机角越过允许窗口。
        cal = YawServoCalibration(0.0, 0.4, 1.0, -90.0, 90.0)
        axes = self.limit.candidate_axes(cal)
        endpoint = next(axis for axis in axes if math.isclose(cal.command_deg(axis), 80.0))
        self.assertAlmostEqual(endpoint, 79.6)
        self.assertTrue(self.limit.allows(-10.0, endpoint, cal))
        self.assertFalse(self.limit.allows(-10.0, round(endpoint), cal))

    def test_negative_slope_preserves_requested_servo_angles(self):
        cal = yaw_calibration(-1.04720792)
        axes = self.limit.candidate_axes(cal)
        self.assertAlmostEqual(cal.command_deg(axes[0]), 60.0)
        self.assertEqual(sorted(round(cal.command_deg(axis), 7) for axis in axes), list(range(40, 81, 5)))
        for axis in axes:
            self.assertTrue(self.limit.allows(-10.0, axis, cal))

    def test_calibrated_travel_and_unique_execution_interval_are_not_bypassed(self):
        restricted = YawServoCalibration(-63.280987, 0.232430, 1.04720792, -50.0, 65.0)
        candidates = self.limit.candidate_axes(restricted)
        self.assertTrue(candidates)
        self.assertTrue(all(restricted.command_deg(axis) <= 65.0 for axis in candidates))
        self.assertFalse(self.limit.allows(-100.0, 20.0, restricted))
        # 执行区间是[-90,90)，90°反解会绕到-90°，必须剔除而不是产生另一分支。
        limit = EdgeYawLimit(True, -20, 80, 100, 90, 5)
        cal = YawServoCalibration(0.0, 0.0, 1.0, -135.0, 135.0)
        self.assertEqual([cal.command_deg(axis) for axis in limit.candidate_axes(cal)], [85.0, 80.0])

    def test_non_aligned_step_still_includes_min_max_and_preferred(self):
        limit = EdgeYawLimit.from_config({**self.config, "sample_step_deg": 13.0})
        servos = [self.yaw.command_deg(axis) for axis in limit.candidate_axes(self.yaw)]
        self.assertAlmostEqual(servos[0], 60.0)
        for expected in (40, 80, 60):
            self.assertTrue(any(math.isclose(value, expected) for value in servos))

    def test_legacy_missing_or_disabled_config_is_compatible(self):
        for config in (None, {"enabled": False}):
            limit = EdgeYawLimit.from_config(config)
            self.assertFalse(limit.active(0.0))
            self.assertEqual(limit.candidate_axes(self.yaw), [])
            self.assertTrue(limit.allows(0.0, self.axis_for_servo(-30.0), self.yaw))

    def test_invalid_config_is_rejected(self):
        invalid = [
            "enabled", {"enabled": True}, {**self.config, "enabled": "true"},
            {**self.config, "servo_min_deg": 65.0}, {**self.config, "servo_max_deg": 55.0},
            {**self.config, "preferred_servo_deg": 100.0}, {**self.config, "sample_step_deg": 0.0},
            {**self.config, "sample_step_deg": -1.0}, {**self.config, "x_threshold_mm": float("nan")},
            {**self.config, "servo_max_deg": float("inf")}, {**self.config, "unknown": 1},
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises((TypeError, ValueError)):
                EdgeYawLimit.from_config(config)

    def test_nonfinite_position_or_yaw_is_not_accepted(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.limit.active(value)
                self.assertFalse(self.limit.allows(value, 0.0, self.yaw))
                self.assertFalse(self.limit.allows(0.0, value, self.yaw))


if __name__ == "__main__":
    unittest.main()
