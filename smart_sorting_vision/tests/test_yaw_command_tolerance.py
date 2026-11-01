"""Yaw拟合小误差压限回归。"""
import unittest

from vision.calibration.yaw_servo import YawServoCalibration


class YawCommandToleranceTests(unittest.TestCase):
    def setUp(self):
        self.cal = YawServoCalibration(
            -63.280987, 0.232430, 1.047207920,
            -96.0, 96.0, 6.0,
        )

    def test_small_positive_overrun_is_clamped_and_quantized_inside_limit(self):
        # 23°协议轴约对应90.58°舵机命令，在±96°范围内可以直接执行。
        self.assertAlmostEqual(self.cal.command_deg(23.0), 90.58, places=1)
        safe_axis = self.cal.safe_wire_axis_deg(23.0)
        self.assertEqual(round(safe_axis), 23)
        self.assertLessEqual(self.cal.command_deg(round(safe_axis)), 96.0)

    def test_large_overrun_is_clamped_to_96(self):
        cal = YawServoCalibration(0.0, 0.0, 2.0, -96.0, 96.0, 6.0)
        self.assertAlmostEqual(cal.command_deg(49.0), 96.0)
        safe_axis = cal.safe_wire_axis_deg(49.0)
        self.assertLessEqual(cal.command_deg(round(safe_axis)), 96.0)


if __name__ == "__main__":
    unittest.main()
