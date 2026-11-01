"""Yaw拟合小误差压限回归。"""
import unittest

from vision.calibration.yaw_servo import YawServoCalibration


class YawCommandToleranceTests(unittest.TestCase):
    def setUp(self):
        self.cal = YawServoCalibration(
            -63.280987, 0.232430, 1.047207920,
            -90.0, 90.0, 2.0,
        )

    def test_small_positive_overrun_is_clamped_and_quantized_inside_limit(self):
        # 23°协议轴约对应90.58°舵机命令，安全协议轴应退到22°。
        self.assertAlmostEqual(self.cal.command_deg(23.0), 90.0)
        safe_axis = self.cal.safe_wire_axis_deg(23.0)
        self.assertEqual(round(safe_axis), 22)
        self.assertLessEqual(self.cal.command_deg(round(safe_axis)), 90.0)

    def test_large_overrun_still_fails(self):
        with self.assertRaises(ValueError):
            self.cal.safe_wire_axis_deg(25.0)


if __name__ == "__main__":
    unittest.main()
