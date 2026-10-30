"""单个 180° 抓取区间与串口协议回归；不连接相机或下位机。"""
from pathlib import Path
import math
import re
import importlib
import unittest

from communication.sorting_protocol import Mode, SortingFrame, SortingProtocol
from runtime.app_config import load_runtime_config
from vision.calibration.yaw_servo import YawServoCalibration


ROOT = Path(__file__).resolve().parents[1]


class SingleYawTests(unittest.TestCase):
    def setUp(self):
        cfg = load_runtime_config(ROOT, ROOT / "config/runtime_config.yaml")
        self.yaw = YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"])

    def test_axial_period_uses_one_physical_window_and_honors_limits(self):
        reachable = 0
        rejected = 0
        for step in range(1801):
            axis = -90.0 + step * 0.1
            physical = self.yaw.physical_axis_deg(axis)
            self.assertGreaterEqual(physical, self.yaw.physical_min_deg - 1e-10)
            self.assertLess(physical, self.yaw.physical_max_deg)
            try:
                command = self.yaw.command_deg(axis)
            except ValueError:
                rejected += 1
                continue
            reachable += 1
            self.assertGreaterEqual(command, self.yaw.safe_command_min_deg)
            self.assertLessEqual(command, self.yaw.safe_command_max_deg)
            for period in (-360.0, -180.0, 180.0, 360.0):
                self.assertAlmostEqual(self.yaw.physical_axis_deg(axis + period), physical)
                self.assertAlmostEqual(self.yaw.command_deg(axis + period), command)
        # 当前实机安全限位为±90°，会拒绝唯一窗口两端的少量超限角度。
        self.assertGreater(reachable, 1600)
        self.assertGreater(rejected, 0)

    def test_reported_minus_82_target_uses_updated_calibration(self):
        # 用户日志中旧参数算到 137.78° 而被拒绝；新标定应约为 -19.16°。
        self.assertAlmostEqual(self.yaw.command_deg(-82.0), -19.370268668183, places=6)

    def test_reference_and_half_open_boundary(self):
        self.assertAlmostEqual(
            self.yaw.command_deg(self.yaw.reference_axis_deg),
            self.yaw.reference_servo_deg,
        )
        self.assertAlmostEqual(
            self.yaw.physical_axis_deg(self.yaw.physical_max_deg),
            self.yaw.physical_min_deg,
        )

    def test_servo_limits_remain_effective(self):
        narrow = YawServoCalibration(
            self.yaw.reference_axis_deg,
            self.yaw.reference_servo_deg,
            self.yaw.servo_deg_per_axis_deg,
            -60.0,
            60.0,
        )
        with self.assertRaises(ValueError):
            narrow.command_deg(narrow.reference_axis_deg + 89.0)
        for invalid in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                self.yaw.command_deg(invalid)

    def test_pick_frame_needs_no_branch_transaction(self):
        frame = SortingFrame(Mode.PICK_TARGET, x_mm=-146.7, y_mm=-88.0, yaw_deg=-82.0)
        recovered = SortingProtocol.unpack(SortingProtocol.pack(frame))
        self.assertEqual(recovered, frame)
        for removed in (0x08, 0x91):
            with self.assertRaises(ValueError):
                Mode(removed)

    def test_mcu_and_yaml_calibration_are_synchronized(self):
        header = ROOT.parent / "Control_fixed/User_File/4_Interaction/ita_robot.h"
        source = header.read_text(encoding="utf-8-sig")
        fields = {
            "Pick_Yaw_Reference_Closing_Axis_Robot_Deg": self.yaw.reference_axis_deg,
            "Pick_Yaw_Reference_Servo_Command_Deg": self.yaw.reference_servo_deg,
            "Pick_Yaw_Servo_Deg_Per_Gripper_Deg": self.yaw.servo_deg_per_axis_deg,
            "Pick_Yaw_Servo_Command_Min_Deg": self.yaw.safe_command_min_deg,
            "Pick_Yaw_Servo_Command_Max_Deg": self.yaw.safe_command_max_deg,
        }
        for name, expected in fields.items():
            matched = re.search(r"\b" + name + r"\s*=\s*([-+\d.eE]+)f?\s*;", source)
            self.assertIsNotNone(matched, f"未找到 MCU 参数 {name}")
            self.assertAlmostEqual(float(matched.group(1)), expected, delta=1e-6)

    def test_last_debug_delegates_use_the_same_single_period(self):
        cfg = load_runtime_config(ROOT, ROOT / "config/runtime_config.yaml")
        for name in ("verify_top_yaw", "verify_top_yaw_tcp_compensated"):
            module = importlib.import_module("scripts.last_use.calibration." + name)
            module.apply_yaw_config(cfg)
            for axis in (-82.0, 60.0):
                expected = self.yaw.command_deg(axis)
                for period in (-180.0, 0.0, 180.0):
                    # 当前舵机位置不再用于在两个物理姿态之间选优。
                    for current in (-100.0, 100.0):
                        self.assertAlmostEqual(module.axis_to_servo_deg(axis + period, current), expected)
            if name.endswith("tcp_compensated"):
                self.assertAlmostEqual(module.TCP_MIN_AXIS_DEG, self.yaw.physical_min_deg)
                self.assertAlmostEqual(module.TCP_MAX_AXIS_DEG, self.yaw.physical_max_deg)
                tcp = cfg["vision"]["top"]["yaw_tcp_compensation"]
                self.assertEqual(list(module.VISUAL_TO_COMMAND_MATRIX), [tuple(row) for row in tcp["visual_to_command_matrix"]])
                self.assertAlmostEqual(module.TCP_REFERENCE_AXIS_DEG, tcp["reference_axis_deg"])


if __name__ == "__main__":
    unittest.main()
