"""边缘限角经过正式规划、补偿、共识和协议的回归；不连接机械设备。"""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from communication.sorting_protocol import Mode, SortingFrame, SortingProtocol
from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.calibration.yaw_servo import YawServoCalibration
from vision.pipeline.top_pick_pipeline import TopPickPipeline


class EdgeYawPipelineTests(unittest.TestCase):
    def make(self, x_mm, enabled=True):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        directory = Path(folder.name)
        (directory / "model.pt").touch()
        matrix = np.array([[0.25, 0, x_mm - 25], [0, 0.25, -125], [0, 0, 1]])
        PixelToRobotTransform(matrix, 200, 200).save(directory / "mapping.json")
        yaw = YawServoCalibration(-63.280987, 0.232430, 1.04720792, -90, 90)
        with patch("vision.pipeline.top_pick_pipeline.TopSegmenter"):
            pipeline = TopPickPipeline(
                model_path=directory / "model.pt", calibration_path=directory / "mapping.json",
                yaw_servo_calibration=yaw, orientation_enabled=False,
                default_grasp_axis_robot_deg=60, grasp_search_enabled=False,
                edge_yaw_config={"enabled": enabled, "x_threshold_mm": -40,
                                 "servo_min_deg": 40, "servo_max_deg": 80,
                                 "preferred_servo_deg": 60, "sample_step_deg": 5},
                x_limits_mm=(-300, 0), y_limits_mm=(-300, 0),
                preopen_configured=True, preopen_command_min_mm=13, preopen_command_max_mm=77,
                preopen_margin_mm=8, preopen_closed_servo_deg=0.710004610,
                preopen_standby_servo_deg=-89.283079760, gripper_collision_enabled=False,
                obstacle_margin=0, border_margin=0, min_safe_area=1, min_clearance=0,
            )
        polygon = np.array([[80, 80], [120, 80], [120, 120], [80, 120]], np.float32)
        pipeline.segmenter.get_objects.return_value = [{"polygon": polygon, "confidence": 0.9,
                                                        "class_name": "object"}]
        return pipeline

    def result(self, pipeline):
        frame = np.zeros((200, 200, 3), np.uint8)
        pipeline.process(frame)
        return pipeline.process(frame)

    def test_edge_replans_real_servo_angle_even_with_global_search_off(self):
        pipeline = self.make(-10)
        result = self.result(pipeline)
        self.assertTrue(result.ready)
        self.assertEqual(result.object_count, 1)
        wire = SortingProtocol.unpack(SortingProtocol.pack(SortingFrame(
            mode=Mode.PICK_TARGET, x_mm=result.x_mm, y_mm=result.y_mm, yaw_deg=result.yaw_deg,
        )))
        self.assertGreater(wire.x_mm, -40)
        self.assertTrue(40 <= pipeline.yaw_servo_calibration.command_deg(wire.yaw_deg) <= 80)
        self.assertNotEqual(wire.yaw_deg, 60)  # 60是舵机偏好角，不能直接当成协议角。
        self.assertTrue(any(r["reason"] == "edge_yaw_outside_servo_range"
                            for r in pipeline.last_gripper_rejections))

    def test_outside_and_exact_boundary_keep_original_direction(self):
        for x in (-50, -40):
            with self.subTest(x=x):
                self.assertEqual(self.result(self.make(x)).yaw_deg, 60)

    def test_disabled_rule_keeps_legacy_grasp_in_edge_region(self):
        self.assertEqual(self.result(self.make(-10, enabled=False)).yaw_deg, 60)

    def test_tcp_adjusted_x_decides_whether_edge_rule_applies(self):
        pipeline = self.make(-22)
        pipeline.yaw_tcp_compensator = SimpleNamespace(
            apply=lambda x, y, yaw: SimpleNamespace(x_mm=x + 4, y_mm=y),
        )
        result = self.result(pipeline)
        self.assertTrue(result.ready)
        self.assertAlmostEqual(result.x_mm, -18)
        self.assertTrue(40 <= pipeline.yaw_servo_calibration.command_deg(result.yaw_deg) <= 80)

    def test_consensus_crossing_x_boundary_rechecks_average_servo_angle(self):
        pipeline = self.make(-40.1)
        frame = np.zeros((200, 200, 3), np.uint8)
        candidate = pipeline._build_candidates(
            frame, pipeline.segmenter.get_objects.return_value,
        )[0]
        # 第一帧在区域外，协议20°对应舵机87.4°，允许保留原姿态；
        # 第二帧已进入区域，协议12°对应舵机79.1°，也合法。
        # 共识中位X=-39.95已在区域内，但均值协议16°对应舵机83.3°，必须拒绝。
        outside = replace(candidate, x_mm=-40.1, yaw_deg=20.0)
        inside = replace(candidate, x_mm=-39.8, yaw_deg=12.0)
        with patch.object(pipeline, "_build_candidates", side_effect=[[outside], [inside]]):
            self.assertFalse(pipeline.process(frame).ready)
            result = pipeline.process(frame)
        self.assertFalse(result.ready)
        self.assertAlmostEqual(result.x_mm, -39.95)
        self.assertIsNone(result.yaw_deg)
        self.assertEqual(result.reason, "edge_yaw_consensus_outside_servo_range")

    def test_no_feasible_opening_is_still_an_object_and_does_not_become_ready(self):
        pipeline = self.make(-10)
        pipeline.grasp_max_opening_mm = 2
        result = self.result(pipeline)
        self.assertFalse(result.ready)
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.object_count, 1)


if __name__ == "__main__":
    unittest.main()
