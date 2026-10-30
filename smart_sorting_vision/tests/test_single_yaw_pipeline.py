"""单个180°执行区间的规划回归；不加载模型、不连接相机或下位机。"""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from communication.sorting_protocol import Mode, SortingFrame, SortingProtocol
from vision.calibration.yaw_servo import YawServoCalibration
from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.grasp.grasp_pose_search import evaluate_target_engagement
from vision.pipeline.top_pick_consensus import TopPickConsensus
from vision.pipeline.top_pick_pipeline import TopPickPipeline


def calibration(minimum=-135.0, maximum=135.0):
    return YawServoCalibration(
        reference_axis_deg=-63.280987,
        reference_servo_deg=0.232430,
        servo_deg_per_axis_deg=1.04720792,
        safe_command_min_deg=minimum,
        safe_command_max_deg=maximum,
    )


class SingleYawPipelineTests(unittest.TestCase):
    def pipeline(self, cal=None):
        pipeline = TopPickPipeline.__new__(TopPickPipeline)
        pipeline.yaw_servo_calibration = calibration() if cal is None else cal
        pipeline.yaw_tcp_compensator = None
        pipeline.x_limits_mm = (-300.0, 0.0)
        pipeline.y_limits_mm = (-300.0, 0.0)
        return pipeline

    def test_protocol_quantization_precedes_physical_pose(self):
        pipeline = self.pipeline()
        # 26.3° 未量化时会位于窗口另一侧；线上26°必须先确定后再做TCP/包络。
        for value in (26.3, 26.7, 89.6, -89.6, 0.5, -0.5):
            wire = pipeline._wire_yaw(value)
            decoded = SortingProtocol.unpack(SortingProtocol.pack(SortingFrame(
                mode=Mode.PICK_TARGET, yaw_deg=wire,
            )))
            self.assertEqual(decoded.yaw_deg, wire)
            self.assertAlmostEqual(pipeline._physical_yaw(wire), calibration().physical_axis_deg(decoded.yaw_deg))
        self.assertGreater(pipeline._physical_yaw(pipeline._wire_yaw(26.3)), 0.0)
        self.assertLess(pipeline._physical_yaw(pipeline._wire_yaw(26.7)), -150.0)
        self.assertEqual(pipeline._wire_yaw(89.6), -90.0)
        self.assertEqual(pipeline._wire_yaw(-89.6), -90.0)

    def test_servo_limit_is_checked_without_tcp_compensation(self):
        pipeline = self.pipeline(calibration(-10.0, 10.0))
        with self.assertRaises(ValueError):
            pipeline._physical_yaw(-82.0)
        pipeline.yaw_servo_calibration = None
        with self.assertRaisesRegex(ValueError, "calibration.yaw_servo"):
            pipeline._physical_yaw(-82.0)

    def test_compensator_receives_only_the_unique_physical_pose(self):
        pipeline = self.pipeline()
        calls = []

        def apply(x, y, physical):
            calls.append((x, y, physical))
            return SimpleNamespace(x_mm=x + 2.0, y_mm=y - 1.0)

        pipeline.yaw_tcp_compensator = SimpleNamespace(apply=apply)
        physical = pipeline._physical_yaw(60.0)
        self.assertEqual(pipeline._compensate_yaw_xy(-100.0, -100.0, physical), (-98.0, -101.0))
        self.assertEqual(calls, [(-100.0, -100.0, physical)])
        self.assertAlmostEqual(physical, -120.0)
        self.assertIsNone(pipeline._compensate_yaw_xy(-1.0, -100.0, physical))

    def test_full_candidate_geometry_uses_the_execution_pose(self):
        # 单个正方形配合非零指爪前后偏置；验证包络与接触检查使用-120°而非轴向60°。
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "model.pt").touch()
            PixelToRobotTransform(np.eye(3), 200, 200).save(directory / "mapping.json")
            with patch("vision.pipeline.top_pick_pipeline.TopSegmenter"):
                pipeline = TopPickPipeline(
                    model_path=directory / "model.pt", calibration_path=directory / "mapping.json",
                    yaw_servo_calibration=calibration(), orientation_enabled=False,
                    default_grasp_axis_robot_deg=60.0, grasp_search_enabled=False,
                    preopen_configured=True, preopen_command_min_mm=13.0, preopen_command_max_mm=77.0,
                    preopen_margin_mm=5.0, preopen_closed_servo_deg=0.710004610,
                    preopen_standby_servo_deg=-89.283079760, gripper_collision_enabled=False,
                    obstacle_margin=1, border_margin=0, min_safe_area=1, min_clearance=0.0,
                    gripper_finger_longitudinal_offset_mm=10.0,
                )
            polygon = np.array([[80, 80], [120, 80], [120, 120], [80, 120]], dtype=np.float32)
            with patch.object(pipeline.gripper_footprint, "check", wraps=pipeline.gripper_footprint.check) as footprint, \
                 patch("vision.pipeline.top_pick_pipeline.evaluate_target_engagement",
                       wraps=evaluate_target_engagement) as engagement:
                candidates = pipeline._build_candidates(
                    np.zeros((200, 200, 3), dtype=np.uint8),
                    [{"polygon": polygon, "confidence": 0.9, "class_name": "object"}],
                )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].yaw_deg, 60.0)
            self.assertAlmostEqual(footprint.call_args.kwargs["yaw_deg"], -120.0)
            self.assertEqual(engagement.call_count, 2)
            for call in engagement.call_args_list:
                self.assertAlmostEqual(call.args[2], -120.0)

    def consensus_pipeline(self):
        pipeline = self.pipeline()
        pipeline.transform = SimpleNamespace(check_frame=lambda _: None)
        pipeline.segmenter = SimpleNamespace(predict=lambda _: None, get_objects=lambda _: [{}])
        pipeline.grasp_search_enabled = False
        pipeline.required_hits = pipeline.window_size = 2
        pipeline.consensus = TopPickConsensus(window_size=2, required_hits=2)
        pipeline._candidate_to_consensus_dict = lambda candidate: candidate
        return pipeline

    def observe(self, pipeline, angle):
        candidate = {
            "center_pixel": (10.0, 10.0), "grasp_x": 10, "grasp_y": 10,
            "x_mm": -100.0, "y_mm": -100.0, "yaw_deg": pipeline._wire_yaw(angle),
            "orientation_reliable": True, "orientation_mode": "parallel_edges",
            "aspect_ratio": 2.0, "rectangularity": 1.0, "confidence": 0.95,
            "clearance": 10.0, "grasp_width_mm": 20.0, "preopen_mm": 25.0,
            "approach_gap_px": 10.0, "score": 1.0, "class_name": "object",
        }
        pipeline._build_candidates = lambda *_args, **_kwargs: [candidate]
        return pipeline.process(np.zeros((20, 20, 3), dtype=np.uint8))

    def test_consensus_reobserves_at_unique_physical_window_seam(self):
        pipeline = self.consensus_pipeline()
        self.assertFalse(self.observe(pipeline, 26.0).ready)
        mixed = self.observe(pipeline, 27.0)
        self.assertFalse(mixed.ready)
        self.assertEqual(mixed.reason, "yaw_window_boundary_unstable")
        stable = self.observe(pipeline, 27.0)
        self.assertTrue(stable.ready)
        self.assertEqual(stable.yaw_deg, 27.0)
        self.assertFalse(hasattr(stable, "yaw_branch"))

    def test_axial_plus_minus_90_seam_is_the_same_execution_pose(self):
        pipeline = self.consensus_pipeline()
        self.assertFalse(self.observe(pipeline, 89.6).ready)
        stable = self.observe(pipeline, -89.6)
        self.assertTrue(stable.ready)
        self.assertEqual(stable.yaw_deg, -90.0)
        self.assertAlmostEqual(pipeline._physical_yaw(stable.yaw_deg), -90.0)


if __name__ == "__main__":
    unittest.main()
