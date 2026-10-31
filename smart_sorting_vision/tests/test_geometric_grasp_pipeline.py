"""多抓点和方向的完整流水线回归；替代推理器，不打开相机和电机。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.calibration.yaw_servo import YawServoCalibration
from vision.grasp.grasp_pose_search import evaluate_target_engagement
from vision.grasp.orientation_planner import axial_difference_deg
from vision.pipeline.top_pick_pipeline import TopPickPipeline


def rectangle(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


class GeometricGraspPipelineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        directory = Path(self.folder.name)
        (directory / "model.pt").touch()
        transform = PixelToRobotTransform(np.diag([0.25, 0.25, 1.0]), 400, 300)
        transform.save(directory / "mapping.json")
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/runtime_config.yaml").read_text(encoding="utf-8"))
        self.geometry = cfg["vision"]["top"]["geometric_grasp"]
        with patch("vision.pipeline.top_pick_pipeline.TopSegmenter"):
            self.pipeline = TopPickPipeline(
                model_path=directory / "model.pt", calibration_path=directory / "mapping.json",
                geometric_grasp_config=self.geometry,
                yaw_servo_calibration=YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"]),
                default_grasp_axis_robot_deg=-85.0, grasp_search_enabled=False,
                preopen_configured=True, preopen_command_min_mm=13.0, preopen_command_max_mm=77.0,
                preopen_margin_mm=8.0, preopen_closed_servo_deg=0.710004610,
                preopen_standby_servo_deg=-89.283079760,
                gripper_collision_enabled=True, gripper_finger_length_mm=5.0,
                gripper_finger_thickness_mm=2.7, gripper_object_margin_mm=0.1,
                gripper_wall_margin_mm=0.0, gripper_opening_uncertainty_mm=0.3,
                gripper_enforce_image_border=False,
                obstacle_margin=0, border_margin=0, min_safe_area=1, min_clearance=1.0,
                grasp_search_hard_min_clearance_mm=0.0,
            )
        self.frame = np.zeros((300, 400, 3), np.uint8)

    def build(self, target, others=()):
        objects = [{"polygon": p, "confidence": 0.9, "class_name": "object", "index": i}
                   for i, p in enumerate([target, *others])]
        return [c for c in self.pipeline._build_candidates(self.frame, objects) if c.object_index == 0]

    def test_rectangle_prefers_long_side_contact_and_checks_multiple_points(self):
        candidates = self.build(rectangle(100, 100, 260, 160))
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertAlmostEqual(candidate.grasp_width_mm, 15.0, places=4)
        self.assertLess(abs(axial_difference_deg(candidate.yaw_deg, -90)), 1.0)
        self.assertGreater(candidate.search_trial_count, 2)
        self.assertLessEqual(candidate.search_trial_count, self.geometry["max_pose_trials"])
        self.assertEqual(candidate.grasp_strategy, "parallel")
        self.assertFalse(hasattr(candidate, "grasp_core_polygon"))

    def test_obstacle_changes_point_or_direction_without_leaving_target(self):
        target = rectangle(100, 100, 260, 160)
        baseline = self.build(target)[0]
        blocked = self.build(target, [rectangle(150, 168, 210, 185)])
        self.assertTrue(blocked)
        selected = blocked[0]
        self.assertTrue(selected.grasp_pixel != baseline.grasp_pixel or selected.yaw_deg != baseline.yaw_deg)
        self.assertGreaterEqual(selected.grasp_pixel[0], 100)
        self.assertLessEqual(selected.grasp_pixel[0], 260)
        self.assertGreater(selected.gripper_clearance_mm, 0.0)

    def test_no_candidate_when_all_finger_channels_are_blocked(self):
        target = rectangle(100, 100, 260, 160)
        obstacles = [rectangle(75, 70, 98, 190), rectangle(262, 70, 285, 190),
                     rectangle(75, 75, 285, 98), rectangle(75, 162, 285, 185)]
        self.assertEqual(self.build(target, obstacles), [])
        self.assertTrue(any(r["object_index"] == 0 for r in self.pipeline.last_gripper_rejections))

    def test_round_object_tries_multiple_axes(self):
        angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        target = np.column_stack((180 + 35 * np.cos(angles), 140 + 35 * np.sin(angles))).astype(np.float32)
        candidates = self.build(target)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].grasp_strategy, "round")
        self.assertGreaterEqual(candidates[0].search_trial_count, 6)

    def test_local_contact_receives_complete_polygon_for_every_pose(self):
        target = np.array([[100, 70], [260, 70], [160, 230]], np.float32)
        with patch("vision.pipeline.top_pick_pipeline.evaluate_target_engagement",
                   wraps=evaluate_target_engagement) as engagement:
            self.build(target)
        expected = self.pipeline.transform.pixels_to_robot(target, apply_tcp_offset=False)
        self.assertGreater(engagement.call_count, 2)
        for call in engagement.call_args_list:
            np.testing.assert_allclose(call.args[0], expected)
            self.assertTrue(call.kwargs["local_contact"])

    def test_target_itself_cannot_intersect_fingers(self):
        target = rectangle(140, 100, 220, 180)
        mask = self.pipeline.planner.polygon_to_mask(target, self.frame.shape)
        result = self.pipeline.gripper_footprint.check(
            grasp_pixel=(180, 140), yaw_deg=0.0, opening_mm=3.0,
            other_polygons=[], image_shape=self.frame.shape, target_mask=mask,
        )
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "target_finger_collision")

    def test_local_footprint_raster_preserves_full_image_collision_and_clearance(self):
        # 用原全图光栅作为参照，检查局部窗口优化没有漏掉旋转包络的碰撞。
        model = self.pipeline.gripper_footprint
        others = [rectangle(227, 120, 245, 160)]
        context = model.prepare_context(other_polygons=others, image_shape=self.frame.shape)
        for yaw in (-85, -40, 0, 37, 85):
            with self.subTest(yaw=yaw):
                result = model.check(
                    grasp_pixel=(180, 140), yaw_deg=yaw, opening_mm=20.0,
                    other_polygons=others, image_shape=self.frame.shape, context=context,
                )
                center = self.pipeline.transform.pixel_to_robot(180, 140, apply_tcp_offset=False)
                expanded = model._make_footprint_robot(
                    center_robot_mm=center, yaw_deg=yaw, opening_mm=20.0,
                    extra_margin_mm=model.object_margin_mm, include_opening_uncertainty=True,
                )
                full = model._polygons_to_mask(model._robot_polygons_to_pixels(expanded), self.frame.shape)
                overlap = int(np.count_nonzero((full > 0) & (context.other_mask > 0)))
                if overlap > model.object_overlap_tolerance_px:
                    self.assertFalse(result.safe)
                    self.assertEqual(result.reason, "other_object_collision")
                    self.assertEqual(result.object_collision_pixels, overlap)
                else:
                    self.assertTrue(result.safe)
                    nominal = model._polygons_to_mask(result.footprint_polygons_px, self.frame.shape)
                    expected = float(context.distance_to_obstacle[nominal > 0].min())
                    self.assertAlmostEqual(result.clearance_px, expected)


if __name__ == "__main__":
    unittest.main()
