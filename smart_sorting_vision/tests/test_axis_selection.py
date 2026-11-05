"""Exercise the real candidate builder without loading YOLO or robot hardware."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.calibration.yaw_servo import YawServoCalibration
from vision.grasp.orientation_planner import axial_difference_deg
from vision.grasp.outward_radial_compensation import OutwardRadialCompensator


def _load_pipeline_without_model():
    # Keep the segmenter stub and the private module local to this import.
    # In particular, later tests must not inherit a fake ultralytics module.
    name = "_axis_selection_pipeline_under_test"
    path = Path(__file__).resolve().parents[1] / "vision/pipeline/top_pick_pipeline.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    segmenter = ModuleType("vision.models.top_segmenter")
    segmenter.TopSegmenter = Mock(name="unused_segmenter")
    with patch.dict(sys.modules, {name: module, "vision.models.top_segmenter": segmenter}):
        spec.loader.exec_module(module)
    return module


PIPELINE = _load_pipeline_without_model()


def rectangle(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def square(x, y, size=40):
    return rectangle(x, y, x + size, y + size)


def densely_sample_one_edge(polygon, count=35):
    """Preserve the closed polygon but deliberately bias vertex density."""
    a, b = polygon[:2]
    edge = [a + (b - a) * i / count for i in range(count)]
    return np.asarray(edge + list(polygon[1:]), dtype=np.float32)


class NominalAxisSelectionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        self.frame = np.zeros((800, 800, 3), dtype=np.uint8)
        (self.directory / "model.pt").touch()

    def pipeline(self, *, scale=0.2, **overrides):
        transform = PixelToRobotTransform(
            np.array([[scale, 0, -200], [0, scale, -200], [0, 0, 1]], dtype=float),
            800, 800,
        )
        path = self.directory / "mapping.json"
        transform.save(path)
        options = dict(
            model_path=self.directory / "model.pt", calibration_path=path,
            camera_config={"width": 800, "height": 800, "safe_roi": [0, 0, 800, 800]},
            yaw_servo_calibration=YawServoCalibration(
                reference_axis_deg=0, reference_servo_deg=0, servo_deg_per_axis_deg=1,
                safe_command_min_deg=-135, safe_command_max_deg=135,
            ),
            edge_yaw_config={"enabled": False}, outward_radial_config={"enabled": False},
            obstacle_margin=0, border_margin=0, min_safe_area=1, min_clearance=0,
            default_grasp_axis_robot_deg=0,
            preopen_configured=True, preopen_margin_mm=8,
            preopen_command_min_mm=13, preopen_command_max_mm=77,
            preopen_closed_servo_deg=0.710004610,
            preopen_standby_servo_deg=-89.283079760,
            gripper_collision_enabled=False, grasp_axis_selection_enabled=True,
            gripper_finger_length_mm=5, gripper_finger_thickness_mm=2.7,
            gripper_opening_uncertainty_mm=0.3,
            gripper_object_margin_mm=0.1, gripper_wall_margin_mm=0.5,
            gripper_enforce_image_border=False,
            grasp_search_enabled=False, grasp_search_yaw_offsets_deg=[0],
            grasp_search_longitudinal_offsets_mm=[0], grasp_search_closing_offsets_mm=[0],
            grasp_search_hard_min_clearance_mm=0.5, grasp_search_preferred_clearance_mm=2,
            grasp_search_min_target_overlap_mm=1.5, grasp_search_min_side_gap_mm=0.15,
            grasp_search_max_trials_per_object=200,
            consensus_window=1, consensus_required=1,
        )
        options.update(overrides)
        with patch.object(PIPELINE, "TopSegmenter"), redirect_stdout(io.StringIO()):
            return PIPELINE.TopPickPipeline(**options)

    @staticmethod
    def objects(polygons):
        return [
            {"index": i, "polygon": poly, "class_name": f"object_{i}", "confidence": 0.95}
            for i, poly in enumerate(polygons)
        ]

    def build(self, pipeline, polygons, *, fast_only=False):
        return pipeline._build_candidates(
            self.frame, self.objects(polygons), fast_only=fast_only,
        )

    def target(self, candidates, index=0):
        matches = [candidate for candidate in candidates if candidate.object_index == index]
        self.assertEqual(len(matches), 1, "expected this target to remain graspable")
        return matches[0]

    def assert_axis(self, candidate, expected):
        self.assertLess(abs(axial_difference_deg(candidate.yaw_deg, expected)), 1.0)

    def test_five_pixel_gap_selects_safe_axis_with_search_on_or_off(self):
        # At .2 mm/px, this 1 mm gap cannot fit the opened right finger.
        polygons = [square(100, 100), square(145, 100)]
        for search in (False, True):
            with self.subTest(search=search):
                pipeline = self.pipeline(grasp_search_enabled=search)
                candidate = self.target(self.build(pipeline, polygons, fast_only=search))
                self.assert_axis(candidate, 90)
                self.assertGreaterEqual(candidate.search_trial_count, 2)

    def test_contact_below_selects_horizontal_axis(self):
        pipeline = self.pipeline(default_grasp_axis_robot_deg=-85)
        candidate = self.target(self.build(pipeline, [square(100, 100), square(100, 140)]))
        self.assert_axis(candidate, 0)

    def test_equal_clear_axes_keep_previous_family_until_it_is_blocked(self):
        pipeline = self.pipeline()
        pipeline._prev_frame_axes = [(120.0, 120.0, -90.0)]
        for _ in range(3):
            candidate = self.target(self.build(pipeline, [square(100, 100)]))
            self.assert_axis(candidate, 90)
        # Continuity must never keep an axis that now intersects a neighbor.
        candidate = self.target(self.build(
            pipeline, [square(100, 100), square(100, 140)],
        ))
        self.assert_axis(candidate, 0)

    def test_long_neighbor_and_sampling_do_not_change_safe_axis(self):
        target, neighbor = square(100, 100), rectangle(100, 140, 700, 180)
        results = []
        for dense_target, dense_neighbor in ((False, False), (True, False), (False, True)):
            with self.subTest(target_sampling=dense_target, neighbor_sampling=dense_neighbor):
                pipeline = self.pipeline(default_grasp_axis_robot_deg=-85)
                candidate = self.target(self.build(pipeline, [
                    densely_sample_one_edge(target) if dense_target else target,
                    densely_sample_one_edge(neighbor) if dense_neighbor else neighbor,
                ]))
                self.assert_axis(candidate, 0)
                results.append((candidate.yaw_deg, candidate.grasp_pixel))
        self.assertTrue(all(item == results[0] for item in results))

    def test_closing_sweep_rejects_an_initially_clear_axis(self):
        # The opened 0-degree finger begins at x=160. This obstacle ends at
        # x=154, so only the subsequent closing motion crosses it.
        pipeline = self.pipeline()
        candidate = self.target(self.build(pipeline, [
            square(100, 100), rectangle(148, 118, 154, 122),
        ]))
        self.assert_axis(candidate, 90)

    def test_changed_family_uses_its_own_midplane_center(self):
        target = np.array(
            [[100, 100], [200, 100], [200, 120], [120, 200], [100, 200]],
            dtype=np.float32,
        )
        pipeline = self.pipeline(scale=0.5, default_grasp_axis_robot_deg=-85)
        candidate = self.target(self.build(pipeline, [target, square(100, 200, 88)]))
        self.assert_axis(candidate, 0)
        # Reusing the old 45-degree family's (130,130) gives a -6 mm gap.
        self.assertEqual(candidate.grasp_pixel, (150, 120))
        self.assertEqual(candidate.radial_base_pixel, (150.0, 120.0))
        self.assertAlmostEqual(candidate.x_mm, -125)
        self.assertAlmostEqual(candidate.y_mm, -140)
        self.assertAlmostEqual(candidate.preopen_mm, 58)

    def test_blocked_target_does_not_hide_other_graspable_objects(self):
        polygons = [
            square(100, 100), square(60, 100), square(140, 100),
            square(100, 60), square(100, 140), square(400, 400),
        ]
        pipeline = self.pipeline(grasp_search_enabled=True)
        candidates = self.build(pipeline, polygons, fast_only=True)
        self.assertNotIn(0, [candidate.object_index for candidate in candidates])
        self.target(candidates, index=5)
        pipeline.segmenter = SimpleNamespace(
            predict=lambda _: None, get_objects=lambda _: self.objects(polygons),
        )
        result = pipeline.process(self.frame)
        self.assertTrue(result.ready)
        self.assertNotEqual(result.class_name, "object_0")

    def test_large_uncertainty_margins_do_not_become_hard_rejection(self):
        pipeline = self.pipeline(
            gripper_opening_uncertainty_mm=100, gripper_object_margin_mm=100,
            gripper_wall_margin_mm=100, grasp_search_hard_min_clearance_mm=100,
        )
        candidate = self.target(self.build(pipeline, [square(100, 100), square(145, 100)]))
        self.assert_axis(candidate, 90)
        self.assertLess(candidate.gripper_clearance_mm, 100)

    def test_explicit_full_collision_keeps_strict_margins(self):
        pipeline = self.pipeline(gripper_collision_enabled=True, gripper_object_margin_mm=5)
        candidates = self.build(pipeline, [square(100, 100), square(145, 100)])
        self.assertEqual(candidates, [])
        self.assertTrue(pipeline.last_gripper_rejections)

    def test_disabled_axis_selection_keeps_primary_only_without_search(self):
        pipeline = self.pipeline(grasp_axis_selection_enabled=False)
        with patch.object(
            pipeline.gripper_footprint, "check_nominal",
            wraps=pipeline.gripper_footprint.check_nominal,
        ) as nominal:
            candidate = self.target(self.build(pipeline, [square(100, 100), square(145, 100)]))
        self.assert_axis(candidate, 0)
        self.assertEqual(candidate.search_trial_count, 1)
        nominal.assert_not_called()

    def test_radial_compensation_is_applied_once_per_family(self):
        pipeline = self.pipeline()
        radial = OutwardRadialCompensator(True, 0.02, 20.0, (0.0, 0.0))
        apply = Mock(wraps=radial.apply)
        pipeline.radial_compensator = SimpleNamespace(apply=apply)
        candidate = self.target(self.build(pipeline, [square(100, 100)]))
        self.assertEqual(apply.call_count, candidate.search_trial_count)
        for call in apply.call_args_list:
            self.assertEqual(call.args[0], (120.0, 120.0))
        self.assertEqual(candidate.radial_base_pixel, (120.0, 120.0))
        self.assertEqual(candidate.grasp_pixel, (122, 122))
        self.assertAlmostEqual(candidate.radial_shift_px, np.hypot(2.4, 2.4))
        self.assertAlmostEqual(candidate.x_mm, -175.52)
        self.assertAlmostEqual(candidate.y_mm, -175.52)

    def test_full_search_applies_local_offsets_before_one_radial_shift(self):
        pipeline = self.pipeline(
            grasp_search_enabled=True, grasp_search_longitudinal_offsets_mm=[0, 1],
            grasp_search_closing_offsets_mm=[0, 1],
        )
        radial = OutwardRadialCompensator(True, 0.02, 20.0, (0.0, 0.0))
        apply = Mock(wraps=radial.apply)
        pipeline.radial_compensator = SimpleNamespace(apply=apply)
        candidate = self.target(self.build(pipeline, [square(100, 100)], fast_only=False))
        # One millimeter is five pixels. Local +closing/+longitudinal are
        # (+x,+y) for 0 degrees and (-y,+x) for the -90 degree family.
        self.assertEqual(candidate.search_trial_count, 8)
        self.assertEqual(apply.call_count, 8)
        self.assertCountEqual(
            [tuple(round(value, 7) for value in call.args[0]) for call in apply.call_args_list],
            [(120.0, 120.0), (120.0, 120.0), (125.0, 120.0), (120.0, 115.0),
             (120.0, 125.0), (125.0, 120.0), (125.0, 125.0), (125.0, 115.0)],
        )

    def test_fast_pass_ignores_configured_extra_rotations(self):
        pipeline = self.pipeline(
            grasp_search_enabled=True, grasp_search_yaw_offsets_deg=[0, 90, -15],
        )
        objects = self.objects([square(100, 100), square(145, 100)])
        pipeline.segmenter = SimpleNamespace(
            predict=lambda _: None, get_objects=lambda _: objects,
        )
        with patch.object(pipeline, "_build_candidates", wraps=pipeline._build_candidates) as build:
            result = pipeline.process(self.frame)
        self.assertEqual(build.call_count, 1)
        self.assertTrue(build.call_args.kwargs["fast_only"])
        candidate = self.target(result.candidates)
        self.assertEqual(candidate.search_trial_count, 2)
        self.assert_axis(candidate, 90)

    def test_empty_fast_pass_still_runs_full_fallback(self):
        pipeline = self.pipeline(
            grasp_search_enabled=True, grasp_search_yaw_offsets_deg=[0, 90, -15],
        )
        # 80 mm exceeds the 77 mm calibrated opening in every trial.
        objects = self.objects([square(100, 100, 400)])
        pipeline.segmenter = SimpleNamespace(
            predict=lambda _: None, get_objects=lambda _: objects,
        )
        with patch.object(pipeline, "_build_candidates", wraps=pipeline._build_candidates) as build:
            result = pipeline.process(self.frame)
        self.assertEqual(
            [call.kwargs["fast_only"] for call in build.call_args_list], [True, False],
        )
        self.assertFalse(result.ready)
        self.assertEqual(result.candidates, [])


if __name__ == "__main__":
    unittest.main()
