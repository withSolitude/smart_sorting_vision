"""指爪局部截面估宽与凹口排除回归；无需模型、相机或下位机。"""
from __future__ import annotations

import unittest

import numpy as np

from vision.grasp.grasp_pose_search import TargetEngagement, evaluate_target_engagement


class LocalTargetEngagementTests(unittest.TestCase):
    @staticmethod
    def triangle() -> np.ndarray:
        return np.array([[-30.0, -30.0], [30.0, -30.0], [0.0, 30.0]])

    def test_triangle_uses_finger_strip_intersections_instead_of_base_width(self):
        # 接触带y=5..15没有任何原始顶点，必须计算斜边与带边界的交点。
        result = evaluate_target_engagement(
            self.triangle(), (0.0, 10.0), 0.0, 33.0,
            finger_length_mm=10.0, local_contact=True,
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.width_mm, 25.0)
        self.assertAlmostEqual(result.longitudinal_overlap_mm, 10.0)
        self.assertAlmostEqual(result.min_side_gap_mm, 4.0)
        self.assertAlmostEqual(result.contact_balance_mm, 0.0)

    def test_global_path_and_old_result_constructor_stay_compatible(self):
        implicit = evaluate_target_engagement(
            self.triangle(), (0.0, 10.0), 0.0, 33.0, finger_length_mm=10.0,
        )
        explicit = evaluate_target_engagement(
            self.triangle(), (0.0, 10.0), 0.0, 33.0,
            finger_length_mm=10.0, local_contact=False,
        )
        self.assertEqual(implicit, explicit)
        self.assertAlmostEqual(implicit.width_mm, 60.0)
        self.assertFalse(implicit.valid)
        legacy = TargetEngagement(20.0, 5.0, 4.0, 4.0, 4.0, True, True)
        self.assertTrue(legacy.valid)
        self.assertEqual(legacy.contact_balance_mm, 0.0)

    def test_rotation_translation_and_nonzero_finger_offset_are_consistent(self):
        angle = 37.0
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        shift = np.array([100.0, -80.0])
        rotated = self.triangle() @ rotation.T + shift
        # 夹爪中心在局部y=7，指爪纵向偏置+3，所以真正接触带仍为y=5..15。
        center = np.array([0.0, 7.0]) @ rotation.T + shift
        result = evaluate_target_engagement(
            rotated, center, angle, 33.0, finger_length_mm=10.0,
            finger_longitudinal_offset_mm=3.0, local_contact=True,
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.width_mm, 25.0)
        self.assertAlmostEqual(result.min_side_gap_mm, 4.0)

    def test_empty_contact_strip_is_invalid_without_global_fallback(self):
        result = evaluate_target_engagement(
            self.triangle(), (0.0, 100.0), 0.0, 1000.0,
            finger_length_mm=10.0, local_contact=True,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.width_mm, 0.0)

    def test_u_shape_gap_is_not_filled_by_outer_extremes(self):
        polygon = np.array([
            [-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [8.0, 20.0],
            [8.0, -8.0], [-8.0, -8.0], [-8.0, 20.0], [-20.0, 20.0],
        ])
        # 接触带跨过U底部，带内有目标也不能让中间的空凹口通过检查。
        result = evaluate_target_engagement(
            polygon, (0.0, 0.0), 0.0, 48.0, finger_length_mm=30.0,
            local_contact=True,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.width_mm, 0.0)

    def test_target_on_only_one_side_of_closing_center_is_invalid(self):
        polygon = np.array([[1.0, -4.0], [3.0, -4.0], [3.0, 4.0], [1.0, 4.0]])
        result = evaluate_target_engagement(
            polygon, (0.0, 0.0), 0.0, 20.0, finger_length_mm=10.0,
            local_contact=True,
        )
        self.assertFalse(result.valid)

    def test_off_center_contact_reports_balance_penalty(self):
        result = evaluate_target_engagement(
            self.triangle(), (2.0, 10.0), 0.0, 33.0,
            finger_length_mm=10.0, local_contact=True,
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.contact_balance_mm, 2.0)
        self.assertAlmostEqual(result.min_side_gap_mm, 2.0)

    def test_too_small_opening_is_rejected_even_for_valid_local_section(self):
        result = evaluate_target_engagement(
            self.triangle(), (0.0, 10.0), 0.0, 20.0,
            finger_length_mm=10.0, local_contact=True,
        )
        self.assertAlmostEqual(result.width_mm, 25.0)
        self.assertFalse(result.valid)
        self.assertLess(result.min_side_gap_mm, 0.0)


if __name__ == "__main__":
    unittest.main()
