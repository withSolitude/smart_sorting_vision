"""完整实例轮廓的多抓取点回归；不加载YOLO，也不连接相机或下位机。"""
import math
import unittest

import cv2
import numpy as np

from vision.grasp.geometric_grasp_candidates import (
    build_grasp_point_seeds,
    describe_geometry,
)


class GeometricGraspPointTests(unittest.TestCase):
    @staticmethod
    def _polygon(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _seeds(mask, *, safe_mask=None, **overrides):
        config = {
            "max_points": 6,
            "min_spacing_px": 12.0,
            "min_radius_ratio": 0.45,
            "min_clearance_px": 3.0,
            "axis_sample_fractions": [-0.3, 0.0, 0.3],
        }
        config.update(overrides)
        return build_grasp_point_seeds(
            GeometricGraspPointTests._polygon(mask),
            mask if safe_mask is None else safe_mask,
            config=config,
        )

    def test_long_rectangle_plateau_uses_middle_instead_of_first_maximum(self):
        mask = np.zeros((160, 300), np.uint8)
        cv2.rectangle(mask, (30, 30), (250, 100), 255, -1)
        seeds = self._seeds(mask)
        self.assertTrue(seeds)
        self.assertEqual(seeds[0].source, "inscribed_peak")
        # 最大内接圆平台横跨长方形中部，不能固定取左端的第一个最大值。
        self.assertAlmostEqual(seeds[0].pixel[0], 140.0, delta=1.0)
        self.assertAlmostEqual(seeds[0].pixel[1], 65.0, delta=1.0)
        self.assertGreaterEqual(len(seeds), 3)
        self.assertTrue(any(s.pixel[0] < 100 for s in seeds))
        self.assertTrue(any(s.pixel[0] > 180 for s in seeds))

    def test_thin_side_extension_does_not_pull_inscribed_candidate_to_full_centroid(self):
        mask = np.zeros((160, 220), np.uint8)
        cv2.rectangle(mask, (70, 55), (150, 115), 255, -1)
        cv2.rectangle(mask, (150, 70), (184, 100), 255, -1)
        before = mask.copy()
        seeds = self._seeds(mask)
        full_y, full_x = np.nonzero(mask)
        true_center = np.asarray([110.0, 85.0])
        peak = next(s for s in seeds if s.source == "inscribed_peak")
        self.assertLess(
            np.linalg.norm(np.asarray(peak.pixel) - true_center),
            np.linalg.norm(np.asarray([full_x.mean(), full_y.mean()]) - true_center),
        )
        # 输入完整mask保留，用于后面的实际夹指接触和碰撞检查。
        np.testing.assert_array_equal(mask, before)

    def test_tapered_object_generates_thick_sections_and_avoids_tip(self):
        mask = np.zeros((320, 320), np.uint8)
        triangle = np.asarray([[150, 30], [250, 280], [50, 280]], np.int32)
        cv2.fillPoly(mask, [triangle], 255)
        seeds = self._seeds(mask)
        self.assertEqual(describe_geometry(self._polygon(mask)), "tapered")
        self.assertGreaterEqual(len(seeds), 2)
        self.assertTrue(any(s.source == "axis_section" for s in seeds))
        # 尖端附近既薄又不利于夹持，不应成为候选落点。
        self.assertTrue(all(s.pixel[1] > 110 for s in seeds))
        self.assertTrue(all(s.clearance_px >= 3.0 for s in seeds))

    def test_circle_keeps_center_and_does_not_invent_three_dimensional_class(self):
        mask = np.zeros((180, 180), np.uint8)
        cv2.circle(mask, (90, 90), 45, 255, -1)
        seeds = self._seeds(mask)
        self.assertEqual(describe_geometry(self._polygon(mask)), "round")
        self.assertEqual(seeds[0].pixel, (90, 90))
        self.assertEqual(len(seeds), 1)

    def test_separate_thick_lobes_keep_multiple_local_maxima(self):
        mask = np.zeros((140, 240), np.uint8)
        cv2.circle(mask, (60, 70), 30, 255, -1)
        cv2.circle(mask, (175, 70), 25, 255, -1)
        cv2.rectangle(mask, (60, 63), (175, 77), 255, -1)
        seeds = self._seeds(mask)
        peaks = [s for s in seeds if s.source == "inscribed_peak"]
        self.assertGreaterEqual(len(peaks), 2)
        self.assertTrue(any(s.pixel[0] < 90 for s in peaks))
        self.assertTrue(any(s.pixel[0] > 150 for s in peaks))

    def test_empty_safe_region_returns_no_points(self):
        mask = np.zeros((120, 180), np.uint8)
        cv2.rectangle(mask, (20, 20), (160, 100), 255, -1)
        seeds = self._seeds(mask, safe_mask=np.zeros_like(mask))
        self.assertEqual(seeds, [])

    def test_all_points_respect_safe_mask_and_minimum_clearance(self):
        mask = np.zeros((160, 300), np.uint8)
        cv2.rectangle(mask, (30, 30), (250, 130), 255, -1)
        safe = mask.copy()
        safe[:, :150] = 0
        before = safe.copy()
        seeds = self._seeds(mask, safe_mask=safe, min_clearance_px=8.0)
        self.assertTrue(seeds)
        for seed in seeds:
            x, y = seed.pixel
            self.assertGreater(safe[y, x], 0)
            self.assertGreaterEqual(seed.clearance_px, 8.0)
            self.assertTrue(math.isfinite(seed.quality))
        np.testing.assert_array_equal(safe, before)

    def test_count_spacing_and_order_are_bounded_and_repeatable(self):
        mask = np.zeros((160, 300), np.uint8)
        cv2.rectangle(mask, (30, 30), (250, 130), 255, -1)
        seeds = self._seeds(mask, max_points=2, min_spacing_px=18.0)
        self.assertLessEqual(len(seeds), 2)
        self.assertEqual(seeds, self._seeds(mask, max_points=2, min_spacing_px=18.0))
        for index, seed in enumerate(seeds):
            for other in seeds[index + 1:]:
                self.assertGreaterEqual(math.dist(seed.pixel, other.pixel), 18.0)

    def test_geometry_describes_parallel_edges_without_claiming_cube(self):
        rectangle = np.asarray([[20, 20], [140, 20], [140, 80], [20, 80]], np.float32)
        self.assertEqual(describe_geometry(rectangle), "parallel")


if __name__ == "__main__":
    unittest.main()
