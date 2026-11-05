"""失败抓取排除：多个失败物体都要记住，且被推走仍能跟踪。"""
from __future__ import annotations

import unittest

import numpy as np

from vision.pipeline.top_pick_pipeline import TopPickPipeline


class FakeTransform:
    def pixel_to_robot(self, x, y, apply_tcp_offset=False):
        return (float(x), float(y))


def square(x0: float, y0: float, size: float = 40.0) -> np.ndarray:
    return np.array(
        [
            [x0, y0],
            [x0 + size, y0],
            [x0 + size, y0 + size],
            [x0, y0 + size],
        ],
        dtype=np.float32,
    )


def make_pipeline() -> TopPickPipeline:
    pipeline = TopPickPipeline.__new__(TopPickPipeline)
    pipeline.transform = FakeTransform()
    pipeline.failed_pick_enabled = True
    pipeline.failed_pick_search_radius_mm = 120.0
    pipeline.failed_pick_area_ratio_min = 0.6
    pipeline.failed_pick_area_ratio_max = 1.6
    pipeline.failed_pick_iou_threshold = 0.35
    pipeline.failed_pick_give_up_frames = 3
    pipeline.failed_pick_permanent_radius_mm = 60.0
    pipeline._last_pick_reference = None
    pipeline._failed_references = []
    pipeline._attempted_points = []
    pipeline._normalize_polygon = lambda poly: np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return pipeline


def add_failure(pipeline: TopPickPipeline, polygon, image_shape=(400, 400)):
    reference = pipeline._reference_from_object({"polygon": polygon}, image_shape)
    pipeline._failed_references.append(reference)
    pipeline._attempted_points.append(reference["robot"].copy())
    return reference


class FailedPickTrackingTests(unittest.TestCase):
    def test_moved_object_is_matched_and_excluded(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        moved = {"polygon": square(130, 100)}
        pipeline._match_failed_object([moved], (400, 400))
        self.assertTrue(moved.get("_blocked"))
        reference = pipeline._failed_references[0]
        # 原始 mask 中心不变，只更新 last_center 用于继续跟随。
        self.assertAlmostEqual(float(reference["center_px"][0]), 120.0, delta=1.0)
        self.assertAlmostEqual(float(reference["last_center_px"][0]), 150.0, delta=1.0)

    def test_two_failed_objects_are_both_remembered(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        add_failure(pipeline, square(100, 160))
        first = {"polygon": square(130, 100)}
        second = {"polygon": square(130, 160)}
        pipeline._match_failed_object([first, second], (400, 400))
        self.assertTrue(first.get("_blocked"))
        self.assertTrue(second.get("_blocked"))

    def test_reference_follows_object_across_frames(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        first_frame = {"polygon": square(130, 100)}
        pipeline._match_failed_object([first_frame], (400, 400))
        self.assertTrue(first_frame.get("_blocked"))
        second_frame = {"polygon": square(160, 100)}
        pipeline._match_failed_object([second_frame], (400, 400))
        self.assertTrue(second_frame.get("_blocked"))
        reference = pipeline._failed_references[0]
        self.assertAlmostEqual(float(reference["center_px"][0]), 120.0, delta=1.0)
        self.assertAlmostEqual(float(reference["last_center_px"][0]), 180.0, delta=1.0)

    def test_permanent_point_blocks_after_tracking_gives_up(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        for _ in range(3):
            pipeline._match_failed_object([{"polygon": square(300, 300)}], (400, 400))
        self.assertEqual(pipeline._failed_references, [])
        stuck = {"polygon": square(105, 100)}
        pipeline._match_failed_object([stuck], (400, 400))
        self.assertTrue(stuck.get("_blocked"))

    def test_far_object_is_not_blocked(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        far = {"polygon": square(320, 320)}
        pipeline._match_failed_object([far], (400, 400))
        self.assertFalse(far.get("_blocked"))

    def test_reset_clears_all_memory(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        pipeline._last_pick_reference = pipeline._reference_from_object(
            {"polygon": square(200, 200)}, (400, 400)
        )
        pipeline.reset_failed_picks()
        self.assertEqual(pipeline._failed_references, [])
        self.assertEqual(pipeline._attempted_points, [])
        self.assertIsNone(pipeline._last_pick_reference)

    def test_mark_pick_failed_appends_reference_and_point(self):
        pipeline = make_pipeline()
        pipeline._last_pick_reference = pipeline._reference_from_object(
            {"polygon": square(100, 100)}, (400, 400)
        )
        self.assertTrue(pipeline.mark_pick_failed())
        self.assertEqual(len(pipeline._failed_references), 1)
        self.assertEqual(len(pipeline._attempted_points), 1)
        pipeline._last_pick_reference = pipeline._reference_from_object(
            {"polygon": square(102, 100)}, (400, 400)
        )
        pipeline.mark_pick_failed()
        self.assertEqual(len(pipeline._failed_references), 1)


if __name__ == "__main__":
    unittest.main()
