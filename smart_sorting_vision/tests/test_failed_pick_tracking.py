"""失败抓取排除：物体被推走仍能匹配，远处/面积不符不误伤。"""
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
    pipeline._failed_reference = None
    pipeline._failed_miss_frames = 0
    pipeline._last_pick_reference = None
    pipeline._normalize_polygon = lambda poly: np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return pipeline


class FailedPickTrackingTests(unittest.TestCase):
    def test_moved_object_is_matched_and_excluded(self):
        pipeline = make_pipeline()
        pipeline._failed_reference = pipeline._reference_from_object(
            {"polygon": square(100, 100)}, (400, 400)
        )
        moved = {"polygon": square(130, 100)}  # 被推走 30px
        pipeline._match_failed_object([moved], (400, 400))
        self.assertTrue(moved.get("_blocked"))
        self.assertAlmostEqual(
            float(pipeline._failed_reference["center_px"][0]), 150.0, delta=1.0
        )

    def test_far_object_is_not_excluded(self):
        pipeline = make_pipeline()
        pipeline._failed_reference = pipeline._reference_from_object(
            {"polygon": square(100, 100)}, (400, 400)
        )
        far = {"polygon": square(300, 300)}
        pipeline._match_failed_object([far], (400, 400))
        self.assertFalse(far.get("_blocked"))

    def test_give_up_after_consecutive_misses(self):
        pipeline = make_pipeline()
        pipeline._failed_reference = pipeline._reference_from_object(
            {"polygon": square(100, 100)}, (400, 400)
        )
        for _ in range(3):
            pipeline._match_failed_object([{"polygon": square(300, 300)}], (400, 400))
        self.assertIsNone(pipeline._failed_reference)

    def test_area_ratio_rejects_too_small_object(self):
        pipeline = make_pipeline()
        pipeline._failed_reference = pipeline._reference_from_object(
            {"polygon": square(100, 100, 40)}, (400, 400)
        )
        small = {"polygon": square(115, 115, 15)}
        pipeline._match_failed_object([small], (400, 400))
        self.assertFalse(small.get("_blocked"))

    def test_mark_pick_failed_copies_last_reference(self):
        pipeline = make_pipeline()
        pipeline._last_pick_reference = pipeline._reference_from_object(
            {"polygon": square(100, 100)}, (400, 400)
        )
        self.assertTrue(pipeline.mark_pick_failed())
        self.assertIsNotNone(pipeline._failed_reference)
        self.assertIsNot(pipeline._failed_reference, pipeline._last_pick_reference)


if __name__ == "__main__":
    unittest.main()
