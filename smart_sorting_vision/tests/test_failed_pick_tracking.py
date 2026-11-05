"""失败抓取：用抓取时的 mask 模板对比，未夹过的优先，全部夹过才允许重试。"""
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
    pipeline._last_pick_reference = None
    pipeline._failed_references = []
    pipeline._normalize_polygon = lambda poly: np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return pipeline


def add_failure(pipeline: TopPickPipeline, polygon, image_shape=(400, 400)):
    reference = pipeline._reference_from_object({"polygon": polygon}, image_shape)
    pipeline._failed_references.append(reference)
    return reference


class FailedPickTrackingTests(unittest.TestCase):
    def test_moved_object_is_matched_and_marked(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        moved = {"polygon": square(130, 100)}
        pipeline._match_failed_object([moved], (400, 400))
        self.assertTrue(moved.get("_blocked"))

    def test_two_failed_objects_are_both_matched(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        add_failure(pipeline, square(100, 160))
        first = {"polygon": square(130, 100)}
        second = {"polygon": square(130, 160)}
        pipeline._match_failed_object([first, second], (400, 400))
        self.assertTrue(first.get("_blocked"))
        self.assertTrue(second.get("_blocked"))

    def test_template_is_fixed_while_object_moves(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        first_frame = {"polygon": square(130, 100)}
        pipeline._match_failed_object([first_frame], (400, 400))
        second_frame = {"polygon": square(160, 100)}
        pipeline._match_failed_object([second_frame], (400, 400))
        self.assertTrue(first_frame.get("_blocked"))
        self.assertTrue(second_frame.get("_blocked"))
        # 原始模板中心始终是抓取时的位置。
        self.assertAlmostEqual(
            float(pipeline._failed_references[0]["center_px"][0]), 120.0, delta=1.0
        )

    def test_far_object_is_not_matched(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        far = {"polygon": square(320, 320)}
        pipeline._match_failed_object([far], (400, 400))
        self.assertFalse(far.get("_blocked"))

    def test_reset_clears_memory(self):
        pipeline = make_pipeline()
        add_failure(pipeline, square(100, 100))
        pipeline._last_pick_reference = pipeline._reference_from_object(
            {"polygon": square(200, 200)}, (400, 400)
        )
        pipeline.reset_failed_picks()
        self.assertEqual(pipeline._failed_references, [])
        self.assertIsNone(pipeline._last_pick_reference)

    def test_mark_pick_failed_appends_and_merges(self):
        pipeline = make_pipeline()
        pipeline._last_pick_reference = pipeline._reference_from_object(
            {"polygon": square(100, 100)}, (400, 400)
        )
        self.assertTrue(pipeline.mark_pick_failed())
        self.assertEqual(len(pipeline._failed_references), 1)
        # 同一物体的重复失败只合并，不重复记录。
        pipeline._last_pick_reference = pipeline._reference_from_object(
            {"polygon": square(102, 100)}, (400, 400)
        )
        pipeline.mark_pick_failed()
        self.assertEqual(len(pipeline._failed_references), 1)


if __name__ == "__main__":
    unittest.main()
