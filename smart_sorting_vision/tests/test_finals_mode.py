"""决赛模式（finals）单元测试：顶部形状族、箱位判定、候选过滤、软结束。"""
from __future__ import annotations

import types
import unittest

from runtime.sorting_runtime import SortingRuntime, RuntimeState
from vision.classify.top_classifier import (
    TopObjectClassifier,
    TopClassResult,
    shape_family,
)
from vision.storage.object_info import ObjectInfo
from vision.storage.storage_manager import StorageManager


class TopClassifierFamilyTests(unittest.TestCase):
    def test_side_shape_to_family(self):
        self.assertEqual(shape_family("三棱柱"), "tri")
        self.assertEqual(shape_family("三棱锥"), "tri")
        self.assertEqual(shape_family("正方体"), "quad")
        self.assertEqual(shape_family("长方体"), "quad")
        self.assertEqual(shape_family("四棱锥"), "quad")
        self.assertEqual(shape_family("四棱台"), "quad")
        self.assertEqual(shape_family("五棱柱"), "pent")
        self.assertEqual(shape_family("六棱柱"), "hex")
        for round_shape in ("圆柱", "圆台", "圆锥", "球"):
            self.assertEqual(shape_family(round_shape), "round")
        self.assertEqual(shape_family("空托盘"), "unknown")
        self.assertEqual(shape_family("不存在"), "unknown")

    def test_geometry_rules(self):
        family = TopObjectClassifier._shape_family(
            rectangularity=0.95, circularity=0.97, hull_vertex_count=12, aspect_ratio=1.0
        )
        self.assertEqual(family[0], "round")

        family = TopObjectClassifier._shape_family(
            rectangularity=0.55, circularity=0.6, hull_vertex_count=3, aspect_ratio=1.0
        )
        self.assertEqual(family[0], "tri")

        family = TopObjectClassifier._shape_family(
            rectangularity=0.92, circularity=0.7, hull_vertex_count=4, aspect_ratio=1.2
        )
        self.assertEqual(family[0], "quad")

        self.assertEqual(
            TopObjectClassifier._shape_family(
                rectangularity=0.4, circularity=0.5, hull_vertex_count=5, aspect_ratio=1.0
            )[0],
            "pent",
        )
        self.assertEqual(
            TopObjectClassifier._shape_family(
                rectangularity=0.4, circularity=0.5, hull_vertex_count=6, aspect_ratio=1.0
            )[0],
            "hex",
        )
        self.assertEqual(
            TopObjectClassifier._shape_family(
                rectangularity=0.1, circularity=0.3, hull_vertex_count=9, aspect_ratio=1.0
            )[0],
            "unknown",
        )


def make_runtime(finals_enabled=True):
    rt = SortingRuntime.__new__(SortingRuntime)
    rt.finals_enabled = finals_enabled
    rt.finals_required_boxes = 6
    rt.finals_min_confidence = 0.45
    rt.finals_no_match_frames = 8
    rt.finals_gamble_enabled = True
    rt.finals_finished = False
    rt._finals_gamble_pick = False
    rt._finals_current_cycle_gamble = False
    rt._finals_no_match = 0
    rt.storage_manager = StorageManager(6)
    rt.on_session_finished = None
    return rt


def fill_boxes(rt, count=6):
    classes = [
        ("red", "正方体"),
        ("blue", "圆柱"),
        ("green", "三棱柱"),
        ("yellow", "六棱柱"),
        ("cyan", "长方体"),
        ("purple", "球"),
    ]
    for color, shape in classes[:count]:
        obj = ObjectInfo(color=color, shape=shape)
        reservation = rt.storage_manager.reserve(obj)
        assert reservation.success
        rt.storage_manager.commit(reservation.reservation_id)


class FinalsRuntimeTests(unittest.TestCase):
    def test_disabled_never_filters(self):
        rt = make_runtime(finals_enabled=False)
        fill_boxes(rt)
        self.assertIsNone(rt._finals_allowed())

    def test_phase1_no_filter_before_boxes_full(self):
        rt = make_runtime()
        fill_boxes(rt, count=5)
        self.assertFalse(rt._finals_boxes_full())
        self.assertIsNone(rt._finals_allowed())

    def test_phase2_filters_by_color_and_family(self):
        rt = make_runtime()
        fill_boxes(rt)
        self.assertTrue(rt._finals_boxes_full())
        allowed = rt._finals_allowed()
        self.assertIsNotNone(allowed)

        hit = types.SimpleNamespace(
            final_color="red", final_shape_family="quad", final_confidence=0.9
        )
        self.assertTrue(allowed(hit))

        wrong_color = types.SimpleNamespace(
            final_color="blue", final_shape_family="quad", final_confidence=0.9
        )
        self.assertFalse(allowed(wrong_color))

        wrong_family = types.SimpleNamespace(
            final_color="red", final_shape_family="round", final_confidence=0.9
        )
        self.assertFalse(allowed(wrong_family))

        low_conf = types.SimpleNamespace(
            final_color="red", final_shape_family="quad", final_confidence=0.2
        )
        self.assertFalse(allowed(low_conf))

        unknown = types.SimpleNamespace(
            final_color="unknown", final_shape_family="unknown", final_confidence=0.9
        )
        self.assertFalse(allowed(unknown))

    def test_gamble_bypasses_filter(self):
        rt = make_runtime()
        fill_boxes(rt)
        rt._finals_gamble_pick = True
        self.assertIsNone(rt._finals_allowed())

    def test_soft_finish_sets_state_and_calls_callback(self):
        rt = make_runtime()
        fill_boxes(rt)
        rt.running = True
        rt.top_pipeline = None
        rt.side_pipeline = None
        rt._clear_cycle_data = lambda **kwargs: None
        rt._safe_reset = lambda pipeline: None
        rt._enter_state = lambda state: setattr(rt, "state", state)
        rt._send_stop_best_effort = lambda: None
        reasons = []
        rt.on_session_finished = reasons.append

        rt._finish_finals("no_available_box:no_available_box")

        self.assertTrue(rt.finals_finished)
        self.assertFalse(rt.running)
        self.assertEqual(rt.state, RuntimeState.IDLE)
        self.assertEqual(reasons, ["no_available_box:no_available_box"])


if __name__ == "__main__":
    unittest.main()
