"""多方向规划必须区分边-边、边-顶点和顶点-顶点接触。"""
from __future__ import annotations

import unittest

import numpy as np

from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.grasp.orientation_planner import plan_parallel_edge_grasp_options


class DirectionContactMetadataTests(unittest.TestCase):
    def options(self, polygon, *, include_vertex_contacts=False, **kwargs):
        return plan_parallel_edge_grasp_options(
            polygon=np.asarray(polygon, dtype=np.float32),
            grasp_pixel=(100.0, 100.0),
            transform=PixelToRobotTransform(np.eye(3), 240, 240),
            include_vertex_contacts=include_vertex_contacts,
            **kwargs,
        )

    def test_triangle_vertex_contacts_are_not_reported_as_parallel_edges(self):
        triangle = [[100, 30], [180, 170], [20, 170]]
        self.assertEqual(self.options(triangle), [])
        options = self.options(triangle, include_vertex_contacts=True)
        self.assertTrue(options)
        for option in options:
            self.assertEqual(option.contact_type, "edge_vertex")
            self.assertEqual(option.mode, "antipodal_edge_vertex")
            self.assertTrue(option.reliable)

    def test_rectangle_keeps_real_parallel_edge_metadata(self):
        rectangle = [[30, 60], [190, 60], [190, 140], [30, 140]]
        for include_vertex in (False, True):
            with self.subTest(include_vertex=include_vertex):
                options = self.options(rectangle, include_vertex_contacts=include_vertex)
                self.assertEqual(len(options), 2)
                self.assertTrue(all(o.contact_type == "edge_edge" for o in options))
                self.assertTrue(all(o.mode == "antipodal_edge_edge" for o in options))

    def test_insufficient_edge_span_is_not_reported_as_edge_contact(self):
        # 将最小可用接触跨度设得大于物体边长，测试低跨度接触的标签。
        options = self.options(
            [[30, 60], [190, 60], [190, 140], [30, 140]],
            include_vertex_contacts=True, min_contact_span_px=1000.0,
        )
        self.assertTrue(options)
        for option in options:
            self.assertEqual(option.contact_type, "vertex_vertex")
            self.assertEqual(option.mode, "antipodal_vertex_vertex")
            self.assertEqual(option.reliable, option.contact_quality >= 0.45)


if __name__ == "__main__":
    unittest.main()
