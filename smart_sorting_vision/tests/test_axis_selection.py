"""确定性选轴：两个正方体贴在一起时，闭合轴应平行于接触面。"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from vision.pipeline.top_pick_pipeline import prefer_axis_by_neighbors


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


def family(axis_image_deg: float) -> SimpleNamespace:
    return SimpleNamespace(closing_axis_image_deg=axis_image_deg)


class PreferAxisTests(unittest.TestCase):
    def test_two_cubes_side_by_side_pick_perpendicular_axis(self):
        target = square(100, 100)
        neighbor = square(140, 100)  # 贴在 +X 方向，接触面法向 = 0°
        # 0° 闭合轴会夹缝隙方向；90° 才是平行于接触面的正确方向。
        chosen = prefer_axis_by_neighbors(
            [family(0.0), family(90.0)],
            target,
            [neighbor],
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.closing_axis_image_deg, 90.0)

    def test_two_cubes_stack_along_y_pick_horizontal_axis(self):
        target = square(100, 100)
        neighbor = square(100, 140)  # 贴在 +Y 方向，接触面法向 = 90°
        chosen = prefer_axis_by_neighbors(
            [family(0.0), family(90.0)],
            target,
            [neighbor],
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.closing_axis_image_deg, 0.0)

    def test_far_neighbor_keeps_primary_axis(self):
        target = square(100, 100)
        neighbor = square(400, 100)  # 离得远，不算贴合
        self.assertIsNone(
            prefer_axis_by_neighbors(
                [family(0.0), family(90.0)],
                target,
                [neighbor],
            )
        )

    def test_no_neighbor_keeps_primary_axis(self):
        self.assertIsNone(
            prefer_axis_by_neighbors(
                [family(0.0), family(90.0)],
                square(100, 100),
                [],
            )
        )

    def test_ambiguous_neighbors_keep_primary_axis(self):
        target = square(100, 100)
        others = [square(140, 100), square(100, 140), square(60, 100), square(100, 60)]
        self.assertIsNone(
            prefer_axis_by_neighbors(
                [family(0.0), family(90.0)],
                target,
                others,
            )
        )


if __name__ == "__main__":
    unittest.main()
