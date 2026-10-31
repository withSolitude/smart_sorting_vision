"""顶部完整 mask 的形状核心抓取点策略回归测试。"""
import unittest

import cv2
import numpy as np

from vision.grasp.collision_aware_grasp_planner import CollisionAwareGraspPlanner


class ShapeCoreMaskTests(unittest.TestCase):
    @staticmethod
    def _mask_with_side_projection() -> np.ndarray:
        # 中央方形近似真实顶部，右侧窄条模拟透视下被带入 mask 的侧面。
        mask = np.zeros((160, 220), dtype=np.uint8)
        cv2.rectangle(mask, (70, 55), (150, 115), 255, -1)
        cv2.rectangle(mask, (150, 70), (184, 100), 255, -1)
        return mask

    def test_core_center_is_less_affected_by_side_projection(self):
        mask = self._mask_with_side_projection()
        ys, xs = np.nonzero(mask > 0)
        full_center = np.array([float(xs.mean()), float(ys.mean())])

        core = CollisionAwareGraspPlanner.build_top_core_mask(
            mask,
            shape_name="rectangular_prism",
        )
        core_ys, core_xs = np.nonzero(core > 0)
        core_center = np.array([float(core_xs.mean()), float(core_ys.mean())])
        true_top_center = np.array([110.0, 85.0])

        self.assertLess(
            float(np.linalg.norm(core_center - true_top_center)),
            float(np.linalg.norm(full_center - true_top_center)),
        )
        self.assertGreater(int(np.count_nonzero(core)), 0)

    def test_shape_strategy_keeps_symmetric_objects_grabbable(self):
        mask = np.zeros((180, 180), dtype=np.uint8)
        cv2.circle(mask, (90, 90), 45, 255, -1)

        core = CollisionAwareGraspPlanner.build_top_core_mask(mask, shape_name="sphere")
        ys, xs = np.nonzero(core > 0)
        self.assertGreater(len(xs), 0)
        self.assertAlmostEqual(float(xs.mean()), 90.0, delta=1.0)
        self.assertAlmostEqual(float(ys.mean()), 90.0, delta=1.0)

    def test_shape_strategy_is_more_conservative_for_flat_prisms(self):
        mask = np.zeros((180, 220), dtype=np.uint8)
        cv2.rectangle(mask, (40, 50), (180, 130), 255, -1)

        prism = CollisionAwareGraspPlanner.build_top_core_mask(
            mask, shape_name="rectangular_prism"
        )
        cylinder = CollisionAwareGraspPlanner.build_top_core_mask(
            mask, shape_name="cylinder"
        )
        sphere = CollisionAwareGraspPlanner.build_top_core_mask(
            mask, shape_name="sphere"
        )
        self.assertLessEqual(np.count_nonzero(prism), np.count_nonzero(cylinder))
        self.assertLessEqual(np.count_nonzero(cylinder), np.count_nonzero(sphere))

    def test_triangle_geometry_uses_pyramid_cone_core_rule(self):
        mask = np.zeros((180, 220), dtype=np.uint8)
        triangle = np.asarray([[55, 135], [110, 40], [165, 135]], dtype=np.int32)
        cv2.fillPoly(mask, [triangle.reshape(-1, 1, 2)], 255)
        auto_core = CollisionAwareGraspPlanner.build_top_core_mask(mask)
        rectangular_core = CollisionAwareGraspPlanner.build_top_core_mask(
            mask, shape_name="rectangular_prism"
        )
        self.assertGreater(np.count_nonzero(auto_core), 0)
        self.assertGreater(np.count_nonzero(auto_core), np.count_nonzero(rectangular_core))

    def test_single_object_square_is_not_misclassified_as_circle(self):
        mask = np.zeros((180, 180), dtype=np.uint8)
        cv2.rectangle(mask, (45, 45), (135, 135), 255, -1)

        auto = CollisionAwareGraspPlanner.build_top_core_mask(
            mask, shape_name="object"
        )
        rectangular = CollisionAwareGraspPlanner.build_top_core_mask(
            mask, shape_name="rectangular_prism"
        )
        circular = CollisionAwareGraspPlanner.build_top_core_mask(
            mask, shape_name="sphere"
        )
        # 方块圆度接近0.785，自动策略应使用矩形的更保守核心。
        self.assertEqual(int(np.count_nonzero(auto)), int(np.count_nonzero(rectangular)))
        self.assertLess(int(np.count_nonzero(auto)), int(np.count_nonzero(circular)))


if __name__ == "__main__":
    unittest.main()
