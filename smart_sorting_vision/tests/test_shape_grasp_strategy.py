"""顶部完整 mask 的形状核心抓取点策略回归测试。"""
import unittest

import cv2
import numpy as np

from vision.grasp.collision_aware_grasp_planner import CollisionAwareGraspPlanner
from vision.grasp.orientation_planner import (
    axial_difference_deg,
    plan_grasp_axis,
    plan_parallel_edge_grasp_options,
)


class _IdentityTransform:
    """离线测试只关心像素几何，不依赖相机或机械标定文件。"""

    @staticmethod
    def pixel_to_robot(x, y, **kwargs):
        return float(x), float(y)

    @staticmethod
    def robot_angle_from_image_angle(x, y, angle_deg):
        return float(angle_deg)


class ShapeCoreMaskTests(unittest.TestCase):
    @staticmethod
    def _polygon(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _planner(config=None):
        return CollisionAwareGraspPlanner(
            obstacle_margin=0,
            border_margin=0,
            min_clearance=1.0,
            shape_core_config=config,
        )

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

    def test_orientation_keeps_the_core_point_instead_of_restoring_full_centroid(self):
        # 旧实现核心抓点在x≈112，但方向规划又覆盖成完整轮廓中点x≈120。
        mask = self._mask_with_side_projection()
        polygon = self._polygon(mask)
        result = self._planner().plan(polygon, [], mask.shape, shape_name="object")
        self.assertTrue(result.valid)
        self.assertLess(result.grasp_point[0], 116)

        legacy_orientation = plan_grasp_axis(
            polygon=polygon,
            grasp_pixel=result.grasp_point,
            transform=_IdentityTransform(),
            safe_mask=result.preferred_mask,
        )
        self.assertGreater(legacy_orientation.grasp_pixel_override[0], 117)

        orientation = plan_grasp_axis(
            polygon=result.planning_polygon,
            grasp_pixel=result.grasp_point,
            transform=_IdentityTransform(),
            safe_mask=result.preferred_mask,
            preserve_grasp_point=True,
        )
        self.assertEqual(orientation.grasp_pixel_override, result.grasp_point)
        # 方向族搜索也必须保留同一个核心点，不能在另一条调用路径中改回去。
        options = plan_parallel_edge_grasp_options(
            polygon=result.planning_polygon,
            grasp_pixel=result.grasp_point,
            transform=_IdentityTransform(),
            safe_mask=result.preferred_mask,
            preserve_grasp_point=True,
        )
        self.assertTrue(options)
        self.assertTrue(all(o.grasp_pixel_override == result.grasp_point for o in options))

    def test_direction_is_computed_from_core_instead_of_side_extension(self):
        # 上方的细长延伸会把完整凸包主夹取轴带成竖直；核心保留左侧粗主体。
        polygon = np.asarray(
            [[30, 30], [170, 30], [170, 60], [90, 60], [90, 150], [30, 150]],
            dtype=np.float32,
        )
        result = self._planner().plan(polygon, [], (200, 220), shape_name="object")
        self.assertTrue(result.valid)
        common = dict(
            grasp_pixel=result.grasp_point,
            transform=_IdentityTransform(),
            safe_mask=result.preferred_mask,
            preserve_grasp_point=True,
        )
        full = plan_grasp_axis(polygon=polygon, **common)
        core = plan_grasp_axis(polygon=result.planning_polygon, **common)
        self.assertLess(abs(axial_difference_deg(core.grasp_axis_robot_deg, 0.0)), 5.0)
        self.assertGreater(
            abs(axial_difference_deg(core.grasp_axis_robot_deg, full.grasp_axis_robot_deg)),
            80.0,
        )
        self.assertEqual(core.grasp_pixel_override, result.grasp_point)

    def test_symmetric_objects_keep_their_center_and_legal_direction(self):
        for shape_name in ("cube", "sphere"):
            with self.subTest(shape_name=shape_name):
                mask = np.zeros((180, 180), dtype=np.uint8)
                if shape_name == "cube":
                    cv2.rectangle(mask, (45, 45), (135, 135), 255, -1)
                else:
                    cv2.circle(mask, (90, 90), 45, 255, -1)
                result = self._planner().plan(
                    self._polygon(mask), [], mask.shape, shape_name=shape_name,
                )
                self.assertTrue(result.valid)
                self.assertEqual(result.grasp_point, (90, 90))
                orientation = plan_grasp_axis(
                    polygon=result.planning_polygon,
                    grasp_pixel=result.grasp_point,
                    transform=_IdentityTransform(),
                    safe_mask=result.preferred_mask,
                    preserve_grasp_point=True,
                )
                self.assertEqual(orientation.grasp_pixel_override, (90, 90))
                self.assertTrue(np.isfinite(orientation.grasp_axis_robot_deg))
                self.assertGreaterEqual(orientation.grasp_axis_robot_deg, -90.0)
                self.assertLess(orientation.grasp_axis_robot_deg, 90.0)

    def test_disabled_strategy_uses_full_geometry(self):
        mask = self._mask_with_side_projection()
        polygon = self._polygon(mask)
        result = self._planner({"enabled": False}).plan(
            polygon, [], mask.shape, shape_name="object",
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.strategy_name, "disabled")
        np.testing.assert_array_equal(result.planning_polygon, polygon)
        np.testing.assert_array_equal(result.preferred_mask, result.safe_mask)
        self.assertGreater(result.grasp_point[0], 117)

    def test_tiny_core_explicitly_falls_back_to_safe_geometry(self):
        polygon = np.asarray([[45, 45], [135, 45], [135, 135], [45, 135]], np.float32)
        # 极强收缩只剩中心一个像素，目标本身仍有足够安全空间。
        result = self._planner({"ratios": {"quadrilateral": 0.99}}).plan(
            polygon, [], (180, 180), shape_name="cube",
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.strategy_name, "safe_fallback")
        self.assertEqual(result.grasp_point, (90, 90))
        np.testing.assert_array_equal(result.planning_polygon, polygon)
        np.testing.assert_array_equal(result.preferred_mask, result.safe_mask)


if __name__ == "__main__":
    unittest.main()
