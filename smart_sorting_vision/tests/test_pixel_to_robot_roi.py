"""ROI重裁剪后的正反映射回归测试；不需要相机、模型或下位机。"""
from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import cv2
import numpy as np

from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.grasp.gripper_footprint import GripperFootprintModel


class PixelToRobotRoiTests(unittest.TestCase):
    def setUp(self):
        self.old_roi = [505, 4, 1739, 1052]
        self.current_roi = [509, 0, 1735, 1049]
        self.H = np.array([[.18, -.006, -200], [.002, .18, -173], [-.00003, .00002, 1.]])
        self.old = PixelToRobotTransform(
            self.H, 1234, 1048, tcp_offset_x_mm=2.5, tcp_offset_y_mm=-1.2,
            metadata={"points_pixel": [[89., 20.], [1123., 34.], [1150., 758.], [108., 963.]]},
        )
        self.full = self.old.to_full_image(self.old_roi, (1920, 1080))
        self.full.configure_camera({"width": 1920, "height": 1080, "safe_roi": self.current_roi})

    def test_legacy_mapping_and_size_check_are_unchanged(self):
        expected = cv2.perspectiveTransform(np.array([[[100., 300.]]]), self.H)[0, 0]
        np.testing.assert_allclose(self.old.pixel_to_robot(100, 300, apply_tcp_offset=False), expected)
        self.old.check_frame(np.zeros((1048, 1234), np.uint8))
        with self.assertRaises(RuntimeError):
            self.old.check_frame(np.zeros((1049, 1226), np.uint8))

    def test_migration_exactly_preserves_original_points(self):
        old_pixels = np.array(self.old.metadata["points_pixel"])
        # 新坐标原点相对旧原点为(+4,-4)，不是按宽高比例缩放。
        current_pixels = old_pixels - [4, -4]
        np.testing.assert_allclose(
            self.full.pixels_to_robot(current_pixels), self.old.pixels_to_robot(old_pixels), atol=1e-10,
        )
        np.testing.assert_allclose(self.full.metadata["points_pixel"], old_pixels + [505, 4])
        self.assertEqual(self.full.tcp_offset_x_mm, self.old.tcp_offset_x_mm)

    def test_roi_changes_do_not_change_physical_coordinates(self):
        full_pixels = np.array([[700., 100.], [1200., 800.]])
        expected = self.full.pixels_to_robot(full_pixels - [509, 0])
        self.full.configure_camera({"width": 1920, "height": 1080, "safe_roi": [400, 20, 1800, 1060]})
        np.testing.assert_allclose(self.full.pixels_to_robot(full_pixels - [400, 20]), expected)
        self.full.check_frame(np.zeros((1040, 1400), np.uint8))

    def test_round_trip_and_footprint_share_current_roi(self):
        points = np.array([[85., 24.], [600., 500.], [1100., 800.]])
        for apply_tcp in (True, False):
            robot = self.full.pixels_to_robot(points, apply_tcp_offset=apply_tcp)
            np.testing.assert_allclose(
                self.full.robot_to_pixels(robot, apply_tcp_offset=apply_tcp), points, atol=1e-9,
            )
        footprint = GripperFootprintModel(transform=self.full)
        robot = self.full.pixels_to_robot(points, apply_tcp_offset=False)
        np.testing.assert_allclose(footprint._robot_to_pixels(robot), points, atol=1e-9)

    def test_yaw_uses_same_physical_axis_after_cropping(self):
        for angle in (-80., -20., 0., 40., 89.):
            before = self.old.robot_angle_from_image_angle(500, 450, angle)
            after = self.full.robot_angle_from_image_angle(496, 454, angle)
            self.assertAlmostEqual(before, after, places=10)

    def test_geometry_errors_are_distinct_from_roi_resize(self):
        with self.assertRaises(ValueError):
            self.full.configure_camera({"width": 1280, "height": 720, "safe_roi": [0, 0, 1000, 700]})
        with self.assertRaises(ValueError):
            self.old.to_full_image([509, 0, 1735, 1049], (1920, 1080))

    def test_calibration_uses_full_undistorted_image_not_raw_or_roi(self):
        from scripts.last_debug import calibrate_pixel_to_robot as entry

        class Camera:
            def open(self):
                pass

            def release(self):
                pass

            def read_undistorted(self):
                return np.zeros((30, 40, 3), np.uint8)

            def read_full(self):
                raise AssertionError("read_full返回RAW，不能用于去畸变坐标标定")

            def read(self):
                raise AssertionError("标定不应绑定当前ROI裁剪")

        points = [(3, 3), (36, 3), (36, 26), (3, 26)]

        def click_four_points(_window, callback):
            for x, y in points:
                callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mapping.json"
            cfg = {"calibration": {"pixel_to_robot": output}}
            with patch.object(entry, "load_context", return_value=(cfg, Camera, None)), \
                 patch.object(entry.cv2, "namedWindow"), \
                 patch.object(entry.cv2, "setMouseCallback", side_effect=click_four_points), \
                 patch.object(entry.cv2, "imshow"), \
                 patch.object(entry.cv2, "waitKey", return_value=13), \
                 patch.object(entry.cv2, "destroyAllWindows"), \
                 patch("sys.argv", ["calibrate_pixel_to_robot"]), \
                 patch("builtins.input", side_effect=["0 0", "10 0", "10 10", "0 10"]):
                entry.main()
            transform = PixelToRobotTransform.load(output)
            self.assertEqual(transform.coordinate_space, "undistorted_full")
            self.assertEqual((transform.image_width, transform.image_height), (40, 30))
            np.testing.assert_allclose(transform.pixels_to_robot(points), [[0, 0], [10, 0], [10, 10], [0, 10]], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
