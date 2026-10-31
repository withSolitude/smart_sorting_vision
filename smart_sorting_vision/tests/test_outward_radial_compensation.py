"""向外径向补偿回归；只用临时内参，不打开相机、模型或下位机。"""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vision.grasp.outward_radial_compensation import OutwardRadialCompensator


class OutwardRadialCompensationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.intrinsics = self.folder / "intrinsics.npz"
        # 两份主点故意不同，用来检测是否误用了去畸变前的原始内参。
        self.new_matrix = np.array([[900.0, 0, 940.0], [0, 910.0, 520.0], [0, 0, 1.0]])
        np.savez(
            self.intrinsics,
            new_camera_matrix=self.new_matrix,
            camera_matrix=np.array([[1000.0, 0, 970.0], [0, 1000.0, 550.0], [0, 0, 1.0]]),
            image_size=np.array([1920, 1080]),
        )
        self.camera = {"width": 1920, "height": 1080, "safe_roi": [400, 40, 1800, 1040]}
        self.config = {"enabled": True, "gain": 0.05, "max_shift_px": 100.0, "center_full_px": None}

    def make(self, **changes):
        return OutwardRadialCompensator.from_config(
            {**self.config, **changes}, camera_config=self.camera, intrinsics_path=self.intrinsics,
        )

    def test_automatic_center_uses_new_intrinsics_and_current_roi_origin(self):
        compensation = self.make()
        self.assertEqual(compensation.center_roi_px, (540.0, 480.0))
        self.assertNotEqual(compensation.center_roi_px, (570.0, 510.0))
        # 不是裁剪图中心(700,500)，只改变ROI不应改变补偿的全图中心。
        self.assertNotEqual(compensation.center_roi_px, (700.0, 500.0))

    def test_all_quadrants_move_only_outward_along_the_same_ray(self):
        compensation = self.make()
        center = np.array(compensation.center_roi_px)
        for radial in ((100, 50), (-100, 50), (-100, -50), (100, -50), (0, 200)):
            with self.subTest(radial=radial):
                direction = np.array(radial, dtype=float)
                point = center + direction
                corrected = np.array(compensation.apply(tuple(point)))
                shift = corrected - point
                np.testing.assert_allclose(shift, direction * 0.05, atol=1e-12)
                self.assertGreater(float(shift @ direction), 0.0)
                self.assertGreater(np.linalg.norm(corrected - center), np.linalg.norm(point - center))

    def test_shift_limit_reduces_length_without_reversing_or_rotating_direction(self):
        compensation = self.make(gain=0.1, max_shift_px=10.0)
        center = np.array(compensation.center_roi_px)
        point = center + [300.0, 400.0]
        shift = np.array(compensation.apply(tuple(point))) - point
        # 原本需要50像素，限幅为10像素；3:4方向必须保持。
        np.testing.assert_allclose(shift, [6.0, 8.0], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(shift)), 10.0)

    def test_center_zero_gain_and_zero_limit_do_not_move_point(self):
        compensation = self.make()
        center = compensation.center_roi_px
        self.assertEqual(compensation.apply(center), center)
        point = (700.25, 700.75)
        self.assertEqual(self.make(gain=0.0).apply(point), point)
        self.assertEqual(self.make(max_shift_px=0.0).apply(point), point)

    def test_disabled_mode_needs_neither_camera_nor_intrinsics(self):
        for config in (None, {"enabled": False}):
            with self.subTest(config=config):
                compensation = OutwardRadialCompensator.from_config(
                    config, camera_config=None, intrinsics_path=self.folder / "missing.npz",
                )
                self.assertFalse(compensation.enabled)
                self.assertEqual(compensation.apply((123.25, 456.75)), (123.25, 456.75))

    def test_manual_full_image_center_does_not_load_intrinsics(self):
        compensation = OutwardRadialCompensator.from_config(
            {**self.config, "center_full_px": [1050.0, 600.0]},
            camera_config=self.camera, intrinsics_path=self.folder / "missing.npz",
        )
        self.assertEqual(compensation.center_roi_px, (650.0, 560.0))
        self.assertEqual(compensation.apply((750.0, 560.0)), (755.0, 560.0))

    def test_roi_recropping_preserves_corrected_full_image_point(self):
        original = self.make(gain=0.1, max_shift_px=10.0)
        changed_camera = {**self.camera, "safe_roi": [500, 20, 1850, 1060]}
        recropped = OutwardRadialCompensator.from_config(
            {**self.config, "gain": 0.1, "max_shift_px": 10.0},
            camera_config=changed_camera, intrinsics_path=self.intrinsics,
        )
        full_point = np.array([1200.0, 700.0])
        first_origin = np.array(self.camera["safe_roi"][:2])
        second_origin = np.array(changed_camera["safe_roi"][:2])
        corrected_first = np.array(original.apply(tuple(full_point - first_origin))) + first_origin
        corrected_second = np.array(recropped.apply(tuple(full_point - second_origin))) + second_origin
        np.testing.assert_allclose(corrected_first, corrected_second, atol=1e-10)

    def test_negative_and_nonfinite_parameters_are_rejected(self):
        for gain in (-0.01, float("nan"), float("inf")):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                self.make(gain=gain)
        for maximum in (-1.0, float("nan"), float("inf")):
            with self.subTest(max_shift_px=maximum), self.assertRaises(ValueError):
                self.make(max_shift_px=maximum)

    def test_incompatible_intrinsics_are_rejected_instead_of_using_wrong_center(self):
        cases = [
            {"new_camera_matrix": self.new_matrix, "image_size": [1280, 720]},
            {"new_camera_matrix": np.eye(2), "image_size": [1920, 1080]},
            {"camera_matrix": self.new_matrix, "image_size": [1920, 1080]},
            {"new_camera_matrix": np.full((3, 3), float("nan"))},
        ]
        for number, fields in enumerate(cases):
            with self.subTest(case=number):
                path = self.folder / f"bad_{number}.npz"
                np.savez(path, **fields)
                with self.assertRaises(ValueError):
                    OutwardRadialCompensator.from_config(
                        self.config, camera_config=self.camera, intrinsics_path=path,
                    )


if __name__ == "__main__":
    unittest.main()
