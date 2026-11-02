"""验证顶部像素标定的旧轴顺序交换。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision.calibration.pixel_to_robot import PixelToRobotTransform


class PixelToRobotAxisSwapTests(unittest.TestCase):
    def test_forward_and_inverse_keep_the_same_point(self):
        transform = PixelToRobotTransform(
            np.array(
                [[2.0, 0.0, 10.0], [0.0, 3.0, -4.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            100,
            100,
            swap_xy=True,
        )
        self.assertEqual(transform.pixel_to_robot(5.0, 6.0), (14.0, 20.0))
        np.testing.assert_allclose(
            transform.pixels_to_robot([(5.0, 6.0)]), [[14.0, 20.0]]
        )
        np.testing.assert_allclose(transform.robot_to_pixel(14.0, 20.0), [5.0, 6.0])

    def test_camera_config_flag_is_used(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mapping.json"
            PixelToRobotTransform(
                np.eye(3),
                20,
                20,
            ).save(path)
            transform = PixelToRobotTransform.load(
                path, camera_config={"swap_robot_xy": True}
            )
            self.assertTrue(transform.swap_xy)
            self.assertEqual(transform.pixel_to_robot(4.0, 5.0), (5.0, 4.0))


if __name__ == "__main__":
    unittest.main()
