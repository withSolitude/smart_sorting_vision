"""标定文件轴顺序修正：正向和反向坐标必须互相一致。"""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from vision.calibration.pixel_to_robot import PixelToRobotTransform


class PixelToRobotAxisSwapTests(unittest.TestCase):
    def test_swap_changes_forward_order_and_preserves_inverse(self):
        # H保存的输出是 [现场Y, 现场X]；swap后对外返回真实 [X,Y]。
        transform = PixelToRobotTransform(
            np.array([[2.0, 0.0, 10.0], [0.0, 3.0, -4.0], [0.0, 0.0, 1.0]]),
            100, 100, swap_xy=True,
        )
        self.assertEqual(transform.pixel_to_robot(5.0, 6.0), (14.0, 20.0))
        np.testing.assert_allclose(transform.pixels_to_robot([(5.0, 6.0)]), [[14.0, 20.0]])
        pixel = transform.robot_to_pixel(14.0, 20.0)
        np.testing.assert_allclose(pixel, [5.0, 6.0])

    def test_camera_config_flag_is_used_when_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mapping.json"
            PixelToRobotTransform(
                np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]),
                20, 20,
            ).save(path)
            transform = PixelToRobotTransform.load(path, camera_config={"swap_robot_xy": True})
            self.assertTrue(transform.swap_xy)
            self.assertEqual(transform.pixel_to_robot(4.0, 5.0), (8.0, 6.0))


if __name__ == "__main__":
    unittest.main()
