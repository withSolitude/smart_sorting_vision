from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)


from ui.sorting_display import (
    SortingDisplayWindow,
)


def make_test_image(bgr):
    """
    生成一张假的 BGR 图像。

    这里故意不 import cv2，
    用于验证 PyQt5 显示系统。
    """

    image = np.zeros(
        (
            360,
            480,
            3,
        ),
        dtype=np.uint8,
    )

    image[60:300, 140:340] = bgr

    return image


def main():

    app = QApplication(
        sys.argv
    )

    window = SortingDisplayWindow(
        fullscreen=True
    )

    def item_1():

        window.add_finished_item(
            box_id=1,
            product_name="正方体_蓝色",
            image_bgr=make_test_image((220, 0, 0)),
            box_count=1,
        )

    def item_2():

        window.add_finished_item(
            box_id=2,
            product_name="三棱柱_红色",
            image_bgr=make_test_image((0, 0, 220)),
            box_count=1,
        )

    def item_3():

        window.add_finished_item(
            box_id=1,
            product_name="正方体_蓝色",
            image_bgr=make_test_image((100, 0, 0)),
            box_count=2,
        )

    QTimer.singleShot(
        1000,
        item_1,
    )

    QTimer.singleShot(
        3000,
        item_2,
    )

    QTimer.singleShot(
        5000,
        item_3,
    )

    sys.exit(
        app.exec_()
    )


if __name__ == "__main__":
    main()
