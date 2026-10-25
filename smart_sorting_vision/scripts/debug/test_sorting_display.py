from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from ui.sorting_display import (
    SortingDisplayWindow,
)


def make_test_image():
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

    # BGR
    image[
        60:300,
        140:340,
        2,
    ] = 220

    return image


def main():

    app = QApplication(
        sys.argv
    )

    window = SortingDisplayWindow(
        fullscreen=True
    )

    image = make_test_image()

    def item_1():

        window.add_finished_item(
            sequence=1,
            product_name="红色五棱柱",
            image_bgr=image,
            success_count=1,
        )

    def item_2():

        window.add_finished_item(
            sequence=2,
            product_name="蓝色圆柱",
            image_bgr=image,
            success_count=2,
        )

    def item_3():

        window.add_finished_item(
            sequence=3,
            product_name="黑色正方体",
            image_bgr=image,
            success_count=3,
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