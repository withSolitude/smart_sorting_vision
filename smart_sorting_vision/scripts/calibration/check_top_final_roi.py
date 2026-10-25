import cv2
import numpy as np

from pathlib import Path

from vision.cameras.top_camera import TopCamera


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CALIBRATION_FILE = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "top_camera_intrinsics.npz"
)


ROI_X1 = 520
ROI_Y1 = 120

ROI_X2 = 1390
ROI_Y2 = 960


def main():

    # ========================================================
    # 1. 加载标定参数
    # ========================================================

    data = np.load(
        CALIBRATION_FILE,
        allow_pickle=False,
    )

    camera_matrix = (
        data["camera_matrix"]
        .astype(np.float64)
    )

    dist_coeffs = (
        data["dist_coeffs"]
        .astype(np.float64)
    )

    new_camera_matrix = (
        data["new_camera_matrix"]
        .astype(np.float64)
    )

    image_size_array = (
        data["image_size"]
        .astype(np.int32)
    )

    image_size = (
        int(image_size_array[0]),
        int(image_size_array[1]),
    )

    # ========================================================
    # 2. 预生成去畸变映射
    # ========================================================

    map1, map2 = (
        cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix,
            image_size,
            cv2.CV_16SC2,
        )
    )

    # ========================================================
    # 3. 相机
    # ========================================================

    camera = TopCamera()

    try:

        camera.open()

        while True:

            # ------------------------------------------------
            # 原始完整图
            # ------------------------------------------------

            frame = (
                camera.read_full()
            )

            # ------------------------------------------------
            # 去畸变
            # ------------------------------------------------

            undistorted = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            # ------------------------------------------------
            # ROI
            # ------------------------------------------------

            roi = undistorted[
                ROI_Y1:ROI_Y2,
                ROI_X1:ROI_X2
            ].copy()

            # ------------------------------------------------
            # 在完整图中画ROI框
            # ------------------------------------------------

            display = (
                undistorted.copy()
            )

            cv2.rectangle(
                display,
                (
                    ROI_X1,
                    ROI_Y1,
                ),
                (
                    ROI_X2,
                    ROI_Y2,
                ),
                (
                    0,
                    255,
                    0,
                ),
                3,
            )

            cv2.putText(
                display,
                (
                    f"ROI "
                    f"({ROI_X1},{ROI_Y1}) "
                    f"- "
                    f"({ROI_X2},{ROI_Y2})"
                ),
                (
                    30,
                    50,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # 为显示缩小完整图
            # ------------------------------------------------

            display_small = (
                cv2.resize(
                    display,
                    (
                        960,
                        540,
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            )

            cv2.imshow(
                "Undistorted Full + ROI",
                display_small,
            )

            cv2.imshow(
                "Final Top ROI",
                roi,
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if (
                key == ord("q")
                or
                key == 27
            ):

                break

    finally:

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()