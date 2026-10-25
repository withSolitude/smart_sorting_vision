import cv2

from vision.cameras.top_camera import TopCamera


# ============================================================
# 当前粗略ROI
# ============================================================

ROI_X1 = 597
ROI_Y1 = 160

ROI_X2 = 1335
ROI_Y2 = 910


def main():

    camera = TopCamera()

    try:

        camera.open()

        while True:

            # ==================================================
            # 一定使用完整1920×1080原图
            # ==================================================

            frame = camera.read_full()

            display = frame.copy()

            # ==================================================
            # ROI框
            # ==================================================

            cv2.rectangle(
                display,
                (ROI_X1, ROI_Y1),
                (ROI_X2, ROI_Y2),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                display,
                (
                    f"ROI: "
                    f"({ROI_X1}, {ROI_Y1}) - "
                    f"({ROI_X2}, {ROI_Y2})"
                ),
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # ==================================================
            # ROI预览
            # ==================================================

            roi = frame[
                ROI_Y1:ROI_Y2,
                ROI_X1:ROI_X2
            ]

            cv2.imshow(
                "Top Camera Full",
                display,
            )

            cv2.imshow(
                "Top Camera ROI",
                roi,
            )

            key = cv2.waitKey(1) & 0xFF

            if (
                key == ord("q")
                or key == 27
            ):
                break

    finally:

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()