import time

import cv2

from vision.cameras.top_camera import TopCamera


def main():

    camera = TopCamera()

    print(
        "========================================"
    )

    print(
        "TOP CAMERA FINAL TEST"
    )

    print(
        "========================================"
    )

    print(
        "1 : RAW FULL"
    )

    print(
        "2 : UNDISTORTED FULL"
    )

    print(
        "3 : FINAL SAFE ROI"
    )

    print(
        "Q / ESC : Quit"
    )

    print(
        "========================================"
    )

    mode = 3

    previous_time = (
        time.perf_counter()
    )

    smooth_fps = 0.0

    try:

        camera.open()

        print()

        print(
            "Stream info:"
        )

        print(
            camera.get_stream_info()
        )

        print()

        print(
            "Fixed controls:"
        )

        print(
            camera.get_fixed_controls()
        )

        while True:

            # ==================================================
            # 读取模式
            # ==================================================

            if mode == 1:

                frame = (
                    camera.read_full()
                )

                mode_name = (
                    "RAW FULL 1920x1080"
                )

            elif mode == 2:

                frame = (
                    camera.read_undistorted()
                )

                mode_name = (
                    "UNDISTORTED FULL 1920x1080"
                )

            else:

                frame = (
                    camera.read()
                )

                mode_name = (
                    "SAFE ROI 870x840"
                )

            # ==================================================
            # FPS
            # ==================================================

            now = (
                time.perf_counter()
            )

            dt = (
                now
                -
                previous_time
            )

            previous_time = (
                now
            )

            if dt > 0:

                fps = (
                    1.0
                    /
                    dt
                )

                if (
                    smooth_fps
                    ==
                    0.0
                ):

                    smooth_fps = (
                        fps
                    )

                else:

                    smooth_fps = (
                        0.9
                        *
                        smooth_fps
                        +
                        0.1
                        *
                        fps
                    )

            # ==================================================
            # 显示
            # ==================================================

            display = (
                frame.copy()
            )

            cv2.putText(
                display,
                mode_name,
                (
                    30,
                    45,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"FPS: "
                    f"{smooth_fps:.1f}"
                ),
                (
                    30,
                    85,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            # --------------------------------------------------
            # 完整1920x1080显示时缩小
            # --------------------------------------------------

            if (
                frame.shape[1]
                >
                1200
            ):

                display = (
                    cv2.resize(
                        display,
                        (
                            960,
                            540,
                        ),
                        interpolation=
                            cv2.INTER_AREA,
                    )
                )

            cv2.imshow(
                "Top Camera",
                display,
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # ==================================================
            # Keyboard
            # ==================================================

            if (
                key == ord("q")
                or
                key == 27
            ):

                break

            elif key == ord("1"):

                mode = 1

            elif key == ord("2"):

                mode = 2

            elif key == ord("3"):

                mode = 3

            elif key == ord("s"):

                save_dir = (
                    "data/calibration/top_camera/robot_mapping"
                )

                import os
                os.makedirs(
                    save_dir,
                    exist_ok=True
                )

                save_path = (
                    save_dir
                    +
                    "/teach_board.jpg"
                )

                cv2.imwrite(
                    save_path,
                    frame
                )

                print(
                    "Saved calibration image:",
                    save_path
                )

    finally:

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()