from pathlib import Path

import cv2

from vision.cameras.top_camera import TopCamera


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SAVE_DIR = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "images"
)


# ============================================================
# 棋盘格内角点
# ============================================================

CHESSBOARD_SIZE = (
    6,
    9,
)


def main():

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera = TopCamera()

    image_index = 0

    print(
        "========================================"
    )

    print(
        "Top Camera Calibration Capture"
    )

    print(
        "========================================"
    )

    print(
        "SPACE : 保存有效棋盘格图片"
    )

    print(
        "Q/ESC : 退出"
    )

    try:

        camera.open()

        while True:

            # ==================================================
            # 标定必须使用完整原始1920×1080图像
            # ==================================================

            frame = camera.read_full()

            display = frame.copy()

            # ==================================================
            # 灰度
            # ==================================================

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            # ==================================================
            # 找棋盘格
            # ==================================================

            found, corners = (
                cv2.findChessboardCorners(
                    gray,
                    CHESSBOARD_SIZE,
                    flags=(
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        |
                        cv2.CALIB_CB_NORMALIZE_IMAGE
                    ),
                )
            )

            if found:

                # ==============================================
                # 亚像素优化
                # ==============================================

                corners_refined = (
                    cv2.cornerSubPix(
                        gray,
                        corners,
                        (11, 11),
                        (-1, -1),
                        (
                            cv2.TERM_CRITERIA_EPS
                            |
                            cv2.TERM_CRITERIA_MAX_ITER,
                            30,
                            0.001,
                        ),
                    )
                )

                cv2.drawChessboardCorners(
                    display,
                    CHESSBOARD_SIZE,
                    corners_refined,
                    found,
                )

                status = (
                    "Chessboard: FOUND"
                )

                color = (
                    0,
                    255,
                    0,
                )

            else:

                status = (
                    "Chessboard: NOT FOUND"
                )

                color = (
                    0,
                    0,
                    255,
                )

            cv2.putText(
                display,
                status,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"Saved: "
                    f"{image_index}"
                ),
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Top Camera Calibration",
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ==================================================
            # SPACE保存
            # ==================================================

            if key == 32:

                if not found:

                    print(
                        "[SKIP] "
                        "当前没有完整检测到棋盘格"
                    )

                    continue

                filename = (
                    SAVE_DIR
                    /
                    f"calib_{image_index:03d}.jpg"
                )

                success = cv2.imwrite(
                    str(filename),
                    frame,
                )

                if success:

                    print(
                        f"[SAVE] {filename}"
                    )

                    image_index += 1

            # ==================================================
            # Q/ESC
            # ==================================================

            elif (
                key == ord("q")
                or key == 27
            ):

                break

    finally:

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()