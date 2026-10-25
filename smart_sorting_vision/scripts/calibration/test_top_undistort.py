from pathlib import Path
import time

import cv2
import numpy as np

from vision.cameras.top_camera import TopCamera


# ============================================================
# 项目根目录
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# 标定参数文件
# ============================================================

CALIBRATION_FILE = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "top_camera_intrinsics.npz"
)


# ============================================================
# 调试图片保存目录
# ============================================================

SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "top_camera"
    / "undistort"
)


# ============================================================
# 显示尺寸
#
# 原始图是 1920x1080
# 左右同时显示会太大，所以缩小一半显示。
#
# 注意：
# 这里只缩放“显示画面”，
# 实际去畸变始终对原始 1920x1080 图处理。
# ============================================================

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


def load_calibration():

    if not CALIBRATION_FILE.exists():

        raise FileNotFoundError(
            "找不到顶部相机标定文件：\n"
            f"{CALIBRATION_FILE}"
        )

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

    valid_roi_array = (
        data["valid_roi"]
        .astype(np.int32)
    )

    valid_roi = tuple(
        int(value)
        for value
        in valid_roi_array
    )

    rms = float(
        data["rms"]
    )

    return (
        camera_matrix,
        dist_coeffs,
        new_camera_matrix,
        image_size,
        valid_roi,
        rms,
    )


def main():

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "========================================"
    )

    print(
        "TOP CAMERA UNDISTORT TEST"
    )

    print(
        "========================================"
    )

    # ========================================================
    # 1. 加载标定参数
    # ========================================================

    (
        camera_matrix,
        dist_coeffs,
        new_camera_matrix,
        image_size,
        valid_roi,
        rms,
    ) = load_calibration()

    print(
        f"Calibration file:"
        f"\n{CALIBRATION_FILE}"
    )

    print()

    print(
        f"Image size:"
        f"\n{image_size}"
    )

    print()

    print(
        f"RMS:"
        f"\n{rms:.6f} px"
    )

    print()

    print(
        "Camera Matrix:"
    )

    print(
        camera_matrix
    )

    print()

    print(
        "Distortion:"
    )

    print(
        dist_coeffs
    )

    print()

    print(
        "New Camera Matrix:"
    )

    print(
        new_camera_matrix
    )

    print()

    print(
        f"Valid ROI:"
        f"\n{valid_roi}"
    )

    # ========================================================
    # 2. 预计算去畸变映射
    #
    # 这一步只运行一次。
    #
    # CV_16SC2：
    # 使用定点映射格式，
    # 对实时 remap 性能比较合适。
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

    print()

    print(
        "[OK] Undistortion maps generated"
    )

    print()

    print(
        "Controls:"
    )

    print(
        "Q / ESC : Quit"
    )

    print(
        "S       : Save original + undistorted"
    )

    print(
        "G       : Toggle guide lines"
    )

    # ========================================================
    # 3. 打开顶部相机
    # ========================================================

    camera = TopCamera()

    save_index = 0

    show_guides = True

    previous_time = (
        time.perf_counter()
    )

    smooth_fps = 0.0

    try:

        camera.open()

        # ====================================================
        # 检查真实相机分辨率和标定分辨率是否一致
        # ====================================================

        info = (
            camera.get_stream_info()
        )

        actual_size = (
            int(info["width"]),
            int(info["height"]),
        )

        if (
            actual_size
            != image_size
        ):

            raise RuntimeError(
                "当前相机分辨率与标定分辨率不一致：\n"
                f"camera      = {actual_size}\n"
                f"calibration = {image_size}\n"
                "\n"
                "不能直接使用这套内参。"
            )

        # ====================================================
        # 4. 实时读取
        # ====================================================

        while True:

            # ------------------------------------------------
            # 原始完整画面
            # ------------------------------------------------

            frame = (
                camera.read_full()
            )

            # ------------------------------------------------
            # 去畸变
            #
            # BORDER_CONSTANT：
            # 超出有效映射范围的位置填黑。
            #
            # 你的 valid_roi 基本就是完整画面，
            # 所以预计几乎看不到黑边。
            # ------------------------------------------------

            undistorted = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            # =================================================
            # FPS
            # =================================================

            current_time = (
                time.perf_counter()
            )

            dt = (
                current_time
                - previous_time
            )

            previous_time = (
                current_time
            )

            if dt > 0:

                current_fps = (
                    1.0 / dt
                )

                if smooth_fps == 0:

                    smooth_fps = (
                        current_fps
                    )

                else:

                    smooth_fps = (
                        0.90
                        * smooth_fps
                        +
                        0.10
                        * current_fps
                    )

            # =================================================
            # 复制用于显示
            # =================================================

            original_display = (
                frame.copy()
            )

            corrected_display = (
                undistorted.copy()
            )

            # =================================================
            # 辅助参考线
            #
            # 这些线只是帮助观察边缘位置变化，
            # 不是用于计算。
            # =================================================

            if show_guides:

                height, width = (
                    frame.shape[:2]
                )

                vertical_positions = (
                    int(width * 0.25),
                    int(width * 0.50),
                    int(width * 0.75),
                )

                horizontal_positions = (
                    int(height * 0.25),
                    int(height * 0.50),
                    int(height * 0.75),
                )

                for x in vertical_positions:

                    cv2.line(
                        original_display,
                        (x, 0),
                        (x, height - 1),
                        (0, 255, 0),
                        1,
                    )

                    cv2.line(
                        corrected_display,
                        (x, 0),
                        (x, height - 1),
                        (0, 255, 0),
                        1,
                    )

                for y in horizontal_positions:

                    cv2.line(
                        original_display,
                        (0, y),
                        (width - 1, y),
                        (0, 255, 0),
                        1,
                    )

                    cv2.line(
                        corrected_display,
                        (0, y),
                        (width - 1, y),
                        (0, 255, 0),
                        1,
                    )

            # =================================================
            # 标题
            # =================================================

            cv2.putText(
                original_display,
                "ORIGINAL",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                corrected_display,
                "UNDISTORTED",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                corrected_display,
                (
                    f"FPS: "
                    f"{smooth_fps:.1f}"
                ),
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # =================================================
            # 缩放，仅用于窗口显示
            # =================================================

            original_small = (
                cv2.resize(
                    original_display,
                    (
                        DISPLAY_WIDTH,
                        DISPLAY_HEIGHT,
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            )

            corrected_small = (
                cv2.resize(
                    corrected_display,
                    (
                        DISPLAY_WIDTH,
                        DISPLAY_HEIGHT,
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            )

            # =================================================
            # 左右拼接
            # =================================================

            comparison = (
                np.hstack(
                    (
                        original_small,
                        corrected_small,
                    )
                )
            )

            cv2.imshow(
                "Top Camera - Original vs Undistorted",
                comparison,
            )

            # =================================================
            # Keyboard
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------------
            # Q / ESC
            # ------------------------------------------------

            if (
                key == ord("q")
                or key == 27
            ):

                break

            # ------------------------------------------------
            # G：辅助线开关
            # ------------------------------------------------

            elif key == ord("g"):

                show_guides = (
                    not show_guides
                )

                print(
                    f"[GUIDES] "
                    f"{show_guides}"
                )

            # ------------------------------------------------
            # S：保存原图和去畸变图
            # ------------------------------------------------

            elif key == ord("s"):

                original_file = (
                    SAVE_DIR
                    /
                    (
                        f"original_"
                        f"{save_index:03d}.jpg"
                    )
                )

                corrected_file = (
                    SAVE_DIR
                    /
                    (
                        f"undistorted_"
                        f"{save_index:03d}.jpg"
                    )
                )

                success_original = (
                    cv2.imwrite(
                        str(
                            original_file
                        ),
                        frame,
                    )
                )

                success_corrected = (
                    cv2.imwrite(
                        str(
                            corrected_file
                        ),
                        undistorted,
                    )
                )

                if (
                    success_original
                    and
                    success_corrected
                ):

                    print(
                        "[SAVE]"
                    )

                    print(
                        original_file
                    )

                    print(
                        corrected_file
                    )

                    save_index += 1

                else:

                    print(
                        "[ERROR] Save failed"
                    )

    finally:

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()