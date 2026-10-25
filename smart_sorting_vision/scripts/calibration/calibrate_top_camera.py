from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 项目路径
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

IMAGE_DIR = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "images"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "top_camera_intrinsics.npz"
)


# ============================================================
# 棋盘参数
#
# 手机标定板：
# 7 × 10 个方格
# 6 × 9 个内角点
# ============================================================

CHESSBOARD_SIZE = (
    6,
    9,
)


# ============================================================
# 单个方格实际尺寸
#
# !!! 必须自己实测手机屏幕上的格子尺寸 !!!
#
# 推荐测连续5格：
#
# SQUARE_SIZE_MM = 五格总长度 / 5
#
# 下面的 9.0 只是示例，不是你的最终值。
# ============================================================

SQUARE_SIZE_MM = 5.2


# ============================================================
# 至少需要多少张有效图片
# ============================================================

MIN_VALID_IMAGES = 12


# ============================================================
# 单张图片重投影误差提醒阈值
#
# 注意：
# 这里只提示，不自动删除。
#
# 大倾角、靠边缘图片有时误差略高，
# 但仍然对畸变估计有价值。
# ============================================================

WARNING_ERROR_PX = 0.8


# ============================================================
# findChessboardCornersSB 参数
# ============================================================

SB_FLAGS = (
    cv2.CALIB_CB_NORMALIZE_IMAGE
    |
    cv2.CALIB_CB_EXHAUSTIVE
    |
    cv2.CALIB_CB_ACCURACY
)


# ============================================================
# 生成棋盘三维坐标
# ============================================================

def create_object_template():

    cols, rows = (
        CHESSBOARD_SIZE
    )

    object_template = np.zeros(
        (
            cols * rows,
            3,
        ),
        dtype=np.float32,
    )

    object_template[:, :2] = (
        np.mgrid[
            0:cols,
            0:rows
        ]
        .T
        .reshape(
            -1,
            2,
        )
    )

    object_template *= (
        SQUARE_SIZE_MM
    )

    return object_template


# ============================================================
# 计算单张图片重投影 RMSE
# ============================================================

def calculate_view_rmse(
    object_points,
    image_points,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
):

    projected_points, _ = (
        cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
    )

    observed = (
        image_points
        .reshape(
            -1,
            2,
        )
        .astype(
            np.float64
        )
    )

    projected = (
        projected_points
        .reshape(
            -1,
            2,
        )
        .astype(
            np.float64
        )
    )

    difference = (
        observed
        -
        projected
    )

    squared_distance = (
        difference[:, 0] ** 2
        +
        difference[:, 1] ** 2
    )

    rmse = float(
        np.sqrt(
            np.mean(
                squared_distance
            )
        )
    )

    return rmse


# ============================================================
# 主程序
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "TOP CAMERA INTRINSIC CALIBRATION"
    )

    print(
        "========================================"
    )

    print(
        f"Image directory : {IMAGE_DIR}"
    )

    print(
        f"Chessboard      : "
        f"{CHESSBOARD_SIZE[0]} x "
        f"{CHESSBOARD_SIZE[1]} inner corners"
    )

    print(
        f"Square size     : "
        f"{SQUARE_SIZE_MM:.4f} mm"
    )

    print()

    # ========================================================
    # 1. 获取全部图片
    # ========================================================

    image_files = sorted(
        list(
            IMAGE_DIR.glob(
                "*.jpg"
            )
        )
        +
        list(
            IMAGE_DIR.glob(
                "*.jpeg"
            )
        )
        +
        list(
            IMAGE_DIR.glob(
                "*.png"
            )
        )
    )

    if not image_files:

        raise RuntimeError(
            "没有找到任何标定图片：\n"
            f"{IMAGE_DIR}"
        )

    print(
        f"Found images: "
        f"{len(image_files)}"
    )

    print()

    # ========================================================
    # 2. 世界坐标模板
    # ========================================================

    object_template = (
        create_object_template()
    )

    object_points = []

    image_points = []

    used_files = []

    failed_files = []

    image_size = None

    # ========================================================
    # 3. 逐张检测棋盘
    # ========================================================

    for filename in image_files:

        image = cv2.imread(
            str(filename),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            print(
                f"[READ FAIL] "
                f"{filename.name}"
            )

            failed_files.append(
                filename.name
            )

            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        current_size = (
            gray.shape[1],
            gray.shape[0],
        )

        # ====================================================
        # 所有标定图片分辨率必须完全一致
        # ====================================================

        if image_size is None:

            image_size = (
                current_size
            )

        elif (
            current_size
            != image_size
        ):

            raise RuntimeError(
                "\n标定图片尺寸不一致：\n"
                f"{filename.name}\n"
                f"actual   = "
                f"{current_size}\n"
                f"expected = "
                f"{image_size}\n"
            )

        # ====================================================
        # 使用更严格的 SB 棋盘检测器
        #
        # findChessboardCornersSB 得到的角点本身已经具有
        # 亚像素精度，因此这里不再调用 cornerSubPix。
        # ====================================================

        try:

            found, corners = (
                cv2.findChessboardCornersSB(
                    gray,
                    CHESSBOARD_SIZE,
                    flags=SB_FLAGS,
                )
            )

        except cv2.error as exc:

            print(
                f"[ERROR] "
                f"{filename.name}: "
                f"{exc}"
            )

            failed_files.append(
                filename.name
            )

            continue

        if (
            not found
            or corners is None
        ):

            print(
                f"[NOT FOUND] "
                f"{filename.name}"
            )

            failed_files.append(
                filename.name
            )

            continue

        expected_corner_count = (
            CHESSBOARD_SIZE[0]
            *
            CHESSBOARD_SIZE[1]
        )

        if (
            len(corners)
            != expected_corner_count
        ):

            print(
                f"[BAD CORNERS] "
                f"{filename.name}: "
                f"{len(corners)}"
            )

            failed_files.append(
                filename.name
            )

            continue

        # ====================================================
        # 保存这一张的对应点
        # ====================================================

        object_points.append(
            object_template.copy()
        )

        image_points.append(
            corners.astype(
                np.float32
            )
        )

        used_files.append(
            filename.name
        )

        print(
            f"[OK] "
            f"{filename.name}"
        )

    # ========================================================
    # 4. 检查有效图片数量
    # ========================================================

    print()

    print(
        "----------------------------------------"
    )

    print(
        f"Valid images : "
        f"{len(used_files)}"
    )

    print(
        f"Failed images: "
        f"{len(failed_files)}"
    )

    print(
        "----------------------------------------"
    )

    if (
        len(object_points)
        <
        MIN_VALID_IMAGES
    ):

        raise RuntimeError(
            "\n有效标定图片不足。\n"
            f"当前：{len(object_points)}\n"
            f"最低建议：{MIN_VALID_IMAGES}\n"
        )

    # ========================================================
    # 5. 相机标定
    # ========================================================

    (
        rms,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    # ========================================================
    # 6. 逐张计算重投影误差
    # ========================================================

    per_view_errors = []

    for i in range(
        len(object_points)
    ):

        view_error = (
            calculate_view_rmse(
                object_points[i],
                image_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs,
            )
        )

        per_view_errors.append(
            view_error
        )

    per_view_errors = np.asarray(
        per_view_errors,
        dtype=np.float64,
    )

    # ========================================================
    # 7. 总体统计
    # ========================================================

    mean_error = float(
        np.mean(
            per_view_errors
        )
    )

    median_error = float(
        np.median(
            per_view_errors
        )
    )

    max_error = float(
        np.max(
            per_view_errors
        )
    )

    max_error_index = int(
        np.argmax(
            per_view_errors
        )
    )

    # ========================================================
    # 8. 计算优化后相机矩阵
    #
    # alpha = 0:
    #
    # 尽量裁去去畸变后产生的无效黑边，
    # 更适合后面固定工位使用。
    # ========================================================

    (
        new_camera_matrix,
        valid_roi,
    ) = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        0.0,
        image_size,
    )

    # ========================================================
    # 9. 打印逐张误差
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "PER-VIEW REPROJECTION ERROR"
    )

    print(
        "========================================"
    )

    sorted_indices = np.argsort(
        per_view_errors
    )

    for index in sorted_indices:

        error = float(
            per_view_errors[
                index
            ]
        )

        filename = (
            used_files[
                index
            ]
        )

        if (
            error
            >=
            WARNING_ERROR_PX
        ):

            status = (
                "  <-- CHECK"
            )

        else:

            status = ""

        print(
            f"{filename:<24} "
            f"{error:8.4f} px"
            f"{status}"
        )

    # ========================================================
    # 10. 输出最终结果
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "CALIBRATION RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"Image size:"
        f"\n{image_size}"
    )

    print()

    print(
        f"Total input images:"
        f"\n{len(image_files)}"
    )

    print()

    print(
        f"Used images:"
        f"\n{len(used_files)}"
    )

    print()

    print(
        f"Failed images:"
        f"\n{len(failed_files)}"
    )

    print()

    print(
        f"OpenCV RMS:"
        f"\n{rms:.8f} px"
    )

    print()

    print(
        f"Mean per-view RMSE:"
        f"\n{mean_error:.8f} px"
    )

    print()

    print(
        f"Median per-view RMSE:"
        f"\n{median_error:.8f} px"
    )

    print()

    print(
        f"Maximum per-view RMSE:"
        f"\n{max_error:.8f} px"
    )

    print()

    print(
        "Worst image:"
    )

    print(
        used_files[
            max_error_index
        ]
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
        "Distortion Coefficients:"
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
        "Valid ROI:"
    )

    print(
        valid_roi
    )

    # ========================================================
    # 11. 参数基本合理性检查
    # ========================================================

    fx = float(
        camera_matrix[0, 0]
    )

    fy = float(
        camera_matrix[1, 1]
    )

    cx = float(
        camera_matrix[0, 2]
    )

    cy = float(
        camera_matrix[1, 2]
    )

    print()

    print(
        "----------------------------------------"
    )

    print(
        "INTRINSIC SUMMARY"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"fx = {fx:.4f}"
    )

    print(
        f"fy = {fy:.4f}"
    )

    print(
        f"cx = {cx:.4f}"
    )

    print(
        f"cy = {cy:.4f}"
    )

    print()

    print(
        f"|fx-fy| / mean = "
        f"{abs(fx - fy) / ((fx + fy) / 2):.6f}"
    )

    # ========================================================
    # 12. 显示失败文件
    # ========================================================

    if failed_files:

        print()

        print(
            "----------------------------------------"
        )

        print(
            "FAILED / REJECTED BY DETECTOR"
        )

        print(
            "----------------------------------------"
        )

        for filename in (
            failed_files
        ):

            print(
                filename
            )

    # ========================================================
    # 13. 保存标定结果
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT_FILE,

        # -----------------------------------------------
        # 核心内参
        # -----------------------------------------------

        camera_matrix=
            camera_matrix,

        dist_coeffs=
            dist_coeffs,

        # -----------------------------------------------
        # 去畸变矩阵
        # -----------------------------------------------

        new_camera_matrix=
            new_camera_matrix,

        valid_roi=
            np.asarray(
                valid_roi,
                dtype=np.int32,
            ),

        # -----------------------------------------------
        # 图像信息
        # -----------------------------------------------

        image_size=
            np.asarray(
                image_size,
                dtype=np.int32,
            ),

        chessboard_size=
            np.asarray(
                CHESSBOARD_SIZE,
                dtype=np.int32,
            ),

        square_size_mm=
            np.asarray(
                SQUARE_SIZE_MM,
                dtype=np.float64,
            ),

        # -----------------------------------------------
        # 误差信息
        # -----------------------------------------------

        rms=
            np.asarray(
                rms,
                dtype=np.float64,
            ),

        mean_reprojection_error=
            np.asarray(
                mean_error,
                dtype=np.float64,
            ),

        median_reprojection_error=
            np.asarray(
                median_error,
                dtype=np.float64,
            ),

        max_reprojection_error=
            np.asarray(
                max_error,
                dtype=np.float64,
            ),

        per_view_errors=
            per_view_errors,

        # -----------------------------------------------
        # 文件信息
        # -----------------------------------------------

        used_files=
            np.asarray(
                used_files
            ),

        failed_files=
            np.asarray(
                failed_files
            ),
    )

    print()

    print(
        "========================================"
    )

    print(
        "CALIBRATION FILE SAVED"
    )

    print(
        "========================================"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()