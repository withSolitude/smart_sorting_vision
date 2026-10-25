from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.cameras.top_camera import TopCamera
from vision.calibration.pixel_to_robot import PixelToRobotTransform


# ================================================================
# 你的上传样图可以检测为 9 x 6 个内角点。
# 如果以后换棋盘格，只修改这里。
# ================================================================
PATTERN_SIZE = (9, 6)  # (columns, rows) 内角点数量

# 每次稳定检测需要的有效帧数。
STABLE_FRAMES = 15
MAX_DETECTION_ATTEMPTS = 60

# RANSAC 在“机械坐标 mm”中的阈值。
RANSAC_THRESHOLD_MM = 2.0

OUTPUT_DIR = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "robot_mapping"
)

TRANSFORM_PATH = OUTPUT_DIR / "pixel_to_robot.json"
POINTS_PATH = OUTPUT_DIR / "calibration_points.csv"
REPORT_PATH = OUTPUT_DIR / "calibration_report.json"
INDEXED_IMAGE_PATH = OUTPUT_DIR / "calibration_board_indexed.jpg"


def select_spread_indices(cols: int, rows: int) -> list[int]:
    """
    自动从棋盘格中选取分布在整个区域的约 15 个角点。
    对 9x6 棋盘格为 3 行 x 5 列。
    """
    selected_cols = np.unique(
        np.round(np.linspace(0, cols - 1, 5)).astype(int)
    )
    selected_rows = np.unique(
        np.round(np.linspace(0, rows - 1, 3)).astype(int)
    )

    indices = []
    for r in selected_rows:
        for c in selected_cols:
            indices.append(int(r * cols + c))
    return indices


def detect_chessboard(frame: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ok, corners = cv2.findChessboardCornersSB(
        gray,
        PATTERN_SIZE,
        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        ),
    )

    if not ok or corners is None:
        return None

    return corners.reshape(-1, 2).astype(np.float64)


def collect_stable_corners(camera: TopCamera) -> tuple[np.ndarray, np.ndarray]:
    """
    连续多帧检测棋盘格，按角点编号对齐后取中位数。
    棋盘格和相机在此过程中必须保持静止。
    """
    samples: list[np.ndarray] = []
    last_frame = None

    print(
        f"[INFO] 正在采集稳定棋盘格角点，"
        f"需要 {STABLE_FRAMES} 个有效帧..."
    )

    for attempt in range(MAX_DETECTION_ATTEMPTS):
        frame = camera.read()
        last_frame = frame
        corners = detect_chessboard(frame)

        preview = frame.copy()
        if corners is not None:
            samples.append(corners)
            for p in corners:
                cv2.circle(
                    preview,
                    (int(round(p[0])), int(round(p[1]))),
                    3,
                    (0, 255, 0),
                    -1,
                )

        cv2.putText(
            preview,
            f"Chessboard stable frames: {len(samples)}/{STABLE_FRAMES}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Top Robot Calibration", preview)
        cv2.waitKey(1)

        if len(samples) >= STABLE_FRAMES:
            break

    if len(samples) < STABLE_FRAMES or last_frame is None:
        raise RuntimeError(
            "棋盘格稳定检测失败。请检查：棋盘完整可见、无遮挡、光照、焦距和 PATTERN_SIZE。"
        )

    stack = np.stack(samples, axis=0)
    median_corners = np.median(stack, axis=0)
    return last_frame, median_corners


def draw_indexed_board(
    frame: np.ndarray,
    corners: np.ndarray,
    selected_indices: list[int],
    active_index: int | None = None,
) -> np.ndarray:
    display = frame.copy()
    selected_set = set(selected_indices)

    for index, (u, v) in enumerate(corners):
        p = (int(round(u)), int(round(v)))

        if index == active_index:
            color = (0, 0, 255)
            radius = 11
            thickness = 3
        elif index in selected_set:
            color = (0, 255, 255)
            radius = 7
            thickness = 2
        else:
            color = (0, 255, 0)
            radius = 3
            thickness = 1

        cv2.circle(display, p, radius, color, thickness)
        cv2.putText(
            display,
            str(index),
            (p[0] + 5, p[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return display


def parse_xy(text: str) -> tuple[float, float] | None:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        return None
    try:
        x_mm = float(parts[0])
        y_mm = float(parts[1])
    except ValueError:
        return None
    if not (np.isfinite(x_mm) and np.isfinite(y_mm)):
        return None
    return x_mm, y_mm


def fit_and_save(
    frame: np.ndarray,
    records: list[dict],
) -> None:
    if len(records) < 4:
        raise RuntimeError("有效对应点少于 4 个，无法计算 Homography")

    image_points = np.asarray(
        [[r["u"], r["v"]] for r in records],
        dtype=np.float64,
    )
    robot_points = np.asarray(
        [[r["x_mm"], r["y_mm"]] for r in records],
        dtype=np.float64,
    )

    H, inlier_mask = cv2.findHomography(
        image_points,
        robot_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_MM,
    )

    if H is None:
        raise RuntimeError("cv2.findHomography() 失败")

    predicted = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2),
        H,
    ).reshape(-1, 2)

    errors = np.linalg.norm(predicted - robot_points, axis=1)
    inliers = (
        inlier_mask.reshape(-1).astype(bool)
        if inlier_mask is not None
        else np.ones(len(records), dtype=bool)
    )

    for i, r in enumerate(records):
        r["pred_x_mm"] = float(predicted[i, 0])
        r["pred_y_mm"] = float(predicted[i, 1])
        r["error_mm"] = float(errors[i])
        r["inlier"] = bool(inliers[i])

    mean_error = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_error = float(np.median(errors))
    max_error = float(np.max(errors))

    transform = PixelToRobotTransform(
        homography=H,
        image_width=frame.shape[1],
        image_height=frame.shape[0],
        source="TopCamera.read() after current distortion correction / ROI",
        metadata={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "pattern_size_inner_corners": list(PATTERN_SIZE),
            "point_count": len(records),
            "inlier_count": int(np.count_nonzero(inliers)),
            "ransac_threshold_mm": RANSAC_THRESHOLD_MM,
            "mean_error_mm": mean_error,
            "rmse_mm": rmse,
            "median_error_mm": median_error,
            "max_error_mm": max_error,
        },
    )
    transform.save(TRANSFORM_PATH)

    with POINTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "point_count": len(records),
        "inlier_count": int(np.count_nonzero(inliers)),
        "mean_error_mm": mean_error,
        "rmse_mm": rmse,
        "median_error_mm": median_error,
        "max_error_mm": max_error,
        "homography": H.tolist(),
        "records": records,
    }
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print("CALIBRATION RESULT")
    print("=" * 72)
    print(f"points       : {len(records)}")
    print(f"inliers      : {np.count_nonzero(inliers)}")
    print(f"mean error   : {mean_error:.3f} mm")
    print(f"RMSE         : {rmse:.3f} mm")
    print(f"median error : {median_error:.3f} mm")
    print(f"max error    : {max_error:.3f} mm")
    print(f"transform    : {TRANSFORM_PATH}")
    print(f"points       : {POINTS_PATH}")
    print(f"report       : {REPORT_PATH}")
    print("=" * 72)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cols, rows = PATTERN_SIZE
    selected_indices = select_spread_indices(cols, rows)

    print("=" * 72)
    print("TOP CAMERA PIXEL -> ROBOT XY CALIBRATION")
    print("=" * 72)
    print("方法：大棋盘格固定在抓取平面 + 夹爪 TCP 临时细针/尖头。")
    print("不需要把棋盘格装进夹爪。")
    print()
    print("准备：")
    print("1. 相机、镜头、焦距、去畸变、ROI全部固定。")
    print("2. 棋盘格平放并贴牢在真实抓取参考平面。")
    print("3. 在夹爪中心夹一根细针/细杆作为 TCP 指针。")
    print("4. 指针尖端尽量接近棋盘平面，但不要刮碰。")
    print("5. 使用 STM32 的手动/JOG 功能移动 XY，禁止执行完整抓取流程。")
    print("=" * 72)

    input("棋盘固定、夹爪移出棋盘视野后，按 Enter 开始采集棋盘角点...")

    camera = TopCamera()
    try:
        camera.open()
        frame, corners = collect_stable_corners(camera)

        indexed = draw_indexed_board(
            frame,
            corners,
            selected_indices,
        )
        cv2.imwrite(str(INDEXED_IMAGE_PATH), indexed)
        cv2.imshow("Top Robot Calibration", indexed)
        cv2.waitKey(100)

        print()
        print(f"检测到 {len(corners)} 个内角点。")
        print(f"本次选取 {len(selected_indices)} 个分布式标定点。")
        print(f"编号图已保存：{INDEXED_IMAGE_PATH}")
        print()
        print("每个点的操作：")
        print("  A. 看窗口中的红色目标角点编号。")
        print("  B. 用 STM32 手动/JOG 将 TCP 细针尖端移动到该角点正上方。")
        print("  C. 保持 Z 高度一致，并尽量让针尖接近棋盘平面。")
        print("  D. 在终端输入此时 STM32/机械坐标：X_mm Y_mm")
        print("  E. 输入 s 跳过当前点；输入 q 终止。")
        print()

        records: list[dict] = []

        for index in selected_indices:
            u, v = corners[index]
            active = draw_indexed_board(
                frame,
                corners,
                selected_indices,
                active_index=index,
            )
            cv2.imshow("Top Robot Calibration", active)
            cv2.waitKey(100)

            while True:
                text = input(
                    f"Corner {index:02d} pixel=({u:.2f},{v:.2f}) | "
                    "输入 X_mm Y_mm [s=skip, q=quit]: "
                ).strip()

                if text.lower() == "s":
                    print(f"[SKIP] corner {index}")
                    break

                if text.lower() == "q":
                    print("[STOP] 用户提前结束采点")
                    fit_and_save(frame, records)
                    return

                xy = parse_xy(text)
                if xy is None:
                    print("[ERROR] 格式错误，例如：120.5 83.0")
                    continue

                x_mm, y_mm = xy
                records.append(
                    {
                        "corner_index": int(index),
                        "u": float(u),
                        "v": float(v),
                        "x_mm": float(x_mm),
                        "y_mm": float(y_mm),
                    }
                )
                print(
                    f"[OK] #{index}: "
                    f"pixel=({u:.2f},{v:.2f}) -> "
                    f"robot=({x_mm:.3f},{y_mm:.3f}) mm"
                )
                break

        fit_and_save(frame, records)

    finally:
        try:
            camera.close()
        except AttributeError:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
