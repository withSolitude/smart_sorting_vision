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
# 8 个托盘固定特征点
#
# 点击顺序必须和机械坐标输入顺序完全一致：
#
# P1  托盘左上角
# P2  托盘右上角
# P3  托盘右下角
# P4  托盘左下角
# P5  左上螺丝中心
# P6  右上螺丝中心
# P7  右下螺丝中心
# P8  左下螺丝中心
# ================================================================

POINTS = [
    ("P1", "tray_top_left"),
    ("P2", "tray_top_right"),
    ("P3", "tray_bottom_right"),
    ("P4", "tray_bottom_left"),
    ("P5", "screw_top_left"),
    ("P6", "screw_top_right"),
    ("P7", "screw_bottom_right"),
    ("P8", "screw_bottom_left"),
]

POINT_COUNT = len(POINTS)

# RANSAC 的阈值处于机械坐标单位：mm。
RANSAC_THRESHOLD_MM = 2.0

OUTPUT_DIR = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "robot_mapping"
)

TRANSFORM_PATH = OUTPUT_DIR / "pixel_to_robot.json"
POINTS_PATH = OUTPUT_DIR / "tray8_calibration_points.csv"
REPORT_PATH = OUTPUT_DIR / "tray8_calibration_report.json"
SNAPSHOT_PATH = OUTPUT_DIR / "tray8_calibration_snapshot.jpg"
INDEXED_PATH = OUTPUT_DIR / "tray8_calibration_indexed.jpg"


class ClickCollector:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.points: list[tuple[float, float]] = []

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= POINT_COUNT:
                return

            self.points.append(
                (
                    float(x),
                    float(y),
                )
            )

            point_id, point_name = POINTS[len(self.points) - 1]

            print(
                f"[CLICK] {point_id} {point_name}: "
                f"pixel=({x}, {y})"
            )

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(
                    f"[UNDO] 删除最后一点："
                    f"({removed[0]:.1f}, {removed[1]:.1f})"
                )

    def draw(self) -> np.ndarray:
        display = self.image.copy()

        for i, (u, v) in enumerate(self.points):
            x = int(round(u))
            y = int(round(v))
            point_id, _ = POINTS[i]

            cv2.circle(
                display,
                (x, y),
                8,
                (0, 0, 255),
                2,
            )

            cv2.drawMarker(
                display,
                (x, y),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )

            cv2.putText(
                display,
                point_id,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        current = len(self.points)

        if current < POINT_COUNT:
            point_id, point_name = POINTS[current]
            line1 = (
                f"CLICK {current + 1}/{POINT_COUNT}: "
                f"{point_id} {point_name}"
            )
        else:
            line1 = "8 points complete - press ENTER to continue"

        cv2.putText(
            display,
            line1,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "Left click=add | Right click/U=undo | R=reset | Q=quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return display


def capture_clean_snapshot(
    camera: TopCamera,
) -> np.ndarray:
    """
    使用和正式视觉相同的 TopCamera.read() 坐标系。

    如果你的畸变矫正不在 TopCamera.read() 内，而是在另一个正式函数中，
    必须把这里改成“正式抓取时实际送给 YOLO 的那一帧”。
    """
    print()
    print("=" * 72)
    print("STEP 1 - CAPTURE CALIBRATION SNAPSHOT")
    print("=" * 72)
    print("请把夹爪移开，不要遮挡托盘四角和四颗螺丝。")
    print("画面稳定后按 SPACE 冻结当前标定图。")
    print("Q / ESC 退出。")
    print("=" * 72)

    cv2.namedWindow(
        "Tray 8-Point Calibration - Live",
        cv2.WINDOW_NORMAL,
    )

    while True:
        frame = camera.read()
        preview = frame.copy()

        cv2.putText(
            preview,
            "SPACE = freeze calibration frame",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Tray 8-Point Calibration - Live",
            preview,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 32:
            cv2.destroyWindow(
                "Tray 8-Point Calibration - Live"
            )
            return frame.copy()

        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt


def collect_image_points(
    frame: np.ndarray,
) -> list[tuple[float, float]]:
    collector = ClickCollector(frame)

    window = "Tray 8-Point Calibration - Click"

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        window,
        collector.mouse_callback,
    )

    print()
    print("=" * 72)
    print("STEP 2 - CLICK 8 IMAGE POINTS")
    print("=" * 72)

    for i, (point_id, point_name) in enumerate(POINTS, start=1):
        print(
            f"{i}. {point_id}: {point_name}"
        )

    print()
    print("点击时尽量放大窗口，并点在：")
    print("  - 螺丝：圆心")
    print("  - 托盘角：两条直边的理论交点")
    print()
    print("右键或 U：撤销最后一点")
    print("R：全部重来")
    print("8点完成后按 ENTER")
    print("=" * 72)

    while True:
        display = collector.draw()

        cv2.imshow(
            window,
            display,
        )

        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt

        if key in (ord("u"), ord("U")):
            if collector.points:
                collector.points.pop()

        if key in (ord("r"), ord("R")):
            collector.points.clear()

        if key in (10, 13):
            if len(collector.points) != POINT_COUNT:
                print(
                    f"[WARN] 当前只有 {len(collector.points)} 个点，"
                    f"必须完成 {POINT_COUNT} 个。"
                )
                continue

            indexed = collector.draw()
            cv2.destroyWindow(window)
            return list(collector.points), indexed


def parse_xy(
    text: str,
) -> tuple[float, float] | None:
    parts = text.replace(",", " ").split()

    if len(parts) != 2:
        return None

    try:
        x_mm = float(parts[0])
        y_mm = float(parts[1])
    except ValueError:
        return None

    if not (
        np.isfinite(x_mm)
        and
        np.isfinite(y_mm)
    ):
        return None

    return (
        x_mm,
        y_mm,
    )


def collect_robot_points(
    image_points: list[tuple[float, float]],
) -> list[dict]:
    records: list[dict] = []

    print()
    print("=" * 72)
    print("STEP 3 - ENTER REAL ROBOT XY FOR EACH LANDMARK")
    print("=" * 72)
    print(
        "对每一个点，用 STM32 的 XY JOG/手动模式让夹爪 TCP "
        "对准该实体特征，然后输入当前机械 X_mm Y_mm。"
    )
    print(
        "建议所有点都从相同方向接近，减少丝杆/同步带回程间隙影响。"
    )
    print(
        "如果某个托盘角夹爪无法实际对准，就不要猜坐标；"
        "应换成一个可准确获取机械坐标的固定特征点。"
    )
    print("=" * 72)

    for i, ((point_id, point_name), (u, v)) in enumerate(
        zip(POINTS, image_points),
        start=1,
    ):
        while True:
            print()
            print(
                f"[{i}/{POINT_COUNT}] "
                f"{point_id} {point_name}"
            )
            print(
                f"image pixel = ({u:.2f}, {v:.2f})"
            )

            text = input(
                "移动 TCP 对准该实体点后，输入 X_mm Y_mm："
            ).strip()

            xy = parse_xy(text)

            if xy is None:
                print(
                    "[WARN] 格式错误，例如：125.4 83.7"
                )
                continue

            x_mm, y_mm = xy

            records.append(
                {
                    "id": point_id,
                    "name": point_name,
                    "u": float(u),
                    "v": float(v),
                    "x_mm": float(x_mm),
                    "y_mm": float(y_mm),
                }
            )
            break

    return records


def fit_and_save(
    frame: np.ndarray,
    indexed_image: np.ndarray,
    records: list[dict],
) -> None:
    image_points = np.asarray(
        [
            [
                record["u"],
                record["v"],
            ]
            for record in records
        ],
        dtype=np.float64,
    )

    robot_points = np.asarray(
        [
            [
                record["x_mm"],
                record["y_mm"],
            ]
            for record in records
        ],
        dtype=np.float64,
    )

    H, inlier_mask = cv2.findHomography(
        image_points,
        robot_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_MM,
    )

    if H is None:
        raise RuntimeError(
            "cv2.findHomography() 计算失败"
        )

    predicted = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2),
        H,
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        predicted - robot_points,
        axis=1,
    )

    if inlier_mask is None:
        inliers = np.ones(
            len(records),
            dtype=bool,
        )
    else:
        inliers = (
            inlier_mask
            .reshape(-1)
            .astype(bool)
        )

    for i, record in enumerate(records):
        record["pred_x_mm"] = float(
            predicted[i, 0]
        )
        record["pred_y_mm"] = float(
            predicted[i, 1]
        )
        record["error_mm"] = float(
            errors[i]
        )
        record["inlier"] = bool(
            inliers[i]
        )

    mean_error = float(
        np.mean(errors)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    median_error = float(
        np.median(errors)
    )

    max_error = float(
        np.max(errors)
    )

    inlier_count = int(
        np.count_nonzero(
            inliers
        )
    )

    transform = PixelToRobotTransform(
        homography=H,
        image_width=frame.shape[1],
        image_height=frame.shape[0],
        source=(
            "TopCamera.read() exact frame used by tray8 calibration"
        ),
        metadata={
            "created_at":
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            "method":
                "tray_4_corners_plus_4_screw_centers",
            "point_count":
                len(records),
            "inlier_count":
                inlier_count,
            "ransac_threshold_mm":
                RANSAC_THRESHOLD_MM,
            "mean_fit_error_mm":
                mean_error,
            "rmse_fit_error_mm":
                rmse,
            "median_fit_error_mm":
                median_error,
            "max_fit_error_mm":
                max_error,
        },
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform.save(
        TRANSFORM_PATH
    )

    cv2.imwrite(
        str(SNAPSHOT_PATH),
        frame,
    )

    cv2.imwrite(
        str(INDEXED_PATH),
        indexed_image,
    )

    with POINTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "name",
                "u",
                "v",
                "x_mm",
                "y_mm",
                "pred_x_mm",
                "pred_y_mm",
                "error_mm",
                "inlier",
            ],
        )

        writer.writeheader()
        writer.writerows(records)

    report = {
        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
        "method":
            "tray_4_corners_plus_4_screw_centers",
        "image_size": [
            int(frame.shape[1]),
            int(frame.shape[0]),
        ],
        "point_count":
            len(records),
        "inlier_count":
            inlier_count,
        "ransac_threshold_mm":
            RANSAC_THRESHOLD_MM,
        "mean_fit_error_mm":
            mean_error,
        "rmse_fit_error_mm":
            rmse,
        "median_fit_error_mm":
            median_error,
        "max_fit_error_mm":
            max_error,
        "homography":
            H.tolist(),
        "records":
            records,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print("TRAY 8-POINT CALIBRATION COMPLETE")
    print("=" * 72)

    for record in records:
        state = (
            "IN"
            if record["inlier"]
            else "OUT"
        )

        print(
            f"{record['id']:>2} "
            f"{record['name']:<20} "
            f"error={record['error_mm']:.3f} mm "
            f"{state}"
        )

    print("-" * 72)
    print(
        f"inliers      : "
        f"{inlier_count}/{len(records)}"
    )
    print(
        f"mean fit err : "
        f"{mean_error:.3f} mm"
    )
    print(
        f"RMSE fit err : "
        f"{rmse:.3f} mm"
    )
    print(
        f"median       : "
        f"{median_error:.3f} mm"
    )
    print(
        f"max fit err  : "
        f"{max_error:.3f} mm"
    )
    print("-" * 72)
    print(
        f"Transform : {TRANSFORM_PATH}"
    )
    print(
        f"Points    : {POINTS_PATH}"
    )
    print(
        f"Report    : {REPORT_PATH}"
    )
    print(
        f"Snapshot  : {SNAPSHOT_PATH}"
    )
    print("=" * 72)

    if inlier_count < 6:
        print(
            "[WARN] 8点中有效内点少于6个。"
            "建议重新检查点的顺序、鼠标点击位置和机械XY。"
        )

    if max_error > 3.0:
        print(
            "[WARN] 拟合点最大误差 > 3 mm。"
            "先不要进入自动抓取，建议重新标定。"
        )

    print(
        "\n注意：上面的 fit error 只是拟合点误差，"
        "不能代替独立验证。下一步必须运行 verify 脚本。"
    )


def main() -> None:
    camera = TopCamera()

    try:
        camera.open()

        frame = capture_clean_snapshot(
            camera
        )

        image_points, indexed = (
            collect_image_points(
                frame
            )
        )

        records = collect_robot_points(
            image_points
        )

        fit_and_save(
            frame,
            indexed,
            records,
        )

    except KeyboardInterrupt:
        print(
            "\n[CANCEL] 标定已取消"
        )

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
