from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "vision").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError(
        "无法自动找到 smart_sorting_vision 项目根目录："
        "根目录下必须存在 vision/ 和 data/"
    )


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.cameras.top_camera import TopCamera
from vision.calibration.pixel_to_robot import PixelToRobotTransform


# ================================================================
# 4 点标定
#
# 默认按四颗螺丝中心：
# P1 左上 -> P2 右上 -> P3 右下 -> P4 左下
#
# 如果你实际测的是四个托盘角，也可以直接用，
# 只要图像点击顺序与机械坐标输入顺序完全一致。
# ================================================================
POINTS = [
    ("P1", "top_left"),
    ("P2", "top_right"),
    ("P3", "bottom_right"),
    ("P4", "bottom_left"),
]
POINT_COUNT = 4

OUTPUT_DIR = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "robot_mapping"
)

TRANSFORM_PATH = OUTPUT_DIR / "pixel_to_robot.json"
POINTS_PATH = OUTPUT_DIR / "tray4_calibration_points.csv"
REPORT_PATH = OUTPUT_DIR / "tray4_calibration_report.json"
SNAPSHOT_PATH = OUTPUT_DIR / "tray4_calibration_snapshot.jpg"
INDEXED_PATH = OUTPUT_DIR / "tray4_calibration_indexed.jpg"


def safe_release_camera(camera) -> None:
    if hasattr(camera, "release"):
        try:
            camera.release()
            return
        except Exception:
            pass

    if hasattr(camera, "close"):
        try:
            camera.close()
        except Exception:
            pass


class ClickCollector:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.points: list[tuple[float, float]] = []

    def mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= POINT_COUNT:
                return

            self.points.append((float(x), float(y)))
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

            cv2.circle(display, (x, y), 8, (0, 0, 255), 2)
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
            line1 = "4 points complete - press ENTER to continue"

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
            "Left=add | Right/U=undo | R=reset | Q=quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return display


def capture_clean_snapshot(camera: TopCamera) -> np.ndarray:
    print()
    print("=" * 72)
    print("STEP 1 - CAPTURE CALIBRATION SNAPSHOT")
    print("=" * 72)
    print("把夹爪移开，保证4个标定点清晰可见。")
    print("画面稳定后按 SPACE 冻结当前图像。")
    print("Q / ESC 退出。")
    print("=" * 72)

    window = "Tray 4-Point Calibration - Live"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

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
        cv2.imshow(window, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            cv2.destroyWindow(window)
            return frame.copy()

        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt


def collect_image_points(frame: np.ndarray):
    collector = ClickCollector(frame)
    window = "Tray 4-Point Calibration - Click"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, collector.mouse_callback)

    print()
    print("=" * 72)
    print("STEP 2 - CLICK 4 IMAGE POINTS")
    print("=" * 72)
    print("严格按顺序：")
    print("P1 = 左上")
    print("P2 = 右上")
    print("P3 = 右下")
    print("P4 = 左下")
    print("如果使用螺丝，就点击螺丝圆心。")
    print("4点完成后按 Enter。")
    print("=" * 72)

    while True:
        cv2.imshow(window, collector.draw())
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt

        if key in (ord("u"), ord("U")) and collector.points:
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


def parse_xy(text: str):
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


def collect_robot_points(image_points):
    records = []

    print()
    print("=" * 72)
    print("STEP 3 - ENTER ROBOT XY")
    print("=" * 72)
    print("用 STM32 XY JOG 让夹爪 TCP 对准同一个实体点。")
    print("然后输入当前机械 X_mm Y_mm。")
    print("建议每个点都从相同方向接近，减小回程间隙影响。")
    print("=" * 72)

    for i, ((point_id, point_name), (u, v)) in enumerate(
        zip(POINTS, image_points),
        start=1,
    ):
        while True:
            print()
            print(f"[{i}/{POINT_COUNT}] {point_id} {point_name}")
            print(f"image pixel = ({u:.2f}, {v:.2f})")

            xy = parse_xy(input("输入 X_mm Y_mm：").strip())
            if xy is None:
                print("[WARN] 格式错误，例如：125.4 83.7")
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


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return abs(float(cv2.contourArea(pts)))


def validate_geometry(image_points, robot_points) -> None:
    image_area = polygon_area(image_points)
    robot_area = polygon_area(robot_points)

    if image_area < 1000.0:
        raise RuntimeError(
            "4个图像标定点覆盖面积太小或顺序异常："
            f"image_area={image_area:.1f} px^2"
        )

    if robot_area < 10.0:
        raise RuntimeError(
            "4个机械标定点覆盖面积太小或顺序异常："
            f"robot_area={robot_area:.3f} mm^2"
        )


def fit_and_save(frame, indexed_image, records) -> None:
    image_points = np.asarray(
        [[r["u"], r["v"]] for r in records],
        dtype=np.float32,
    )
    robot_points = np.asarray(
        [[r["x_mm"], r["y_mm"]] for r in records],
        dtype=np.float32,
    )

    validate_geometry(image_points, robot_points)

    # 只有4个点时，不使用RANSAC。
    # 4个点正好唯一决定一个Homography，没有冗余点可剔除。
    H = cv2.getPerspectiveTransform(image_points, robot_points)

    predicted = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2),
        H,
    ).reshape(-1, 2)

    errors = np.linalg.norm(predicted - robot_points, axis=1)

    for i, record in enumerate(records):
        record["pred_x_mm"] = float(predicted[i, 0])
        record["pred_y_mm"] = float(predicted[i, 1])
        record["error_mm"] = float(errors[i])

    transform = PixelToRobotTransform(
        homography=H,
        image_width=frame.shape[1],
        image_height=frame.shape[0],
        source="TopCamera.read() exact frame used by tray4 calibration",
        metadata={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "method": "4_point_direct_homography",
            "point_count": 4,
            "note": (
                "Four points exactly determine a homography; "
                "independent validation is mandatory."
            ),
        },
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform.save(TRANSFORM_PATH)
    cv2.imwrite(str(SNAPSHOT_PATH), frame)
    cv2.imwrite(str(INDEXED_PATH), indexed_image)

    with POINTS_PATH.open("w", newline="", encoding="utf-8-sig") as f:
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
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "4_point_direct_homography",
        "image_size": [int(frame.shape[1]), int(frame.shape[0])],
        "point_count": 4,
        "image_polygon_area_px2": polygon_area(image_points),
        "robot_polygon_area_mm2": polygon_area(robot_points),
        "homography": H.tolist(),
        "records": records,
        "note": (
            "4-point fit error is not a useful accuracy metric. "
            "Use independent validation points."
        ),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print("TRAY 4-POINT CALIBRATION COMPLETE")
    print("=" * 72)

    for record in records:
        print(
            f"{record['id']} {record['name']:<14} "
            f"pixel=({record['u']:.1f}, {record['v']:.1f}) "
            f"robot=({record['x_mm']:.3f}, {record['y_mm']:.3f})"
        )

    print("-" * 72)
    print(f"Transform : {TRANSFORM_PATH}")
    print(f"Points    : {POINTS_PATH}")
    print(f"Report    : {REPORT_PATH}")
    print(f"Snapshot  : {SNAPSHOT_PATH}")
    print("=" * 72)
    print()
    print(
        "[IMPORTANT] 只有4个点时，Homography 会几乎精确穿过这4个点，"
        "所以不能靠这4点的拟合误差判断精度。"
    )
    print("下一步必须用 5~10 个未参与标定的内部点做独立验证。")


def main() -> None:
    camera = TopCamera()

    try:
        camera.open()
        frame = capture_clean_snapshot(camera)
        image_points, indexed = collect_image_points(frame)
        records = collect_robot_points(image_points)
        fit_and_save(frame, indexed, records)

    except KeyboardInterrupt:
        print("\n[CANCEL] 标定已取消")

    finally:
        safe_release_camera(camera)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
