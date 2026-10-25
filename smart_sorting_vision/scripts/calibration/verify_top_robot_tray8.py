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


CALIBRATION_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "robot_mapping"
    / "pixel_to_robot.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "robot_mapping"
)

VERIFY_CSV_PATH = (
    OUTPUT_DIR
    / "tray8_validation_points.csv"
)

VERIFY_REPORT_PATH = (
    OUTPUT_DIR
    / "tray8_validation_report.json"
)

# 建议至少 5 个，不参与标定的内部点。
RECOMMENDED_VERIFY_POINTS = 5


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


class VerificationUI:
    def __init__(
        self,
        transform: PixelToRobotTransform,
    ) -> None:
        self.transform = transform
        self.clicked: tuple[float, float] | None = None

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = (
                float(x),
                float(y),
            )


def main() -> None:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"找不到标定文件：{CALIBRATION_PATH}"
        )

    transform = PixelToRobotTransform.load(
        CALIBRATION_PATH
    )

    camera = TopCamera()
    ui = VerificationUI(
        transform
    )

    records: list[dict] = []

    try:
        camera.open()

        cv2.namedWindow(
            "Tray Calibration Verify",
            cv2.WINDOW_NORMAL,
        )

        cv2.setMouseCallback(
            "Tray Calibration Verify",
            ui.mouse_callback,
        )

        print()
        print("=" * 72)
        print("TRAY 8-POINT CALIBRATION VERIFICATION")
        print("=" * 72)
        print(
            "选择 5~10 个没有参与标定的内部固定特征点。"
        )
        print(
            "可临时在托盘内部贴 2~3 mm 小圆点/胶带标记，"
            "但不要移动相机或托盘。"
        )
        print()
        print(
            "1. 鼠标左键点击一个验证点"
        )
        print(
            "2. 程序输出预测 X/Y"
        )
        print(
            "3. 用 STM32 JOG 把 TCP 实际对准同一个实体点"
        )
        print(
            "4. 回到终端输入此时实际 X_mm Y_mm"
        )
        print()
        print(
            "Q / ESC：结束并生成报告"
        )
        print("=" * 72)

        while True:
            frame = camera.read()
            transform.check_frame(
                frame
            )

            display = frame.copy()

            cv2.putText(
                display,
                (
                    f"validation points: "
                    f"{len(records)} | "
                    f"left click=new point | Q=finish"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            for record in records:
                p = (
                    int(round(record["u"])),
                    int(round(record["v"])),
                )

                cv2.circle(
                    display,
                    p,
                    6,
                    (0, 255, 255),
                    2,
                )

            if ui.clicked is not None:
                u, v = ui.clicked

                x_pred, y_pred = (
                    transform.pixel_to_robot(
                        u,
                        v,
                        apply_tcp_offset=False,
                    )
                )

                cv2.drawMarker(
                    display,
                    (
                        int(round(u)),
                        int(round(v)),
                    ),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    26,
                    2,
                )

                cv2.imshow(
                    "Tray Calibration Verify",
                    display,
                )

                cv2.waitKey(1)

                print()
                print(
                    f"[VERIFY {len(records) + 1}]"
                )
                print(
                    f"pixel     = ({u:.2f}, {v:.2f})"
                )
                print(
                    f"predicted = "
                    f"X={x_pred:.3f} mm, "
                    f"Y={y_pred:.3f} mm"
                )

                while True:
                    text = input(
                        "TCP实际对准该实体点后，"
                        "输入实际 X_mm Y_mm；"
                        "输入 s 跳过："
                    ).strip()

                    if text.lower() == "s":
                        break

                    xy = parse_xy(
                        text
                    )

                    if xy is None:
                        print(
                            "[WARN] 格式错误，例如：125.4 83.7"
                        )
                        continue

                    x_actual, y_actual = xy

                    error = float(
                        np.hypot(
                            x_pred - x_actual,
                            y_pred - y_actual,
                        )
                    )

                    records.append(
                        {
                            "index":
                                len(records) + 1,
                            "u":
                                float(u),
                            "v":
                                float(v),
                            "pred_x_mm":
                                float(x_pred),
                            "pred_y_mm":
                                float(y_pred),
                            "actual_x_mm":
                                float(x_actual),
                            "actual_y_mm":
                                float(y_actual),
                            "error_mm":
                                error,
                        }
                    )

                    print(
                        f"error = {error:.3f} mm"
                    )
                    break

                ui.clicked = None

            cv2.imshow(
                "Tray Calibration Verify",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

    finally:
        camera.close()
        cv2.destroyAllWindows()

    if not records:
        print(
            "[INFO] 没有保存任何验证点"
        )
        return

    errors = np.asarray(
        [
            r["error_mm"]
            for r in records
        ],
        dtype=np.float64,
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

    median = float(
        np.median(errors)
    )

    p95 = float(
        np.percentile(
            errors,
            95,
        )
    )

    max_error = float(
        np.max(errors)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with VERIFY_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                records[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            records
        )

    report = {
        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
        "point_count":
            len(records),
        "mean_error_mm":
            mean_error,
        "rmse_mm":
            rmse,
        "median_error_mm":
            median,
        "p95_error_mm":
            p95,
        "max_error_mm":
            max_error,
        "records":
            records,
    }

    with VERIFY_REPORT_PATH.open(
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
    print("VALIDATION RESULT")
    print("=" * 72)
    print(
        f"points : {len(records)}"
    )
    print(
        f"mean   : {mean_error:.3f} mm"
    )
    print(
        f"RMSE   : {rmse:.3f} mm"
    )
    print(
        f"median : {median:.3f} mm"
    )
    print(
        f"P95    : {p95:.3f} mm"
    )
    print(
        f"max    : {max_error:.3f} mm"
    )
    print("=" * 72)

    if len(records) < RECOMMENDED_VERIFY_POINTS:
        print(
            "[WARN] 验证点少于5个，建议继续补充。"
        )

    if mean_error <= 2.0 and p95 <= 3.0:
        print(
            "[PASS] 当前平面标定精度可进入真实物体 XY 空走验证。"
        )
    else:
        print(
            "[WARN] 当前验证误差偏大，"
            "先检查点击、机械TCP对准、回程间隙和相机/托盘固定。"
        )


if __name__ == "__main__":
    main()
