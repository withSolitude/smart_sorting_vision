from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from communication.sorting_protocol import Mode, SortingFrame
from communication.stm32_link import STM32Link
from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.calibration.pixel_to_robot import PixelToRobotTransform


# ----------------------------------------------------------------------
# 这次 2026-09-24 固定点标定对应的新 ROI:
# safe_roi = [580, 120, 1330, 927]
# ROI size = 750 x 807
#
# 注意：
#   真正运行时仍以 runtime_config.yaml 为唯一相机配置来源；
#   这里仅做一致性检查，不在脚本里覆盖 TopCamera 的 ROI。
# ----------------------------------------------------------------------
EXPECTED_SAFE_ROI = (580, 120, 1330, 927)
EXPECTED_FRAME_SIZE = (750, 807)  # width, height

DEFAULT_SESSION = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "robot_mapping"
    / "teach_sessions"
    / "20260924_231006"
)

DEFAULT_CANDIDATE = DEFAULT_SESSION / "pixel_to_robot_candidate.json"
DEFAULT_TAUGHT_POINTS = DEFAULT_SESSION / "taught_points.csv"

# 优先复查本次拟合中误差较大的点。
DEFAULT_MARKERS = ("P04", "P05", "P06", "P07")


class UserStopRequested(RuntimeError):
    pass


def _project_path(value: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Top XY calibration verification for the 20260924 candidate mapping. "
            "Press N to verify preset markers such as P04/P05/P06/P07, or click any point."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "runtime_config.yaml"),
        help="runtime_config.yaml path",
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CANDIDATE),
        help=(
            "mapping JSON to verify; default is the 20260924_231006 "
            "pixel_to_robot_candidate.json"
        ),
    )
    parser.add_argument(
        "--taught-points",
        default=str(DEFAULT_TAUGHT_POINTS),
        help="taught_points.csv used to locate P01..P12 in the ROI image",
    )
    parser.add_argument(
        "--markers",
        default=",".join(DEFAULT_MARKERS),
        help="preset marker queue, e.g. P04,P05,P06,P07",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.5,
        help="default manual micro-adjust step in mm (default: 0.5)",
    )
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    parser.add_argument("--reach-timeout", type=float, default=15.0)
    return parser.parse_args()


def inside_workspace(
    x_mm: float,
    y_mm: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> bool:
    return (
        x_limits[0] <= x_mm <= x_limits[1]
        and y_limits[0] <= y_mm <= y_limits[1]
    )


def handle_safety_frame(frame: SortingFrame) -> None:
    if frame.mode == Mode.MCU_ERROR:
        raise RuntimeError(
            "STM32 MCU_ERROR: "
            f"x={frame.x_mm:.1f}, y={frame.y_mm:.1f}, code={frame.box_id}"
        )

    if frame.mode == Mode.MCU_STOP_WORK:
        raise UserStopRequested("STM32 physical STOP_WORK received")


def drain_before_command(link: STM32Link) -> None:
    for frame in link.poll():
        print(f"[OLD RX] {link.describe(frame)}")
        handle_safety_frame(frame)


def _coord_matches(
    frame: SortingFrame,
    x_mm: float,
    y_mm: float,
    tol_mm: float,
) -> bool:
    return (
        abs(frame.x_mm - x_mm) <= tol_mm
        and abs(frame.y_mm - y_mm) <= tol_mm
    )


def send_calib_move_and_wait(
    link: STM32Link,
    x_mm: float,
    y_mm: float,
    *,
    ack_timeout: float,
    reach_timeout: float,
) -> SortingFrame:
    drain_before_command(link)

    tx = SortingFrame(
        mode=Mode.CALIB_MOVE,
        x_mm=float(x_mm),
        y_mm=float(y_mm),
        box_id=0,
    )
    raw = link.send_frame(tx)
    print(
        f"[TX] CALIB_MOVE X={x_mm:.3f} Y={y_mm:.3f} | "
        f"{raw.hex(' ').upper()}"
    )

    start = time.monotonic()
    ack_deadline = start + float(ack_timeout)
    reach_deadline = start + float(reach_timeout)

    ack_seen = False
    reached: SortingFrame | None = None

    while True:
        now = time.monotonic()

        for frame in link.poll():
            print(f"[RX] {link.describe(frame)}")
            handle_safety_frame(frame)

            if frame.mode == Mode.MCU_BUSY:
                raise RuntimeError(
                    "STM32 returned MCU_BUSY. CALIB_MOVE is only accepted while "
                    "sorting handshake is STOPPED; do not press START during verification."
                )

            if frame.mode == Mode.MCU_CALIB_MOVE_ACK:
                if _coord_matches(frame, x_mm, y_mm, 0.2):
                    ack_seen = True
                else:
                    print(
                        "[IGNORE] stale/mismatched CALIB_ACK: "
                        f"target=({x_mm:.3f},{y_mm:.3f}), "
                        f"rx=({frame.x_mm:.3f},{frame.y_mm:.3f})"
                    )

            elif frame.mode == Mode.MCU_CALIB_REACHED:
                if _coord_matches(frame, x_mm, y_mm, 0.5):
                    reached = frame
                else:
                    print(
                        "[IGNORE] stale/mismatched CALIB_REACHED: "
                        f"target=({x_mm:.3f},{y_mm:.3f}), "
                        f"rx=({frame.x_mm:.3f},{frame.y_mm:.3f})"
                    )

        if ack_seen and reached is not None:
            return reached

        if not ack_seen and now >= ack_deadline:
            raise TimeoutError(
                f"Timeout waiting MCU_CALIB_MOVE_ACK for ({x_mm:.3f}, {y_mm:.3f})"
            )

        if now >= reach_deadline:
            raise TimeoutError(
                f"Timeout waiting MCU_CALIB_REACHED for ({x_mm:.3f}, {y_mm:.3f})"
            )

        time.sleep(0.002)


class VerificationUI:
    def __init__(self) -> None:
        self.clicked: tuple[float, float] | None = None

    def mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = (float(x), float(y))


def draw_text(
    image: np.ndarray,
    text: str,
    y: int,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    scale: float = 0.58,
) -> None:
    cv2.putText(
        image,
        text,
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def load_marker_points(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到示教点文件：{path}")

    points: dict[str, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            point_id = str(row["point_id"]).strip().upper()
            points[point_id] = {
                "u": float(row["u"]),
                "v": float(row["v"]),
                "taught_x_mm": float(row["x_mm"]),
                "taught_y_mm": float(row["y_mm"]),
            }
    return points


def compute_stats(records: list[dict]) -> dict[str, float]:
    errors = np.asarray([r["error_mm"] for r in records], dtype=np.float64)
    return {
        "mean_error_mm": float(np.mean(errors)),
        "rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
        "median_error_mm": float(np.median(errors)),
        "p95_error_mm": float(np.percentile(errors, 95)),
        "max_error_mm": float(np.max(errors)),
        "mean_dx_mm": float(np.mean([r["dx_mm"] for r in records])),
        "mean_dy_mm": float(np.mean([r["dy_mm"] for r in records])),
    }


def save_results(
    records: list[dict],
    *,
    calibration_path: Path,
    csv_path: Path,
    report_path: Path,
) -> None:
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    stats = compute_stats(records)
    calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

    try:
        calibration_label = str(calibration_path.relative_to(ROOT))
    except ValueError:
        calibration_label = str(calibration_path)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "measurement_method": (
            "candidate H -> marker/click -> auto CALIB_MOVE -> "
            "manual micro-adjust -> accept corrected robot XY"
        ),
        "safe_roi_expected": list(EXPECTED_SAFE_ROI),
        "frame_size_expected": list(EXPECTED_FRAME_SIZE),
        "calibration_file": calibration_label,
        "calibration_sha256": calibration_sha256,
        "point_count": len(records),
        **stats,
        "records": records,
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def verify_roi_and_mapping(
    cfg: dict,
    calibration_path: Path,
) -> tuple[int, int]:
    roi = tuple(int(v) for v in cfg["camera"]["top"]["safe_roi"])
    if len(roi) != 4:
        raise ValueError(f"camera.top.safe_roi 必须为4个数，当前：{roi}")

    x1, y1, x2, y2 = roi
    roi_w = x2 - x1
    roi_h = y2 - y1

    with calibration_path.open("r", encoding="utf-8") as f:
        mapping_json = json.load(f)

    map_w = int(mapping_json["image_width"])
    map_h = int(mapping_json["image_height"])

    print(f"[CONFIG] YAML safe_roi = {list(roi)}")
    print(f"[CONFIG] ROI size      = {roi_w} x {roi_h}")
    print(f"[CONFIG] Mapping size  = {map_w} x {map_h}")

    if roi != EXPECTED_SAFE_ROI:
        print(
            "[WARN] 当前 YAML safe_roi 与本次脚本记录的新 ROI 不同："
            f"expected={list(EXPECTED_SAFE_ROI)}, actual={list(roi)}"
        )

    if (roi_w, roi_h) != (map_w, map_h):
        raise RuntimeError(
            "ROI尺寸与候选 pixel_to_robot 映射尺寸不一致，禁止继续运动："
            f"ROI={roi_w}x{roi_h}, mapping={map_w}x{map_h}"
        )

    if (map_w, map_h) != EXPECTED_FRAME_SIZE:
        print(
            "[WARN] 当前 mapping 尺寸不是本次预期的 750x807："
            f"{map_w}x{map_h}"
        )

    return roi_w, roi_h


def main() -> int:
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be > 0")

    cfg = load_runtime_config(ROOT, args.config)

    calibration_path = _project_path(args.calibration)
    taught_points_path = _project_path(args.taught_points)

    if not calibration_path.is_file():
        raise FileNotFoundError(f"找不到标定文件：{calibration_path}")

    marker_points = load_marker_points(taught_points_path)

    marker_queue = [
        item.strip().upper()
        for item in str(args.markers).replace(";", ",").split(",")
        if item.strip()
    ]
    for point_id in marker_queue:
        if point_id not in marker_points:
            raise ValueError(
                f"--markers 中的 {point_id} 不存在于 {taught_points_path.name}"
            )

    verify_roi_and_mapping(cfg, calibration_path)

    output_dir = calibration_path.parent
    verify_csv_path = output_dir / "candidate_marker_validation_points.csv"
    verify_report_path = output_dir / "candidate_marker_validation_report.json"

    x_limits = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["x"])
    y_limits = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["y"])

    transform = PixelToRobotTransform.load(calibration_path)

    # 正式相机参数只从 YAML 读取，包括新 safe_roi=[580,120,1330,927]。
    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )

    camera = TopCamera()
    link = STM32Link(
        device=cfg["stm32"]["device"],
        baudrate=cfg["stm32"]["baud"],
    )

    ui = VerificationUI()
    records: list[dict] = []

    step_mm = float(args.step)
    active: dict | None = None
    finish = False
    next_marker_index = 0

    def start_verification(
        u: float,
        v: float,
        *,
        point_id: str = "",
    ) -> dict | None:
        x_pred, y_pred = transform.pixel_to_robot(
            float(u),
            float(v),
            apply_tcp_offset=False,
        )

        print()
        label = point_id or f"CLICK-{len(records) + 1}"
        print(f"[POINT {label}] pixel=({u:.2f}, {v:.2f})")
        print(f"[H PRED] X={x_pred:.3f} mm, Y={y_pred:.3f} mm")

        if point_id and point_id in marker_points:
            p = marker_points[point_id]
            print(
                f"[OLD TEACH] X={p['taught_x_mm']:.3f} "
                f"Y={p['taught_y_mm']:.3f} mm"
            )

        if not inside_workspace(x_pred, y_pred, x_limits, y_limits):
            print(
                "[BLOCK] H预测坐标超出机械工作范围，不下发："
                f"X={x_pred:.3f}, Y={y_pred:.3f}"
            )
            return None

        send_calib_move_and_wait(
            link,
            x_pred,
            y_pred,
            ack_timeout=args.ack_timeout,
            reach_timeout=args.reach_timeout,
        )

        print(
            "[REACHED] 已到 H 预测位置。观察 TCP 是否对准黑点；"
            "不准就微调，准了按 ENTER。"
        )

        return {
            "point_id": point_id,
            "u": float(u),
            "v": float(v),
            "pred_x_mm": float(x_pred),
            "pred_y_mm": float(y_pred),
            "cmd_x_mm": float(x_pred),
            "cmd_y_mm": float(y_pred),
        }

    print()
    print("=" * 84)
    print("TOP XY CANDIDATE VERIFY: P04/P05/P06/P07 + MANUAL MICRO-ADJUST")
    print("=" * 84)
    print(f"workspace : X={x_limits} mm, Y={y_limits} mm")
    print(f"serial    : {cfg['stm32']['device']} @ {cfg['stm32']['baud']}")
    print(f"mapping   : {calibration_path}")
    print(f"markers   : {marker_queue}")
    print()
    print("操作：")
    print("  N       -> 按队列测试下一个固定点（默认 P04 -> P05 -> P06 -> P07）")
    print("  左键     -> 也可以测试任意图像点")
    print("  J / L   -> X- / X+")
    print("  I / K   -> Y+ / Y-")
    print("  1~5     -> 步长 0.1 / 0.2 / 0.5 / 1.0 / 2.0 mm")
    print("  M       -> 终端输入任意增量 dx dy")
    print("  R       -> 回到 H 预测位置")
    print("  ENTER   -> 当前点对准后保存")
    print("  C       -> 取消当前点")
    print("  Q / ESC -> 结束")
    print()
    print("注意：验证期间 MCU 必须保持 STOPPED，不要按 START。")
    print("=" * 84)

    try:
        link.open()
        camera.open()

        cv2.namedWindow("Tray Calibration Verify", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Tray Calibration Verify", ui.mouse_callback)

        while not finish:
            frame = camera.read()
            transform.check_frame(frame)
            display = frame.copy()

            # 把 12 个旧示教点都画出来，重点点用更醒目的标记。
            for point_id, p in marker_points.items():
                x = int(round(p["u"]))
                y = int(round(p["v"]))
                highlight = point_id in marker_queue
                color = (0, 255, 255) if highlight else (120, 120, 120)
                radius = 7 if highlight else 4
                cv2.circle(display, (x, y), radius, color, 2)
                if highlight:
                    cv2.putText(
                        display,
                        point_id,
                        (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # 已保存点再画绿色圈。
            for record in records:
                p = (int(round(record["u"])), int(round(record["v"])))
                cv2.circle(display, p, 9, (0, 255, 0), 2)

            queue_text = (
                marker_queue[next_marker_index]
                if next_marker_index < len(marker_queue)
                else "DONE"
            )
            draw_text(
                display,
                (
                    f"saved={len(records)} | step={step_mm:.1f}mm | "
                    f"N=next({queue_text}) | click=any | Q=finish"
                ),
                30,
            )

            if active is None:
                draw_text(
                    display,
                    "IDLE: press N for preset marker, or left-click any point",
                    58,
                    color=(0, 255, 255),
                )
            else:
                u = float(active["u"])
                v = float(active["v"])
                cv2.drawMarker(
                    display,
                    (int(round(u)), int(round(v))),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    26,
                    2,
                )
                label = active["point_id"] or "CLICK"
                draw_text(
                    display,
                    (
                        f"{label} H: X={active['pred_x_mm']:.2f} "
                        f"Y={active['pred_y_mm']:.2f} | "
                        f"CMD: X={active['cmd_x_mm']:.2f} "
                        f"Y={active['cmd_y_mm']:.2f}"
                    ),
                    58,
                    color=(0, 255, 255),
                )
                draw_text(
                    display,
                    (
                        f"dX={active['cmd_x_mm'] - active['pred_x_mm']:+.2f} "
                        f"dY={active['cmd_y_mm'] - active['pred_y_mm']:+.2f} mm | "
                        "J/L X-/X+  I/K Y+/Y-  ENTER accept"
                    ),
                    86,
                    color=(0, 255, 255),
                    scale=0.54,
                )

            cv2.imshow("Tray Calibration Verify", display)

            if active is not None and ui.clicked is not None:
                ui.clicked = None

            # 鼠标任意点验证。
            if active is None and ui.clicked is not None:
                u, v = ui.clicked
                ui.clicked = None
                active = start_verification(u, v)

            key = cv2.waitKey(10) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), ord("Q"), 27):
                finish = True
                continue

            # N：按队列逐个验证 P04/P05/P06/P07。
            if active is None and key in (ord("n"), ord("N")):
                if next_marker_index >= len(marker_queue):
                    print("[INFO] 预设 marker 队列已经全部测试完。")
                    continue

                point_id = marker_queue[next_marker_index]
                p = marker_points[point_id]
                active = start_verification(
                    p["u"],
                    p["v"],
                    point_id=point_id,
                )
                if active is not None:
                    next_marker_index += 1
                continue

            if active is None:
                continue

            preset = {
                ord("1"): 0.1,
                ord("2"): 0.2,
                ord("3"): 0.5,
                ord("4"): 1.0,
                ord("5"): 2.0,
            }
            if key in preset:
                step_mm = preset[key]
                print(f"[STEP] {step_mm:.1f} mm")
                continue

            new_x = float(active["cmd_x_mm"])
            new_y = float(active["cmd_y_mm"])
            move_requested = False

            if key in (ord("j"), ord("J")):
                new_x -= step_mm
                move_requested = True
            elif key in (ord("l"), ord("L")):
                new_x += step_mm
                move_requested = True
            elif key in (ord("i"), ord("I")):
                new_y += step_mm
                move_requested = True
            elif key in (ord("k"), ord("K")):
                new_y -= step_mm
                move_requested = True
            elif key in (ord("r"), ord("R")):
                new_x = float(active["pred_x_mm"])
                new_y = float(active["pred_y_mm"])
                move_requested = True
            elif key in (ord("m"), ord("M")):
                text = input(
                    "输入本次增量 dx_mm dy_mm（例如 0.3 -0.2），回车取消："
                ).strip()
                if text:
                    parts = text.replace(",", " ").split()
                    if len(parts) != 2:
                        print("[WARN] 格式应为：dx dy")
                    else:
                        try:
                            dx = float(parts[0])
                            dy = float(parts[1])
                        except ValueError:
                            print("[WARN] dx/dy 必须是数字")
                        else:
                            if np.isfinite(dx) and np.isfinite(dy):
                                new_x += dx
                                new_y += dy
                                move_requested = True
                            else:
                                print("[WARN] dx/dy 必须为有限数")

            elif key in (10, 13):
                dx = float(active["cmd_x_mm"] - active["pred_x_mm"])
                dy = float(active["cmd_y_mm"] - active["pred_y_mm"])
                error = float(np.hypot(dx, dy))

                point_id = str(active["point_id"])
                taught_x = ""
                taught_y = ""
                if point_id and point_id in marker_points:
                    taught_x = marker_points[point_id]["taught_x_mm"]
                    taught_y = marker_points[point_id]["taught_y_mm"]

                record = {
                    "index": len(records) + 1,
                    "point_id": point_id,
                    "u": float(active["u"]),
                    "v": float(active["v"]),
                    "taught_x_mm": taught_x,
                    "taught_y_mm": taught_y,
                    "pred_x_mm": float(active["pred_x_mm"]),
                    "pred_y_mm": float(active["pred_y_mm"]),
                    "actual_x_mm": float(active["cmd_x_mm"]),
                    "actual_y_mm": float(active["cmd_y_mm"]),
                    "final_x_mm": float(active["cmd_x_mm"]),
                    "final_y_mm": float(active["cmd_y_mm"]),
                    "dx_mm": dx,
                    "dy_mm": dy,
                    "error_mm": error,
                    "accepted_at": datetime.now().isoformat(timespec="seconds"),
                }
                records.append(record)

                save_results(
                    records,
                    calibration_path=calibration_path,
                    csv_path=verify_csv_path,
                    report_path=verify_report_path,
                )

                print(
                    f"[SAVE] {point_id or record['index']}: "
                    f"dX={dx:+.3f} dY={dy:+.3f} |dXY|={error:.3f} mm"
                )
                active = None
                continue

            elif key in (ord("c"), ord("C")):
                print("[CANCEL] 当前点不保存。")
                active = None
                continue

            if move_requested:
                if not inside_workspace(new_x, new_y, x_limits, y_limits):
                    print(
                        "[BLOCK] 微调后目标超出工作范围，不下发："
                        f"X={new_x:.3f}, Y={new_y:.3f}; "
                        f"workspace X={x_limits}, Y={y_limits}"
                    )
                    continue

                send_calib_move_and_wait(
                    link,
                    new_x,
                    new_y,
                    ack_timeout=args.ack_timeout,
                    reach_timeout=args.reach_timeout,
                )
                active["cmd_x_mm"] = float(new_x)
                active["cmd_y_mm"] = float(new_y)
                print(
                    f"[ADJUSTED] X={new_x:.3f} Y={new_y:.3f} | "
                    f"dX={new_x - active['pred_x_mm']:+.3f} "
                    f"dY={new_y - active['pred_y_mm']:+.3f}"
                )

    except UserStopRequested as exc:
        print(f"\n[STOP] {exc}")
        return 2
    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt")
        return 130
    finally:
        try:
            if link.is_open:
                try:
                    link.send_frame(SortingFrame(mode=Mode.STOP))
                except Exception as exc:
                    print(f"[WARN] STOP send failed during cleanup: {exc}")
        finally:
            link.close()
            camera.release()
            cv2.destroyAllWindows()

    if not records:
        print("[INFO] 没有保存任何验证点。")
        return 0

    save_results(
        records,
        calibration_path=calibration_path,
        csv_path=verify_csv_path,
        report_path=verify_report_path,
    )

    stats = compute_stats(records)

    print()
    print("=" * 84)
    print("VALIDATION RESULT (candidate H + manual micro-adjust)")
    print("=" * 84)
    print(f"points : {len(records)}")
    print(f"mean   : {stats['mean_error_mm']:.3f} mm")
    print(f"RMSE   : {stats['rmse_mm']:.3f} mm")
    print(f"median : {stats['median_error_mm']:.3f} mm")
    print(f"P95    : {stats['p95_error_mm']:.3f} mm")
    print(f"max    : {stats['max_error_mm']:.3f} mm")
    print(f"mean dX: {stats['mean_dx_mm']:+.3f} mm")
    print(f"mean dY: {stats['mean_dy_mm']:+.3f} mm")
    print(f"CSV    : {verify_csv_path}")
    print(f"report : {verify_report_path}")
    print("=" * 84)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
