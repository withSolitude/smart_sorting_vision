from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from communication.sorting_protocol import Mode, SortingFrame, make_calib_yaw
from communication.stm32_link import STM32Link
from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.calibration.pixel_to_robot import PixelToRobotTransform
from vision.calibration.yaw_tcp_compensation import YawTcpCompensator


# Must match the final firmware parameters in ita_robot.h.
# ============================================================
# Final Yaw calibration - 2026-09-25
# calibration XY = (-30, 3)
# ============================================================

YAW_REF_AXIS_DEG = 43.590200
YAW_REF_SERVO_DEG = -0.622672
YAW_SERVO_PER_AXIS = -1.10205336

YAW_SERVO_MIN_DEG = -60.0
YAW_SERVO_MAX_DEG = 135.0


# ============================================================
# Compensated verification targets
#
# TCP fit was obtained mainly from the reliable positive-angle
# region around XY=(-30,3).
# ============================================================

DEFAULT_TARGETS_DEG = [
    0.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
]


# ============================================================
# Reference yaw of the NEW XY/H teaching pose
#
# New H teaching kept Yaw/Lift/Claw geometry fixed.
# Standby Yaw servo command during that calibration = 110 deg.
#
# Using the final yaw calibration:
# physical closing axis at servo=110 deg ~= -56.788489 deg
# ============================================================

TCP_REFERENCE_AXIS_DEG = -56.788489


# ============================================================
# TCP rotating-offset fit
# XY=(-30,3), positive-angle fit
#
# radius = 15.446985 mm
# fit RMSE = 0.100565 mm
# fit MAX  = 0.164872 mm
# ============================================================

TCP_OFFSET_AXIS0_X_MM = 0.2362879328
TCP_OFFSET_AXIS0_Y_MM = 15.4451777630

# Verification range only.
# 0 deg is a very small extrapolation from the 10..60 deg fit
# and was independently visible/testable.
TCP_MIN_AXIS_DEG = 0.0
TCP_MAX_AXIS_DEG = 60.0


# ============================================================
# New XY command <-> visual TCP Jacobian
# ============================================================

VISUAL_TO_COMMAND_MATRIX = (
    (0.72533465, 0.01675850),
    (0.00479271, 0.64797763),
)

COMMAND_TO_VISUAL_TCP_MATRIX = (
    (1.37890961, -0.03566244),
    (-0.01019898, 1.54352692),
)
class UserStopRequested(RuntimeError):
    pass


def normalize_axial_deg(angle_deg: float) -> float:
    """Normalize an undirected axis angle into [-90, 90)."""
    value = (float(angle_deg) + 90.0) % 180.0 - 90.0
    if abs(value) < 1e-12:
        return 0.0
    return value


def axial_error_deg(measured_deg: float, target_deg: float) -> float:
    """Shortest signed error between two undirected axes, in [-90, 90)."""
    return normalize_axial_deg(float(measured_deg) - float(target_deg))


def axis_to_servo_deg(target_axis_deg: float, current_servo_deg: float = 0.0) -> float:
    """Mirror production firmware's unique directed-yaw branch."""
    target_axis_deg = normalize_axial_deg(target_axis_deg)
    q = YAW_REF_SERVO_DEG + YAW_SERVO_PER_AXIS * (target_axis_deg - YAW_REF_AXIS_DEG)
    if not (YAW_SERVO_MIN_DEG <= q <= YAW_SERVO_MAX_DEG):
        raise ValueError(
            f"unique yaw branch is outside calibrated servo range: axis={target_axis_deg:+.2f}, q={q:+.2f}"
        )
    return float(q)


def parse_targets(text: str) -> list[float]:
    parts = [v.strip() for v in text.split(",") if v.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("target list cannot be empty")
    try:
        return [normalize_axial_deg(float(v)) for v in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the final Yaw-TCP compensation. The base X/Y is the H/reference-yaw command; "
            "for each yaw this tool applies deltaXY = R(ref)d - R(theta)d before moving the stage, "
            "then checks whether the measured TCP center stays fixed."
        )
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "runtime_config.yaml"))
    parser.add_argument("--x", type=float, default=-70.0, help="base H/reference-yaw X command in mm")
    parser.add_argument("--y", type=float, default=20.0, help="base H/reference-yaw Y command in mm")
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=DEFAULT_TARGETS_DEG,
        help='comma-separated physical closing-axis targets, e.g. "-45,-55,-65,-75,-85"',
    )
    parser.add_argument("--settle", type=float, default=1.5, help="extra seconds after MCU YAW REACHED")
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    parser.add_argument("--reach-timeout", type=float, default=3.0)
    parser.add_argument(
        "--min-click-span",
        type=float,
        default=20.0,
        help="minimum distance between the two physical reference clicks, in mm",
    )
    parser.add_argument(
        "--max-pairwise",
        type=float,
        default=2.5,
        help="PASS threshold for maximum pairwise compensated TCP-center drift, in mm",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=1.5,
        help="PASS threshold for RMS distance of compensated TCP centers from their mean, in mm",
    )
    return parser.parse_args()


def handle_safety_frame(frame: SortingFrame) -> None:
    if frame.mode == Mode.MCU_ERROR:
        raise RuntimeError(
            f"STM32 MCU_ERROR: code={frame.box_id}, x={frame.x_mm:.1f}, y={frame.y_mm:.1f}"
        )
    if frame.mode == Mode.MCU_STOP_WORK:
        raise UserStopRequested("STM32 physical STOP_WORK received")


def drain_before_command(link: STM32Link) -> None:
    for frame in link.poll():
        print(f"[OLD RX] {link.describe(frame)}")
        handle_safety_frame(frame)


def send_calib_move_and_wait(
    link: STM32Link,
    x_mm: float,
    y_mm: float,
    *,
    ack_timeout: float,
    reach_timeout: float,
) -> None:
    drain_before_command(link)
    raw = link.send_frame(SortingFrame(mode=Mode.CALIB_MOVE, x_mm=x_mm, y_mm=y_mm))
    print(f"[TX] CALIB_MOVE X={x_mm:.2f} Y={y_mm:.2f} | {raw.hex(' ').upper()}")
    start = time.monotonic()
    ack_seen = False
    reached = False
    while True:
        now = time.monotonic()
        for frame in link.poll():
            print(f"[RX] {link.describe(frame)}")
            handle_safety_frame(frame)
            if frame.mode == Mode.MCU_BUSY:
                raise RuntimeError("MCU_BUSY: TCP check requires sorting handshake STOPPED")
            if frame.mode == Mode.MCU_CALIB_MOVE_ACK:
                if abs(frame.x_mm - x_mm) <= 0.2 and abs(frame.y_mm - y_mm) <= 0.2:
                    ack_seen = True
            elif frame.mode == Mode.MCU_CALIB_REACHED:
                if abs(frame.x_mm - x_mm) <= 0.5 and abs(frame.y_mm - y_mm) <= 0.5:
                    reached = True
        if ack_seen and reached:
            return
        if not ack_seen and now - start >= ack_timeout:
            raise TimeoutError("timeout waiting MCU_CALIB_MOVE_ACK")
        if now - start >= reach_timeout:
            raise TimeoutError("timeout waiting MCU_CALIB_REACHED")
        time.sleep(0.002)


def send_calib_yaw_and_wait(
    link: STM32Link,
    servo_deg: float,
    *,
    ack_timeout: float,
    reach_timeout: float,
) -> None:
    drain_before_command(link)
    raw = link.send_frame(make_calib_yaw(float(servo_deg)))
    print(f"[TX] CALIB_YAW servo={servo_deg:+.3f}deg | {raw.hex(' ').upper()}")
    start = time.monotonic()
    ack_seen = False
    reached = False
    while True:
        now = time.monotonic()
        for frame in link.poll():
            print(f"[RX] {link.describe(frame)}")
            handle_safety_frame(frame)
            if frame.mode == Mode.MCU_BUSY:
                raise RuntimeError("MCU_BUSY: CALIB_YAW only works while sorting is STOPPED")
            if frame.mode == Mode.MCU_CALIB_YAW_ACK and abs(frame.x_mm - servo_deg) <= 0.2:
                ack_seen = True
            elif frame.mode == Mode.MCU_CALIB_YAW_REACHED and abs(frame.x_mm - servo_deg) <= 0.2:
                reached = True
        if ack_seen and reached:
            return
        if not ack_seen and now - start >= ack_timeout:
            raise TimeoutError(f"timeout waiting MCU_CALIB_YAW_ACK for {servo_deg:+.2f} deg")
        if now - start >= reach_timeout:
            raise TimeoutError(f"timeout waiting MCU_CALIB_YAW_REACHED for {servo_deg:+.2f} deg")
        time.sleep(0.002)


class ClickUI:
    def __init__(self) -> None:
        self.points: list[tuple[float, float]] = []

    def clear(self) -> None:
        self.points.clear()

    def mouse_callback(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= 2:
            self.points.clear()
        self.points.append((float(x), float(y)))


def measure_pose_from_two_points(
    points: list[tuple[float, float]],
    transform: PixelToRobotTransform,
    *,
    min_span_mm: float,
) -> dict:
    """
    Convert two repeatable physical points to robot coordinates.

    The line between the two points must follow the gripper closing axis and, critically,
    the midpoint of the two chosen physical points must represent the actual gripping/TCP
    center. A good choice is one symmetric point on each finger at the same longitudinal
    position.
    """
    if len(points) != 2:
        raise ValueError("need exactly two clicks")
    (u1, v1), (u2, v2) = points
    x1, y1 = transform.pixel_to_robot(u1, v1, apply_tcp_offset=False)
    x2, y2 = transform.pixel_to_robot(u2, v2, apply_tcp_offset=False)
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    span = float(math.hypot(dx, dy))
    if span < min_span_mm:
        raise ValueError(
            f"click span {span:.1f} mm < {min_span_mm:.1f} mm; choose farther symmetric points"
        )
    measured_axis = normalize_axial_deg(math.degrees(math.atan2(dy, dx)))
    center_x = float((x1 + x2) * 0.5)
    center_y = float((y1 + y2) * 0.5)
    return {
        "u1": float(u1),
        "v1": float(v1),
        "u2": float(u2),
        "v2": float(v2),
        "x1_mm": float(x1),
        "y1_mm": float(y1),
        "x2_mm": float(x2),
        "y2_mm": float(y2),
        "axis_click_span_mm": span,
        "measured_axis_deg": float(measured_axis),
        "tcp_center_x_mm": center_x,
        "tcp_center_y_mm": center_y,
    }


def draw_text(image: np.ndarray, text: str, y: int, color=(0, 255, 0), scale=0.50) -> None:
    cv2.putText(
        image,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def fit_rotating_offset(rows: list[dict]) -> dict:
    """Fit p(theta) = c + R(theta) d using measured physical axis angles."""
    if len(rows) < 3:
        raise ValueError("at least 3 samples are required to fit a rotating TCP offset")

    a_rows: list[list[float]] = []
    b_vals: list[float] = []
    for row in rows:
        theta = math.radians(float(row["measured_axis_deg"]))
        c = math.cos(theta)
        s = math.sin(theta)
        # x = cx + c*dx - s*dy
        a_rows.append([1.0, 0.0, c, -s])
        b_vals.append(float(row["tcp_center_x_mm"]))
        # y = cy + s*dx + c*dy
        a_rows.append([0.0, 1.0, s, c])
        b_vals.append(float(row["tcp_center_y_mm"]))

    A = np.asarray(a_rows, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, dx0, dy0 = [float(v) for v in params]

    residuals: list[float] = []
    predicted: list[tuple[float, float]] = []
    for row in rows:
        theta = math.radians(float(row["measured_axis_deg"]))
        c = math.cos(theta)
        s = math.sin(theta)
        px = cx + c * dx0 - s * dy0
        py = cy + s * dx0 + c * dy0
        predicted.append((px, py))
        residuals.append(
            float(math.hypot(px - float(row["tcp_center_x_mm"]), py - float(row["tcp_center_y_mm"])))
        )

    radius = float(math.hypot(dx0, dy0))
    rms = float(math.sqrt(np.mean(np.square(residuals))))
    max_res = float(max(residuals))
    return {
        "rotation_center_x_mm": cx,
        "rotation_center_y_mm": cy,
        "tcp_offset_at_axis_0_x_mm": dx0,
        "tcp_offset_at_axis_0_y_mm": dy0,
        "fitted_tcp_offset_radius_mm": radius,
        "fit_residual_rmse_mm": rms,
        "fit_residual_max_mm": max_res,
    }


def pairwise_max_distance(points_xy: np.ndarray) -> float:
    if len(points_xy) <= 1:
        return 0.0
    max_d = 0.0
    for i in range(len(points_xy)):
        diff = points_xy[i + 1 :] - points_xy[i]
        if len(diff):
            max_d = max(max_d, float(np.max(np.linalg.norm(diff, axis=1))))
    return max_d


def build_report(rows: list[dict], args: argparse.Namespace, mapping_path: Path) -> dict:
    centers = np.asarray(
        [[float(r["tcp_center_x_mm"]), float(r["tcp_center_y_mm"])] for r in rows],
        dtype=np.float64,
    )
    mean_center = np.mean(centers, axis=0)
    from_mean = np.linalg.norm(centers - mean_center, axis=1)
    max_pairwise = pairwise_max_distance(centers)
    rms = float(math.sqrt(np.mean(np.square(from_mean))))
    passed = bool(max_pairwise <= float(args.max_pairwise) and rms <= float(args.rms_threshold))
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": (
            "base H/reference-yaw XY -> apply calibrated yaw-TCP deltaXY -> move stage -> "
            "set unique physical yaw -> click two symmetric gripper points -> verify TCP center stays fixed"
        ),
        "base_reference_xy_mm": {"x": float(args.x), "y": float(args.y)},
        "pixel_to_robot_file": str(mapping_path.relative_to(ROOT)),
        "tcp_compensation": {
            "reference_axis_deg": TCP_REFERENCE_AXIS_DEG,
            "offset_axis0_x_mm": TCP_OFFSET_AXIS0_X_MM,
            "offset_axis0_y_mm": TCP_OFFSET_AXIS0_Y_MM,
            "min_axis_deg": TCP_MIN_AXIS_DEG,
            "max_axis_deg": TCP_MAX_AXIS_DEG,
            "visual_to_command_matrix": [list(r) for r in VISUAL_TO_COMMAND_MATRIX],
            "command_to_visual_tcp_matrix": [list(r) for r in COMMAND_TO_VISUAL_TCP_MATRIX],
        },
        "sample_count": len(rows),
        "mean_tcp_center_mm": {"x": float(mean_center[0]), "y": float(mean_center[1])},
        "max_distance_from_mean_mm": float(np.max(from_mean)),
        "rms_distance_from_mean_mm": rms,
        "max_pairwise_tcp_drift_mm": float(max_pairwise),
        "thresholds_mm": {
            "max_pairwise": float(args.max_pairwise),
            "rms_from_mean": float(args.rms_threshold),
        },
        "passed": passed,
        "samples": rows,
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.settle < 0 or args.min_click_span <= 0:
        raise ValueError("--settle must be >=0 and --min-click-span must be >0")

    cfg = load_runtime_config(ROOT, args.config)
    x_lim = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["x"])
    y_lim = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["y"])
    if not (x_lim[0] <= args.x <= x_lim[1] and y_lim[0] <= args.y <= y_lim[1]):
        raise ValueError(f"verification XY out of workspace: ({args.x}, {args.y})")

    mapping_path = Path(cfg["calibration"]["pixel_to_robot"])
    transform = PixelToRobotTransform.load(mapping_path)
    output_dir = mapping_path.parent
    csv_path = output_dir / "yaw_tcp_compensated_verification_samples.csv"
    report_path = output_dir / "yaw_tcp_compensated_verification_report.json"

    TopCamera.configure(cfg["camera"]["top"], calibration_file=cfg["calibration"]["top_intrinsics"])
    camera = TopCamera()
    link = STM32Link(device=cfg["stm32"]["device"], baudrate=cfg["stm32"]["baud"])
    clicks = ClickUI()
    rows: list[dict] = []
    current_servo = 0.0
    compensator = YawTcpCompensator(
        reference_axis_deg=TCP_REFERENCE_AXIS_DEG,
        offset_axis0_x_mm=TCP_OFFSET_AXIS0_X_MM,
        offset_axis0_y_mm=TCP_OFFSET_AXIS0_Y_MM,
        min_axis_deg=TCP_MIN_AXIS_DEG,
        max_axis_deg=TCP_MAX_AXIS_DEG,
        visual_to_command_matrix=VISUAL_TO_COMMAND_MATRIX,
    )

    print("=" * 92)
    print("YAW TCP COMPENSATION INDEPENDENT VERIFICATION")
    print("=" * 92)
    print(f"Base H/reference-yaw XY = ({args.x:.2f}, {args.y:.2f}) mm")
    print(f"Yaw targets = {', '.join(f'{v:+.1f}' for v in args.targets)} deg")
    print("For EVERY pose, click two repeatable physical points, one on each finger.")
    print("Their connecting line must be the closing axis, and their MIDPOINT must be the true gripping center.")
    print(f"Two-click span must be >= {args.min_click_span:.1f} mm.")
    print("ENTER: save pose   C: clear/re-click   N: skip pose   Q/ESC: abort")
    print("=" * 92)

    try:
        link.open()
        camera.open()
        cv2.namedWindow("Yaw TCP Compensation Verify", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Yaw TCP Compensation Verify", clicks.mouse_callback)

        for idx, target_axis in enumerate(args.targets, start=1):
            target_axis = normalize_axial_deg(target_axis)
            corrected = compensator.apply(float(args.x), float(args.y), target_axis)
            cmd_x = float(corrected.x_mm)
            cmd_y = float(corrected.y_mm)
            if not (x_lim[0] <= cmd_x <= x_lim[1] and y_lim[0] <= cmd_y <= y_lim[1]):
                raise ValueError(
                    f"compensated XY outside workspace at yaw {target_axis:+.2f}: "
                    f"({cmd_x:.2f}, {cmd_y:.2f})"
                )
            servo_cmd = axis_to_servo_deg(target_axis, current_servo)
            print(
                f"\n[{idx}/{len(args.targets)}] target_axis={target_axis:+.2f} deg "
                f"-> visual_dXY=({corrected.visual_delta_x_mm:+.3f},{corrected.visual_delta_y_mm:+.3f}) mm "
                f"-> cmd_dXY=({corrected.delta_x_mm:+.3f},{corrected.delta_y_mm:+.3f}) mm "
                f"-> XY=({cmd_x:+.3f},{cmd_y:+.3f}) -> servo={servo_cmd:+.3f} deg"
            )
            send_calib_move_and_wait(
                link,
                cmd_x,
                cmd_y,
                ack_timeout=args.ack_timeout,
                reach_timeout=max(args.reach_timeout, 15.0),
            )
            send_calib_yaw_and_wait(
                link,
                servo_cmd,
                ack_timeout=args.ack_timeout,
                reach_timeout=args.reach_timeout,
            )
            current_servo = servo_cmd
            if args.settle:
                print(f"[WAIT] extra settle {args.settle:.1f}s")
                time.sleep(args.settle)

            clicks.clear()
            while True:
                frame = camera.read()
                transform.check_frame(frame)
                display = frame.copy()
                measurement = None
                problem = None
                if len(clicks.points) == 2:
                    try:
                        measurement = measure_pose_from_two_points(
                            clicks.points,
                            transform,
                            min_span_mm=args.min_click_span,
                        )
                    except ValueError as exc:
                        problem = str(exc)

                for pidx, point in enumerate(clicks.points):
                    p = (int(round(point[0])), int(round(point[1])))
                    cv2.circle(display, p, 6, (0, 255, 255), 2)
                    cv2.putText(
                        display,
                        f"P{pidx+1}",
                        (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if len(clicks.points) == 2:
                    p1 = tuple(int(round(v)) for v in clicks.points[0])
                    p2 = tuple(int(round(v)) for v in clicks.points[1])
                    cv2.line(display, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)
                    mid = (int(round((p1[0] + p2[0]) * 0.5)), int(round((p1[1] + p2[1]) * 0.5)))
                    cv2.drawMarker(
                        display,
                        mid,
                        (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=18,
                        thickness=2,
                    )

                draw_text(
                    display,
                    (f"[{idx}/{len(args.targets)}] axis={target_axis:+.1f}deg servo={servo_cmd:+.2f}deg "
                     f"XY=({cmd_x:+.1f},{cmd_y:+.1f})"),
                    28,
                )
                draw_text(
                    display,
                    "Click TWO symmetric finger points; red cross = measured gripping-center midpoint",
                    54,
                    color=(0, 255, 255),
                )
                draw_text(display, "ENTER save | C clear | N skip | Q/ESC abort", 80)

                if measurement is not None:
                    axis_err = axial_error_deg(measurement["measured_axis_deg"], target_axis)
                    draw_text(
                        display,
                        (
                            f"measured axis={measurement['measured_axis_deg']:+.2f}deg "
                            f"err={axis_err:+.2f}deg span={measurement['axis_click_span_mm']:.1f}mm"
                        ),
                        106,
                        color=(255, 255, 0),
                    )
                    draw_text(
                        display,
                        (
                            f"TCP center=({measurement['tcp_center_x_mm']:+.2f}, "
                            f"{measurement['tcp_center_y_mm']:+.2f}) mm"
                        ),
                        132,
                        color=(255, 255, 0),
                    )
                elif problem:
                    draw_text(display, problem, 106, color=(0, 0, 255))

                cv2.imshow("Yaw TCP Compensation Verify", display)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
                if key in (ord("c"), ord("C")):
                    clicks.clear()
                    continue
                if key in (ord("n"), ord("N")):
                    print("[SKIP] current yaw target")
                    break
                if key in (10, 13):
                    if measurement is None:
                        print(f"[NOT SAVED] {problem or 'click two valid symmetric points first'}")
                        continue
                    axis_err = axial_error_deg(measurement["measured_axis_deg"], target_axis)
                    row = {
                        "index": len(rows) + 1,
                        "target_axis_deg": float(target_axis),
                        "servo_command_deg": float(servo_cmd),
                        "command_x_mm": cmd_x,
                        "command_y_mm": cmd_y,
                        "visual_comp_delta_x_mm": float(corrected.visual_delta_x_mm),
                        "visual_comp_delta_y_mm": float(corrected.visual_delta_y_mm),
                        "comp_delta_x_mm": float(corrected.delta_x_mm),
                        "comp_delta_y_mm": float(corrected.delta_y_mm),
                        "measured_axis_deg": float(measurement["measured_axis_deg"]),
                        "axis_error_deg": float(axis_err),
                        **measurement,
                        "accepted_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    rows.append(row)
                    print(
                        f"[SAVE] axis target={target_axis:+.2f} measured={measurement['measured_axis_deg']:+.2f} "
                        f"err={axis_err:+.2f} | center=({measurement['tcp_center_x_mm']:+.3f}, "
                        f"{measurement['tcp_center_y_mm']:+.3f}) mm"
                    )
                    break

        if len(rows) < 3:
            print("Need at least 3 accepted poses; no TCP-offset report written.")
            return 2

        save_csv(csv_path, rows)
        report = build_report(rows, args, mapping_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n" + "=" * 92)
        print("TCP COMPENSATION VERIFICATION RESULT")
        print("=" * 92)
        print(f"samples                  : {len(rows)}")
        print(f"max pairwise TCP drift   : {report['max_pairwise_tcp_drift_mm']:.3f} mm")
        print(f"max distance from mean   : {report['max_distance_from_mean_mm']:.3f} mm")
        print(f"RMS distance from mean   : {report['rms_distance_from_mean_mm']:.3f} mm")
        print(f"PASS                     : {report['passed']}")
        print(f"CSV                      : {csv_path}")
        print(f"JSON                     : {report_path}")
        if report["passed"]:
            print("Compensation verified. You may enable vision.top.yaw_tcp_compensation.enabled=true.")
        else:
            print("Do NOT enable production compensation yet; review this report first.")
        return 0

    except UserStopRequested as exc:
        print(f"[STOP] {exc}")
        return 3
    except KeyboardInterrupt:
        print("\n[ABORT] user aborted TCP offset check")
        return 130
    finally:
        try:
            camera.release()
        except Exception:
            pass
        try:
            link.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
