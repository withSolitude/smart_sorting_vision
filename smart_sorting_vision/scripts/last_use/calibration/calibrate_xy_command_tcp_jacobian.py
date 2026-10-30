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

# Must match the final firmware yaw calibration.
YAW_REF_AXIS_DEG = 43.590200
YAW_REF_SERVO_DEG = -0.622672
YAW_SERVO_PER_AXIS = -1.10205336
YAW_SERVO_MIN_DEG = -60.0
YAW_SERVO_MAX_DEG = 135.0


class UserStopRequested(RuntimeError):
    pass


def normalize_axial_deg(angle_deg: float) -> float:
    value = (float(angle_deg) + 90.0) % 180.0 - 90.0
    return 0.0 if abs(value) < 1e-12 else value


def axis_to_servo_deg(target_axis_deg: float) -> float:
    axis = normalize_axial_deg(target_axis_deg)
    q = YAW_REF_SERVO_DEG + YAW_SERVO_PER_AXIS * (axis - YAW_REF_AXIS_DEG)
    if not (YAW_SERVO_MIN_DEG <= q <= YAW_SERVO_MAX_DEG):
        raise ValueError(
            f"yaw target outside calibrated servo range: axis={axis:+.2f}deg -> servo={q:+.2f}deg"
        )
    return float(q)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Calibrate local 2x2 Jacobian from stage XY command-mm to top-camera/H measured TCP-mm. "
            "Yaw is held fixed; five points are measured: center, X+, X-, Y+, Y-."
        )
    )
    p.add_argument("--config", default=str(ROOT / "config" / "runtime_config.yaml"))
    p.add_argument("--x", type=float, default=-70.0, help="base stage X in mm")
    p.add_argument("--y", type=float, default=20.0, help="base stage Y in mm")
    p.add_argument("--yaw", type=float, default=-65.0, help="fixed physical closing-axis yaw in deg")
    p.add_argument("--step", type=float, default=5.0, help="stage perturbation in mm")
    p.add_argument("--settle", type=float, default=1.2, help="extra settle time after motion, seconds")
    p.add_argument("--ack-timeout", type=float, default=1.0)
    p.add_argument("--reach-timeout", type=float, default=15.0)
    p.add_argument("--min-click-span", type=float, default=20.0)
    p.add_argument("--max-fit-rmse", type=float, default=0.8,
                   help="warning threshold for TCP-mm affine-fit RMSE")
    p.add_argument("--max-condition", type=float, default=3.0,
                   help="warning threshold for Jacobian condition number")
    return p.parse_args()


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
    print(f"[TX] CALIB_MOVE X={x_mm:.3f} Y={y_mm:.3f} | {raw.hex(' ').upper()}")
    start = time.monotonic()
    ack_seen = False
    reached = False
    while True:
        now = time.monotonic()
        for frame in link.poll():
            print(f"[RX] {link.describe(frame)}")
            handle_safety_frame(frame)
            if frame.mode == Mode.MCU_BUSY:
                raise RuntimeError("MCU_BUSY: calibration requires sorting handshake STOPPED")
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
                raise RuntimeError("MCU_BUSY: CALIB_YAW requires sorting handshake STOPPED")
            if frame.mode == Mode.MCU_CALIB_YAW_ACK and abs(frame.x_mm - servo_deg) <= 0.2:
                ack_seen = True
            elif frame.mode == Mode.MCU_CALIB_YAW_REACHED and abs(frame.x_mm - servo_deg) <= 0.2:
                reached = True
        if ack_seen and reached:
            return
        if not ack_seen and now - start >= ack_timeout:
            raise TimeoutError("timeout waiting MCU_CALIB_YAW_ACK")
        if now - start >= reach_timeout:
            raise TimeoutError("timeout waiting MCU_CALIB_YAW_REACHED")
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


def measure_tcp(points: list[tuple[float, float]], transform: PixelToRobotTransform, min_span_mm: float) -> dict:
    if len(points) != 2:
        raise ValueError("need exactly two clicks")
    (u1, v1), (u2, v2) = points
    x1, y1 = transform.pixel_to_robot(u1, v1, apply_tcp_offset=False)
    x2, y2 = transform.pixel_to_robot(u2, v2, apply_tcp_offset=False)
    span = float(math.hypot(x2 - x1, y2 - y1))
    if span < min_span_mm:
        raise ValueError(f"click span {span:.1f}mm < {min_span_mm:.1f}mm")
    axis = normalize_axial_deg(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    return {
        "u1": float(u1), "v1": float(v1), "u2": float(u2), "v2": float(v2),
        "x1_mm": float(x1), "y1_mm": float(y1), "x2_mm": float(x2), "y2_mm": float(y2),
        "axis_click_span_mm": span,
        "measured_axis_deg": float(axis),
        "tcp_center_x_mm": float((x1 + x2) * 0.5),
        "tcp_center_y_mm": float((y1 + y2) * 0.5),
    }


def draw_text(image: np.ndarray, text: str, y: int, color=(0, 255, 0), scale=0.50) -> None:
    cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def fit_jacobian(rows: list[dict], base_x: float, base_y: float) -> dict:
    # Fit measured_tcp = intercept + J * (command - base) using all five points.
    A = []
    bx = []
    by = []
    for r in rows:
        dx = float(r["command_x_mm"]) - base_x
        dy = float(r["command_y_mm"]) - base_y
        A.append([1.0, dx, dy])
        bx.append(float(r["tcp_center_x_mm"]))
        by.append(float(r["tcp_center_y_mm"]))
    A = np.asarray(A, dtype=np.float64)
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    px, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    py, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
    J = np.asarray([[px[1], px[2]], [py[1], py[2]]], dtype=np.float64)
    pred = np.column_stack((A @ px, A @ py))
    obs = np.column_stack((bx, by))
    residuals = np.linalg.norm(pred - obs, axis=1)
    det = float(np.linalg.det(J))
    cond = float(np.linalg.cond(J))
    if abs(det) < 1e-6:
        raise RuntimeError(f"Jacobian is singular/near-singular: det={det:.6g}")
    J_inv = np.linalg.inv(J)

    # Also report symmetric finite-difference columns; useful for sanity checking.
    by_label = {str(r["label"]): r for r in rows}
    symmetric = None
    needed = {"X+", "X-", "Y+", "Y-"}
    if needed.issubset(by_label):
        xp = np.array([by_label["X+"]["tcp_center_x_mm"], by_label["X+"]["tcp_center_y_mm"]], float)
        xm = np.array([by_label["X-"]["tcp_center_x_mm"], by_label["X-"]["tcp_center_y_mm"]], float)
        yp = np.array([by_label["Y+"]["tcp_center_x_mm"], by_label["Y+"]["tcp_center_y_mm"]], float)
        ym = np.array([by_label["Y-"]["tcp_center_x_mm"], by_label["Y-"]["tcp_center_y_mm"]], float)
        sx = float(by_label["X+"]["command_x_mm"] - by_label["X-"]["command_x_mm"])
        sy = float(by_label["Y+"]["command_y_mm"] - by_label["Y-"]["command_y_mm"])
        J_fd = np.column_stack(((xp - xm) / sx, (yp - ym) / sy))
        symmetric = J_fd.tolist()

    return {
        "matrix_command_to_visual_tcp": J.tolist(),
        "inverse_visual_tcp_to_command": J_inv.tolist(),
        "determinant": det,
        "condition_number": cond,
        "fit_rmse_mm": float(math.sqrt(np.mean(np.square(residuals)))),
        "fit_max_residual_mm": float(np.max(residuals)),
        "fit_residuals_mm": [float(v) for v in residuals],
        "symmetric_finite_difference_matrix": symmetric,
        "intercept_tcp_at_base_mm": [float(px[0]), float(py[0])],
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.step <= 0 or args.min_click_span <= 0 or args.settle < 0:
        raise ValueError("--step and --min-click-span must be >0; --settle must be >=0")

    cfg = load_runtime_config(ROOT, args.config)
    x_lim = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["x"])
    y_lim = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["y"])
    poses = [
        ("CENTER", args.x, args.y),
        ("X+", args.x + args.step, args.y),
        ("X-", args.x - args.step, args.y),
        ("Y+", args.x, args.y + args.step),
        ("Y-", args.x, args.y - args.step),
    ]
    for label, x, y in poses:
        if not (x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1]):
            raise ValueError(f"{label} pose outside workspace: ({x:.2f},{y:.2f})")

    target_axis = normalize_axial_deg(args.yaw)
    servo_cmd = axis_to_servo_deg(target_axis)
    mapping_path = Path(cfg["calibration"]["pixel_to_robot"])
    transform = PixelToRobotTransform.load(mapping_path, camera_config=cfg["camera"]["top"])
    output_dir = mapping_path.parent
    csv_path = output_dir / "xy_command_tcp_jacobian_samples.csv"
    report_path = output_dir / "xy_command_tcp_jacobian_report.json"

    TopCamera.configure(cfg["camera"]["top"], calibration_file=cfg["calibration"]["top_intrinsics"])
    camera = TopCamera()
    link = STM32Link(device=cfg["stm32"]["device"], baudrate=cfg["stm32"]["baud"])
    clicks = ClickUI()
    rows: list[dict] = []

    print("=" * 92)
    print("XY COMMAND -> VISUAL TCP LOCAL JACOBIAN CALIBRATION")
    print("=" * 92)
    print(f"Base XY=({args.x:+.2f},{args.y:+.2f})mm  step={args.step:.2f}mm")
    print(f"Fixed physical closing-axis yaw={target_axis:+.2f}deg -> servo={servo_cmd:+.3f}deg")
    print("At each pose click the SAME two symmetric finger features used in TCP calibration.")
    print("ENTER save | C clear/re-click | Q/ESC abort")
    print("=" * 92)

    try:
        link.open()
        camera.open()
        cv2.namedWindow("XY Command-TCP Jacobian", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("XY Command-TCP Jacobian", clicks.mouse_callback)

        # Set yaw once and keep it fixed for all five XY moves.
        send_calib_yaw_and_wait(link, servo_cmd, ack_timeout=args.ack_timeout, reach_timeout=3.0)
        if args.settle:
            time.sleep(args.settle)

        for idx, (label, cmd_x, cmd_y) in enumerate(poses, start=1):
            print(f"\n[{idx}/5] {label}: move XY=({cmd_x:+.3f},{cmd_y:+.3f})")
            send_calib_move_and_wait(
                link, cmd_x, cmd_y,
                ack_timeout=args.ack_timeout,
                reach_timeout=args.reach_timeout,
            )
            if args.settle:
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
                        measurement = measure_tcp(clicks.points, transform, args.min_click_span)
                    except ValueError as exc:
                        problem = str(exc)

                for pidx, point in enumerate(clicks.points):
                    p = (int(round(point[0])), int(round(point[1])))
                    cv2.circle(display, p, 6, (0, 255, 255), 2)
                    cv2.putText(display, f"P{pidx+1}", (p[0]+8, p[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)
                if len(clicks.points) == 2:
                    p1 = tuple(int(round(v)) for v in clicks.points[0])
                    p2 = tuple(int(round(v)) for v in clicks.points[1])
                    cv2.line(display, p1, p2, (255,255,0), 2, cv2.LINE_AA)
                    mid = (int(round((p1[0]+p2[0])*0.5)), int(round((p1[1]+p2[1])*0.5)))
                    cv2.drawMarker(display, mid, (0,0,255), cv2.MARKER_CROSS, 18, 2)

                draw_text(display, f"[{idx}/5] {label} command XY=({cmd_x:+.2f},{cmd_y:+.2f}) mm", 28)
                draw_text(display, f"Fixed yaw={target_axis:+.1f}deg; click SAME two symmetric finger points", 54,
                          color=(0,255,255))
                draw_text(display, "ENTER save | C clear | Q/ESC abort", 80)
                if measurement:
                    draw_text(display,
                              f"TCP=({measurement['tcp_center_x_mm']:+.3f},{measurement['tcp_center_y_mm']:+.3f}) "
                              f"span={measurement['axis_click_span_mm']:.1f}mm",
                              106, color=(255,255,0))
                elif problem:
                    draw_text(display, problem, 106, color=(0,0,255))

                cv2.imshow("XY Command-TCP Jacobian", display)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord('q'), 27):
                    raise KeyboardInterrupt
                if key in (ord('c'), ord('C')):
                    clicks.clear()
                    continue
                if key in (10, 13):
                    if measurement is None:
                        print(f"[NOT SAVED] {problem or 'click two valid points first'}")
                        continue
                    row = {
                        "index": len(rows)+1,
                        "label": label,
                        "command_x_mm": float(cmd_x),
                        "command_y_mm": float(cmd_y),
                        "fixed_target_axis_deg": float(target_axis),
                        "fixed_servo_command_deg": float(servo_cmd),
                        **measurement,
                        "accepted_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    rows.append(row)
                    print(f"[SAVE] {label} TCP=({measurement['tcp_center_x_mm']:+.4f},"
                          f"{measurement['tcp_center_y_mm']:+.4f}) mm")
                    break

        fit = fit_jacobian(rows, float(args.x), float(args.y))
        quality_ok = bool(
            fit["fit_rmse_mm"] <= args.max_fit_rmse
            and fit["condition_number"] <= args.max_condition
            and abs(fit["determinant"]) > 0.05
        )
        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "method": "fixed yaw -> stage center/X+/X-/Y+/Y- -> two-click TCP midpoint -> fit measured_tcp = b + J*(command-base)",
            "base_stage_xy_mm": {"x": float(args.x), "y": float(args.y)},
            "step_mm": float(args.step),
            "fixed_target_axis_deg": float(target_axis),
            "fixed_servo_command_deg": float(servo_cmd),
            "pixel_to_robot_file": str(mapping_path.relative_to(ROOT)),
            "jacobian": fit,
            "quality_thresholds": {
                "max_fit_rmse_mm": float(args.max_fit_rmse),
                "max_condition_number": float(args.max_condition),
                "min_abs_determinant": 0.05,
            },
            "quality_ok": quality_ok,
            "samples": rows,
        }
        save_csv(csv_path, rows)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        J = np.asarray(fit["matrix_command_to_visual_tcp"], dtype=float)
        Ji = np.asarray(fit["inverse_visual_tcp_to_command"], dtype=float)
        print("\n" + "=" * 92)
        print("JACOBIAN RESULT")
        print("=" * 92)
        print("J command->visual TCP:")
        print(J)
        print("J^-1 visual TCP->command:")
        print(Ji)
        print(f"det={fit['determinant']:.6f}  cond={fit['condition_number']:.3f}")
        print(f"fit RMSE={fit['fit_rmse_mm']:.3f}mm  max residual={fit['fit_max_residual_mm']:.3f}mm")
        print(f"QUALITY_OK={quality_ok}")
        print(f"CSV  : {csv_path}")
        print(f"JSON : {report_path}")
        if quality_ok:
            print("Next: send xy_command_tcp_jacobian_report.json back; use J^-1 in yaw-TCP compensation.")
        else:
            print("Do not use this J yet; repeat with identical click features and stable camera/mechanics.")
        return 0

    except UserStopRequested as exc:
        print(f"[STOP] {exc}")
        return 3
    except KeyboardInterrupt:
        print("\n[ABORT] user aborted")
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
