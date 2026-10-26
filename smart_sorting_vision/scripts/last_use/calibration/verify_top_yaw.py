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


# Must match the final firmware parameters in ita_robot.h.
# Must match the final firmware parameters in ita_robot.h.
YAW_REF_AXIS_DEG = 43.590200
YAW_REF_SERVO_DEG = -0.622672
YAW_SERVO_PER_AXIS = -1.10205336

YAW_SERVO_MIN_DEG = -60.0
YAW_SERVO_MAX_DEG = 135.0

DEFAULT_TARGETS_DEG = [75.0, 45.0, 15.0, -15.0, -45.0, -75.0]


def apply_yaw_config(cfg: dict) -> None:
    """从 YAML 读取 Yaw 映射，避免验证继续使用旧硬编码。"""
    values = cfg.get("calibration", {}).get("yaw_servo", {})
    required = (
        "reference_axis_deg", "reference_servo_deg",
        "servo_deg_per_axis_deg", "safe_command_min_deg",
        "safe_command_max_deg",
    )
    if not all(key in values for key in required):
        raise ValueError("calibration.yaw_servo 配置不完整")
    global YAW_REF_AXIS_DEG, YAW_REF_SERVO_DEG, YAW_SERVO_PER_AXIS
    global YAW_SERVO_MIN_DEG, YAW_SERVO_MAX_DEG
    YAW_REF_AXIS_DEG = float(values["reference_axis_deg"])
    YAW_REF_SERVO_DEG = float(values["reference_servo_deg"])
    YAW_SERVO_PER_AXIS = float(values["servo_deg_per_axis_deg"])
    YAW_SERVO_MIN_DEG = float(values["safe_command_min_deg"])
    YAW_SERVO_MAX_DEG = float(values["safe_command_max_deg"])


class UserStopRequested(RuntimeError):
    pass


def normalize_axial_deg(angle_deg: float) -> float:
    """Normalize an undirected axis angle into [-90, 90)."""
    value = (float(angle_deg) + 90.0) % 180.0 - 90.0
    if abs(value) < 1e-12:
        return 0.0
    return value


def axis_to_servo_deg(target_axis_deg: float, current_servo_deg: float) -> float:
    """Mirror the firmware's 180-degree-equivalent solution selection."""
    target_axis_deg = normalize_axial_deg(target_axis_deg)
    raw_delta = target_axis_deg - YAW_REF_AXIS_DEG
    candidates: list[float] = []
    for n in range(-4, 5):
        q = YAW_REF_SERVO_DEG + YAW_SERVO_PER_AXIS * (raw_delta + 180.0 * n)
        if YAW_SERVO_MIN_DEG <= q <= YAW_SERVO_MAX_DEG:
            candidates.append(float(q))
    if not candidates:
        raise ValueError(f"no calibrated servo solution for target axis {target_axis_deg:+.2f} deg")
    return min(candidates, key=lambda q: abs(q - current_servo_deg))


def parse_targets(text: str) -> list[float]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("target list cannot be empty")
    try:
        return [normalize_axial_deg(float(v)) for v in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent verification of the final XY-frame gripper closing-axis yaw mapping."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "runtime_config.yaml"))
    parser.add_argument("--x", type=float, default=-20.0, help="safe verification X in mm")
    parser.add_argument("--y", type=float, default=20.0, help="safe verification Y in mm")
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=DEFAULT_TARGETS_DEG,
        help='comma-separated physical closing-axis targets, e.g. "75,45,15,-15,-45,-75"',
    )
    parser.add_argument("--settle", type=float, default=1.5, help="extra seconds after MCU REACHED")
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    parser.add_argument("--reach-timeout", type=float, default=3.0)
    parser.add_argument("--min-click-span", type=float, default=20.0, help="minimum two-click span in mm")
    parser.add_argument("--pass-mae", type=float, default=3.0)
    parser.add_argument("--pass-rmse", type=float, default=4.0)
    parser.add_argument("--pass-max", type=float, default=7.0)
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
                raise RuntimeError("MCU_BUSY: verification requires sorting handshake STOPPED")
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


def measure_axis(
    points: list[tuple[float, float]],
    transform: PixelToRobotTransform,
    *,
    min_span_mm: float,
) -> tuple[float, dict]:
    if len(points) != 2:
        raise ValueError("need exactly two clicks")
    (u1, v1), (u2, v2) = points
    x1, y1 = transform.pixel_to_robot(u1, v1, apply_tcp_offset=False)
    x2, y2 = transform.pixel_to_robot(u2, v2, apply_tcp_offset=False)
    span = float(math.hypot(x2 - x1, y2 - y1))
    if span < min_span_mm:
        raise ValueError(
            f"click span {span:.1f} mm < {min_span_mm:.1f} mm; choose farther points"
        )
    theta = normalize_axial_deg(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    return theta, {
        "u1": u1,
        "v1": v1,
        "u2": u2,
        "v2": v2,
        "x1_mm": float(x1),
        "y1_mm": float(y1),
        "x2_mm": float(x2),
        "y2_mm": float(y2),
        "axis_click_span_mm": span,
    }


def draw_text(image: np.ndarray, text: str, y: int, color=(0, 255, 0), scale=0.52) -> None:
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


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict], args: argparse.Namespace, mapping_path: Path) -> dict:
    errors = np.asarray([float(r["axis_error_deg"]) for r in rows], dtype=np.float64)
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_abs = float(np.max(abs_errors))
    passed = mae <= args.pass_mae and rmse <= args.pass_rmse and max_abs <= args.pass_max
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "physical closing-axis target -> calibrated servo command -> top-camera two-click independent verification",
        "verification_xy_mm": {"x": float(args.x), "y": float(args.y)},
        "pixel_to_robot_file": str(mapping_path.relative_to(ROOT)),
        "firmware_yaw_parameters": {
            "reference_closing_axis_robot_deg": YAW_REF_AXIS_DEG,
            "reference_servo_command_deg": YAW_REF_SERVO_DEG,
            "servo_deg_per_gripper_deg": YAW_SERVO_PER_AXIS,
            "servo_min_deg": YAW_SERVO_MIN_DEG,
            "servo_max_deg": YAW_SERVO_MAX_DEG,
        },
        "thresholds_deg": {
            "mae": float(args.pass_mae),
            "rmse": float(args.pass_rmse),
            "max_abs": float(args.pass_max),
        },
        "mean_abs_error_deg": mae,
        "rmse_deg": rmse,
        "max_abs_error_deg": max_abs,
        "passed": bool(passed),
        "sample_count": len(rows),
        "samples": rows,
    }


def main() -> int:
    args = parse_args()
    if args.settle < 0 or args.min_click_span <= 0:
        raise ValueError("--settle must be >=0 and --min-click-span must be >0")

    cfg = load_runtime_config(ROOT, args.config)
    apply_yaw_config(cfg)
    x_lim = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["x"])
    y_lim = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["y"])
    if not (x_lim[0] <= args.x <= x_lim[1] and y_lim[0] <= args.y <= y_lim[1]):
        raise ValueError(f"verification XY out of workspace: ({args.x}, {args.y})")

    mapping_path = Path(cfg["calibration"]["pixel_to_robot"])
    transform = PixelToRobotTransform.load(mapping_path)
    output_dir = mapping_path.parent
    csv_path = output_dir / "yaw_verification_samples.csv"
    report_path = output_dir / "yaw_verification_report.json"

    TopCamera.configure(cfg["camera"]["top"], calibration_file=cfg["calibration"]["top_intrinsics"])
    camera = TopCamera()
    link = STM32Link(device=cfg["stm32"]["device"], baudrate=cfg["stm32"]["baud"])
    clicks = ClickUI()
    rows: list[dict] = []
    current_servo = 0.0

    print("=" * 86)
    print("YAW INDEPENDENT VERIFICATION")
    print("=" * 86)
    print(f"XY = ({args.x:.2f}, {args.y:.2f}) mm")
    print(f"targets = {', '.join(f'{v:+.1f}' for v in args.targets)} deg")
    print("每个目标：程序自动计算并下发舵机角 -> 到位后再等待 settle -> 沿夹爪闭合轴点两点 -> ENTER保存。")
    print(f"两点击点机械距离必须 >= {args.min_click_span:.1f} mm。")
    print("N: skip current target   C: clear clicks   Q/ESC: abort")
    print("=" * 86)

    try:
        link.open()
        camera.open()
        send_calib_move_and_wait(
            link,
            float(args.x),
            float(args.y),
            ack_timeout=args.ack_timeout,
            reach_timeout=max(args.reach_timeout, 15.0),
        )

        cv2.namedWindow("Yaw Verification", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Yaw Verification", clicks.mouse_callback)

        for idx, target_axis in enumerate(args.targets, start=1):
            target_axis = normalize_axial_deg(target_axis)
            servo_cmd = axis_to_servo_deg(target_axis, current_servo)
            print(
                f"\n[{idx}/{len(args.targets)}] target_axis={target_axis:+.2f} deg "
                f"-> servo_cmd={servo_cmd:+.3f} deg"
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
                measured = None
                details = None
                problem = None
                if len(clicks.points) == 2:
                    try:
                        measured, details = measure_axis(
                            clicks.points, transform, min_span_mm=args.min_click_span
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

                draw_text(
                    display,
                    f"[{idx}/{len(args.targets)}] target axis={target_axis:+.2f}deg | servo={servo_cmd:+.2f}deg",
                    30,
                    color=(0, 255, 255),
                )
                if measured is not None:
                    err = normalize_axial_deg(measured - target_axis)
                    draw_text(
                        display,
                        f"measured={measured:+.2f}deg | error={err:+.2f}deg | ENTER save",
                        58,
                        color=(0, 255, 0),
                    )
                    draw_text(
                        display,
                        f"click span={details['axis_click_span_mm']:.1f}mm",
                        86,
                        color=(255, 255, 0),
                    )
                elif problem:
                    draw_text(display, problem, 58, color=(0, 0, 255), scale=0.47)
                else:
                    draw_text(display, "click TWO far-apart points along CLOSING AXIS", 58, color=(0, 255, 255))

                cv2.imshow("Yaw Verification", display)
                key = cv2.waitKey(10) & 0xFF
                if key == 255:
                    continue
                if key in (ord("q"), ord("Q"), 27):
                    raise KeyboardInterrupt
                if key in (ord("c"), ord("C")):
                    clicks.clear()
                    continue
                if key in (ord("n"), ord("N")):
                    print("[SKIP] current target")
                    break
                if key in (10, 13):
                    if measured is None or details is None:
                        print("[WARN] need a valid two-click measurement first")
                        continue
                    err = normalize_axial_deg(measured - target_axis)
                    row = {
                        "index": idx,
                        "target_closing_axis_deg": float(target_axis),
                        "servo_command_deg": float(servo_cmd),
                        "measured_closing_axis_deg": float(measured),
                        "axis_error_deg": float(err),
                        **details,
                        "accepted_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    rows.append(row)
                    print(
                        f"[SAVE] target={target_axis:+.2f} measured={measured:+.2f} "
                        f"error={err:+.2f} span={details['axis_click_span_mm']:.1f}mm"
                    )
                    break

        if not rows:
            print("[WARN] no samples saved")
            return 1

        report = build_report(rows, args, mapping_path)
        save_csv(csv_path, rows)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n" + "=" * 72)
        print("YAW VERIFICATION RESULT")
        print("=" * 72)
        print(f"MAE      = {report['mean_abs_error_deg']:.3f} deg")
        print(f"RMSE     = {report['rmse_deg']:.3f} deg")
        print(f"MAX ABS  = {report['max_abs_error_deg']:.3f} deg")
        print(f"PASS     = {report['passed']}")
        print(f"CSV      = {csv_path}")
        print(f"REPORT   = {report_path}")
        print("=" * 72)
        return 0 if report["passed"] else 3

    except UserStopRequested as exc:
        print(f"[STOP] {exc}")
        return 2
    except KeyboardInterrupt:
        print("[STOP] verification aborted")
        return 130
    finally:
        try:
            if link.is_open:
                try:
                    link.send_frame(SortingFrame(mode=Mode.STOP))
                except Exception as exc:
                    print(f"[WARN] cleanup STOP failed: {exc}")
        finally:
            link.close()
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
