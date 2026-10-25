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


class UserStopRequested(RuntimeError):
    pass


def normalize_axial_deg(angle_deg: float) -> float:
    value = (float(angle_deg) + 90.0) % 180.0 - 90.0
    if abs(value) < 1e-12:
        return 0.0
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Yaw mechanical calibration: move XY to a safe point, directly command the "
            "Yaw servo, click two points along the gripper closing axis, and fit the "
            "servo-command / physical-axis relationship."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "runtime_config.yaml"),
        help="runtime_config.yaml path",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="safe calibration X in mm; default is workspace midpoint",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=None,
        help="safe calibration Y in mm; default is workspace midpoint",
    )
    parser.add_argument(
        "--initial-yaw",
        type=float,
        default=0.0,
        help="initial direct servo command in deg (default: 0)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
        help="initial yaw adjustment step in deg (default: 5)",
    )
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    parser.add_argument("--reach-timeout", type=float, default=3.0)
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
    raw = link.send_frame(
        SortingFrame(mode=Mode.CALIB_MOVE, x_mm=float(x_mm), y_mm=float(y_mm))
    )
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
                raise RuntimeError("MCU_BUSY: calibration requires the handshake to stay STOPPED")
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
    if not (-135.0 <= servo_deg <= 135.0):
        raise ValueError(f"Yaw servo command out of range: {servo_deg:.2f} deg")

    drain_before_command(link)
    raw = link.send_frame(make_calib_yaw(float(servo_deg)))
    print(f"[TX] CALIB_YAW servo={servo_deg:.2f}deg | {raw.hex(' ').upper()}")

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
            if frame.mode == Mode.MCU_CALIB_YAW_ACK:
                if abs(frame.x_mm - servo_deg) <= 0.2:
                    ack_seen = True
            elif frame.mode == Mode.MCU_CALIB_YAW_REACHED:
                if abs(frame.x_mm - servo_deg) <= 0.2:
                    reached = True

        if ack_seen and reached:
            return
        if not ack_seen and now - start >= ack_timeout:
            raise TimeoutError(f"timeout waiting MCU_CALIB_YAW_ACK for {servo_deg:.2f}deg")
        if now - start >= reach_timeout:
            raise TimeoutError(f"timeout waiting MCU_CALIB_YAW_REACHED for {servo_deg:.2f}deg")
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


def draw_text(
    image: np.ndarray,
    text: str,
    y: int,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    scale: float = 0.54,
) -> None:
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


def measure_closing_axis_robot_deg(
    points: list[tuple[float, float]],
    transform: PixelToRobotTransform,
) -> tuple[float, dict]:
    if len(points) != 2:
        raise ValueError("need exactly two image points")

    (u1, v1), (u2, v2) = points
    x1, y1 = transform.pixel_to_robot(u1, v1, apply_tcp_offset=False)
    x2, y2 = transform.pixel_to_robot(u2, v2, apply_tcp_offset=False)
    length_mm = float(math.hypot(x2 - x1, y2 - y1))
    if length_mm < 5.0:
        raise ValueError(
            f"two clicks are too close ({length_mm:.2f} mm); choose two farther points along the closing axis"
        )

    theta = normalize_axial_deg(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    details = {
        "u1": float(u1),
        "v1": float(v1),
        "u2": float(u2),
        "v2": float(v2),
        "x1_mm": float(x1),
        "y1_mm": float(y1),
        "x2_mm": float(x2),
        "y2_mm": float(y2),
        "axis_click_span_mm": length_mm,
    }
    return float(theta), details


def fit_yaw_mapping(samples: list[dict]) -> dict:
    if len(samples) < 3:
        raise ValueError("at least 3 samples are required for yaw fit; 5~7 are recommended")

    ordered = sorted(samples, key=lambda r: float(r["servo_command_deg"]))
    q = np.asarray([r["servo_command_deg"] for r in ordered], dtype=np.float64)
    theta_axial = np.asarray([r["closing_axis_robot_deg"] for r in ordered], dtype=np.float64)

    # The measured closing axis is 180-degree periodic.  Double the angles, unwrap,
    # then divide by two to recover a continuous axis sequence as the servo moves.
    theta_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta_axial * 2.0))) / 2.0

    A = np.column_stack([np.ones_like(theta_unwrapped), theta_unwrapped])
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    intercept = float(coef[0])
    k = float(coef[1])

    if not np.isfinite(k) or abs(k) < 1e-6:
        raise RuntimeError("fitted yaw ratio is invalid")

    q_fit = intercept + k * theta_unwrapped
    q_err = q - q_fit
    theta_err = q_err / k

    # Use the sample closest to servo 0 as a human-friendly reference, but put it
    # exactly on the fitted line so q0/k/theta0 remain internally consistent.
    ref_idx = int(np.argmin(np.abs(q)))
    theta_ref_unwrapped = float(theta_unwrapped[ref_idx])
    theta_ref_axial = normalize_axial_deg(theta_ref_unwrapped)
    q_ref_fit = float(intercept + k * theta_ref_unwrapped)

    fit_samples = []
    for row, theta_u, qf, qe, te in zip(ordered, theta_unwrapped, q_fit, q_err, theta_err):
        item = dict(row)
        item.update(
            {
                "closing_axis_unwrapped_deg": float(theta_u),
                "servo_fit_deg": float(qf),
                "servo_residual_deg": float(qe),
                "axis_equivalent_error_deg": float(te),
            }
        )
        fit_samples.append(item)

    return {
        "reference_closing_axis_robot_deg": float(theta_ref_axial),
        "reference_servo_command_deg": float(q_ref_fit),
        "servo_deg_per_gripper_deg": float(k),
        "fit_intercept_deg": float(intercept),
        "servo_rmse_deg": float(np.sqrt(np.mean(q_err ** 2))),
        "servo_max_abs_residual_deg": float(np.max(np.abs(q_err))),
        "axis_rmse_deg": float(np.sqrt(np.mean(theta_err ** 2))),
        "axis_max_abs_error_deg": float(np.max(np.abs(theta_err))),
        "tested_servo_min_deg": float(np.min(q)),
        "tested_servo_max_deg": float(np.max(q)),
        "sample_count": int(len(samples)),
        "samples": fit_samples,
    }


def save_samples_csv(path: Path, samples: list[dict]) -> None:
    if not samples:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(samples[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def save_report(path: Path, fit: dict, *, xy: tuple[float, float], mapping_path: Path) -> None:
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": (
            "direct servo yaw command -> top-camera two-click measurement of gripper closing axis "
            "-> axial unwrap -> least-squares q = a + k*theta"
        ),
        "calibration_xy_mm": {"x": float(xy[0]), "y": float(xy[1])},
        "pixel_to_robot_file": str(mapping_path.relative_to(ROOT)),
        "important_note": (
            "MCU_CALIB_YAW_REACHED is time-based in this firmware because the yaw servo is not "
            "periodically read back. The physical closing-axis angle is measured from the top camera."
        ),
        **fit,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be > 0")

    cfg = load_runtime_config(ROOT, args.config)
    x_lim = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["x"])
    y_lim = tuple(float(v) for v in cfg["mechanics"]["workspace_mm"]["y"])

    x_cal = float(args.x) if args.x is not None else (x_lim[0] + x_lim[1]) / 2.0
    y_cal = float(args.y) if args.y is not None else (y_lim[0] + y_lim[1]) / 2.0
    if not (x_lim[0] <= x_cal <= x_lim[1] and y_lim[0] <= y_cal <= y_lim[1]):
        raise ValueError(f"calibration XY out of workspace: ({x_cal}, {y_cal})")

    current_yaw = float(args.initial_yaw)
    if not (-135.0 <= current_yaw <= 135.0):
        raise ValueError("--initial-yaw must be within [-135, 135]")

    mapping_path = Path(cfg["calibration"]["pixel_to_robot"])
    transform = PixelToRobotTransform.load(mapping_path)

    output_dir = mapping_path.parent
    samples_path = output_dir / "yaw_calibration_samples.csv"
    report_path = output_dir / "yaw_calibration_report.json"

    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()
    link = STM32Link(device=cfg["stm32"]["device"], baudrate=cfg["stm32"]["baud"])

    clicks = ClickUI()
    samples: list[dict] = []
    step_deg = float(args.step)
    measured_theta: float | None = None
    measured_details: dict | None = None

    print("=" * 86)
    print("YAW CALIBRATION: DIRECT SERVO COMMAND -> TWO-CLICK PHYSICAL CLOSING AXIS")
    print("=" * 86)
    print("下位机必须使用本补丁，并保持自动分拣 STOPPED。")
    print(f"XY calibration position: X={x_cal:.2f} mm, Y={y_cal:.2f} mm")
    print("请确保此XY位置下：夹爪抬高、不接触物体、Yaw全范围转动不会碰撞/拉线。")
    print("两个点击点必须沿『夹爪闭合轴』，即两夹指相互靠近的方向；点的先后顺序无所谓。")
    print()
    print("A/D : servo yaw -/+ step and send")
    print("1/2/3/4/5 : step = 0.5 / 1 / 2 / 5 / 10 deg")
    print("M   : terminal exact servo angle")
    print("SPACE: resend current servo angle")
    print("LEFT CLICK x2: measure physical closing axis")
    print("ENTER: save current (servo command, measured axis) sample")
    print("C: clear current two clicks   U: undo last sample")
    print("F: fit and save report        Q/ESC: finish")
    print("建议至少 5 个点，覆盖例如 -60,-30,0,+30,+60deg；确认安全后再扩大范围。")
    print("=" * 86)

    try:
        link.open()
        camera.open()

        print("[1/2] move XY to safe yaw-calibration position...")
        send_calib_move_and_wait(
            link,
            x_cal,
            y_cal,
            ack_timeout=args.ack_timeout,
            reach_timeout=max(args.reach_timeout, 15.0),
        )

        print(f"[2/2] move yaw servo to initial {current_yaw:.2f}deg...")
        send_calib_yaw_and_wait(
            link,
            current_yaw,
            ack_timeout=args.ack_timeout,
            reach_timeout=args.reach_timeout,
        )

        cv2.namedWindow("Yaw Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Yaw Calibration", clicks.mouse_callback)

        while True:
            frame = camera.read()
            transform.check_frame(frame)
            display = frame.copy()

            # Recalculate from the latest two clicks if available.
            if len(clicks.points) == 2:
                try:
                    measured_theta, measured_details = measure_closing_axis_robot_deg(
                        clicks.points, transform
                    )
                except ValueError as exc:
                    measured_theta = None
                    measured_details = None
                    draw_text(display, str(exc), 112, color=(0, 0, 255), scale=0.48)
            else:
                measured_theta = None
                measured_details = None

            for i, point in enumerate(clicks.points):
                p = (int(round(point[0])), int(round(point[1])))
                cv2.circle(display, p, 6, (0, 255, 255), 2)
                cv2.putText(
                    display,
                    f"P{i+1}",
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
                f"servo_cmd={current_yaw:+.1f}deg | step={step_deg:.1f}deg | saved={len(samples)}",
                30,
                color=(0, 255, 255),
            )
            if measured_theta is None:
                draw_text(
                    display,
                    "click TWO points along gripper CLOSING AXIS",
                    58,
                    color=(0, 255, 255),
                )
            else:
                draw_text(
                    display,
                    f"measured closing axis = {measured_theta:+.2f}deg | ENTER save sample",
                    58,
                    color=(0, 255, 0),
                )

            if len(samples) >= 3:
                try:
                    live_fit = fit_yaw_mapping(samples)
                    draw_text(
                        display,
                        (
                            f"fit k={live_fit['servo_deg_per_gripper_deg']:+.4f} "
                            f"axis_RMSE={live_fit['axis_rmse_deg']:.2f}deg | F save report"
                        ),
                        86,
                        color=(255, 255, 0),
                        scale=0.50,
                    )
                except Exception:
                    pass

            cv2.imshow("Yaw Calibration", display)
            key = cv2.waitKey(10) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), ord("Q"), 27):
                break

            preset = {
                ord("1"): 0.5,
                ord("2"): 1.0,
                ord("3"): 2.0,
                ord("4"): 5.0,
                ord("5"): 10.0,
            }
            if key in preset:
                step_deg = preset[key]
                print(f"[STEP] {step_deg:.1f}deg")
                continue

            target: float | None = None
            if key in (ord("a"), ord("A")):
                target = current_yaw - step_deg
            elif key in (ord("d"), ord("D")):
                target = current_yaw + step_deg
            elif key == 32:  # SPACE
                target = current_yaw
            elif key in (ord("m"), ord("M")):
                text = input("输入直接Yaw舵机命令角度[-135,135] deg，回车取消：").strip()
                if text:
                    try:
                        target = float(text)
                    except ValueError:
                        print("[WARN] angle must be numeric")

            if target is not None:
                if not (-135.0 <= target <= 135.0):
                    print(f"[BLOCK] target {target:.2f}deg outside [-135,135]")
                    continue
                send_calib_yaw_and_wait(
                    link,
                    float(target),
                    ack_timeout=args.ack_timeout,
                    reach_timeout=args.reach_timeout,
                )
                current_yaw = float(target)
                clicks.clear()
                measured_theta = None
                measured_details = None
                continue

            if key in (ord("c"), ord("C")):
                clicks.clear()
                measured_theta = None
                measured_details = None
                print("[CLEAR] current measurement points cleared")
                continue

            if key in (ord("u"), ord("U")):
                if samples:
                    removed = samples.pop()
                    save_samples_csv(samples_path, samples)
                    print(
                        f"[UNDO] q={removed['servo_command_deg']:+.2f}, "
                        f"theta={removed['closing_axis_robot_deg']:+.2f}"
                    )
                continue

            if key in (10, 13):
                if measured_theta is None or measured_details is None:
                    print("[WARN] click two valid points along closing axis first")
                    continue
                if any(abs(float(s["servo_command_deg"]) - current_yaw) < 0.05 for s in samples):
                    print("[WARN] this servo command already has a saved sample; undo it first if replacing")
                    continue

                record = {
                    "index": len(samples) + 1,
                    "servo_command_deg": float(current_yaw),
                    "closing_axis_robot_deg": float(measured_theta),
                    **measured_details,
                    "accepted_at": datetime.now().isoformat(timespec="seconds"),
                }
                samples.append(record)
                save_samples_csv(samples_path, samples)
                print(
                    f"[SAVE] #{len(samples)} servo={current_yaw:+.2f}deg -> "
                    f"closing_axis={measured_theta:+.2f}deg"
                )
                clicks.clear()
                measured_theta = None
                measured_details = None
                continue

            if key in (ord("f"), ord("F")):
                if len(samples) < 3:
                    print("[WARN] need at least 3 samples; 5~7 recommended")
                    continue
                fit = fit_yaw_mapping(samples)
                save_report(report_path, fit, xy=(x_cal, y_cal), mapping_path=mapping_path)
                print("=" * 72)
                print("YAW FIT")
                print("=" * 72)
                print(
                    "Pick_Yaw_Reference_Closing_Axis_Robot_Deg = "
                    f"{fit['reference_closing_axis_robot_deg']:+.6f}f;"
                )
                print(
                    "Pick_Yaw_Reference_Servo_Command_Deg      = "
                    f"{fit['reference_servo_command_deg']:+.6f}f;"
                )
                print(
                    "Pick_Yaw_Servo_Deg_Per_Gripper_Deg        = "
                    f"{fit['servo_deg_per_gripper_deg']:+.8f}f;"
                )
                print(f"axis RMSE = {fit['axis_rmse_deg']:.3f}deg")
                print(f"axis max  = {fit['axis_max_abs_error_deg']:.3f}deg")
                print(f"report    = {report_path}")
                print("=" * 72)

    except UserStopRequested as exc:
        print(f"[STOP] {exc}")
        return 2
    except KeyboardInterrupt:
        print("[STOP] KeyboardInterrupt")
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

    if len(samples) >= 3:
        fit = fit_yaw_mapping(samples)
        save_report(report_path, fit, xy=(x_cal, y_cal), mapping_path=mapping_path)
        print()
        print("Final fitted firmware parameters:")
        print(
            f"  Pick_Yaw_Reference_Closing_Axis_Robot_Deg = "
            f"{fit['reference_closing_axis_robot_deg']:+.6f}f;"
        )
        print(
            f"  Pick_Yaw_Reference_Servo_Command_Deg      = "
            f"{fit['reference_servo_command_deg']:+.6f}f;"
        )
        print(
            f"  Pick_Yaw_Servo_Deg_Per_Gripper_Deg        = "
            f"{fit['servo_deg_per_gripper_deg']:+.8f}f;"
        )
        print(f"  axis RMSE = {fit['axis_rmse_deg']:.3f}deg")
        print(f"  axis max  = {fit['axis_max_abs_error_deg']:.3f}deg")
        print(f"  report    = {report_path}")
        print("请把 yaw_calibration_report.json 发给我，我再把三项参数写入下位机并做多角度验证。")
    else:
        print(f"[INFO] only {len(samples)} sample(s); no fit report generated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
