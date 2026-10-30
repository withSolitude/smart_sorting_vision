from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from communication.sorting_protocol import Mode, SortingFrame, make_pick_target, make_stop
from communication.stm32_link import STM32Link
from communication.claw_preopen_transaction import latch_preopen
from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.pipeline.top_pick_pipeline import TopPickPipeline
from vision.calibration.yaw_tcp_compensation import YawTcpCompensator
from vision.calibration.descent_drift_compensation import DescentDriftCompensator


class UserStopRequested(RuntimeError):
    pass


def normalize_axial_deg(angle_deg: float) -> float:
    value = (float(angle_deg) + 90.0) % 180.0 - 90.0
    return 0.0 if abs(value) < 1e-12 else value


def build_compensator(top_cfg: dict) -> YawTcpCompensator | None:
    comp = top_cfg.get("yaw_tcp_compensation", {})
    if not bool(comp.get("enabled", False)):
        return None
    offset = comp["offset_axis0_mm"]
    return YawTcpCompensator(
        reference_axis_deg=float(comp["reference_axis_deg"]),
        offset_axis0_x_mm=float(offset[0]),
        offset_axis0_y_mm=float(offset[1]),
        min_axis_deg=float(comp.get("min_axis_deg", -87.0)),
        max_axis_deg=float(comp.get("max_axis_deg", 89.0)),
        visual_to_command_matrix=comp.get(
            "visual_to_command_matrix", [[1.0, 0.0], [0.0, 1.0]]
        ),
    )


def build_descent_drift_compensator(
    top_cfg: dict,
    *,
    override_signed_drift_mm: float | None = None,
    override_relative_angle_deg: float | None = None,
) -> DescentDriftCompensator | None:
    drift_cfg = top_cfg.get("descent_drift_compensation", {})
    enabled = bool(drift_cfg.get("enabled", False)) or override_signed_drift_mm is not None
    if not enabled:
        return None

    signed_drift_mm = (
        float(override_signed_drift_mm)
        if override_signed_drift_mm is not None
        else float(drift_cfg.get("signed_drift_mm", 0.0))
    )
    relative_angle_deg = (
        float(override_relative_angle_deg)
        if override_relative_angle_deg is not None
        else float(drift_cfg.get("relative_angle_deg", 90.0))
    )

    matrix = [[1.0, 0.0], [0.0, 1.0]]
    if bool(drift_cfg.get("use_visual_to_command_matrix", True)):
        matrix = top_cfg.get("yaw_tcp_compensation", {}).get(
            "visual_to_command_matrix", matrix
        )

    return DescentDriftCompensator(
        signed_drift_mm=signed_drift_mm,
        relative_angle_deg=relative_angle_deg,
        visual_to_command_matrix=matrix,
    )


def build_pipeline(cfg: dict) -> TopPickPipeline:
    top_model_cfg = cfg["model"]["top"]
    top_vision_cfg = cfg["vision"]["top"]
    consensus = top_vision_cfg["consensus"]
    orient = top_vision_cfg.get("grasp_orientation", {})
    x_limits = cfg["mechanics"]["workspace_mm"]["x"]
    y_limits = cfg["mechanics"]["workspace_mm"]["y"]
    comp = build_compensator(top_vision_cfg)
    preopen = orient.get("dynamic_preopen", {}) or {}

    return TopPickPipeline(
        model_path=top_model_cfg["path"],
        calibration_path=cfg["calibration"]["pixel_to_robot"],
        camera_config=cfg["camera"]["top"],
        conf=top_model_cfg["conf"],
        imgsz=top_model_cfg["imgsz"],
        device=top_model_cfg["device"],
        obstacle_margin=top_vision_cfg["obstacle_margin_px"],
        border_margin=top_vision_cfg["border_margin_px"],
        min_safe_area=top_vision_cfg["min_safe_area_px"],
        min_clearance=top_vision_cfg["min_clearance_px"],
        consensus_window=consensus["window"],
        consensus_required=consensus["required"],
        max_track_center_shift_px=consensus["max_track_center_shift_px"],
        max_grasp_spread_mm=consensus["max_grasp_spread_mm"],
        reset_after_misses=consensus["reset_after_misses"],
        orientation_enabled=orient.get("enabled", True),
        grip_long_sides=orient.get("grip_long_sides", True),
        min_orientation_aspect_ratio=orient.get("min_aspect_ratio", 1.15),
        default_grasp_axis_robot_deg=orient.get("default_grasp_axis_robot_deg", -85.0),
        yaw_alignment_offset_deg=orient.get("yaw_alignment_offset_deg", 0.0),
        max_yaw_spread_deg=orient.get("max_yaw_spread_deg", 12.0),
        grasp_width_slack_ratio=orient.get("width_slack_ratio", 0.15),
        grasp_support_tolerance_px=orient.get("support_tolerance_px", 3.0),
        grasp_min_contact_span_px=orient.get("min_contact_span_px", 6.0),
        grasp_circle_circularity_threshold=orient.get("circle_circularity_threshold", 0.985),
        grasp_circle_min_vertices=orient.get("circle_min_vertices", 8),
        grasp_min_opening_mm=orient.get("min_opening_mm", None),
        grasp_max_opening_mm=orient.get("max_opening_mm", None),
        dynamic_preopen_enabled=preopen.get("enabled", True),
        preopen_margin_mm=preopen.get("margin_mm", 6.0),
        preopen_command_min_mm=preopen.get("min_command_mm", None),
        preopen_command_max_mm=preopen.get("max_command_mm", None),
        preopen_profile_kind=preopen.get("profile_kind", ""),
        preopen_configured=preopen.get("configured", False),
        preopen_closed_servo_deg=preopen.get("closed_servo_deg", -15.0),
        preopen_standby_servo_deg=preopen.get("standby_servo_deg", -95.0),
        x_limits_mm=x_limits,
        y_limits_mm=y_limits,
        yaw_tcp_compensator=comp,
    )


def draw_axis(image, center, angle_deg, length_px, color, thickness=2):
    theta = math.radians(float(angle_deg))
    dx = 0.5 * float(length_px) * math.cos(theta)
    dy = 0.5 * float(length_px) * math.sin(theta)
    cx, cy = center
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def poll_control_frames(link: STM32Link, state: dict) -> None:
    for frame in link.poll():
        print(f"[RX] {link.describe(frame)}")
        if frame.mode == Mode.MCU_START_WORK:
            state["started"] = True
            state["last_event"] = "MCU_START_WORK"
        elif frame.mode in (Mode.MCU_STOP_WORK, Mode.MCU_STOP_ACK):
            state["started"] = False
            state["last_event"] = frame.mode.name
        elif frame.mode == Mode.MCU_ERROR:
            state["last_error"] = int(frame.box_id)
            state["last_event"] = f"MCU_ERROR({frame.box_id})"


def force_stopped(link: STM32Link, timeout_s: float = 1.5) -> None:
    try:
        link.send_frame(make_stop())
    except Exception as exc:
        print(f"[WARN] startup STOP send failed: {exc}")
        return
    end = time.monotonic() + float(timeout_s)
    while time.monotonic() < end:
        for frame in link.poll():
            print(f"[RX] {link.describe(frame)}")
            if frame.mode in (Mode.MCU_STOP_ACK, Mode.MCU_STOP_WORK):
                return
        time.sleep(0.003)


def stop_and_wait(link: STM32Link, timeout_s: float = 2.0) -> None:
    try:
        raw = link.send_frame(make_stop())
        print(f"[TX] STOP | {raw.hex(' ').upper()}")
    except Exception as exc:
        print(f"[WARN] STOP send failed: {exc}")
        return
    end = time.monotonic() + float(timeout_s)
    while time.monotonic() < end:
        for frame in link.poll():
            print(f"[RX] {link.describe(frame)}")
            if frame.mode in (Mode.MCU_STOP_ACK, Mode.MCU_STOP_WORK):
                return
        time.sleep(0.003)


def execute_pick_first_stage(
    link: STM32Link,
    target: dict,
    *,
    ack_timeout_s: float,
    ready_timeout_s: float,
) -> dict:
    x = float(target["x_mm"])
    y = float(target["y_mm"])
    yaw = normalize_axial_deg(float(target["yaw_deg"]))

    result = {
        "ack": False,
        "recognition_ready": False,
        "mcu_error": None,
        "busy": False,
        "elapsed_s": None,
    }

    if target.get("preopen_mm") is None:
        raise ValueError("No measured preopen command; real pick blocked")
    result["preopen_mm"] = latch_preopen(link, target["preopen_mm"], ack_timeout_s)
    result["preopen_ack"] = True
    frame = make_pick_target(x, y, yaw)
    raw = link.send_frame(frame)
    print(
        f"[TX] PICK_TARGET X={x:.2f} Y={y:.2f} Yaw={yaw:+.2f} | "
        f"{raw.hex(' ').upper()}"
    )

    start = time.monotonic()
    ack_deadline = start + float(ack_timeout_s)
    ready_deadline = start + float(ready_timeout_s)

    while time.monotonic() < ready_deadline:
        now = time.monotonic()
        for rx in link.poll():
            print(f"[RX] {link.describe(rx)}")

            if rx.mode == Mode.MCU_STOP_WORK:
                raise UserStopRequested("STM32 physical STOP_WORK received during pick")

            if rx.mode == Mode.MCU_ERROR:
                result["mcu_error"] = int(rx.box_id)
                result["elapsed_s"] = now - start
                return result

            if rx.mode == Mode.MCU_BUSY:
                result["busy"] = True
                result["elapsed_s"] = now - start
                return result

            if rx.mode == Mode.MCU_PICK_TARGET_ACK:
                # 允许协议0.1mm/1deg量化误差。
                if (
                    abs(float(rx.x_mm) - x) <= 0.2
                    and abs(float(rx.y_mm) - y) <= 0.2
                    and abs(normalize_axial_deg(float(rx.yaw_deg) - yaw)) <= 1.1
                ):
                    result["ack"] = True
                    print("[OK] PICK_TARGET accepted by STM32")

            if rx.mode == Mode.MCU_RECOGNITION_READY:
                result["recognition_ready"] = True
                result["elapsed_s"] = now - start
                print("[OK] first-stage pick completed: object should now be on side-recognition tray")
                return result

        if not result["ack"] and now >= ack_deadline:
            result["elapsed_s"] = now - start
            return result

        time.sleep(0.003)

    result["elapsed_s"] = time.monotonic() - start
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "ONE-object real pick validation. It uses the production PICK_TARGET path: "
            "XY+Yaw -> pre-close -> TOF descent -> close -> lift -> move to side tray -> release -> return."
        )
    )
    p.add_argument("--config", default=str(ROOT / "config" / "runtime_config.yaml"))
    p.add_argument("--ack-timeout", type=float, default=2.0)
    p.add_argument("--ready-timeout", type=float, default=35.0)
    p.add_argument("--no-start-reset", action="store_true", help="do not send STOP once at startup")
    p.add_argument(
        "--descent-drift-mm",
        type=float,
        default=None,
        help=(
            "override signed descent drift mm for this run. Positive means the actual "
            "TCP drifts toward closing_axis + relative_angle; negative means opposite."
        ),
    )
    p.add_argument(
        "--descent-relative-angle",
        type=float,
        default=None,
        help="override drift direction relative to closing axis; default config is +90 deg",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(ROOT, Path(args.config))

    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()
    pipeline = build_pipeline(cfg)
    drift_compensator = build_descent_drift_compensator(
        cfg["vision"]["top"],
        override_signed_drift_mm=args.descent_drift_mm,
        override_relative_angle_deg=args.descent_relative_angle,
    )
    link = STM32Link(device=cfg["stm32"]["device"], baudrate=cfg["stm32"]["baud"])

    out_dir = ROOT / "data" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "single_object_pick_report.json"

    state = {
        "started": False,
        "last_event": "startup",
        "last_error": None,
    }
    frozen: dict | None = None
    frozen_frame = None
    report: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "top vision -> production PICK_TARGET -> wait PICK_ACK -> wait RECOGNITION_READY -> STOP",
        "important_note": (
            "PICK_TARGET executes the lower-controller production first-stage sequence, including Z descent, "
            "claw closing, lift, transfer to the side-recognition tray, release and return. No SORT_RESULT is sent."
        ),
        "target": None,
        "descent_drift_compensation": (
            None
            if drift_compensator is None
            else {
                "signed_drift_mm": drift_compensator.signed_drift_mm,
                "relative_angle_deg": drift_compensator.relative_angle_deg,
                "visual_to_command_matrix": drift_compensator.visual_to_command_matrix.tolist(),
            }
        ),
        "result": None,
    }

    window = "Single Object REAL Pick - press physical START, P=execute / R=reacquire / S=STOP / Q=quit"

    try:
        camera.open()
        link.open()
        if not args.no_start_reset:
            force_stopped(link)

        print("\n[SAFETY] Put exactly ONE easy object in the top tray.")
        print("[SAFETY] Keep the side-recognition tray clear and ready to receive the object.")
        print("[SAFETY] P executes REAL Z descent + claw close + lift + transfer + release.")
        print("[START] After this script is running, press the STM32 physical START button.")
        print("[KEY] P=execute frozen target, R=reacquire, S=STOP, Q=quit\n")

        while True:
            poll_control_frames(link, state)

            frame = camera.read()
            show = frame.copy()

            if frozen is None:
                result = pipeline.process(frame)
                status = (
                    f"START={'YES' if state['started'] else 'NO'} | objects={result.object_count} "
                    f"candidates={result.candidate_count} votes={result.votes}/{result.required_hits} "
                    f"reason={result.reason}"
                )
                cv2.putText(show, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

                if result.object_count != 1:
                    cv2.putText(
                        show,
                        "REAL-PICK test requires exactly ONE segmented object",
                        (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.56,
                        (0, 0, 255),
                        2,
                    )
                elif (
                    result.ready
                    and result.grasp_pixel is not None
                    and result.x_mm is not None
                    and result.y_mm is not None
                    and result.yaw_deg is not None
                    and result.candidates
                ):
                    best = max(result.candidates, key=lambda c: c.score)
                    gx, gy = map(int, result.grasp_pixel)
                    cx, cy = map(float, best.center_pixel)
                    yaw_value = normalize_axial_deg(float(result.yaw_deg))
                    base_x = float(result.x_mm)
                    base_y = float(result.y_mm)
                    drift_info = None
                    pick_x = base_x
                    pick_y = base_y
                    if drift_compensator is not None:
                        drift_info = drift_compensator.apply(base_x, base_y, yaw_value)
                        pick_x = float(drift_info.x_mm)
                        pick_y = float(drift_info.y_mm)

                    frozen = {
                        "x_mm": pick_x,
                        "y_mm": pick_y,
                        "alignment_x_mm": base_x,
                        "alignment_y_mm": base_y,
                        "yaw_deg": yaw_value,
                        "preopen_mm": getattr(result, "preopen_mm", None),
                        "descent_precomp_dx_mm": (0.0 if drift_info is None else float(drift_info.command_delta_x_mm)),
                        "descent_precomp_dy_mm": (0.0 if drift_info is None else float(drift_info.command_delta_y_mm)),
                        "predicted_descent_drift_x_mm": (0.0 if drift_info is None else float(drift_info.predicted_drift_x_mm)),
                        "predicted_descent_drift_y_mm": (0.0 if drift_info is None else float(drift_info.predicted_drift_y_mm)),
                        "grasp_pixel": [gx, gy],
                        "center_pixel": [cx, cy],
                        "class_name": result.class_name,
                        "confidence": float(result.confidence),
                        "orientation_mode": str(getattr(best, "orientation_mode", "unknown")),
                        "contact_type": str(getattr(best, "contact_type", "none")),
                        "contact_quality": float(getattr(best, "contact_quality", 0.0)),
                        "grasp_width_mm": (
                            None
                            if getattr(best, "grasp_width_mm", None) is None
                            else float(best.grasp_width_mm)
                        ),
                        "grasp_width_px": float(getattr(best, "grasp_width_px", 0.0)),
                        "closing_axis_image_deg": float(getattr(best, "closing_axis_image_deg", 0.0)),
                        "support_min_point_px": getattr(best, "support_min_point_px", None),
                        "support_max_point_px": getattr(best, "support_max_point_px", None),
                    }
                    frozen_frame = frame.copy()
                    print(
                        "[FROZEN] "
                        f"ALIGN=({frozen['alignment_x_mm']:.2f},{frozen['alignment_y_mm']:.2f}) "
                        f"PICK=({frozen['x_mm']:.2f},{frozen['y_mm']:.2f}) "
                        f"pre=({frozen['descent_precomp_dx_mm']:+.2f},{frozen['descent_precomp_dy_mm']:+.2f}) "
                        f"Yaw={frozen['yaw_deg']:+.2f} mode={frozen['orientation_mode']} "
                        f"contact={frozen['contact_type']} width={frozen['grasp_width_mm']}mm"
                    )
            else:
                if frozen_frame is not None:
                    h, w = show.shape[:2]
                    thumb_w = min(290, w // 3)
                    scale = thumb_w / frozen_frame.shape[1]
                    thumb = cv2.resize(
                        frozen_frame,
                        (thumb_w, max(1, int(frozen_frame.shape[0] * scale))),
                    )
                    th, tw = thumb.shape[:2]
                    show[0:th, w - tw:w] = thumb

                gx, gy = frozen["grasp_pixel"]
                cx, cy = frozen["center_pixel"]
                cv2.drawMarker(show, (int(round(cx)), int(round(cy))), (255, 0, 0), cv2.MARKER_CROSS, 24, 2)
                cv2.drawMarker(show, (gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                draw_axis(
                    show,
                    (cx, cy),
                    float(frozen["closing_axis_image_deg"]),
                    120.0,
                    (0, 255, 255),
                )
                pmin = frozen.get("support_min_point_px")
                pmax = frozen.get("support_max_point_px")
                if pmin is not None:
                    cv2.drawMarker(show, (int(round(pmin[0])), int(round(pmin[1]))), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
                if pmax is not None:
                    cv2.drawMarker(show, (int(round(pmax[0])), int(round(pmax[1]))), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)

                lines = [
                    f"START={'YES' if state['started'] else 'NO'} | REAL PICK ARMED",
                    f"ALIGN X={frozen['alignment_x_mm']:.2f} Y={frozen['alignment_y_mm']:.2f} Yaw={frozen['yaw_deg']:+.2f}",
                    f"DESC pre dXY=({frozen['descent_precomp_dx_mm']:+.2f},{frozen['descent_precomp_dy_mm']:+.2f}) -> PICK=({frozen['x_mm']:.2f},{frozen['y_mm']:.2f})",
                    f"mode={frozen['orientation_mode']} contact={frozen['contact_type']} Q={frozen['contact_quality']:.2f}",
                    f"grasp width={frozen['grasp_width_mm']}mm ({frozen['grasp_width_px']:.1f}px)",
                    "P: REAL PICK to side tray | R: reacquire | S: STOP | Q: quit",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(show, text, (12, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.imshow(window, show)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                print("[USER] STOP")
                stop_and_wait(link)
                state["started"] = False
                continue

            if key == ord("r"):
                pipeline.reset()
                frozen = None
                frozen_frame = None
                continue

            if key == ord("p") and frozen is not None:
                if not state["started"]:
                    print("[BLOCK] STM32 START has not been observed. Press the physical START button first.")
                    continue

                print("\n[EXECUTE] REAL first-stage pick starts NOW.")
                print("           Do not touch the mechanism. Use the physical STOP button if abnormal.\n")

                frozen["sent_at"] = datetime.now().isoformat(timespec="seconds")
                report["target"] = dict(frozen)
                pick_result = execute_pick_first_stage(
                    link,
                    frozen,
                    ack_timeout_s=args.ack_timeout,
                    ready_timeout_s=args.ready_timeout,
                )
                pick_result["finished_at"] = datetime.now().isoformat(timespec="seconds")
                report["result"] = pick_result
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

                if pick_result.get("recognition_ready"):
                    # First-stage success means object is already on side tray and gripper returned.
                    # We deliberately do NOT send SORT_RESULT in this validation.
                    stop_and_wait(link)
                    print(f"[PASS] object reached side-recognition tray. Report: {report_path}")
                    return 0

                if pick_result.get("busy"):
                    print("[FAIL] MCU_BUSY. The usual cause is that physical START was not active / handshake not WAIT_PICK_TARGET.")
                elif pick_result.get("mcu_error") is not None:
                    print(f"[FAIL] MCU_ERROR code={pick_result['mcu_error']}")
                elif not pick_result.get("ack"):
                    print("[FAIL] PICK_ACK timeout")
                else:
                    print("[FAIL] RECOGNITION_READY timeout; send STOP and inspect lower-controller grasp/TOF state.")

                stop_and_wait(link)
                return 3

    except UserStopRequested as exc:
        print(f"[STOP] {exc}")
        report["result"] = {"physical_stop": True, "message": str(exc)}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2
    except (RuntimeError, ValueError, TimeoutError) as exc:
        print(f"[FAIL] {exc}")
        report["result"] = {"failed": True, "message": str(exc)}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 3
    finally:
        try:
            if link.is_open:
                link.send_frame(make_stop())
                print("[STOP] cleanup STOP sent; confirm safe return on the machine")
        except Exception as exc:
            print(f"[WARN] cleanup STOP failed: {exc}; use physical STOP")
        try:
            link.close()
        except Exception:
            pass
        try:
            camera.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
