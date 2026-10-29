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

from communication.sorting_protocol import Mode, SortingFrame, make_calib_yaw
from communication.stm32_link import STM32Link
from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.pipeline.top_pick_pipeline import TopPickPipeline
from vision.calibration.yaw_tcp_compensation import YawTcpCompensator
from vision.grasp.orientation_planner import polygon_long_axis_image_deg


# Must match the final firmware Yaw calibration in ita_robot.h.
YAW_REF_AXIS_DEG = 42.105382
YAW_REF_SERVO_DEG = -1.338941
YAW_SERVO_PER_AXIS = -1.055363
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
            f"Yaw unique branch outside calibrated servo range: axis={axis:+.2f} deg, q={q:+.2f} deg"
        )
    return float(q)


def handle_safety_frame(frame: SortingFrame) -> None:
    if frame.mode == Mode.MCU_ERROR:
        raise RuntimeError(
            f"STM32 MCU_ERROR: code={frame.box_id}, x={frame.x_mm:.1f}, y={frame.y_mm:.1f}"
        )
    if frame.mode == Mode.MCU_STOP_WORK:
        raise UserStopRequested("STM32 physical STOP_WORK received")


def drain(link: STM32Link) -> None:
    for frame in link.poll():
        print(f"[OLD RX] {link.describe(frame)}")
        handle_safety_frame(frame)


def ensure_stopped(link: STM32Link, timeout_s: float = 1.5) -> None:
    """Best-effort force the sorting handshake into STOPPED before CALIB_* commands."""
    drain(link)
    try:
        raw = link.send_frame(SortingFrame(mode=Mode.STOP))
        print(f"[TX] STOP | {raw.hex(' ').upper()}")
    except Exception as exc:
        print(f"[WARN] STOP send failed: {exc}")
        return

    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        for frame in link.poll():
            print(f"[RX] {link.describe(frame)}")
            if frame.mode == Mode.MCU_ERROR:
                handle_safety_frame(frame)
            if frame.mode in (Mode.MCU_STOP_ACK, Mode.MCU_STOP_WORK):
                time.sleep(0.15)
                return
        time.sleep(0.003)
    print("[WARN] STOP_ACK not observed; CALIB command will still verify MCU_BUSY explicitly.")


def send_calib_move_and_wait(
    link: STM32Link,
    x_mm: float,
    y_mm: float,
    *,
    ack_timeout: float,
    reach_timeout: float,
) -> None:
    drain(link)
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
                raise RuntimeError("MCU_BUSY: run this validation only while sorting is STOPPED")
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
    drain(link)
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


def build_compensator(top_cfg: dict) -> YawTcpCompensator:
    comp = top_cfg.get("yaw_tcp_compensation", {})
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
        closing_bias_mm=float(comp.get("closing_bias_mm", 0.0)),
        longitudinal_bias_mm=float(comp.get("longitudinal_bias_mm", 0.0)),
    )


def build_pipeline(cfg: dict, comp: YawTcpCompensator) -> TopPickPipeline:
    top_model_cfg = cfg["model"]["top"]
    top_vision_cfg = cfg["vision"]["top"]
    consensus = top_vision_cfg["consensus"]
    orient = top_vision_cfg.get("grasp_orientation", {})
    x_limits = cfg["mechanics"]["workspace_mm"]["x"]
    y_limits = cfg["mechanics"]["workspace_mm"]["y"]
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
        min_orientation_aspect_ratio=orient.get("min_aspect_ratio", 1.20),
        default_grasp_axis_robot_deg=orient.get("default_grasp_axis_robot_deg", 0.0),
        yaw_alignment_offset_deg=orient.get("yaw_alignment_offset_deg", 0.0),
        max_yaw_spread_deg=orient.get("max_yaw_spread_deg", 12.0),
        grasp_width_slack_ratio=orient.get("width_slack_ratio", 0.15),
        grasp_support_tolerance_px=orient.get("support_tolerance_px", 3.0),
        grasp_min_contact_span_px=orient.get("min_contact_span_px", 3.0),
        grasp_circle_circularity_threshold=orient.get("circle_circularity_threshold", 0.985),
        grasp_circle_min_vertices=orient.get("circle_min_vertices", 8),
        grasp_min_opening_mm=orient.get("min_opening_mm", None),
        grasp_max_opening_mm=orient.get("max_opening_mm", None),
        x_limits_mm=x_limits,
        y_limits_mm=y_limits,
        yaw_tcp_compensator=comp,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Real-object approach-only validation: top vision -> H XY/Yaw -> Yaw-TCP compensation -> "
            "CALIB_MOVE + CALIB_YAW. It NEVER commands Z descent or claw closing."
        )
    )
    p.add_argument("--config", default=str(ROOT / "config" / "runtime_config.yaml"))
    p.add_argument("--ack-timeout", type=float, default=1.0)
    p.add_argument("--reach-timeout", type=float, default=4.0)
    p.add_argument("--settle", type=float, default=1.0)
    p.add_argument(
        "--force-yaw",
        type=float,
        default=None,
        help="optional test yaw; by default use the top-vision antipodal/fallback grasp axis",
    )
    p.add_argument(
        "--require-dynamic-yaw",
        action="store_true",
        help=(
            "diagnostic mode: reject only shapes whose grasp direction is genuinely ambiguous "
            "(normally a near-circle). Default is OFF so every valid segmented shape still "
            "produces XY; polygons use antipodal geometry and circles use fallback Yaw."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(ROOT, Path(args.config))
    top_cfg = cfg["vision"]["top"]
    comp_cfg = top_cfg.get("yaw_tcp_compensation", {})
    if not bool(comp_cfg.get("enabled", False)):
        raise RuntimeError(
            "vision.top.yaw_tcp_compensation.enabled is false. Use the supplied config patch or enable it explicitly."
        )

    comp = build_compensator(top_cfg)
    print(
        "[TCP BIAS] "
        f"closing={comp.closing_bias_mm:+.3f} mm, "
        f"longitudinal={comp.longitudinal_bias_mm:+.3f} mm"
    )

    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()
    pipeline = build_pipeline(cfg, comp)
    link = STM32Link(device=cfg["stm32"]["device"], baudrate=cfg["stm32"]["baud"])

    output_dir = ROOT / "data" / "calibration" / "top_camera" / "robot_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "real_object_alignment_report.json"
    records: list[dict] = []

    frozen = None
    frozen_frame = None
    window = "Real Object Alignment - M move / R reacquire / G good / B bad / Q quit"

    try:
        camera.open()
        link.open()
        ensure_stopped(link)
        print("\n[SAFETY] This tool only sends CALIB_MOVE + CALIB_YAW. No Z descent and no claw closing.")
        print("[USE] Put ONE object in the tray. Simple polygons use antipodal min-width/support-pair Yaw; near-circles use fallback Yaw.")
        if args.require_dynamic_yaw:
            print("[MODE] --require-dynamic-yaw enabled: only geometry-reliable antipodal grasps may move.")
        else:
            print("[MODE] general-shape mode: XY is available for any valid segmented object; unreliable orientation uses fallback Yaw.")
        print("[KEY] M=move to frozen target, R=reacquire, G=mark good, B=mark bad, Q=quit\n")

        while True:
            frame = camera.read()
            show = frame.copy()

            if frozen is None:
                result = pipeline.process(frame)
                status = (
                    f"objects={result.object_count} candidates={result.candidate_count} "
                    f"votes={result.votes}/{result.required_hits} reason={result.reason}"
                )
                cv2.putText(show, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

                # Show the best instantaneous candidate even before consensus is READY.
                # This makes it obvious whether a shape uses a real antipodal support
                # pair or the near-circle fallback orientation.
                preview_candidate = max(result.candidates, key=lambda c: c.score) if result.candidates else None
                if preview_candidate is not None:
                    mode = str(getattr(preview_candidate, "orientation_mode", "dynamic" if bool(preview_candidate.orientation_reliable) else "fallback")).upper()
                    preview = (
                        f"preview: aspect={float(preview_candidate.aspect_ratio):.2f} "
                        f"hull={int(getattr(preview_candidate, 'hull_vertex_count', 0))} "
                        f"mode={mode} contact={getattr(preview_candidate, 'contact_type', 'none')} "
                        f"width={float(getattr(preview_candidate, 'grasp_width_px', 0.0)):.1f}px "
                        f"yaw={float(preview_candidate.yaw_deg):+.1f}"
                    )
                    cv2.putText(show, preview, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 0), 2)

                if result.ready and result.grasp_pixel is not None and result.x_mm is not None and result.y_mm is not None and result.yaw_deg is not None:
                    best_candidate = max(result.candidates, key=lambda c: c.score) if result.candidates else None
                    if best_candidate is None:
                        continue

                    candidate_yaw = normalize_axial_deg(float(best_candidate.yaw_deg))
                    stable_yaw = normalize_axial_deg(float(result.yaw_deg))
                    orientation_reliable = bool(best_candidate.orientation_reliable)
                    orientation_mode = str(getattr(best_candidate, "orientation_mode", "dynamic" if orientation_reliable else "fallback"))
                    rectangularity = float(getattr(best_candidate, "rectangularity", 0.0))
                    aspect_ratio = float(best_candidate.aspect_ratio)
                    circularity = float(getattr(best_candidate, "circularity", 0.0))
                    hull_vertex_count = int(getattr(best_candidate, "hull_vertex_count", 0))
                    closing_axis_image_deg = float(getattr(best_candidate, "closing_axis_image_deg", 0.0))
                    grasp_width_px = float(getattr(best_candidate, "grasp_width_px", 0.0))
                    grasp_width_mm = getattr(best_candidate, "grasp_width_mm", None)
                    contact_type = str(getattr(best_candidate, "contact_type", "none"))
                    contact_quality = float(getattr(best_candidate, "contact_quality", 0.0))
                    support_overlap_px = float(getattr(best_candidate, "support_overlap_px", 0.0))
                    support_min_point_px = getattr(best_candidate, "support_min_point_px", None)
                    support_max_point_px = getattr(best_candidate, "support_max_point_px", None)

                    # General-shape policy:
                    #   triangle/polygon/rectangle -> antipodal min-width/support-pair Yaw
                    #   true near-circle            -> configured fallback Yaw
                    # XY/grasp-point generation must NOT depend on whether orientation
                    # is meaningful.  The old validation tool intentionally rejected
                    # unreliable orientation, which is why some non-elongated shapes used
                    # to appear as if they had "no coordinates" despite a valid mask.
                    if args.require_dynamic_yaw and args.force_yaw is None and not orientation_reliable:
                        warn = (
                            f"WAIT dynamic-Yaw required; aspect={aspect_ratio:.2f}. "
                            "Remove --require-dynamic-yaw to test this shape with fallback Yaw."
                        )
                        cv2.putText(show, warn, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
                        continue

                    yaw_delta = abs(normalize_axial_deg(stable_yaw - candidate_yaw))
                    if args.force_yaw is None and orientation_reliable and yaw_delta > 6.0:
                        warn = (
                            f"WAIT stable/candidate Yaw mismatch: stable={stable_yaw:+.1f} "
                            f"candidate={candidate_yaw:+.1f}"
                        )
                        cv2.putText(show, warn, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 255), 2)
                        continue

                    detected_yaw = stable_yaw
                    target_yaw = normalize_axial_deg(args.force_yaw if args.force_yaw is not None else detected_yaw)
                    gx, gy = map(int, result.grasp_pixel)
                    raw_x, raw_y = pipeline.transform.pixel_to_robot(gx, gy, apply_tcp_offset=True)
                    applied = comp.apply(raw_x, raw_y, target_yaw)
                    center_pixel = (
                        float(best_candidate.center_pixel[0]),
                        float(best_candidate.center_pixel[1]),
                    )
                    long_axis_image_deg = None
                    try:
                        long_axis_image_deg, _ = polygon_long_axis_image_deg(best_candidate.polygon)
                    except Exception:
                        long_axis_image_deg = None

                    frozen = {
                        "grasp_pixel": [gx, gy],
                        "center_pixel": [center_pixel[0], center_pixel[1]],
                        "long_axis_image_deg": (
                            None if long_axis_image_deg is None else float(long_axis_image_deg)
                        ),
                        "raw_h_x_mm": float(raw_x),
                        "raw_h_y_mm": float(raw_y),
                        "detected_yaw_deg": float(detected_yaw),
                        "candidate_yaw_deg": float(candidate_yaw),
                        "orientation_reliable": bool(orientation_reliable),
                        "yaw_mode": orientation_mode,
                        "rectangularity": float(rectangularity),
                        "aspect_ratio": float(aspect_ratio),
                        "circularity": float(circularity),
                        "hull_vertex_count": int(hull_vertex_count),
                        "closing_axis_image_deg": float(closing_axis_image_deg),
                        "grasp_width_px": float(grasp_width_px),
                        "grasp_width_mm": (None if grasp_width_mm is None else float(grasp_width_mm)),
                        "contact_type": contact_type,
                        "contact_quality": float(contact_quality),
                        "support_overlap_px": float(support_overlap_px),
                        "support_min_point_px": support_min_point_px,
                        "support_max_point_px": support_max_point_px,
                        "object_long_axis_robot_deg": float(best_candidate.object_long_axis_deg),
                        "target_yaw_deg": float(target_yaw),
                        "command_x_mm": float(applied.x_mm),
                        "command_y_mm": float(applied.y_mm),
                        "comp_dx_mm": float(applied.delta_x_mm),
                        "comp_dy_mm": float(applied.delta_y_mm),
                        "class_name": result.class_name,
                        "confidence": float(result.confidence),
                        "moved": False,
                        "judgement": None,
                    }
                    frozen_frame = frame.copy()
                    print(
                        "[TARGET] "
                        f"pixel=({gx},{gy}) rawH=({raw_x:.2f},{raw_y:.2f}) "
                        f"aspect={aspect_ratio:.2f} hull={hull_vertex_count} circ={circularity:.3f} "
                        f"reliable={orientation_reliable} mode={orientation_mode} "
                        f"contact={contact_type} width={grasp_width_px:.1f}px "
                        f"yaw_candidate={candidate_yaw:+.2f} yaw_stable={stable_yaw:+.2f} "
                        f"yaw_target={target_yaw:+.2f} "
                        f"comp=({applied.delta_x_mm:+.2f},{applied.delta_y_mm:+.2f}) "
                        f"cmd=({applied.x_mm:.2f},{applied.y_mm:.2f})"
                    )
            else:
                if frozen_frame is not None:
                    # Keep a stable pre-move image as a small inset for reference.
                    h, w = show.shape[:2]
                    thumb_w = min(290, w // 3)
                    scale = thumb_w / frozen_frame.shape[1]
                    thumb = cv2.resize(frozen_frame, (thumb_w, max(1, int(frozen_frame.shape[0] * scale))))
                    th, tw = thumb.shape[:2]
                    show[0:th, w - tw:w] = thumb
                gx, gy = frozen["grasp_pixel"]
                cx, cy = frozen.get("center_pixel", [gx, gy])
                # BLUE = polygon centroid, RED = final grasp point.
                cv2.drawMarker(
                    show,
                    (int(round(cx)), int(round(cy))),
                    (255, 0, 0),
                    cv2.MARKER_CROSS,
                    24,
                    2,
                )
                cv2.drawMarker(show, (gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                long_img = frozen.get("long_axis_image_deg")
                if long_img is not None:
                    def _draw_axis(angle_deg, length_px, color):
                        theta = math.radians(float(angle_deg))
                        dx = 0.5 * length_px * math.cos(theta)
                        dy = 0.5 * length_px * math.sin(theta)
                        p1 = (int(round(cx - dx)), int(round(cy - dy)))
                        p2 = (int(round(cx + dx)), int(round(cy + dy)))
                        cv2.line(show, p1, p2, color, 2, cv2.LINE_AA)
                    _draw_axis(long_img, 150.0, (0, 255, 0))

                    # Yellow = actual closing axis chosen by the generic antipodal planner.
                    mode = str(frozen.get("yaw_mode", "symmetric_fallback"))
                    if mode.startswith("antipodal_"):
                        _draw_axis(float(frozen.get("closing_axis_image_deg", 0.0)), 120.0, (0, 255, 255))

                # Cyan/magenta markers show the two supporting contact features
                # used to evaluate the parallel-jaw grasp.
                pmin = frozen.get("support_min_point_px")
                pmax = frozen.get("support_max_point_px")
                if pmin is not None:
                    cv2.drawMarker(show, (int(round(pmin[0])), int(round(pmin[1]))), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
                if pmax is not None:
                    cv2.drawMarker(show, (int(round(pmax[0])), int(round(pmax[1]))), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
                lines = [
                    "BLUE=centroid RED=grasp GREEN=box-long YELLOW=closing CYAN/MAGENTA=supports",
                    f"RAW H: X={frozen['raw_h_x_mm']:.2f} Y={frozen['raw_h_y_mm']:.2f}",
                    f"Yaw det={frozen['detected_yaw_deg']:+.2f} target={frozen['target_yaw_deg']:+.2f}",
                    f"mode={frozen.get('yaw_mode', 'unknown')} contact={frozen.get('contact_type', 'none')} hull={frozen.get('hull_vertex_count', 0)} circ={frozen.get('circularity', 0.0):.3f}",
                    f"width={frozen.get('grasp_width_px', 0.0):.1f}px / {frozen.get('grasp_width_mm', None)}mm contactQ={frozen.get('contact_quality', 0.0):.2f} overlap={frozen.get('support_overlap_px', 0.0):.1f}px",
                    f"TCP dXY=({frozen['comp_dx_mm']:+.2f},{frozen['comp_dy_mm']:+.2f})",
                    f"CMD: X={frozen['command_x_mm']:.2f} Y={frozen['command_y_mm']:.2f}",
                    "M: move only XY+Yaw | G/B: record | R: reacquire | Q: quit",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(show, text, (12, 30 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.imshow(window, show)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                pipeline.reset()
                frozen = None
                frozen_frame = None
                continue
            if frozen is None:
                continue

            if key == ord('m'):
                x = float(frozen["command_x_mm"])
                y = float(frozen["command_y_mm"])
                yaw = float(frozen["target_yaw_deg"])
                q = axis_to_servo_deg(yaw)
                print(f"[MOVE] approach-only -> X={x:.2f}, Y={y:.2f}, axis={yaw:+.2f}, servo={q:+.2f}")
                send_calib_move_and_wait(
                    link, x, y,
                    ack_timeout=args.ack_timeout,
                    reach_timeout=args.reach_timeout,
                )
                send_calib_yaw_and_wait(
                    link, q,
                    ack_timeout=args.ack_timeout,
                    reach_timeout=args.reach_timeout,
                )
                time.sleep(max(0.0, args.settle))
                frozen["moved"] = True
                frozen["moved_at"] = datetime.now().isoformat(timespec="seconds")
                print("[CHECK] No descent/closing was commanded. Visually judge whether the gripper center/axis is aligned with the object.")

            if key in (ord('g'), ord('b')):
                judgement = "good" if key == ord('g') else "bad"
                item = dict(frozen)
                item["judgement"] = judgement
                item["recorded_at"] = datetime.now().isoformat(timespec="seconds")
                records.append(item)
                report = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "method": "real object top-vision target -> production Yaw-TCP compensation -> CALIB_MOVE + CALIB_YAW only",
                    "safety_note": "No Z descent or claw-close command is issued by this script.",
                    "sample_count": len(records),
                    "good_count": sum(r.get("judgement") == "good" for r in records),
                    "bad_count": sum(r.get("judgement") == "bad" for r in records),
                    "samples": records,
                }
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[SAVE] {judgement.upper()} -> {report_path}")
                pipeline.reset()
                frozen = None
                frozen_frame = None

    except UserStopRequested as exc:
        print(f"[STOP] {exc}")
        return 2
    finally:
        try:
            link.close()
        except Exception:
            pass
        try:
            camera.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    print(f"[DONE] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
