from __future__ import annotations

"""
Top-only real grasp diagnostic test.

Purpose
-------
1. Use the SAME top-camera / TopPickPipeline / TCP / single-period yaw / dynamic-preopen
   configuration as the production program.
2. Execute only the production first-stage pick sequence.
3. Never open the side camera and never run side recognition / SORT_RESULT.
4. Keep a top-camera debug window visible so physical grasp error can be compared
   against the commanded grasp point.
5. Save one JSON report + raw/annotated pre-pick images for every real pick.

Important lower-controller behavior
-----------------------------------
PICK_TARGET is the existing production first-stage command, so the STM32 will still:
    move to target -> grasp -> lift -> move to side-recognition tray -> release ->
    return to recognition-ready posture -> MCU_RECOGNITION_READY.
This script intentionally stops there. It does NOT classify the side image and does
NOT send SORT_RESULT.

Controls
--------
P : execute the currently READY top target (real robot motion)
R : reset top consensus / reacquire target
S : send STOP immediately
Q / ESC : STOP and quit

Recommended use
---------------
- Start this script.
- After startup STOP synchronization, press the physical START button on the machine.
- Wait until the overlay says READY.
- Film the physical gripper and screen together if possible.
- Press P once. After the first-stage pick completes the script sends STOP.
- Remove the object from the side tray, press the physical START button again, and
  repeat for another test point.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "vision").is_dir() and (parent / "communication").is_dir() and (parent / "config").is_dir():
            return parent
    raise RuntimeError("Cannot find smart_sorting_vision project root")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from communication.sorting_protocol import (  # noqa: E402
    Mode,
    SortingFrame,
    SortingProtocol,
    make_pick_target,
    make_stop,
)
from communication.stm32_link import STM32Link  # noqa: E402
from runtime.app_config import load_runtime_config  # noqa: E402
from vision.cameras.top_camera import TopCamera  # noqa: E402
from vision.pipeline.top_pick_pipeline import TopPickPipeline  # noqa: E402
from vision.calibration.yaw_tcp_compensation import YawTcpCompensator  # noqa: E402
from vision.calibration.yaw_servo import YawServoCalibration  # noqa: E402
from vision.calibration.descent_drift_compensation import DescentDriftCompensator  # noqa: E402


WINDOW_NAME = "TOP PICK ONLY DIAGNOSTIC | P pick | R reset | S stop | Q quit"


class UserQuit(RuntimeError):
    pass


class PhysicalStop(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real top-camera grasp test. Side camera/recognition is never opened."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "runtime_config.yaml"),
        help="runtime_config.yaml path",
    )
    parser.add_argument(
        "--ack-timeout",
        type=float,
        default=None,
        help="Override ACK timeout. Default: runtime.pick_ack_timeout_s",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=None,
        help="Override MCU_RECOGNITION_READY timeout. Default: runtime.recognition_ready_timeout_s",
    )
    parser.add_argument(
        "--save-dir",
        default=str(ROOT / "outputs" / "debug" / "top_pick_only_diagnostic"),
        help="Directory for JSON and pre-pick images",
    )
    return parser.parse_args()


def normalize_axial_deg(angle_deg: float) -> float:
    value = (float(angle_deg) + 90.0) % 180.0 - 90.0
    return 0.0 if abs(value) < 1.0e-12 else float(value)


def build_tcp_compensator(top_cfg: dict, yaw: YawServoCalibration) -> YawTcpCompensator | None:
    comp = top_cfg.get("yaw_tcp_compensation", {}) or {}
    if not bool(comp.get("enabled", False)):
        return None

    offset = comp.get("offset_axis0_mm", [0.0, 0.0])
    return YawTcpCompensator(
        reference_axis_deg=float(comp["reference_axis_deg"]),
        offset_axis0_x_mm=float(offset[0]),
        offset_axis0_y_mm=float(offset[1]),
        min_axis_deg=yaw.physical_min_deg,
        max_axis_deg=yaw.physical_max_deg,
        visual_to_command_matrix=comp.get(
            "visual_to_command_matrix", [[1.0, 0.0], [0.0, 1.0]]
        ),
        closing_bias_mm=float(comp.get("closing_bias_mm", 0.0)),
        longitudinal_bias_mm=float(comp.get("longitudinal_bias_mm", 0.0)),
    )


def build_descent_drift_compensator(top_cfg: dict) -> DescentDriftCompensator | None:
    drift_cfg = top_cfg.get("descent_drift_compensation", {}) or {}
    if not bool(drift_cfg.get("enabled", False)):
        return None

    tcp_cfg = top_cfg.get("yaw_tcp_compensation", {}) or {}
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    if bool(drift_cfg.get("use_visual_to_command_matrix", True)):
        matrix = tcp_cfg.get("visual_to_command_matrix", matrix)

    return DescentDriftCompensator(
        signed_drift_mm=float(drift_cfg.get("signed_drift_mm", 0.0)),
        relative_angle_deg=float(drift_cfg.get("relative_angle_deg", 90.0)),
        visual_to_command_matrix=matrix,
    )


def build_pipeline(cfg: dict) -> tuple[TopPickPipeline, YawServoCalibration]:
    top_model_cfg = cfg["model"]["top"]
    top_vision_cfg = cfg["vision"]["top"]
    top_consensus = top_vision_cfg["consensus"]
    orient = top_vision_cfg.get("grasp_orientation", {}) or {}
    preopen = orient.get("dynamic_preopen", {}) or {}
    collision = top_vision_cfg.get("gripper_collision", {}) or {}
    search = collision.get("grasp_search", {}) or {}
    mechanics_cfg = cfg.get("mechanics", {}) or {}

    allowed_polygon_mm = collision.get(
        "allowed_polygon_mm",
        mechanics_cfg.get("allowed_polygon_mm", None),
    )

    yaw = YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"])

    pipeline = TopPickPipeline(
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
        geometric_grasp_config=top_vision_cfg.get("geometric_grasp"),
        consensus_window=top_consensus["window"],
        consensus_required=top_consensus["required"],
        max_track_center_shift_px=top_consensus["max_track_center_shift_px"],
        max_grasp_spread_mm=top_consensus["max_grasp_spread_mm"],
        reset_after_misses=top_consensus["reset_after_misses"],
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
        dynamic_preopen_enabled=preopen.get("enabled", True),
        preopen_margin_mm=preopen.get("margin_mm", 6.0),
        preopen_command_min_mm=preopen.get("min_command_mm", None),
        preopen_command_max_mm=preopen.get("max_command_mm", None),
        preopen_profile_kind=preopen.get("profile_kind", "cad_four_bar"),
        preopen_configured=bool(preopen.get("configured", False)),
        preopen_closed_servo_deg=float(preopen.get("closed_servo_deg", -35.0)),
        preopen_standby_servo_deg=float(preopen.get("standby_servo_deg", -95.0)),
        gripper_collision_enabled=collision.get("enabled", True),
        gripper_finger_length_mm=collision.get("finger_length_mm", 20.0),
        gripper_finger_thickness_mm=collision.get("finger_thickness_mm", 9.0),
        gripper_finger_longitudinal_offset_mm=collision.get("finger_longitudinal_offset_mm", 0.0),
        gripper_opening_uncertainty_mm=collision.get("opening_uncertainty_mm", 1.5),
        gripper_object_margin_mm=collision.get("object_margin_mm", 1.0),
        gripper_wall_margin_mm=collision.get("wall_margin_mm", 2.0),
        gripper_fallback_opening_mm=collision.get("fallback_opening_mm", 38.3),
        gripper_forbidden_zones_mm=mechanics_cfg.get("forbidden_zones_mm", []) or [],
        gripper_allowed_region_mm=collision.get("allowed_region_mm", None),
        gripper_allowed_polygon_mm=allowed_polygon_mm,
        gripper_object_overlap_tolerance_px=collision.get("object_overlap_tolerance_px", 4),
        gripper_enforce_image_border=collision.get("enforce_image_border", True),
        gripper_image_border_margin_px=collision.get("image_border_margin_px", 2),
        gripper_body_enabled=(collision.get("body", {}) or {}).get("enabled", False),
        gripper_body_width_mm=(collision.get("body", {}) or {}).get("width_mm", 50.0),
        gripper_body_length_mm=(collision.get("body", {}) or {}).get("length_mm", 20.0),
        gripper_body_longitudinal_offset_mm=(collision.get("body", {}) or {}).get("longitudinal_offset_mm", -25.0),
        gripper_debug_log_rejections=collision.get("debug_log_rejections", False),
        grasp_search_enabled=search.get("enabled", True),
        grasp_search_yaw_offsets_deg=search.get("yaw_offsets_deg", [0, -5, 5, -10, 10, -15, 15]),
        grasp_search_longitudinal_offsets_mm=search.get("longitudinal_offsets_mm", [0, -2, 2, -4, 4]),
        grasp_search_closing_offsets_mm=search.get("closing_offsets_mm", [0]),
        grasp_search_hard_min_clearance_mm=search.get("hard_min_clearance_mm", 1.0),
        grasp_search_preferred_clearance_mm=search.get("preferred_clearance_mm", 4.0),
        grasp_search_min_target_overlap_mm=search.get("min_target_overlap_mm", 1.5),
        grasp_search_min_side_gap_mm=search.get("min_side_gap_mm", 0.25),
        grasp_search_yaw_penalty_per_deg=search.get("yaw_penalty_per_deg", 0.12),
        grasp_search_center_penalty_per_mm=search.get("center_penalty_per_mm", 0.35),
        grasp_search_max_trials_per_object=search.get("max_trials_per_object", 40),
        x_limits_mm=cfg["mechanics"]["workspace_mm"]["x"],
        y_limits_mm=cfg["mechanics"]["workspace_mm"]["y"],
        yaw_tcp_compensator=build_tcp_compensator(top_vision_cfg, yaw),
        yaw_servo_calibration=yaw,
    )
    return pipeline, yaw


def draw_axis(image: np.ndarray, center: tuple[float, float], angle_deg: float, length_px: float, color, thickness: int = 2) -> None:
    theta = math.radians(float(angle_deg))
    dx = 0.5 * float(length_px) * math.cos(theta)
    dy = 0.5 * float(length_px) * math.sin(theta)
    cx, cy = map(float, center)
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def put_lines(image: np.ndarray, lines: list[str], *, x: int = 12, y: int = 28, dy: int = 27) -> None:
    for i, line in enumerate(lines):
        org = (x, y + i * dy)
        cv2.putText(image, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def candidate_for_result(result) -> Any | None:
    if not result.candidates:
        return None
    if result.grasp_pixel is None:
        return max(result.candidates, key=lambda c: float(c.score))
    gx, gy = map(float, result.grasp_pixel)
    return min(
        result.candidates,
        key=lambda c: (float(c.grasp_pixel[0]) - gx) ** 2 + (float(c.grasp_pixel[1]) - gy) ** 2,
    )


def draw_pipeline_overlay(frame: np.ndarray, result, *, state_text: str, started: bool) -> np.ndarray:
    vis = frame.copy()
    overlay = vis.copy()

    selected = candidate_for_result(result) if result is not None else None

    if result is not None:
        for candidate in result.candidates:
            poly = np.asarray(candidate.polygon, dtype=np.int32).reshape(-1, 1, 2)
            is_selected = selected is candidate
            fill_color = (40, 180, 40) if not is_selected else (0, 180, 255)
            cv2.fillPoly(overlay, [poly], fill_color)
            cv2.polylines(vis, [poly], True, fill_color, 2 if not is_selected else 3, cv2.LINE_AA)

            cx, cy = map(int, map(round, candidate.center_pixel))
            gx, gy = map(int, candidate.grasp_pixel)
            cv2.drawMarker(vis, (cx, cy), (255, 80, 80), cv2.MARKER_CROSS, 18, 2)
            cv2.drawMarker(vis, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            draw_axis(vis, (gx, gy), float(candidate.closing_axis_image_deg), 110.0, (0, 255, 255), 2)

            for fp in getattr(candidate, "gripper_footprint_polygons_px", []) or []:
                contour = np.asarray(fp, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [contour], True, (255, 255, 0), 2, cv2.LINE_AA)

            label = (
                f"#{candidate.object_index} {candidate.class_name} {candidate.confidence:.2f} "
                f"XY=({candidate.x_mm:.1f},{candidate.y_mm:.1f}) "
                f"Y={candidate.yaw_deg:+.1f} "
                f"gap={candidate.gripper_clearance_mm:.1f}mm"
            )
            tx = max(5, min(vis.shape[1] - 620, gx - 250))
            ty = max(24, gy - 24)
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.14, vis, 0.86, 0.0, vis)

    lines = [
        f"STATE={state_text} | MCU_START={'YES' if started else 'NO'}",
    ]
    if result is not None:
        lines.append(
            f"TOP objects={result.object_count} candidates={result.candidate_count} "
            f"READY={result.ready} votes={result.votes}/{result.required_hits} reason={result.reason}"
        )
        if result.ready:
            lines.append(
                f"READY TARGET: X={float(result.x_mm):.2f} Y={float(result.y_mm):.2f} "
                f"Yaw={float(result.yaw_deg):+.1f} "
                f"preopen={result.preopen_mm}mm"
            )
    lines.append("P=REAL PICK | R=reacquire | S=STOP | Q=quit")
    put_lines(vis, lines)
    return vis


def draw_frozen_overlay(frame: np.ndarray, frozen: dict, *, phase: str, started: bool) -> np.ndarray:
    vis = frame.copy()
    poly = frozen.get("polygon")
    if poly is not None:
        contour = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [contour], (0, 180, 255))
        cv2.addWeighted(overlay, 0.14, vis, 0.86, 0.0, vis)
        cv2.polylines(vis, [contour], True, (0, 180, 255), 3, cv2.LINE_AA)

    cx, cy = frozen["center_pixel"]
    gx, gy = frozen["grasp_pixel"]
    cv2.drawMarker(vis, (int(round(cx)), int(round(cy))), (255, 80, 80), cv2.MARKER_CROSS, 22, 2)
    cv2.drawMarker(vis, (int(round(gx)), int(round(gy))), (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
    draw_axis(vis, (gx, gy), float(frozen["closing_axis_image_deg"]), 120.0, (0, 255, 255), 3)

    for fp in frozen.get("gripper_footprint_polygons_px", []) or []:
        contour = np.asarray(fp, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [contour], True, (255, 255, 0), 2, cv2.LINE_AA)

    lines = [
        f"PHASE={phase} | MCU_START={'YES' if started else 'NO'}",
        f"VIS/TCP ALIGN: X={frozen['alignment_x_mm']:.2f} Y={frozen['alignment_y_mm']:.2f}",
        f"FINAL CMD: X={frozen['command_x_mm']:.2f} Y={frozen['command_y_mm']:.2f} "
        f"dXY=({frozen['descent_precomp_dx_mm']:+.2f},{frozen['descent_precomp_dy_mm']:+.2f})",
        f"Yaw(axis)={frozen['yaw_deg']:+.1f} "
        f"physical={frozen['physical_yaw_deg']:+.1f} preopen={frozen['preopen_mm']}mm",
        f"pixel center=({cx:.1f},{cy:.1f}) grasp=({gx:.0f},{gy:.0f}) "
        f"width={frozen['grasp_width_mm']}mm clearance={frozen['gripper_clearance_mm']:.2f}mm",
        "Yellow = planned closing axis | RED cross = commanded grasp point",
        "S=STOP | Q=STOP+quit",
    ]
    put_lines(vis, lines)
    return vis


def show_frame_and_get_key(camera: TopCamera, frozen: dict, phase: str, started: bool) -> int:
    frame = camera.read()
    if frame is None:
        return -1
    vis = draw_frozen_overlay(frame, frozen, phase=phase, started=started)
    cv2.imshow(WINDOW_NAME, vis)
    return cv2.waitKey(1) & 0xFF


def quantized(frame: SortingFrame) -> SortingFrame:
    return SortingProtocol.unpack(SortingProtocol.pack(frame))


def check_common_rx(link: STM32Link, rx: SortingFrame, state: dict) -> None:
    print(f"[RX] {link.describe(rx)}")
    if rx.mode == Mode.MCU_START_WORK:
        state["started"] = True
    elif rx.mode in (Mode.MCU_STOP_WORK, Mode.MCU_STOP_ACK):
        state["started"] = False
        if rx.mode == Mode.MCU_STOP_WORK:
            raise PhysicalStop("physical STOP received")
    elif rx.mode == Mode.MCU_ERROR:
        raise RuntimeError(f"MCU_ERROR code={int(rx.box_id)}")


def wait_for_specific_ack(
    link: STM32Link,
    camera: TopCamera,
    frozen: dict,
    state: dict,
    *,
    command: SortingFrame,
    ack_mode: Mode,
    timeout_s: float,
    phase: str,
    retry_interval_s: float,
) -> SortingFrame:
    expected = quantized(command)
    raw = link.send_frame(expected)
    print(f"[TX] {phase} | {raw.hex(' ').upper()}")
    start = time.monotonic()
    deadline = start + float(timeout_s)
    last_send = start

    while time.monotonic() < deadline:
        now = time.monotonic()
        for rx in link.poll():
            check_common_rx(link, rx, state)
            if rx.mode == Mode.MCU_BUSY:
                continue
            if rx.mode != ack_mode:
                continue

            if ack_mode == Mode.MCU_PICK_PREOPEN_ACK:
                if abs(float(rx.x_mm) - float(expected.x_mm)) > 0.051:
                    raise RuntimeError(
                        f"PICK_PREOPEN_ACK mismatch expected={expected.x_mm:.1f} got={rx.x_mm:.1f}"
                    )
            return rx

        if now - last_send >= float(retry_interval_s):
            raw = link.send_frame(expected)
            print(f"[TX][RETRY] {phase} | {raw.hex(' ').upper()}")
            last_send = now

        key = show_frame_and_get_key(camera, frozen, phase, bool(state["started"]))
        if key in (ord("q"), ord("Q"), 27):
            raise UserQuit()
        if key in (ord("s"), ord("S")):
            raise PhysicalStop("user requested STOP")
        time.sleep(0.002)

    raise TimeoutError(f"{phase} timeout after {timeout_s:.2f}s")


def execute_pick(
    link: STM32Link,
    camera: TopCamera,
    frozen: dict,
    state: dict,
    *,
    ack_timeout_s: float,
    ready_timeout_s: float,
    retry_interval_s: float,
) -> dict:
    result = {
        "preopen_ack": frozen["preopen_mm"] is None,
        "pick_ack": False,
        "recognition_ready": False,
        "elapsed_s": None,
    }

    start = time.monotonic()

    if frozen["preopen_mm"] is not None:
        wait_for_specific_ack(
            link,
            camera,
            frozen,
            state,
            command=SortingFrame(mode=Mode.PICK_PREOPEN, x_mm=float(frozen["preopen_mm"])),
            ack_mode=Mode.MCU_PICK_PREOPEN_ACK,
            timeout_s=ack_timeout_s,
            phase="WAIT_PREOPEN_ACK",
            retry_interval_s=retry_interval_s,
        )
        result["preopen_ack"] = True

    pick_command = quantized(
        make_pick_target(
            float(frozen["command_x_mm"]),
            float(frozen["command_y_mm"]),
            float(frozen["yaw_deg"]),
        )
    )
    raw = link.send_frame(pick_command)
    print(
        f"[TX] PICK_TARGET X={pick_command.x_mm:.1f} Y={pick_command.y_mm:.1f} "
        f"Yaw={pick_command.yaw_deg:+.0f} | {raw.hex(' ').upper()}"
    )

    ack_deadline = time.monotonic() + float(ack_timeout_s)
    ready_deadline = time.monotonic() + float(ready_timeout_s)
    last_send = time.monotonic()

    while time.monotonic() < ready_deadline:
        now = time.monotonic()
        for rx in link.poll():
            check_common_rx(link, rx, state)
            if rx.mode == Mode.MCU_BUSY:
                continue

            if rx.mode == Mode.MCU_PICK_TARGET_ACK:
                if (
                    abs(float(rx.x_mm) - float(pick_command.x_mm)) > 0.051
                    or abs(float(rx.y_mm) - float(pick_command.y_mm)) > 0.051
                    or abs(float(rx.yaw_deg) - float(pick_command.yaw_deg)) > 0.51
                ):
                    raise RuntimeError(
                        "PICK_TARGET_ACK mismatch: "
                        f"expected=({pick_command.x_mm:.1f},{pick_command.y_mm:.1f},{pick_command.yaw_deg:+.0f}) "
                        f"got=({rx.x_mm:.1f},{rx.y_mm:.1f},{rx.yaw_deg:+.0f})"
                    )
                result["pick_ack"] = True
                print("[OK] PICK_TARGET ACK")

            if rx.mode == Mode.MCU_RECOGNITION_READY:
                if (
                    abs(float(rx.x_mm) - float(pick_command.x_mm)) > 0.051
                    or abs(float(rx.y_mm) - float(pick_command.y_mm)) > 0.051
                ):
                    raise RuntimeError(
                        "RECOGNITION_READY mismatch: "
                        f"expected=({pick_command.x_mm:.1f},{pick_command.y_mm:.1f}) "
                        f"got=({rx.x_mm:.1f},{rx.y_mm:.1f})"
                    )
                result["recognition_ready"] = True
                result["elapsed_s"] = time.monotonic() - start
                print("[OK] first-stage PICK complete; side recognition intentionally skipped")
                return result

        if not result["pick_ack"] and now >= ack_deadline:
            raise TimeoutError(f"PICK_TARGET_ACK timeout after {ack_timeout_s:.2f}s")

        if not result["pick_ack"] and now - last_send >= float(retry_interval_s):
            raw = link.send_frame(pick_command)
            print(f"[TX][RETRY] PICK_TARGET | {raw.hex(' ').upper()}")
            last_send = now

        key = show_frame_and_get_key(camera, frozen, "PICKING / WAIT_RECOGNITION_READY", bool(state["started"]))
        if key in (ord("q"), ord("Q"), 27):
            raise UserQuit()
        if key in (ord("s"), ord("S")):
            raise PhysicalStop("user requested STOP")
        time.sleep(0.002)

    raise TimeoutError(f"MCU_RECOGNITION_READY timeout after {ready_timeout_s:.2f}s")


def send_stop_and_wait(link: STM32Link, state: dict, timeout_s: float = 2.0) -> None:
    try:
        raw = link.send_frame(make_stop())
        print(f"[TX] STOP | {raw.hex(' ').upper()}")
    except Exception as exc:
        print(f"[WARN] STOP send failed: {exc}")
        return

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        for rx in link.poll():
            print(f"[RX] {link.describe(rx)}")
            if rx.mode in (Mode.MCU_STOP_ACK, Mode.MCU_STOP_WORK):
                state["started"] = False
                return
        time.sleep(0.003)


def make_frozen_target(result, yaw: YawServoCalibration, drift_compensator: DescentDriftCompensator | None) -> dict:
    if not result.ready:
        raise ValueError("top result is not READY")
    if result.x_mm is None or result.y_mm is None or result.yaw_deg is None:
        raise ValueError("READY result is missing XY/Yaw")
    if result.preopen_mm is None:
        raise ValueError("READY result is missing calibrated preopen")

    selected = candidate_for_result(result)
    if selected is None:
        raise ValueError("READY result has no candidate")

    alignment_x = float(result.x_mm)
    alignment_y = float(result.y_mm)
    yaw_deg = normalize_axial_deg(float(result.yaw_deg))
    # 与生产程序一致：先量化协议角度，再确定唯一物理姿态并校验舵机限位。
    yaw_deg = quantized(make_pick_target(alignment_x, alignment_y, yaw_deg)).yaw_deg
    physical_yaw = yaw.physical_axis_deg(yaw_deg)
    yaw.command_deg(yaw_deg)

    command_x = alignment_x
    command_y = alignment_y
    pre_dx = 0.0
    pre_dy = 0.0
    predicted_dx = 0.0
    predicted_dy = 0.0

    if drift_compensator is not None:
        drift = drift_compensator.apply(alignment_x, alignment_y, physical_yaw)
        command_x = float(drift.x_mm)
        command_y = float(drift.y_mm)
        pre_dx = float(drift.command_delta_x_mm)
        pre_dy = float(drift.command_delta_y_mm)
        predicted_dx = float(drift.predicted_drift_x_mm)
        predicted_dy = float(drift.predicted_drift_y_mm)

    polygon = np.asarray(selected.polygon, dtype=float).tolist()
    footprints = [
        np.asarray(poly, dtype=float).tolist()
        for poly in (getattr(selected, "gripper_footprint_polygons_px", []) or [])
    ]

    return {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "alignment_x_mm": alignment_x,
        "alignment_y_mm": alignment_y,
        "command_x_mm": command_x,
        "command_y_mm": command_y,
        "descent_precomp_dx_mm": pre_dx,
        "descent_precomp_dy_mm": pre_dy,
        "predicted_descent_drift_x_mm": predicted_dx,
        "predicted_descent_drift_y_mm": predicted_dy,
        "yaw_deg": yaw_deg,
        "physical_yaw_deg": physical_yaw,
        "preopen_mm": None if result.preopen_mm is None else float(result.preopen_mm),
        "grasp_width_mm": None if result.grasp_width_mm is None else float(result.grasp_width_mm),
        "class_name": result.class_name,
        "confidence": float(result.confidence),
        "grasp_pixel": [int(result.grasp_pixel[0]), int(result.grasp_pixel[1])],
        "center_pixel": [float(selected.center_pixel[0]), float(selected.center_pixel[1])],
        "closing_axis_image_deg": float(selected.closing_axis_image_deg),
        "orientation_mode": str(selected.orientation_mode),
        "contact_type": str(selected.contact_type),
        "contact_quality": float(selected.contact_quality),
        "gripper_clearance_mm": float(selected.gripper_clearance_mm),
        "search_yaw_offset_deg": float(selected.search_yaw_offset_deg),
        "search_longitudinal_offset_mm": float(selected.search_longitudinal_offset_mm),
        "search_closing_offset_mm": float(selected.search_closing_offset_mm),
        "score": float(selected.score),
        "polygon": polygon,
        "gripper_footprint_polygons_px": footprints,
    }


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(ROOT, args.config)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = cfg.get("runtime", {}) or {}
    ack_timeout_s = float(args.ack_timeout or runtime_cfg.get("pick_ack_timeout_s", 1.5))
    ready_timeout_s = float(args.ready_timeout or runtime_cfg.get("recognition_ready_timeout_s", 18.0))
    retry_interval_s = float(runtime_cfg.get("retry_interval_s", 0.25))

    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()
    pipeline, yaw_calibration = build_pipeline(cfg)
    drift_compensator = build_descent_drift_compensator(cfg["vision"]["top"])

    link = STM32Link(
        device=cfg["stm32"]["device"],
        baudrate=cfg["stm32"]["baud"],
    )

    state = {"started": False}
    pick_index = 0

    print("=" * 82)
    print("TOP-ONLY REAL GRASP DIAGNOSTIC")
    print("- Side camera: NOT OPENED")
    print("- Side recognition: DISABLED")
    print("- SORT_RESULT: NEVER SENT")
    print("- Real PICK_TARGET: ENABLED (physical robot WILL MOVE)")
    print("=" * 82)
    print(f"config: {args.config}")
    print(f"output: {save_dir}")
    print(f"ACK timeout={ack_timeout_s:.2f}s, RECOGNITION_READY timeout={ready_timeout_s:.2f}s")
    print("After startup STOP sync, press the PHYSICAL START button, wait for READY, then press P.")
    print("For every successful pick the script sends STOP after MCU_RECOGNITION_READY.")
    print("=" * 82)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        link.open()
        send_stop_and_wait(link, state, timeout_s=2.0)
        camera.open()

        # Top model warmup only. Side model/camera is intentionally untouched.
        warm = camera.read()
        if warm is None:
            raise RuntimeError("Top camera returned no frame during warmup")
        if hasattr(pipeline.segmenter, "warmup"):
            try:
                stats = pipeline.segmenter.warmup(warm, runs=2)
                print(f"[WARMUP] top: {stats}")
            except Exception as exc:
                print(f"[WARN] top warmup skipped/failed: {exc}")
        pipeline.reset()

        latest_ready = None
        latest_result = None

        while True:
            # Poll physical START / STOP / error before allowing any motion.
            try:
                for rx in link.poll():
                    print(f"[RX] {link.describe(rx)}")
                    if rx.mode == Mode.MCU_START_WORK:
                        if not state["started"]:
                            print("[READY] MCU START observed; real pick is now allowed")
                        state["started"] = True
                    elif rx.mode == Mode.MCU_STOP_WORK:
                        state["started"] = False
                        pipeline.reset()
                        latest_ready = None
                        print("[STOP] physical STOP_WORK")
                    elif rx.mode == Mode.MCU_STOP_ACK:
                        state["started"] = False
                    elif rx.mode == Mode.MCU_ERROR:
                        raise RuntimeError(f"MCU_ERROR code={int(rx.box_id)}")
            except PhysicalStop:
                state["started"] = False

            frame = camera.read()
            if frame is None:
                continue

            result = pipeline.process(frame)
            latest_result = result
            if result.ready:
                latest_ready = make_frozen_target(result, yaw_calibration, drift_compensator)

            state_text = "WAIT PHYSICAL START" if not state["started"] else "TOP DETECT / READY FOR P"
            vis = draw_pipeline_overlay(frame, result, state_text=state_text, started=bool(state["started"]))
            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("s"), ord("S")):
                print("[USER] STOP")
                send_stop_and_wait(link, state)
                pipeline.reset()
                latest_ready = None
                continue

            if key in (ord("r"), ord("R")):
                print("[USER] reacquire")
                pipeline.reset()
                latest_ready = None
                continue

            if key not in (ord("p"), ord("P")):
                continue

            if not state["started"]:
                print("[BLOCK] Physical START has not been observed. Press the machine START button first.")
                continue
            if latest_result is None or not latest_result.ready or latest_ready is None:
                print("[BLOCK] Top pipeline is not READY. Wait for stable target first.")
                continue

            # Freeze exactly the target visible when P is pressed.
            frozen = make_frozen_target(latest_result, yaw_calibration, drift_compensator)
            pick_index += 1
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            stem = f"pick_{pick_index:03d}_{stamp}"
            raw_path = save_dir / f"{stem}_raw.jpg"
            overlay_path = save_dir / f"{stem}_overlay.jpg"
            report_path = save_dir / f"{stem}.json"

            cv2.imwrite(str(raw_path), frame)
            cv2.imwrite(
                str(overlay_path),
                draw_frozen_overlay(frame, frozen, phase="PRE-PICK FROZEN", started=True),
            )

            report = {
                "test": "top_pick_only_diagnostic",
                "config_path": str(Path(args.config)),
                "pick_index": pick_index,
                "target": frozen,
                "result": None,
                "files": {
                    "raw": str(raw_path),
                    "overlay": str(overlay_path),
                },
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

            print("\n" + "-" * 82)
            print(
                "[REAL PICK] "
                f"ALIGN=({frozen['alignment_x_mm']:.2f},{frozen['alignment_y_mm']:.2f}) "
                f"CMD=({frozen['command_x_mm']:.2f},{frozen['command_y_mm']:.2f}) "
                f"Yaw={frozen['yaw_deg']:+.1f} "
                f"physical={frozen['physical_yaw_deg']:+.1f} preopen={frozen['preopen_mm']}"
            )
            print("[REAL PICK] Robot motion starts now. Physical STOP remains the emergency stop.")

            try:
                pick_result = execute_pick(
                    link,
                    camera,
                    frozen,
                    state,
                    ack_timeout_s=ack_timeout_s,
                    ready_timeout_s=ready_timeout_s,
                    retry_interval_s=retry_interval_s,
                )
                report["result"] = pick_result
                report["finished_at"] = datetime.now().isoformat(timespec="milliseconds")
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

                # The object is now on the side tray. Deliberately do NOT perform side recognition.
                # STOP makes the single diagnostic cycle deterministic and prevents any sort motion.
                send_stop_and_wait(link, state)
                print(f"[DONE] first-stage pick complete; side recognition skipped. Report: {report_path}")
                print("       Remove the object from the side tray, press physical START again, then repeat.")

            except (PhysicalStop, UserQuit) as exc:
                report["result"] = {"aborted": True, "reason": str(exc)}
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                send_stop_and_wait(link, state)
                if isinstance(exc, UserQuit):
                    raise
                print(f"[ABORT] {exc}")
            except Exception as exc:
                report["result"] = {"failed": True, "reason": str(exc)}
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                send_stop_and_wait(link, state)
                print(f"[FAIL] {exc}")

            pipeline.reset()
            latest_ready = None
            latest_result = None
            print("-" * 82 + "\n")

    except UserQuit:
        pass
    finally:
        try:
            if link.is_open:
                send_stop_and_wait(link, state, timeout_s=1.5)
        except Exception as exc:
            print(f"[WARN] cleanup STOP failed: {exc}")
        try:
            camera.close()
        except Exception:
            pass
        try:
            link.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
