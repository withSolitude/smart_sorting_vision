from __future__ import annotations

import argparse
import inspect
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.pipeline.top_pick_pipeline import TopPickPipeline
from vision.calibration.yaw_tcp_compensation import YawTcpCompensator

DEFAULT_CONFIG = ROOT / "config" / "runtime_config.yaml"

REASON_CN = {
    "grasp_center_outside_target": "中心点跑出目标",
    "invalid_trial_width": "目标宽度计算无效",
    "preopen_out_of_calibrated_range": "预开口超标定范围",
    "target_not_between_fingers": "目标不能落在两指之间",
    "target_contact_overlap_too_small": "夹持接触重叠不足",
    "image_border_collision": "夹爪碰图像边界",
    "allowed_region_collision": "夹爪超允许区域",
    "forbidden_zone_collision": "夹爪碰禁入区",
    "other_object_collision": "夹爪碰其它物体",
    "clearance_below_hard_min": "安全间隙不足",
    "candidate_robot_to_pixel_failed": "机械坐标转像素失败",
    "candidate_pixel_transform_failed": "像素转机械坐标失败",
    "candidate_xy_not_finite": "机械坐标无效",
    "yaw_tcp_branch_unavailable": "Yaw/TCP方向不可用",
    "machine_workspace_limit": "超XY工作范围",
    "target_collision": "夹爪与目标几何冲突",
    "body_collision": "夹爪主体碰撞",
}


def reason_text(reason: str) -> str:
    reason = str(reason)
    cn = REASON_CN.get(reason)
    return f"{reason} | {cn}" if cn else reason


def draw_polygon(image, polygon, color, thickness=2):
    if polygon is None:
        return
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return
    points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)


def fill_polygon_alpha(image, polygon, color, alpha):
    if polygon is None:
        return
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return
    points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def draw_cross(image, point, color, size=8, thickness=2):
    x = int(point[0])
    y = int(point[1])
    cv2.line(image, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def draw_axis(image, center, angle_deg, half_length_px, color, thickness=2):
    cx = float(center[0])
    cy = float(center[1])
    theta = math.radians(float(angle_deg))
    dx = float(half_length_px) * math.cos(theta)
    dy = float(half_length_px) * math.sin(theta)
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def put_label(
    image,
    text,
    origin,
    *,
    color=(255, 255, 255),
    background=(0, 0, 0),
    scale=0.42,
    thickness=1,
):
    x = int(origin[0])
    y = int(origin[1])
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = max(0, min(x, max(0, image.shape[1] - width - 8)))
    y = max(height + 6, min(y, image.shape[0] - baseline - 4))
    cv2.rectangle(
        image,
        (x - 3, y - height - 4),
        (x + width + 4, y + baseline + 3),
        background,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def put_panel(image, lines, *, x=10, y=10, line_height=22, scale=0.50):
    if not lines:
        return
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        for line in lines
    ]
    width = max(w for w, _ in sizes) + 18
    height = line_height * len(lines) + 12
    cv2.rectangle(
        image,
        (x, y),
        (min(image.shape[1] - 1, x + width), min(image.shape[0] - 1, y + height)),
        (0, 0, 0),
        -1,
    )
    yy = y + 20
    for line in lines:
        cv2.putText(
            image,
            line,
            (x + 7, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        yy += line_height


def build_yaw_tcp_compensator(top_vision_cfg: dict):
    top_tcp_comp = top_vision_cfg.get("yaw_tcp_compensation", {}) or {}
    if not bool(top_tcp_comp.get("enabled", False)):
        return None

    offset_axis0 = top_tcp_comp.get("offset_axis0_mm", [0.0, 0.0])
    visual_to_command = top_tcp_comp.get(
        "visual_to_command_matrix", [[1.0, 0.0], [0.0, 1.0]]
    )

    return YawTcpCompensator(
        reference_axis_deg=float(top_tcp_comp["reference_axis_deg"]),
        offset_axis0_x_mm=float(offset_axis0[0]),
        offset_axis0_y_mm=float(offset_axis0[1]),
        min_axis_deg=float(top_tcp_comp.get("min_axis_deg", -90.0)),
        max_axis_deg=float(top_tcp_comp.get("max_axis_deg", 90.0)),
        visual_to_command_matrix=visual_to_command,
        closing_bias_mm=float(top_tcp_comp.get("closing_bias_mm", 0.0)),
        longitudinal_bias_mm=float(top_tcp_comp.get("longitudinal_bias_mm", 0.0)),
    )


def build_top_pipeline(cfg: dict) -> TopPickPipeline:
    top_model_cfg = cfg["model"]["top"]
    top_vision_cfg = cfg["vision"]["top"]
    top_consensus = top_vision_cfg["consensus"]
    top_orientation = top_vision_cfg.get("grasp_orientation", {}) or {}
    top_preopen = top_orientation.get("dynamic_preopen", {}) or {}
    top_gripper_collision = top_vision_cfg.get("gripper_collision", {}) or {}
    top_grasp_search = top_gripper_collision.get("grasp_search", {}) or {}
    top_tcp_comp = top_vision_cfg.get("yaw_tcp_compensation", {}) or {}
    mechanics_cfg = cfg.get("mechanics", {}) or {}
    body_cfg = top_gripper_collision.get("body", {}) or {}

    x_limits = mechanics_cfg["workspace_mm"]["x"]
    y_limits = mechanics_cfg["workspace_mm"]["y"]
    forbidden_zones_mm = mechanics_cfg.get("forbidden_zones_mm", []) or []

    # If the gripper layer explicitly has allowed_polygon_mm: null, keep it disabled.
    if "allowed_polygon_mm" in top_gripper_collision:
        allowed_polygon_mm = top_gripper_collision["allowed_polygon_mm"]
    else:
        allowed_polygon_mm = mechanics_cfg.get("allowed_polygon_mm", None)

    yaw_tcp_compensator = build_yaw_tcp_compensator(top_vision_cfg)

    kwargs = dict(
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

        consensus_window=top_consensus["window"],
        consensus_required=top_consensus["required"],
        max_track_center_shift_px=top_consensus["max_track_center_shift_px"],
        max_grasp_spread_mm=top_consensus["max_grasp_spread_mm"],
        reset_after_misses=top_consensus["reset_after_misses"],

        orientation_enabled=top_orientation.get("enabled", True),
        grip_long_sides=top_orientation.get("grip_long_sides", True),
        min_orientation_aspect_ratio=top_orientation.get("min_aspect_ratio", 1.20),
        default_grasp_axis_robot_deg=top_orientation.get(
            "default_grasp_axis_robot_deg", 0.0
        ),
        yaw_alignment_offset_deg=top_orientation.get("yaw_alignment_offset_deg", 0.0),
        max_yaw_spread_deg=top_orientation.get("max_yaw_spread_deg", 12.0),
        grasp_width_slack_ratio=top_orientation.get("width_slack_ratio", 0.15),
        grasp_support_tolerance_px=top_orientation.get("support_tolerance_px", 3.0),
        grasp_min_contact_span_px=top_orientation.get("min_contact_span_px", 3.0),
        grasp_circle_circularity_threshold=top_orientation.get(
            "circle_circularity_threshold", 0.985
        ),
        grasp_circle_min_vertices=top_orientation.get("circle_min_vertices", 8),
        grasp_min_opening_mm=top_orientation.get("min_opening_mm", None),
        grasp_max_opening_mm=top_orientation.get("max_opening_mm", None),

        dynamic_preopen_enabled=top_preopen.get("enabled", True),
        preopen_margin_mm=top_preopen.get("margin_mm", 6.0),
        preopen_command_min_mm=top_preopen.get("min_command_mm", None),
        preopen_command_max_mm=top_preopen.get("max_command_mm", None),
        preopen_profile_kind=top_preopen.get("profile_kind", "cad_four_bar"),
        preopen_configured=bool(top_preopen.get("configured", False)),
        preopen_closed_servo_deg=float(top_preopen.get("closed_servo_deg", -35.0)),
        preopen_standby_servo_deg=float(top_preopen.get("standby_servo_deg", -95.0)),

        gripper_collision_enabled=top_gripper_collision.get("enabled", True),
        gripper_finger_length_mm=top_gripper_collision.get("finger_length_mm", 20.0),
        gripper_finger_thickness_mm=top_gripper_collision.get(
            "finger_thickness_mm", 9.0
        ),
        gripper_finger_longitudinal_offset_mm=top_gripper_collision.get(
            "finger_longitudinal_offset_mm", 0.0
        ),
        gripper_opening_uncertainty_mm=top_gripper_collision.get(
            "opening_uncertainty_mm", 1.5
        ),
        gripper_object_margin_mm=top_gripper_collision.get("object_margin_mm", 1.0),
        gripper_wall_margin_mm=top_gripper_collision.get("wall_margin_mm", 2.0),
        gripper_fallback_opening_mm=top_gripper_collision.get(
            "fallback_opening_mm", 38.3
        ),
        gripper_forbidden_zones_mm=forbidden_zones_mm,
        gripper_allowed_region_mm=top_gripper_collision.get("allowed_region_mm", None),
        gripper_allowed_polygon_mm=allowed_polygon_mm,
        gripper_object_overlap_tolerance_px=top_gripper_collision.get(
            "object_overlap_tolerance_px", 4
        ),
        gripper_enforce_image_border=top_gripper_collision.get(
            "enforce_image_border", True
        ),
        gripper_image_border_margin_px=top_gripper_collision.get(
            "image_border_margin_px", 2
        ),
        gripper_body_enabled=body_cfg.get("enabled", False),
        gripper_body_width_mm=body_cfg.get("width_mm", 50.0),
        gripper_body_length_mm=body_cfg.get("length_mm", 20.0),
        gripper_body_longitudinal_offset_mm=body_cfg.get(
            "longitudinal_offset_mm", -25.0
        ),

        # Debug viewer: force rejection records ON.
        gripper_debug_log_rejections=True,

        grasp_search_enabled=top_grasp_search.get("enabled", True),
        grasp_search_yaw_offsets_deg=top_grasp_search.get(
            "yaw_offsets_deg", [0, -5, 5, -10, 10, -15, 15]
        ),
        grasp_search_longitudinal_offsets_mm=top_grasp_search.get(
            "longitudinal_offsets_mm", [0, -2, 2, -4, 4]
        ),
        grasp_search_closing_offsets_mm=top_grasp_search.get(
            "closing_offsets_mm", [0]
        ),
        grasp_search_hard_min_clearance_mm=top_grasp_search.get(
            "hard_min_clearance_mm", 1.0
        ),
        grasp_search_preferred_clearance_mm=top_grasp_search.get(
            "preferred_clearance_mm", 4.0
        ),
        grasp_search_min_target_overlap_mm=top_grasp_search.get(
            "min_target_overlap_mm", 1.5
        ),
        grasp_search_min_side_gap_mm=top_grasp_search.get("min_side_gap_mm", 0.25),
        grasp_search_yaw_penalty_per_deg=top_grasp_search.get(
            "yaw_penalty_per_deg", 0.12
        ),
        grasp_search_center_penalty_per_mm=top_grasp_search.get(
            "center_penalty_per_mm", 0.35
        ),
        grasp_search_max_trials_per_object=top_grasp_search.get(
            "max_trials_per_object", 40
        ),

        x_limits_mm=x_limits,
        y_limits_mm=y_limits,
        yaw_tcp_compensator=yaw_tcp_compensator,

        # Current dual-branch build supports these. Older builds filter them out.
        branch1_axis_correction_deg=float(
            top_tcp_comp.get("branch1_axis_correction_deg", 0.0)
        ),
        branch1_closing_bias_mm=float(
            top_tcp_comp.get("branch1_closing_bias_mm", 0.0)
        ),
        branch1_longitudinal_bias_mm=float(
            top_tcp_comp.get("branch1_longitudinal_bias_mm", 0.0)
        ),
    )

    signature = inspect.signature(TopPickPipeline.__init__)
    supported = {name for name in signature.parameters if name != "self"}
    kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return TopPickPipeline(**kwargs)


def choose_display_best(result):
    if result is None or not result.candidates:
        return None

    if result.grasp_pixel is not None:
        gx = float(result.grasp_pixel[0])
        gy = float(result.grasp_pixel[1])
        return min(
            result.candidates,
            key=lambda candidate:
                (float(candidate.grasp_pixel[0]) - gx) ** 2
                + (float(candidate.grasp_pixel[1]) - gy) ** 2,
        )

    return max(
        result.candidates,
        key=lambda candidate: float(getattr(candidate, "score", 0.0)),
    )


def draw_rejection(image, record, *, show_label):
    polygons = record.get("footprint_polygons_px", []) or []

    for polygon in polygons:
        fill_polygon_alpha(image, polygon, (0, 0, 255), 0.12)
        draw_polygon(image, polygon, (0, 0, 255), 2)

    grasp_pixel = record.get("grasp_pixel", None)
    if grasp_pixel is None:
        return

    point = (int(grasp_pixel[0]), int(grasp_pixel[1]))
    draw_cross(image, point, (0, 0, 255), 6, 2)

    if show_label:
        # OpenCV Hershey font does not render Chinese reliably.
        # Use the original ASCII rejection code on the image.
        put_label(
            image,
            str(record.get("reason", "unknown")),
            (point[0] + 7, point[1] - 8),
            color=(255, 255, 255),
            background=(0, 0, 100),
            scale=0.40,
        )


def draw_safe_alternative(image, record):
    polygons = record.get("footprint_polygons_px", []) or []
    for polygon in polygons:
        fill_polygon_alpha(image, polygon, (0, 190, 255), 0.08)
        draw_polygon(image, polygon, (0, 190, 255), 1)

    grasp_pixel = record.get("grasp_pixel", None)
    if grasp_pixel is not None:
        draw_cross(
            image,
            (int(grasp_pixel[0]), int(grasp_pixel[1])),
            (0, 190, 255),
            4,
            1,
        )


def draw_best_candidate(image, candidate, *, show_object_polygon):
    if show_object_polygon:
        draw_polygon(image, candidate.polygon, (0, 255, 0), 2)

    for polygon in candidate.gripper_footprint_polygons_px:
        fill_polygon_alpha(image, polygon, (0, 255, 0), 0.20)
        draw_polygon(image, polygon, (0, 255, 0), 2)

    grasp_pixel = (int(candidate.grasp_pixel[0]), int(candidate.grasp_pixel[1]))
    draw_cross(image, grasp_pixel, (0, 255, 0), 8, 2)

    closing_axis_image_deg = getattr(candidate, "closing_axis_image_deg", None)
    if closing_axis_image_deg is not None:
        draw_axis(
            image,
            grasp_pixel,
            float(closing_axis_image_deg),
            45.0,
            (255, 0, 255),
            2,
        )

    label = (
        f"BEST obj={candidate.object_index} "
        f"yaw={candidate.yaw_deg:+.1f} "
        f"gap={candidate.gripper_clearance_mm:.1f}mm"
    )

    yaw_branch = getattr(candidate, "yaw_branch", None)
    physical_yaw = getattr(candidate, "physical_yaw_deg", None)

    if yaw_branch is not None:
        label += f" B{int(yaw_branch)}"
    if physical_yaw is not None:
        label += f" phy={float(physical_yaw):+.1f}"

    put_label(
        image,
        label,
        (grasp_pixel[0] + 9, grasp_pixel[1] + 22),
        background=(0, 80, 0),
        scale=0.43,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Top-camera gripper footprint/rejection viewer. "
            "It NEVER opens STM32 and NEVER moves the robot."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--show-safe-alternatives", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()

    cfg = load_runtime_config(ROOT, config_path)

    # Correct production TopCamera construction:
    # configure(...) -> TopCamera() -> open()
    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()

    pipeline = build_top_pipeline(cfg)

    window_name = "TOP GRIPPER FOOTPRINT + REJECTION DEBUG"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if args.fullscreen:
        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
    else:
        cv2.resizeWindow(window_name, 1500, 900)

    show_all = bool(args.show_all)
    show_safe_alternatives = bool(args.show_safe_alternatives)
    show_rejections = True
    show_rejection_labels = True
    show_object_polygon = True
    paused = False

    last_frame = None
    last_result = None
    save_index = 0

    fps_t0 = time.perf_counter()
    fps_count = 0
    fps_display = 0.0

    print("=" * 78)
    print("TOP GRIPPER FOOTPRINT + REJECTION DEBUG")
    print("This program does NOT open STM32 and cannot move the robot.")
    print("=" * 78)
    print("GREEN  : best safe gripper footprint")
    print("YELLOW : safe but unselected search pose")
    print("RED    : rejected pose / rejected footprint")
    print("A : show/hide all per-object safe candidates")
    print("Y : show/hide safe search alternatives")
    print("R : show/hide rejected poses")
    print("L : show/hide rejection labels")
    print("O : show/hide object polygon")
    print("P : pause/resume")
    print("I : print current frame details")
    print("S : save current debug frame")
    print("Q/ESC : quit")
    print("=" * 78)

    try:
        camera.open()

        while True:
            if not paused or last_frame is None:
                frame = camera.read()
                if frame is None:
                    continue

                last_frame = frame.copy()
                last_result = pipeline.process(frame)

                fps_count += 1
                now = time.perf_counter()
                elapsed = now - fps_t0
                if elapsed >= 1.0:
                    fps_display = fps_count / elapsed
                    fps_count = 0
                    fps_t0 = now

            frame = last_frame.copy()
            result = last_result
            best = choose_display_best(result)

            if result is not None and show_all:
                for candidate in result.candidates:
                    if candidate is best:
                        continue

                    if show_object_polygon:
                        draw_polygon(frame, candidate.polygon, (0, 190, 255), 1)

                    for polygon in candidate.gripper_footprint_polygons_px:
                        fill_polygon_alpha(frame, polygon, (0, 190, 255), 0.07)
                        draw_polygon(frame, polygon, (0, 190, 255), 1)

            if show_safe_alternatives:
                for record in (
                    getattr(pipeline, "last_gripper_search_alternatives", [])
                    or []
                ):
                    draw_safe_alternative(frame, record)

            rejection_records = (
                getattr(pipeline, "last_gripper_rejections", [])
                or []
            )

            if show_rejections:
                for record in rejection_records:
                    draw_rejection(
                        frame,
                        record,
                        show_label=show_rejection_labels,
                    )

            if best is not None:
                draw_best_candidate(
                    frame,
                    best,
                    show_object_polygon=show_object_polygon,
                )

            reason_counter = Counter(
                str(record.get("reason", "unknown"))
                for record in rejection_records
            )

            panel_lines = [
                (
                    f"FPS={fps_display:.1f} "
                    f"infer={0.0 if result is None else result.inference_ms:.1f}ms "
                    f"paused={paused}"
                ),
                (
                    f"objects={0 if result is None else result.object_count} "
                    f"candidates={0 if result is None else result.candidate_count} "
                    f"READY={False if result is None else result.ready}"
                ),
                (
                    f"show all={show_all} yellow={show_safe_alternatives} "
                    f"reject={show_rejections} labels={show_rejection_labels}"
                ),
            ]

            if result is not None:
                panel_lines.append(
                    f"consensus={result.votes}/{result.required_hits} "
                    f"reason={result.reason}"
                )

            if best is not None:
                line = (
                    f"BEST xy=({best.x_mm:.1f},{best.y_mm:.1f}) "
                    f"yaw={best.yaw_deg:+.1f}"
                )

                yaw_branch = getattr(best, "yaw_branch", None)
                physical_yaw = getattr(best, "physical_yaw_deg", None)

                if yaw_branch is not None:
                    line += f" B{int(yaw_branch)}"
                if physical_yaw is not None:
                    line += f" physical={float(physical_yaw):+.1f}"

                panel_lines.append(line)
                panel_lines.append(
                    f"preopen={best.preopen_mm}mm "
                    f"gripper_gap={best.gripper_clearance_mm:.2f}mm "
                    f"trials={best.search_trial_count} "
                    f"safe={best.search_safe_count}"
                )

            if reason_counter:
                panel_lines.append(
                    "shown rejects: "
                    + " | ".join(
                        f"{reason}:{count}"
                        for reason, count in reason_counter.most_common(5)
                    )
                )

            put_panel(
                frame,
                panel_lines,
                x=10,
                y=10,
                line_height=22,
                scale=0.50,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("a"), ord("A")):
                show_all = not show_all
            elif key in (ord("y"), ord("Y")):
                show_safe_alternatives = not show_safe_alternatives
            elif key in (ord("r"), ord("R")):
                show_rejections = not show_rejections
            elif key in (ord("l"), ord("L")):
                show_rejection_labels = not show_rejection_labels
            elif key in (ord("o"), ord("O")):
                show_object_polygon = not show_object_polygon
            elif key in (ord("p"), ord("P")):
                paused = not paused
            elif key in (ord("s"), ord("S")):
                save_index += 1
                save_dir = (
                    ROOT / "outputs" / "debug" / "top_gripper_rejection"
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"gripper_debug_{save_index:03d}.png"
                cv2.imwrite(str(save_path), frame)
                print(f"[SAVE] {save_path}")
            elif key in (ord("i"), ord("I")):
                print("")
                print("=" * 78)
                print("[FRAME DEBUG]")

                if result is None:
                    print("result = None")
                else:
                    print("objects    =", result.object_count)
                    print("candidates =", result.candidate_count)
                    print(
                        "READY      =",
                        result.ready,
                        f"{result.votes}/{result.required_hits}",
                    )
                    print("reason     =", result.reason)

                if best is None:
                    print("BEST       = None")
                else:
                    print("BEST obj   =", best.object_index)
                    print("pixel      =", best.grasp_pixel)
                    print(
                        "robot XY   =",
                        f"({best.x_mm:.3f}, {best.y_mm:.3f}) mm",
                    )
                    print("yaw        =", f"{best.yaw_deg:+.3f} deg")
                    if hasattr(best, "yaw_branch"):
                        print("branch     =", best.yaw_branch)
                    if hasattr(best, "physical_yaw_deg"):
                        print(
                            "physical   =",
                            f"{best.physical_yaw_deg:+.3f} deg",
                        )
                    print("preopen    =", best.preopen_mm)
                    print(
                        "clearance  =",
                        f"{best.gripper_clearance_mm:.3f} mm",
                    )
                    print("trials     =", best.search_trial_count)
                    print("safe       =", best.search_safe_count)
                    print("score      =", f"{best.score:.4f}")

                print("")
                print("[REJECTIONS STORED FOR PREVIEW]")

                for index, record in enumerate(
                    rejection_records,
                    start=1,
                ):
                    print(
                        f"{index:02d}. "
                        f"obj={record.get('object_index')} "
                        f"px={record.get('grasp_pixel')} "
                        f"yaw={record.get('yaw_deg')} "
                        f"opening={record.get('opening_mm')} "
                        f"reason={reason_text(record.get('reason', 'unknown'))}"
                    )

                print("=" * 78)

    finally:
        try:
            camera.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
