#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.pipeline.top_pick_pipeline import TopPickPipeline
from vision.calibration.yaw_tcp_compensation import YawTcpCompensator
from vision.calibration.yaw_servo import YawServoCalibration


def build_pipeline(cfg: dict, *, debug_rejections: bool) -> TopPickPipeline:
    top_model = cfg["model"]["top"]
    top_cfg = cfg["vision"]["top"]
    consensus = top_cfg["consensus"]
    orientation = top_cfg.get("grasp_orientation", {}) or {}
    preopen = orientation.get("dynamic_preopen", {}) or {}
    gc = top_cfg.get("gripper_collision", {}) or {}
    search = gc.get("grasp_search", {}) or {}
    body = gc.get("body", {}) or {}
    yaw_servo = YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"])

    mechanics = cfg.get("mechanics", {}) or {}
    x_limits = mechanics["workspace_mm"]["x"]
    y_limits = mechanics["workspace_mm"]["y"]
    allowed_polygon_mm = gc.get("allowed_polygon_mm", mechanics.get("allowed_polygon_mm", None))

    tcp_cfg = top_cfg.get("yaw_tcp_compensation", {}) or {}
    yaw_comp = None
    if bool(tcp_cfg.get("enabled", False)):
        d0 = tcp_cfg.get("offset_axis0_mm", [0.0, 0.0])
        yaw_comp = YawTcpCompensator(
            reference_axis_deg=float(tcp_cfg["reference_axis_deg"]),
            offset_axis0_x_mm=float(d0[0]),
            offset_axis0_y_mm=float(d0[1]),
            min_axis_deg=float(tcp_cfg.get("min_axis_deg", -87.0)),
            max_axis_deg=float(tcp_cfg.get("max_axis_deg", 89.0)),
            visual_to_command_matrix=tcp_cfg.get(
                "visual_to_command_matrix", [[1.0, 0.0], [0.0, 1.0]]
            ),
        )

    return TopPickPipeline(
        model_path=top_model["path"],
        calibration_path=cfg["calibration"]["pixel_to_robot"],
        camera_config=cfg["camera"]["top"],
        conf=top_model["conf"],
        imgsz=top_model["imgsz"],
        device=top_model["device"],
        obstacle_margin=top_cfg["obstacle_margin_px"],
        border_margin=top_cfg["border_margin_px"],
        min_safe_area=top_cfg["min_safe_area_px"],
        min_clearance=top_cfg["min_clearance_px"],
        outward_radial_config=top_cfg.get("outward_radial_compensation"),
        top_intrinsics_path=cfg["calibration"]["top_intrinsics"],
        consensus_window=consensus["window"],
        consensus_required=consensus["required"],
        max_track_center_shift_px=consensus["max_track_center_shift_px"],
        max_grasp_spread_mm=consensus["max_grasp_spread_mm"],
        reset_after_misses=consensus["reset_after_misses"],
        orientation_enabled=orientation.get("enabled", True),
        grip_long_sides=orientation.get("grip_long_sides", True),
        min_orientation_aspect_ratio=orientation.get("min_aspect_ratio", 1.2),
        default_grasp_axis_robot_deg=orientation.get("default_grasp_axis_robot_deg", 0.0),
        yaw_alignment_offset_deg=orientation.get("yaw_alignment_offset_deg", 0.0),
        max_yaw_spread_deg=orientation.get("max_yaw_spread_deg", 12.0),
        grasp_width_slack_ratio=orientation.get("width_slack_ratio", 0.15),
        grasp_support_tolerance_px=orientation.get("support_tolerance_px", 3.0),
        grasp_min_contact_span_px=orientation.get("min_contact_span_px", 3.0),
        grasp_circle_circularity_threshold=orientation.get("circle_circularity_threshold", 0.985),
        grasp_circle_min_vertices=orientation.get("circle_min_vertices", 8),
        grasp_min_opening_mm=orientation.get("min_opening_mm", None),
        grasp_max_opening_mm=orientation.get("max_opening_mm", None),
        dynamic_preopen_enabled=preopen.get("enabled", True),
        preopen_margin_mm=preopen.get("margin_mm", 6.0),
        preopen_command_min_mm=preopen.get("min_command_mm", None),
        preopen_command_max_mm=preopen.get("max_command_mm", None),
        preopen_profile_kind=preopen.get("profile_kind", ""),
        preopen_configured=preopen.get("configured", False),
        preopen_closed_servo_deg=preopen.get("closed_servo_deg", -15.0),
        preopen_standby_servo_deg=preopen.get("standby_servo_deg", -95.0),
        gripper_collision_enabled=gc.get("enabled", True),
        gripper_finger_length_mm=gc.get("finger_length_mm", 20.0),
        gripper_finger_thickness_mm=gc.get("finger_thickness_mm", 9.0),
        gripper_finger_longitudinal_offset_mm=gc.get("finger_longitudinal_offset_mm", 0.0),
        gripper_opening_uncertainty_mm=gc.get("opening_uncertainty_mm", 1.5),
        gripper_object_margin_mm=gc.get("object_margin_mm", 1.0),
        gripper_wall_margin_mm=gc.get("wall_margin_mm", 2.0),
        gripper_fallback_opening_mm=gc.get("fallback_opening_mm", 38.3),
        gripper_forbidden_zones_mm=mechanics.get("forbidden_zones_mm", []),
        gripper_allowed_region_mm=gc.get("allowed_region_mm", None),
        gripper_allowed_polygon_mm=allowed_polygon_mm,
        gripper_object_overlap_tolerance_px=gc.get("object_overlap_tolerance_px", 4),
        gripper_enforce_image_border=gc.get("enforce_image_border", True),
        gripper_image_border_margin_px=gc.get("image_border_margin_px", 2),
        gripper_body_enabled=body.get("enabled", False),
        gripper_body_width_mm=body.get("width_mm", 50.0),
        gripper_body_length_mm=body.get("length_mm", 20.0),
        gripper_body_longitudinal_offset_mm=body.get("longitudinal_offset_mm", -25.0),
        gripper_debug_log_rejections=bool(debug_rejections),
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
        x_limits_mm=x_limits,
        y_limits_mm=y_limits,
        yaw_tcp_compensator=yaw_comp,
        yaw_servo_calibration=yaw_servo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="只看视觉，不驱动STM32：预览真实夹爪footprint")
    parser.add_argument("--log-reject", action="store_true", help="打印因夹爪碰撞被淘汰的候选")
    args = parser.parse_args()

    cfg = load_runtime_config(ROOT, ROOT / "config" / "runtime_config.yaml")
    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()
    camera.open()
    pipeline = build_pipeline(cfg, debug_rejections=args.log_reject)

    print("[PREVIEW] no STM32 command will be sent")
    print("[PREVIEW] green=selected, yellow=safe alternative, red=rejected, cyan=tray allowed polygon")
    print("[TUNE] [/] finger_length -/+1mm | -/= thickness -/+0.5mm")
    print("[TUNE] ,/. object_margin -/+0.25mm | ;/' wall_margin -/+0.25mm")
    print("[TUNE] P print current YAML | q/ESC quit")

    try:
        warm = camera.read()
        if warm is not None:
            pipeline.segmenter.warmup(warm, runs=2)
            pipeline.reset()

        while True:
            frame = camera.read()
            if frame is None:
                continue
            result = pipeline.process(frame)
            vis = frame.copy()

            # 托盘允许区域：青色。这里的四边形内部才是可进入区域。
            allowed_px = pipeline.gripper_footprint.allowed_polygon_pixels()
            if allowed_px is not None:
                ap = np.round(allowed_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [ap], True, (255, 255, 0), 2)

            # 真正的固定禁入区：红色边框。不要把整个托盘填进 forbidden_zones。
            for zone_name, zone_px in pipeline.gripper_footprint.forbidden_zone_polygons_pixels():
                zp = np.round(zone_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [zp], True, (0, 0, 255), 2)

            # 同一物体中“安全但未被选中”的多姿态候选画黄色细框。
            for alternative in pipeline.last_gripper_search_alternatives:
                for fp in alternative.get("footprint_polygons_px", []):
                    ap = np.round(fp).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [ap], True, (0, 220, 220), 1)

            # 被碰撞检查拒绝的候选也画出来，避免 safe_candidates=0 时完全看不到原因。
            for rejected in pipeline.last_gripper_rejections:
                for fp in rejected.get("footprint_polygons_px", []):
                    rp = np.round(fp).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [rp], True, (0, 0, 255), 1)
                gx, gy = rejected.get("grasp_pixel", (0, 0))
                cv2.putText(
                    vis,
                    str(rejected.get("reason", "reject")),
                    (int(gx) + 6, max(16, int(gy) + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            for candidate in result.candidates:
                poly = np.round(candidate.polygon).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [poly], True, (255, 180, 0), 1)

                for fp in candidate.gripper_footprint_polygons_px:
                    p = np.round(fp).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [p], True, (0, 255, 0), 2)

                gx, gy = candidate.grasp_pixel
                cv2.circle(vis, (gx, gy), 4, (0, 0, 255), -1)
                text = (
                    f"yaw={candidate.yaw_deg:+.1f} dY={candidate.search_yaw_offset_deg:+.0f} "
                    f"dL={candidate.search_longitudinal_offset_mm:+.1f} "
                    f"open={candidate.preopen_mm if candidate.preopen_mm is not None else -1:.1f} "
                    f"gap={candidate.gripper_clearance_mm:.1f}mm "
                    f"safe={candidate.search_safe_count}/{candidate.search_trial_count}"
                )
                cv2.putText(
                    vis,
                    text,
                    (gx + 8, max(18, gy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                vis,
                f"objects={result.object_count} safe_candidates={result.candidate_count}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            gf = pipeline.gripper_footprint
            tune_text = (
                f"finger={gf.finger_length_mm:.1f}x{gf.finger_thickness_mm:.1f}mm "
                f"objM={gf.object_margin_mm:.2f} wallM={gf.wall_margin_mm:.2f}"
            )
            cv2.putText(
                vis, tune_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 255), 1, cv2.LINE_AA
            )

            cv2.imshow("Gripper Footprint Preview", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key == ord("["):
                gf.finger_length_mm = max(4.0, gf.finger_length_mm - 1.0)
            elif key == ord("]"):
                gf.finger_length_mm += 1.0
            elif key == ord("-"):
                gf.finger_thickness_mm = max(2.0, gf.finger_thickness_mm - 0.5)
            elif key in (ord("="), ord("+")):
                gf.finger_thickness_mm += 0.5
            elif key == ord(","):
                gf.object_margin_mm = max(0.0, gf.object_margin_mm - 0.25)
            elif key == ord("."):
                gf.object_margin_mm += 0.25
            elif key == ord(";"):
                gf.wall_margin_mm = max(0.0, gf.wall_margin_mm - 0.25)
            elif key == ord("'"):
                gf.wall_margin_mm += 0.25
            elif key in (ord("p"), ord("P")):
                print("\n[GRIPPER_COLLISION YAML]")
                print(f"finger_length_mm: {gf.finger_length_mm:.2f}")
                print(f"finger_thickness_mm: {gf.finger_thickness_mm:.2f}")
                print(f"opening_uncertainty_mm: {gf.opening_uncertainty_mm:.2f}")
                print(f"object_margin_mm: {gf.object_margin_mm:.2f}")
                print(f"wall_margin_mm: {gf.wall_margin_mm:.2f}")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
