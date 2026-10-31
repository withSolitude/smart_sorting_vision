from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.pipeline.top_pick_pipeline import TopPickPipeline
from vision.grasp.orientation_planner import polygon_long_axis_image_deg
from vision.calibration.yaw_servo import YawServoCalibration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Preview top segmentation + dynamic pickup yaw only. "
            "This script NEVER opens the STM32 serial link and NEVER commands any motor."
        )
    )
    p.add_argument("--config", default=str(ROOT / "config" / "runtime_config.yaml"))
    p.add_argument(
        "--calibration",
        default=None,
        help="Optional pixel_to_robot.json/candidate path. Default: runtime_config calibration.pixel_to_robot",
    )
    return p.parse_args()


def _draw_axis(img, center, angle_deg, length_px, color, thickness=2):
    cx, cy = map(float, center)
    t = math.radians(float(angle_deg))
    dx = 0.5 * float(length_px) * math.cos(t)
    dy = 0.5 * float(length_px) * math.sin(t)
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(ROOT, args.config)
    top_model_cfg = cfg["model"]["top"]
    top_vision_cfg = cfg["vision"]["top"]
    top_consensus = top_vision_cfg["consensus"]
    top_orientation = top_vision_cfg.get("grasp_orientation", {})
    top_preopen = top_orientation.get("dynamic_preopen", {}) or {}
    yaw_servo = YawServoCalibration.from_config(cfg["calibration"]["yaw_servo"])
    calibration = args.calibration or cfg["calibration"]["pixel_to_robot"]

    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()
    pipeline = TopPickPipeline(
        model_path=top_model_cfg["path"],
        calibration_path=calibration,
        camera_config=cfg["camera"]["top"],
        conf=top_model_cfg["conf"],
        imgsz=top_model_cfg["imgsz"],
        device=top_model_cfg["device"],
        obstacle_margin=top_vision_cfg["obstacle_margin_px"],
        border_margin=top_vision_cfg["border_margin_px"],
        min_safe_area=top_vision_cfg["min_safe_area_px"],
        min_clearance=top_vision_cfg["min_clearance_px"],
        shape_core_config=top_vision_cfg.get("shape_core"),
        consensus_window=top_consensus["window"],
        consensus_required=top_consensus["required"],
        max_track_center_shift_px=top_consensus["max_track_center_shift_px"],
        max_grasp_spread_mm=top_consensus["max_grasp_spread_mm"],
        reset_after_misses=top_consensus["reset_after_misses"],
        orientation_enabled=top_orientation.get("enabled", True),
        grip_long_sides=top_orientation.get("grip_long_sides", True),
        min_orientation_aspect_ratio=top_orientation.get("min_aspect_ratio", 1.20),
        default_grasp_axis_robot_deg=top_orientation.get("default_grasp_axis_robot_deg", 0.0),
        max_yaw_spread_deg=top_orientation.get("max_yaw_spread_deg", 12.0),
        dynamic_preopen_enabled=top_preopen.get("enabled", True),
        preopen_margin_mm=top_preopen.get("margin_mm", 6.0),
        preopen_command_min_mm=top_preopen.get("min_command_mm", None),
        preopen_command_max_mm=top_preopen.get("max_command_mm", None),
        preopen_profile_kind=top_preopen.get("profile_kind", ""),
        preopen_configured=top_preopen.get("configured", False),
        preopen_closed_servo_deg=top_preopen.get("closed_servo_deg", -15.0),
        preopen_standby_servo_deg=top_preopen.get("standby_servo_deg", -95.0),
        x_limits_mm=cfg["mechanics"]["workspace_mm"]["x"],
        y_limits_mm=cfg["mechanics"]["workspace_mm"]["y"],
        yaw_servo_calibration=yaw_servo,
    )

    win = "TOP DYNAMIC YAW PREVIEW - Q quit / R reset consensus"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        camera.open()
        print("This preview does NOT open STM32 and cannot move the robot.")
        print("Green box/line = object long axis; yellow cross = grasp point.")
        print("Text grasp_axis = desired gripper CLOSING axis in robot XY coordinates.")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            result = pipeline.process(frame)
            vis = frame.copy()

            for c in result.candidates:
                poly = np.asarray(c.polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [poly], True, (0, 220, 0), 2, cv2.LINE_AA)
                gx, gy = c.grasp_pixel
                cv2.drawMarker(vis, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                try:
                    long_image_deg, _ = polygon_long_axis_image_deg(c.polygon)
                    _draw_axis(vis, (gx, gy), long_image_deg, 90, (0, 220, 0), 3)
                    # In image space this dotted-looking perpendicular is only a visual hint;
                    # the actual command angle is computed after Homography into robot XY.
                    _draw_axis(vis, (gx, gy), long_image_deg + 90.0, 60, (255, 200, 0), 2)
                except Exception:
                    pass
                label = (
                    f"AR={c.aspect_ratio:.2f} longXY={c.object_long_axis_deg:+.1f} "
                    f"grasp_axis={c.yaw_deg:+.1f}deg "
                    f"{'ORI' if c.orientation_reliable else 'DEFAULT'}"
                )
                cv2.putText(vis, label, (max(5, gx - 170), max(22, gy - 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

            status = (
                f"READY={result.ready} votes={result.votes}/{result.required_hits} "
                f"reason={result.reason}"
            )
            if result.ready and result.yaw_deg is not None:
                status += f" | SEND grasp_axis={result.yaw_deg:+.1f}deg"
            cv2.putText(vis, status, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('r'), ord('R')):
                pipeline.reset()
        return 0
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
