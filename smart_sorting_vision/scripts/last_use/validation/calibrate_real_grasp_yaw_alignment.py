from __future__ import annotations

import argparse
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

from communication.stm32_link import STM32Link
from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.grasp.orientation_planner import polygon_long_axis_image_deg

# Reuse the already-tested STOP/CALIB_MOVE/CALIB_YAW and pipeline construction.
from scripts.last_use.validation.validate_real_object_alignment import (
    UserStopRequested,
    axis_to_servo_deg,
    build_compensator,
    build_pipeline,
    ensure_stopped,
    normalize_axial_deg,
    send_calib_move_and_wait,
    send_calib_yaw_and_wait,
)


def draw_axis(
    image: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
    *,
    length_px: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    theta = math.radians(float(angle_deg))
    dx = 0.5 * float(length_px) * math.cos(theta)
    dy = 0.5 * float(length_px) * math.sin(theta)
    p1 = (int(round(center[0] - dx)), int(round(center[1] - dy)))
    p2 = (int(round(center[0] + dx)), int(round(center[1] + dy)))
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Calibrate a constant visual grasp-yaw alignment offset with one elongated real object. "
            "Only CALIB_MOVE + CALIB_YAW are sent; no Z descent and no claw closing."
        )
    )
    p.add_argument("--config", default=str(ROOT / "config" / "runtime_config.yaml"))
    p.add_argument("--ack-timeout", type=float, default=1.0)
    p.add_argument("--reach-timeout", type=float, default=4.0)
    p.add_argument("--settle", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config(ROOT, Path(args.config))
    top_cfg = cfg["vision"]["top"]
    orient_cfg = top_cfg.setdefault("grasp_orientation", {})

    # The purpose of this tool is to solve the offset itself. Do not recursively
    # include an old value while estimating a new one.
    previous_offset = float(orient_cfg.get("yaw_alignment_offset_deg", 0.0))
    orient_cfg["yaw_alignment_offset_deg"] = 0.0

    comp_cfg = top_cfg.get("yaw_tcp_compensation", {})
    if not bool(comp_cfg.get("enabled", False)):
        raise RuntimeError(
            "vision.top.yaw_tcp_compensation.enabled is false; keep the verified TCP compensation enabled for this test."
        )
    comp = build_compensator(top_cfg)

    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()
    pipeline = build_pipeline(cfg, comp)
    link = STM32Link(device=cfg["stm32"]["device"], baudrate=cfg["stm32"]["baud"])

    out_dir = ROOT / "data" / "calibration" / "top_camera" / "robot_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "grasp_yaw_alignment_report.json"

    frozen: dict | None = None
    frozen_frame: np.ndarray | None = None
    offset_deg = 0.0
    step_deg = 1.0
    window = "Grasp Yaw Alignment - J/L offset - M move - ENTER save - R reacquire - Q quit"

    try:
        camera.open()
        link.open()
        ensure_stopped(link)

        print("\n[SAFETY] Only XY + Yaw are moved. No Z descent and no claw closing.")
        print("[OBJECT] Use ONE elongated rectangle with a clear long/short axis.")
        print("[GOAL] Make the REAL gripper closing direction parallel to the object's SHORT axis.")
        print("       Equivalently, the two finger bodies should be parallel to the object's LONG sides.")
        print("[KEY] J/L=-/+ offset | 1=0.5deg 2=1deg 3=2deg 4=5deg | M=move | ENTER=save | R=reacquire | Q=quit")
        print(f"[INFO] Config currently had yaw_alignment_offset_deg={previous_offset:+.2f}; calibration starts from 0.00 deg.\n")

        while True:
            frame = camera.read()
            show = frame.copy()

            if frozen is None:
                result = pipeline.process(frame)
                status = (
                    f"objects={result.object_count} candidates={result.candidate_count} "
                    f"votes={result.votes}/{result.required_hits}"
                )
                cv2.putText(show, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                if (
                    result.ready
                    and result.grasp_pixel is not None
                    and result.yaw_deg is not None
                    and result.candidates
                ):
                    # This calibration is intentionally for one object. Pick the
                    # highest-score candidate from the current frame for drawing.
                    candidate = max(result.candidates, key=lambda c: c.score)
                    if not candidate.orientation_reliable:
                        cv2.putText(
                            show,
                            "Orientation unreliable: use a more elongated rectangle",
                            (12, 58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        gx, gy = map(int, result.grasp_pixel)
                        raw_x, raw_y = pipeline.transform.pixel_to_robot(gx, gy, apply_tcp_offset=True)
                        base_yaw = normalize_axial_deg(float(result.yaw_deg))
                        long_img_deg, aspect = polygon_long_axis_image_deg(candidate.polygon)
                        frozen = {
                            "grasp_pixel": [gx, gy],
                            "center_pixel": [float(candidate.center_pixel[0]), float(candidate.center_pixel[1])],
                            "raw_h_x_mm": float(raw_x),
                            "raw_h_y_mm": float(raw_y),
                            "base_planned_yaw_deg": float(base_yaw),
                            "object_long_axis_robot_deg": float(candidate.object_long_axis_deg),
                            "object_long_axis_image_deg": float(long_img_deg),
                            "aspect_ratio": float(aspect),
                            "class_name": candidate.class_name,
                            "confidence": float(candidate.confidence),
                        }
                        frozen_frame = frame.copy()
                        offset_deg = 0.0
                        print(
                            "[FREEZE] "
                            f"center=({candidate.center_pixel[0]:.1f},{candidate.center_pixel[1]:.1f}) "
                            f"grasp=({gx},{gy}) long_robot={candidate.object_long_axis_deg:+.2f} "
                            f"base_closing_axis={base_yaw:+.2f} aspect={aspect:.2f}"
                        )
            else:
                if frozen_frame is not None:
                    show = frozen_frame.copy()

                cx, cy = frozen["center_pixel"]
                gx, gy = frozen["grasp_pixel"]
                long_img = float(frozen["object_long_axis_image_deg"])
                short_img = long_img + 90.0

                # BLUE = polygon centroid; RED = selected safe grasp point.
                cv2.drawMarker(show, (int(round(cx)), int(round(cy))), (255, 0, 0), cv2.MARKER_CROSS, 24, 2)
                cv2.drawMarker(show, (int(gx), int(gy)), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                # GREEN = object long axis; YELLOW = desired closing direction in image geometry.
                draw_axis(show, (cx, cy), long_img, length_px=150, color=(0, 255, 0), thickness=2)
                draw_axis(show, (cx, cy), short_img, length_px=120, color=(0, 255, 255), thickness=2)

                target_yaw = normalize_axial_deg(float(frozen["base_planned_yaw_deg"]) + offset_deg)
                valid = True
                error_text = ""
                try:
                    q = axis_to_servo_deg(target_yaw)
                    applied = comp.apply(
                        float(frozen["raw_h_x_mm"]),
                        float(frozen["raw_h_y_mm"]),
                        target_yaw,
                    )
                except Exception as exc:
                    valid = False
                    q = float("nan")
                    applied = None
                    error_text = str(exc)

                lines = [
                    f"BLUE centroid=({cx:.1f},{cy:.1f})  RED grasp=({gx},{gy})",
                    f"GREEN long axis | YELLOW desired closing axis",
                    f"base yaw={frozen['base_planned_yaw_deg']:+.2f}  offset={offset_deg:+.2f}  target={target_yaw:+.2f}",
                    f"step={step_deg:.1f} deg  J/L=-/+  1/2/3/4 step  M=move  ENTER=save  R=reacquire",
                ]
                if valid and applied is not None:
                    lines.append(
                        f"CMD XY=({applied.x_mm:.2f},{applied.y_mm:.2f}) servo={q:+.2f}"
                    )
                else:
                    lines.append("INVALID: " + error_text[:90])

                for i, text in enumerate(lines):
                    color = (255, 255, 255) if i < 4 else ((255, 255, 255) if valid else (0, 0, 255))
                    cv2.putText(show, text, (12, 28 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

            cv2.imshow(window, show)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                pipeline.reset()
                frozen = None
                frozen_frame = None
                offset_deg = 0.0
                continue
            if frozen is None:
                continue

            if key == ord("1"):
                step_deg = 0.5
            elif key == ord("2"):
                step_deg = 1.0
            elif key == ord("3"):
                step_deg = 2.0
            elif key == ord("4"):
                step_deg = 5.0
            elif key == ord("j"):
                offset_deg = normalize_axial_deg(offset_deg - step_deg)
                print(f"[OFFSET] {offset_deg:+.2f} deg")
            elif key == ord("l"):
                offset_deg = normalize_axial_deg(offset_deg + step_deg)
                print(f"[OFFSET] {offset_deg:+.2f} deg")
            elif key == ord("m"):
                target_yaw = normalize_axial_deg(float(frozen["base_planned_yaw_deg"]) + offset_deg)
                q = axis_to_servo_deg(target_yaw)
                applied = comp.apply(
                    float(frozen["raw_h_x_mm"]),
                    float(frozen["raw_h_y_mm"]),
                    target_yaw,
                )
                print(
                    f"[MOVE] offset={offset_deg:+.2f} -> axis={target_yaw:+.2f}, "
                    f"servo={q:+.2f}, XY=({applied.x_mm:.2f},{applied.y_mm:.2f})"
                )
                send_calib_move_and_wait(
                    link,
                    float(applied.x_mm),
                    float(applied.y_mm),
                    ack_timeout=args.ack_timeout,
                    reach_timeout=args.reach_timeout,
                )
                send_calib_yaw_and_wait(
                    link,
                    q,
                    ack_timeout=args.ack_timeout,
                    reach_timeout=args.reach_timeout,
                )
                time.sleep(max(0.0, args.settle))
                print("[CHECK] Adjust J/L until REAL closing direction matches the object's SHORT axis, then ENTER.")
            elif key in (10, 13):
                target_yaw = normalize_axial_deg(float(frozen["base_planned_yaw_deg"]) + offset_deg)
                report = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "method": "one elongated object -> production orientation plan with zero offset -> manually tune constant physical yaw alignment offset",
                    "safety_note": "Only CALIB_MOVE + CALIB_YAW were used; no Z descent and no claw closing.",
                    "selected_offset_deg": float(offset_deg),
                    "base_planned_yaw_deg": float(frozen["base_planned_yaw_deg"]),
                    "final_target_yaw_deg": float(target_yaw),
                    "object_long_axis_robot_deg": float(frozen["object_long_axis_robot_deg"]),
                    "object_long_axis_image_deg": float(frozen["object_long_axis_image_deg"]),
                    "aspect_ratio": float(frozen["aspect_ratio"]),
                    "raw_h_xy_mm": [float(frozen["raw_h_x_mm"]), float(frozen["raw_h_y_mm"])],
                    "config_to_set": {
                        "vision.top.grasp_orientation.yaw_alignment_offset_deg": float(offset_deg)
                    },
                    "previous_config_offset_deg": float(previous_offset),
                }
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[SAVE] selected_offset_deg={offset_deg:+.3f} deg")
                print(f"[SAVE] report: {report_path}")
                print("[NEXT] Put this value into vision.top.grasp_orientation.yaw_alignment_offset_deg, then rerun real-object alignment.")
                return 0

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
