#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.calibration.pixel_to_robot import PixelToRobotTransform


@dataclass
class ClickPoint:
    px: int
    py: int
    rx: float
    ry: float


def _signed_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _order_quad(points: list[ClickPoint]) -> tuple[np.ndarray, np.ndarray]:
    if len(points) != 4:
        raise ValueError("exactly four points are required")
    px = np.asarray([[p.px, p.py] for p in points], dtype=np.float64)
    rb = np.asarray([[p.rx, p.ry] for p in points], dtype=np.float64)
    center = px.mean(axis=0)
    order = np.argsort(np.arctan2(px[:, 1] - center[1], px[:, 0] - center[0]))
    px = px[order]
    rb = rb[order]
    if _signed_area(px) > 0.0:
        px = px[::-1]
        rb = rb[::-1]
    start = int(np.argmin(px[:, 0] + px[:, 1]))
    return np.roll(px, -start, axis=0), np.roll(rb, -start, axis=0)


def _yaml_line(robot_poly: np.ndarray) -> str:
    pts = ", ".join(f"[{float(x):.2f}, {float(y):.2f}]" for x, y in robot_poly)
    return f"allowed_polygon_mm: [{pts}]"


def main() -> None:
    cfg = load_runtime_config(ROOT, ROOT / "config" / "runtime_config.yaml")
    TopCamera.configure(cfg["camera"]["top"], calibration_file=cfg["calibration"]["top_intrinsics"])
    camera = TopCamera()
    camera.open()
    transform = PixelToRobotTransform.load(cfg["calibration"]["pixel_to_robot"])

    window = "Tray Allowed Polygon 4-Point Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    points: list[ClickPoint] = []
    final_px: np.ndarray | None = None
    final_rb: np.ndarray | None = None

    def on_mouse(event, x, y, flags, userdata):
        nonlocal points, final_px, final_rb
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 4:
                points = []
                final_px = None
                final_rb = None
            try:
                rx, ry = transform.pixel_to_robot(x, y, apply_tcp_offset=False)
            except Exception as exc:
                print(f"[CLICK][ERROR] {exc}")
                return
            points.append(ClickPoint(int(x), int(y), float(rx), float(ry)))
            print(f"[CLICK {len(points)}/4] pixel=({x},{y}) robot=({rx:.3f},{ry:.3f}) mm")
            if len(points) == 4:
                px, rb = _order_quad(points)
                if abs(_signed_area(px)) < 50.0 or abs(_signed_area(rb)) < 10.0:
                    print("[TRAY][REJECT] 四点面积太小或点重复，请重标")
                    points = []
                    return
                final_px, final_rb = px, rb
                print("\n[TRAY] 四点允许区域完成。请把这一行放到 vision.top.gripper_collision：")
                print("  " + _yaml_line(final_rb))
                print("[IMPORTANT] 托盘四边形是允许区域，不要再填入 mechanics.forbidden_zones_mm。\n")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                final_px = None
                final_rb = None
                print(f"[CLICK] 撤销，当前 {len(points)}/4")

    cv2.setMouseCallback(window, on_mouse)

    print("=" * 74)
    print("托盘可抓区域四点标定（不是禁入区）")
    print("左键依次点托盘内部平面四个真实边界角点，顺序任意；第4点自动排序。")
    print("右键撤销；C清空；S打印YAML；Q/ESC退出。")
    print("四个点尽量点在夹爪真正不能越过的内侧边界，不要点外侧墙顶。")
    print("=" * 74)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            vis = frame.copy()

            if final_px is not None:
                p = np.round(final_px).astype(np.int32).reshape(-1, 1, 2)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [p], (255, 255, 0))
                cv2.addWeighted(overlay, 0.10, vis, 0.90, 0, vis)
                cv2.polylines(vis, [p], True, (255, 255, 0), 2)
                cv2.putText(vis, "ALLOWED TRAY", tuple(np.mean(final_px, axis=0).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            for idx, pt in enumerate(points, start=1):
                cv2.circle(vis, (pt.px, pt.py), 6, (0, 255, 255), -1)
                cv2.putText(vis, str(idx), (pt.px + 7, pt.py - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if 2 <= len(points) < 4:
                arr = np.asarray([[p.px, p.py] for p in points], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [arr], False, (0, 255, 255), 2)

            cv2.putText(vis, f"tray points {len(points)}/4", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('c'), ord('C')):
                points = []
                final_px = None
                final_rb = None
                print("[TRAY] 已清空")
            elif key in (ord('s'), ord('S')) and final_rb is not None:
                print("  " + _yaml_line(final_rb))
    finally:
        if final_rb is not None:
            print("\n[FINAL YAML]\nvision:\n  top:\n    gripper_collision:")
            print("      " + _yaml_line(final_rb))
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
