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


@dataclass
class CalibratedZone:
    name: str
    robot_polygon: np.ndarray  # Nx2, mm
    pixel_polygon: np.ndarray  # Nx2, px


def _signed_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _order_quad(points: list[ClickPoint]) -> tuple[np.ndarray, np.ndarray]:
    """将四个任意点击顺序的角点整理成不自交的环形顺序。"""
    if len(points) != 4:
        raise ValueError("exactly four points are required")

    px = np.asarray([[p.px, p.py] for p in points], dtype=np.float64)
    rb = np.asarray([[p.rx, p.ry] for p in points], dtype=np.float64)

    center = px.mean(axis=0)
    angles = np.arctan2(px[:, 1] - center[1], px[:, 0] - center[0])
    order = np.argsort(angles)
    px = px[order]
    rb = rb[order]

    # 统一成顺时针显示顺序；对于 fillPoly 来说方向无所谓，统一只是便于核对 YAML。
    if _signed_area(px) > 0.0:
        px = px[::-1]
        rb = rb[::-1]

    # 从图像中最靠左上（x+y最小）的点作为第一个点，输出更稳定、更容易人工核对。
    start = int(np.argmin(px[:, 0] + px[:, 1]))
    px = np.roll(px, -start, axis=0)
    rb = np.roll(rb, -start, axis=0)
    return px, rb


def _format_zone_yaml(zone: CalibratedZone) -> str:
    pts = ", ".join(
        f"[{float(x):.2f}, {float(y):.2f}]" for x, y in zone.robot_polygon
    )
    return f"  - {{name: {zone.name}, polygon: [{pts}]}}"


def _print_all_zones(zones: list[CalibratedZone]) -> None:
    print("\n" + "=" * 72)
    print("请把下面内容替换/填入 runtime_config.yaml：")
    print("mechanics:")
    print("  forbidden_zones_mm:")
    if not zones:
        print("    []")
    else:
        for zone in zones:
            print("  " + _format_zone_yaml(zone).lstrip())
    print("=" * 72 + "\n")


def main() -> None:
    cfg = load_runtime_config(ROOT, ROOT / "config" / "runtime_config.yaml")

    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    camera = TopCamera()
    camera.open()
    transform = PixelToRobotTransform.load(cfg["calibration"]["pixel_to_robot"])

    window = "Forbidden Zone 4-Point Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    pending: list[ClickPoint] = []
    zones: list[CalibratedZone] = []

    def on_mouse(event, x, y, flags, userdata):
        nonlocal pending, zones

        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                rx, ry = transform.pixel_to_robot(x, y, apply_tcp_offset=False)
            except Exception as exc:
                print(f"[CLICK][ERROR] {exc}")
                return

            point = ClickPoint(int(x), int(y), float(rx), float(ry))
            pending.append(point)
            print(
                f"[CLICK {len(pending)}/4] pixel=({x},{y}) "
                f"robot=({rx:.3f},{ry:.3f}) mm"
            )

            if len(pending) == 4:
                try:
                    px_poly, robot_poly = _order_quad(pending)
                except Exception as exc:
                    print(f"[ZONE][ERROR] {exc}")
                    pending = []
                    return

                pixel_area = abs(_signed_area(px_poly))
                robot_area = abs(_signed_area(robot_poly))
                if pixel_area < 20.0 or robot_area < 1.0:
                    print(
                        "[ZONE][REJECT] 四点面积太小/角点重复，请重新标定 "
                        f"(pixel_area={pixel_area:.1f}, robot_area={robot_area:.2f})"
                    )
                    pending = []
                    return

                zone = CalibratedZone(
                    name=f"wall_{len(zones) + 1}",
                    robot_polygon=robot_poly,
                    pixel_polygon=px_poly,
                )
                zones.append(zone)
                print("[ZONE] 完成四点禁入区：")
                print(_format_zone_yaml(zone))
                pending = []

        elif event == cv2.EVENT_RBUTTONDOWN:
            if pending:
                removed = pending.pop()
                print(
                    f"[CLICK] 撤销最后一个角点 pixel=({removed.px},{removed.py})，"
                    f"当前 {len(pending)}/4"
                )
            else:
                print("[CLICK] 当前没有未完成角点")

    cv2.setMouseCallback(window, on_mouse)

    print("=" * 72)
    print("固定障碍物禁入区：四点标定（不要用于整个托盘边界）")
    print("左键：点击同一块挡板/禁入区的 4 个真实角点，顺序任意")
    print("      第 4 点后程序会自动整理成不自交四边形并输出 polygon YAML")
    print("右键：撤销当前最后一个未完成角点")
    print("U：删除最后一个已经完成的禁入区")
    print("C：清空本次所有禁入区")
    print("S：重新打印当前全部 YAML")
    print("Q / ESC：退出（退出时也会打印一次全部 YAML）")
    print("注意：整个托盘边界请运行 calibrate_tray_allowed_polygon.py，不能把托盘内部填成 forbidden zone。")
    print("=" * 72)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            vis = frame.copy()

            # 已完成区域：按四边形真实形状画出来。
            for i, zone in enumerate(zones, start=1):
                pxi = np.round(zone.pixel_polygon).astype(np.int32).reshape(-1, 1, 2)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pxi], (0, 0, 255))
                cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
                cv2.polylines(vis, [pxi], True, (0, 0, 255), 2)
                c = np.mean(zone.pixel_polygon, axis=0).astype(int)
                cv2.putText(
                    vis,
                    zone.name,
                    tuple(c),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

            # 当前正在点击的 1~3 个点。
            if pending:
                pxy = np.asarray([[p.px, p.py] for p in pending], dtype=np.int32)
                for idx, p in enumerate(pending, start=1):
                    cv2.circle(vis, (p.px, p.py), 6, (0, 255, 255), -1)
                    cv2.putText(
                        vis,
                        str(idx),
                        (p.px + 8, p.py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                if len(pxy) >= 2:
                    cv2.polylines(vis, [pxy.reshape(-1, 1, 2)], False, (0, 255, 255), 2)

            cv2.putText(
                vis,
                f"pending {len(pending)}/4 | zones {len(zones)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                pending = []
                zones = []
                print("[ZONE] 已清空本次全部标定")
            elif key in (ord("u"), ord("U")):
                pending = []
                if zones:
                    removed = zones.pop()
                    print(f"[ZONE] 已删除 {removed.name}")
                else:
                    print("[ZONE] 当前没有已完成区域")
            elif key in (ord("s"), ord("S")):
                _print_all_zones(zones)
    finally:
        _print_all_zones(zones)
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
