"""用四个固定点标定顶部完整去畸变图像到 XY 坐标的单应矩阵。

先在画面中依次点击左上、右上、右下、左下四个固定点，再输入每个点的
机械 XY。保存完整图坐标，正式流水线自动补上ROI原点，后续裁剪不必重标。
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from vision.calibration.pixel_to_robot import PixelToRobotTransform

from .common import ROOT, load_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runtime_config.yaml")
    args = parser.parse_args()
    cfg, TopCamera, _ = load_context(ROOT / args.config)
    output = ROOT / cfg["calibration"]["pixel_to_robot"]

    camera = TopCamera()
    try:
        camera.open()
        # 内参已经生效，但此处不裁ROI、不使用多边形掩膜。
        frame = camera.read_undistorted()
        points: list[tuple[float, float]] = []

        def on_click(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((float(x), float(y)))
                print(f"[PIXEL] P{len(points)} = ({float(x):.1f}, {float(y):.1f})")

        window = "pixel to robot: click 4 points"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_click)
        while True:
            view = frame.copy()
            for i, (u, v) in enumerate(points, start=1):
                cv2.circle(view, (round(u), round(v)), 6, (0, 0, 255), 2)
                cv2.putText(view, f"P{i}", (round(u) + 8, round(v) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            if len(points) < 4:
                instruction = f"click {len(points) + 1}/4; R clear; Backspace undo; Q quit"
            else:
                instruction = "4/4 complete; Enter continue; R clear; Backspace undo; Q quit"
            cv2.putText(view, instruction, (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return
            if key in (ord("r"), ord("R")):
                points.clear()
                print("[PIXEL] 已清空，重新点击 4 个点")
                continue
            if key in (8, 127):
                if points:
                    removed = points.pop()
                    print(f"[PIXEL] 撤销 P{len(points) + 1}: {removed}")
                continue
            if len(points) == 4 and key in (10, 13):
                break

        robot = []
        for i, (u, v) in enumerate(points, start=1):
            x, y = map(float, input(f"P{i} pixel=({u:.1f},{v:.1f}) 输入 X_mm Y_mm: ").split())
            robot.append((x, y))

        H = cv2.getPerspectiveTransform(np.float32(points), np.float32(robot))
        transform = PixelToRobotTransform(
            H, frame.shape[1], frame.shape[0], source="TopCamera.read_undistorted()",
            metadata={"point_count": 4, "points_pixel": points, "points_robot": robot},
            coordinate_space="undistorted_full",
        )
        transform.save(output)
        output.with_suffix(".points.json").write_text(
            json.dumps({"pixel": points, "robot": robot}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[SAVE] {output}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
