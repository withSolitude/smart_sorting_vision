"""实时标注顶部/侧面相机的手工多边形 ROI。

运行示例：
    python -m scripts.last_debug.select_roi --camera top --write
    python -m scripts.last_debug.select_roi --camera side --write

窗口操作：
    鼠标左键单击：添加一个多边形顶点；按住左键沿圆弧拖动：连续采点
    R：清空当前顶点，重新标注
    Backspace：删除最后一个顶点
    Enter：确认并写入 YAML（不加 --write 时只打印）
    Q / Esc：取消

窗口中的坐标是当前实际显示帧的完整图坐标。顶部相机优先显示去畸变图；
没有内参时自动退回原始完整图。保存时同时写入多边形和它的外接矩形，
这样运行时仍保持原有 ROI 图像尺寸。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .common import ROOT, load_context


class PolygonPicker:
    """在实时画面上逐点标注多边形。"""

    def __init__(self, window: str, frame_size: tuple[int, int], initial: list[list[int]] | None = None):
        self.window = window
        self.width, self.height = frame_size
        self.points: list[tuple[int, int]] = [
            (int(point[0]), int(point[1])) for point in (initial or [])
        ]
        self.scale = 1.0
        self.frame = None
        self.dragging = False
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self._append_point(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self._append_point(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _append_point(self, x: int, y: int) -> None:
        # 显示窗口可能被缩放，点击坐标需要换回完整图坐标。
        px = max(0, min(self.width - 1, int(round(x / self.scale))))
        py = max(0, min(self.height - 1, int(round(y / self.scale))))
        # 拖动时只保留相隔至少3像素的点，避免鼠标事件产生大量重复点。
        if self.points and (px - self.points[-1][0]) ** 2 + (py - self.points[-1][1]) ** 2 < 9:
            return
        self.points.append((px, py))

    def draw(self, frame):
        self.height, self.width = frame.shape[:2]
        max_w, max_h = 1600, 900
        self.scale = min(1.0, max_w / max(self.width, 1), max_h / max(self.height, 1))
        display = frame
        if self.scale < 1.0:
            display = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        for index, (px, py) in enumerate(self.points, start=1):
            dx, dy = int(round(px * self.scale)), int(round(py * self.scale))
            cv2.circle(display, (dx, dy), 6, (0, 0, 255), -1)
            cv2.putText(display, str(index), (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(self.points) >= 2:
            points = [(int(round(x * self.scale)), int(round(y * self.scale))) for x, y in self.points]
            cv2.polylines(display, [np.array(points, dtype="int32")], False, (0, 255, 0), 2)
        if len(self.points) >= 3:
            points = [(int(round(x * self.scale)), int(round(y * self.scale))) for x, y in self.points]
            cv2.polylines(display, [np.array(points, dtype="int32")], True, (0, 255, 0), 3)
        cv2.putText(
            display,
            f"points={len(self.points)} | click/drag draw | Backspace undo | R clear | Enter save | Q cancel",
            (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )
        return display


def _read_frame(name: str, camera):
    if name == "top":
        # 顶部标定完成后 ROI 坐标属于去畸变图；从零调试时没有 map 就看原图。
        try:
            return camera.read_undistorted()
        except RuntimeError:
            return camera.read_full()
    return camera.read_full()


def _bounding_rect(points: list[tuple[int, int]]) -> list[int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="实时标注手工多边形 ROI")
    parser.add_argument("--camera", choices=("top", "side"), required=True)
    parser.add_argument("--config", default="config/runtime_config.yaml")
    parser.add_argument("--write", action="store_true", help="按 Enter 时写回 YAML")
    args = parser.parse_args()

    config_path = ROOT / args.config
    cfg, TopCamera, SideCamera = load_context(config_path)
    camera = TopCamera() if args.camera == "top" else SideCamera()
    camera_cfg = cfg["camera"][args.camera]
    polygon_key = "safe_roi_polygon" if args.camera == "top" else "roi_polygon"
    rect_key = "safe_roi" if args.camera == "top" else "roi"
    initial = camera_cfg.get(polygon_key)
    picker = None
    try:
        camera.open(require_calibration=False) if args.camera == "top" else camera.open()
        first = _read_frame(args.camera, camera)
        picker = PolygonPicker(f"select {args.camera} polygon ROI", (first.shape[1], first.shape[0]), initial)
        while True:
            frame = first if picker.frame is None else _read_frame(args.camera, camera)
            picker.frame = frame
            cv2.imshow(picker.window, picker.draw(frame))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[CANCEL] 未写入 ROI")
                return
            if key in (ord("r"), ord("R")):
                picker.points.clear()
                continue
            if key in (8, 127):
                if picker.points:
                    picker.points.pop()
                continue
            if key in (10, 13):
                if len(picker.points) < 3:
                    print("[WARN] 至少点击 3 个点后才能保存")
                    continue
                rect = _bounding_rect(picker.points)
                print(f"camera.{args.camera}.{polygon_key} = {[list(p) for p in picker.points]}")
                print(f"camera.{args.camera}.{rect_key} = {rect}")
                if not args.write:
                    return
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                raw.setdefault("camera", {}).setdefault(args.camera, {})[polygon_key] = [list(p) for p in picker.points]
                raw["camera"][args.camera][rect_key] = rect
                config_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
                print(f"[WRITE] {config_path}")
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
