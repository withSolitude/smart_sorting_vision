"""last_debug 共用入口：所有相机参数只从 runtime_config.yaml 读取。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "runtime_config.yaml"


def load_context(config_path: str | Path = CONFIG_PATH):
    """加载统一配置，并按配置初始化两个相机类。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runtime.app_config import load_runtime_config
    from vision.cameras.side_camera import SideCamera
    from vision.cameras.top_camera import TopCamera

    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = ROOT / config_file
    cfg = load_runtime_config(ROOT, config_file)
    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )
    SideCamera.configure(cfg["camera"]["side"])
    return cfg, TopCamera, SideCamera


def raw_top_camera_class(TopCamera):
    """返回不读取旧内参的顶部相机类，用于从零标定/原始预览。"""
    class RawTopCamera(TopCamera):
        def _load_calibration(self):
            pass

        def _create_undistort_maps(self):
            pass

    return RawTopCamera


def _polygon_from_config(camera_cfg: dict) -> tuple[tuple[int, int], ...] | None:
    value = camera_cfg.get("safe_roi_polygon", camera_cfg.get("roi_polygon"))
    if not value:
        return None
    return tuple((int(point[0]), int(point[1])) for point in value)


def crop(frame, camera_cfg: dict):
    """按 YAML 中 ROI 外接矩形截取，并把多边形外的像素置零。"""
    key = "safe_roi" if "safe_roi" in camera_cfg else "roi"
    x1, y1, x2, y2 = map(int, camera_cfg[key])
    roi = frame[y1:y2, x1:x2].copy()
    polygon = _polygon_from_config(camera_cfg)
    if polygon is not None:
        points = np.asarray(polygon, dtype=np.int32)
        points[:, 0] -= x1
        points[:, 1] -= y1
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
        roi[mask == 0] = 0
    return roi, (x1, y1, x2, y2)


def draw_roi(frame, roi, label: str, polygon: Iterable[Iterable[int]] | None = None):
    """在完整图上画 ROI 外接框和可选多边形边界。"""
    x1, y1, x2, y2 = roi
    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 0), 2)
    if polygon is not None:
        points = np.asarray(list(polygon), dtype=np.int32).reshape(-1, 1, 2)
        if len(points) >= 3:
            cv2.polylines(out, [points], True, (0, 255, 255), 3)
    cv2.putText(
        out,
        f"{label} ROI=({x1},{y1})-({x2},{y2})",
        (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return out
