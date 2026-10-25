"""last_debug 共用入口：所有相机参数只从 runtime_config.yaml 读取。"""
from __future__ import annotations

import sys
from pathlib import Path


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


def crop(frame, camera_cfg: dict):
    """按 YAML 中的 safe_roi/roi 截取画面。"""
    key = "safe_roi" if "safe_roi" in camera_cfg else "roi"
    x1, y1, x2, y2 = map(int, camera_cfg[key])
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def draw_roi(frame, roi, label: str):
    import cv2

    x1, y1, x2, y2 = roi
    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(out, f"{label} ROI=({x1},{y1})-({x2},{y2})", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return out
