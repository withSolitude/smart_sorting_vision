"""顶部/侧面相机画面和 ROI 预览。

运行：
    python -m scripts.last_debug.preview_cameras --camera top
    python -m scripts.last_debug.preview_cameras --camera side
    python -m scripts.last_debug.preview_cameras --camera both

没有顶部内参文件时加 --raw，只看原始完整图。

按 S 保存当前完整图和 ROI，按 Q/ESC 退出。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from .common import ROOT, crop, draw_roi, load_context, raw_top_camera_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=("top", "side", "both"), default="both")
    parser.add_argument("--config", default="config/runtime_config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--raw", action="store_true", help="顶部相机不读取旧内参")
    args = parser.parse_args()

    cfg, TopCamera, SideCamera = load_context(ROOT / args.config)
    if args.raw and args.camera in ("top", "both"):
        TopCamera = raw_top_camera_class(TopCamera)
    cameras = {}
    if args.camera in ("top", "both"):
        cameras["top"] = TopCamera()
    if args.camera in ("side", "both"):
        cameras["side"] = SideCamera()

    output = ROOT / (args.output or cfg["debug"]["frames_dir"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        for name, camera in cameras.items():
            camera.open(require_calibration=False) if name == "top" else camera.open()
        # 顶部正式 ROI 位于去畸变坐标系；有内参时预览也使用同一坐标系。
        # 没有内参或显式 --raw 时，直接显示镜头原始画面。
        top_undistorted = (
            "top" in cameras
            and not args.raw
            and getattr(cameras.get("top"), "_map1", None) is not None
        )
        while True:
            for name, camera in cameras.items():
                full = (
                    camera.read_undistorted()
                    if name == "top" and top_undistorted
                    else camera.read_full()
                )
                roi, rect = crop(full, cfg["camera"][name])
                polygon_key = "safe_roi_polygon" if name == "top" else "roi_polygon"
                cv2.imshow(
                    f"{name} full",
                    draw_roi(full, rect, name, cfg["camera"][name].get(polygon_key)),
                )
                cv2.imshow(f"{name} roi", roi)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord("S")):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                for name, camera in cameras.items():
                    full = (
                        camera.read_undistorted()
                        if name == "top" and top_undistorted
                        else camera.read_full()
                    )
                    roi, _ = crop(full, cfg["camera"][name])
                    cv2.imwrite(str(output / f"{name}_{stamp}_full.jpg"), full)
                    cv2.imwrite(str(output / f"{name}_{stamp}_roi.jpg"), roi)
                print(f"[SAVE] {output}/*_{stamp}_*.jpg")
    finally:
        for camera in cameras.values():
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
