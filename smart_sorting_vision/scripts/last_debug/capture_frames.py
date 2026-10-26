"""采集顶部/侧面完整图和 ROI 图。

按 S 保存，Q/ESC 退出。保存目录和相机参数均来自命令行/YAML。
"""
from __future__ import annotations

import argparse
from datetime import datetime

import cv2

from .common import ROOT, crop, draw_roi, load_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=("top", "side"), required=True)
    parser.add_argument("--config", default="config/runtime_config.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg, TopCamera, SideCamera = load_context(ROOT / args.config)
    camera = TopCamera() if args.camera == "top" else SideCamera()
    output = ROOT / (args.output or cfg["debug"]["dataset_dir"]) / args.camera
    output.mkdir(parents=True, exist_ok=True)
    index = len(list(output.glob("*.jpg"))) // 2
    try:
        camera.open(require_calibration=False) if args.camera == "top" else camera.open()
        top_undistorted = (
            args.camera == "top"
            and getattr(camera, "_map1", None) is not None
        )
        while True:
            full = camera.read_undistorted() if top_undistorted else camera.read_full()
            roi, rect = crop(full, cfg["camera"][args.camera])
            polygon_key = "safe_roi_polygon" if args.camera == "top" else "roi_polygon"
            cv2.imshow(
                f"{args.camera} full",
                draw_roi(full, rect, args.camera, cfg["camera"][args.camera].get(polygon_key)),
            )
            cv2.imshow(f"{args.camera} roi", roi)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord("S")):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(output / f"{index:06d}_{stamp}_full.jpg"), full)
                cv2.imwrite(str(output / f"{index:06d}_{stamp}_roi.jpg"), roi)
                print(f"[SAVE] {output} index={index}")
                index += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
