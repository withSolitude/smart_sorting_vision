"""顶部相机圆点阵列图片采集，供 calibrate_top_intrinsics.py 使用。"""
from __future__ import annotations

import argparse
from datetime import datetime

import cv2

from .common import ROOT, load_context, raw_top_camera_class
from .calibration_pattern import find_pattern, read_pattern_config


def main() -> None:
    parser = argparse.ArgumentParser(description="采集顶部相机圆点阵列图片")
    parser.add_argument("--config", default="config/runtime_config.yaml")
    args = parser.parse_args()
    cfg, TopCamera, _ = load_context(ROOT / args.config)
    TopCamera = raw_top_camera_class(TopCamera)
    cal = cfg["calibration"]["top_camera"]
    pattern_type, pattern_size, spacing_mm = read_pattern_config(cal)
    output = ROOT / cal["image_dir"]
    output.mkdir(parents=True, exist_ok=True)
    camera = TopCamera()
    saved = len(list(output.glob("*.jpg")))
    try:
        camera.open()
        print(
            f"圆点板 {pattern_size[0]}x{pattern_size[1]}，"
            f"圆心距={spacing_mm:g}mm；S/SPACE 保存，Q/ESC 退出。"
        )
        while True:
            frame = camera.read_full()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_pattern(gray, pattern_type, pattern_size)
            view = frame.copy()
            if found:
                cv2.drawChessboardCorners(view, pattern_size, corners, found)
            cv2.putText(view, f"circles={found} saved={saved}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("top intrinsic capture", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord("S"), 32) and found:
                path = output / f"calib_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                print(f"[SAVE] {path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
