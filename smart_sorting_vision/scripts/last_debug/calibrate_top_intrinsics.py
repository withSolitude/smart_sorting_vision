"""使用 YAML 中的圆点阵列参数计算顶部相机内参。"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from .common import ROOT, load_context
from .calibration_pattern import find_pattern, make_object_points, read_pattern_config


def main() -> None:
    parser = argparse.ArgumentParser(description="计算顶部相机内参")
    parser.add_argument("--config", default="config/runtime_config.yaml")
    args = parser.parse_args()
    cfg, _, _ = load_context(ROOT / args.config)
    cal = cfg["calibration"]["top_camera"]
    pattern_type, pattern_size, spacing_mm = read_pattern_config(cal)
    files = sorted((ROOT / cal["image_dir"]).glob("*.jpg"))
    if len(files) < int(cal["min_valid_images"]):
        raise RuntimeError(f"标定图片至少需要 {cal['min_valid_images']} 张，当前 {len(files)} 张")

    template = make_object_points(pattern_size, spacing_mm)
    objects, images = [], []
    size = None
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        size = (image.shape[1], image.shape[0])
        found, corners = find_pattern(image, pattern_type, pattern_size)
        if found:
            objects.append(template.copy())
            images.append(corners)
    if len(images) < int(cal["min_valid_images"]):
        raise RuntimeError(f"有效棋盘图片只有 {len(images)} 张")

    rms, matrix, dist, _, _ = cv2.calibrateCamera(objects, images, size, None, None)
    new_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(matrix, dist, size, 1.0, size)
    output = ROOT / cfg["calibration"]["top_intrinsics"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        camera_matrix=matrix,
        dist_coeffs=dist,
        new_camera_matrix=new_matrix,
        image_size=np.array(size),
        valid_roi=np.array(valid_roi),
        rms=np.array([rms]),
        pattern_type=np.array(pattern_type),
        pattern_size=np.array(pattern_size),
        circle_spacing_mm=np.array([spacing_mm]),
    )
    warning = float(cal.get("warning_error_px", 0.8))
    status = "PASS" if rms <= warning else "WARN"
    print(
        f"[SAVE] {output}\n"
        f"pattern={pattern_type} size={pattern_size} spacing={spacing_mm:g}mm\n"
        f"RMS={rms:.4f}px valid={len(images)}/{len(files)} [{status}, limit={warning:g}px]"
    )


if __name__ == "__main__":
    main()
