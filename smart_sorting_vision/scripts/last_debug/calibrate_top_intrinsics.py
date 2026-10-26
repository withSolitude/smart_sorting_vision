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
    objects, images, image_paths = [], [], []
    size = None
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        current_size = (image.shape[1], image.shape[0])
        if size is None:
            size = current_size
        elif current_size != size:
            raise RuntimeError(
                f"标定图片分辨率不一致：{path.name}={current_size}，"
                f"期望={size}；不能混入手机图或其它相机图片"
            )
        found, corners = find_pattern(image, pattern_type, pattern_size)
        if found:
            objects.append(template.copy())
            images.append(corners)
            image_paths.append(path)
    if len(images) < int(cal["min_valid_images"]):
        raise RuntimeError(f"有效棋盘图片只有 {len(images)} 张")

    calibration_flags = (
        cv2.CALIB_FIX_K3
        if bool(cal.get("fix_k3", True))
        else 0
    )
    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        objects,
        images,
        size,
        None,
        None,
        flags=calibration_flags,
    )
    view_errors = []
    for object_points, image_points, rvec, tvec, path in zip(
        objects,
        images,
        rvecs,
        tvecs,
        image_paths,
    ):
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            matrix,
            dist,
        )
        measured = image_points.reshape(-1, 2)
        predicted = projected.reshape(-1, 2)
        error = np.linalg.norm(measured - predicted, axis=1)
        view_errors.append((float(np.sqrt(np.mean(error**2))), path))

    max_view_error = float(cal.get("max_view_error_px", 2.0))
    bad_views = [item for item in view_errors if item[0] > max_view_error]
    if bad_views:
        details = ", ".join(
            f"{path.name}={error:.3f}px"
            for error, path in bad_views
        )
        raise RuntimeError(
            f"存在异常标定图片，单图重投影误差超过 {max_view_error:g}px：{details}；"
            "请移走这些图片后重新采集"
        )
    alpha = float(cal.get("new_camera_alpha", 0.0))
    new_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
        matrix,
        dist,
        size,
        alpha,
        size,
    )
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
        calibration_flags=np.array([calibration_flags], dtype=np.int32),
        new_camera_alpha=np.array([alpha]),
    )
    warning = float(cal.get("warning_error_px", 0.8))
    status = "PASS" if rms <= warning else "WARN"
    print(
        f"[SAVE] {output}\n"
        f"pattern={pattern_type} size={pattern_size} spacing={spacing_mm:g}mm\n"
        f"flags={calibration_flags} alpha={alpha:g}\n"
        f"RMS={rms:.4f}px valid={len(images)}/{len(files)} [{status}, limit={warning:g}px]\n"
        f"view_rmse={min(error for error, _ in view_errors):.4f}/"
        f"{max(error for error, _ in view_errors):.4f}px\n"
        f"dist={np.array2string(dist.reshape(-1), precision=6)}"
    )


if __name__ == "__main__":
    main()
