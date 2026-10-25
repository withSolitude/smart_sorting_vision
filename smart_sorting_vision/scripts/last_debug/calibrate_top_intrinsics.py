"""使用 YAML 中的棋盘参数计算顶部相机内参。"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from .common import ROOT, load_context


def main() -> None:
    parser = argparse.ArgumentParser(description="计算顶部相机内参")
    parser.add_argument("--config", default="config/runtime_config.yaml")
    args = parser.parse_args()
    cfg, _, _ = load_context(ROOT / args.config)
    cal = cfg["calibration"]["top_camera"]
    board = tuple(map(int, cal["chessboard_size"]))
    square = float(cal["square_size_mm"])
    files = sorted((ROOT / cal["image_dir"]).glob("*.jpg"))
    if len(files) < int(cal["min_valid_images"]):
        raise RuntimeError(f"标定图片至少需要 {cal['min_valid_images']} 张，当前 {len(files)} 张")

    template = np.zeros((board[0] * board[1], 3), np.float32)
    template[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    objects, images = [], []
    size = None
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        size = (image.shape[1], image.shape[0])
        found, corners = cv2.findChessboardCornersSB(image, board)
        if found:
            objects.append(template.copy())
            images.append(corners)
    if len(images) < int(cal["min_valid_images"]):
        raise RuntimeError(f"有效棋盘图片只有 {len(images)} 张")

    rms, matrix, dist, _, _ = cv2.calibrateCamera(objects, images, size, None, None)
    new_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(matrix, dist, size, 1.0, size)
    output = ROOT / cfg["calibration"]["top_intrinsics"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, camera_matrix=matrix, dist_coeffs=dist, new_camera_matrix=new_matrix,
             image_size=np.array(size), valid_roi=np.array(valid_roi), rms=np.array([rms]))
    print(f"[SAVE] {output}\nRMS={rms:.4f}px valid={len(images)}/{len(files)}")


if __name__ == "__main__":
    main()
