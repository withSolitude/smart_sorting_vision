"""采集侧面空托盘背景并写入 YAML 指定文件。"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from .common import ROOT, load_context


def main() -> None:
    parser = argparse.ArgumentParser(description="采集侧面空托盘背景")
    parser.add_argument("--config", default="config/runtime_config.yaml")
    args = parser.parse_args()
    cfg, _, SideCamera = load_context(ROOT / args.config)
    capture_cfg = cfg["calibration"]["side_background_capture"]
    settle = int(capture_cfg["settle_frames"])
    count = int(capture_cfg["frame_count"])
    output = ROOT / cfg["calibration"]["side_background"]

    camera = SideCamera()
    try:
        camera.open()
        for _ in range(settle):
            camera.read()
        frames = []
        for _ in range(count):
            frame = camera.read()
            if frame is None:
                raise RuntimeError("读取侧面相机失败")
            frames.append(frame)
        background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), background):
            raise RuntimeError(f"保存背景失败: {output}")
        print(f"[SAVE] {output} frames={count} shape={background.shape}")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
