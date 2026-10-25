"""鼠标选择顶部或侧面 ROI，并可回写 runtime_config.yaml。

运行后拖拽框选区域，按 Enter 打印坐标；加 --write 才会写回 YAML。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from .common import ROOT, load_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=("top", "side"), required=True)
    parser.add_argument("--config", default="config/runtime_config.yaml")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config
    cfg, TopCamera, SideCamera = load_context(config_path)
    camera = TopCamera() if args.camera == "top" else SideCamera()
    try:
        camera.open(require_calibration=False) if args.camera == "top" else camera.open()
        frame = camera.read_full()
        x, y, width, height = cv2.selectROI(
            f"select {args.camera} ROI", frame, showCrosshair=True, fromCenter=False
        )
        cv2.destroyAllWindows()
        roi = [int(x), int(y), int(x + width), int(y + height)]
        print(f"camera.{args.camera}.{'safe_roi' if args.camera == 'top' else 'roi'} = {roi}")
        if not args.write:
            return

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        key = "safe_roi" if args.camera == "top" else "roi"
        raw["camera"][args.camera][key] = roi
        config_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"[WRITE] {config_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
