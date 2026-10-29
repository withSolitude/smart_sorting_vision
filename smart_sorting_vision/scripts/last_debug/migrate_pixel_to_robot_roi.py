"""把已有ROI局部坐标标定换算到完整图坐标，不重新采点拟合。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

from runtime.app_config import load_runtime_config
from vision.calibration.pixel_to_robot import PixelToRobotTransform
from .common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/runtime_config.yaml")
    parser.add_argument("--old-roi", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--input", type=Path, help="旧标定文件；默认使用YAML中的pixel_to_robot")
    parser.add_argument("--output", type=Path, help="新文件；默认备份后更新原文件")
    args = parser.parse_args()
    cfg = load_runtime_config(ROOT, ROOT / args.config)
    source = args.input.resolve() if args.input else Path(cfg["calibration"]["pixel_to_robot"])
    destination = args.output.resolve() if args.output else source
    transform = PixelToRobotTransform.load(source)
    if transform.coordinate_space == "undistorted_full":
        print(f"[OK] 已是完整图坐标，无需再次迁移：{source}")
        return
    camera = cfg["camera"]["top"]
    converted = transform.to_full_image(args.old_roi, (camera["width"], camera["height"]))
    converted.metadata["migrated_from"] = source.name
    if destination.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = destination.with_name(f"{destination.stem}.roi_backup_{stamp}.json")
        shutil.copy2(destination, backup)
        print(f"[BACKUP] {backup}")
    converted.save(destination)
    print(f"[SAVE] {destination}")
    print("H已换算到完整去畸变图坐标。保持相机几何和去畸变不变，后续ROI可独立调整。")


if __name__ == "__main__":
    main()
