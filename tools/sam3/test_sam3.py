"""先用一张图片验证官方权重；保存预览与 YOLO 分割标签。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from sam3_interface import BASE, Sam3Interface, load_config
from sam3_dataset import draw_preview, process_masks, to_yolo_text


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="官方 SAM 3 单图测试")
    parser.add_argument("image", type=Path, help="要测试的图片路径")
    parser.add_argument("--config", type=Path, default=BASE / "sam3_config.yaml")
    parser.add_argument("--prompt", action="append", help="本次提示词，可重复指定")
    parser.add_argument("--show", action="store_true", help="打开预览窗口")
    args = parser.parse_args()
    config = load_config(args.config)
    model = Sam3Interface(config["model"])
    prompts = args.prompt or config["inference"]["prompts"]
    with Image.open(args.image) as opened:
        image = opened.convert("RGB")
    objects = process_masks(model.predict(image, prompts), config["export"])
    preview = draw_preview(np.asarray(image)[:, :, ::-1].copy(), objects)
    output = config["dataset"]["output_dir"] / (
        "single_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True)
    # imencode+tofile 支持 Windows 中文路径。
    ok, encoded = cv2.imencode(".jpg", preview)
    if not ok:
        raise RuntimeError("预览编码失败")
    encoded.tofile(output / "preview.jpg")
    (output / f"{args.image.stem}.txt").write_text(
        to_yolo_text(objects, image.width, image.height), encoding="utf-8"
    )
    print(f"[DONE] {len(objects)} 个实例；输出：{output}")
    if args.show:
        cv2.namedWindow("SAM 3 preview", cv2.WINDOW_NORMAL)
        cv2.imshow("SAM 3 preview", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
