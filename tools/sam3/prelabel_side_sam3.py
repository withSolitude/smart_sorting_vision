"""按侧面形状目录赋类别，用官方 SAM 3 生成完整可见物体的 YOLO-seg 预标注。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

from sam3_classes import class_for_path, class_names
from sam3_dataset import export_dataset, process_masks, scan_dataset
from sam3_interface import BASE, Sam3Interface, load_config


class SidePredictor:
    """每个目录表示图中的一个主要目标；形状标签来自目录而不是文字预测。"""

    def __init__(self, model: Sam3Interface, config: dict, source: Path):
        self.model, self.config, self.source = model, config, source
        self.names = class_names(config)
        self.definitions = config["annotation"]["classes"]
        self.target = np.asarray(config["inference"]["target_point"], dtype=float)
        if self.target.shape != (2,) or not np.isfinite(self.target).all() or (
            (self.target < 0).any() or (self.target >= 1).any()
        ):
            raise ValueError("inference.target_point 必须为[0,1)内的两个归一化坐标")

    def __call__(self, image, path: Path) -> list[dict]:
        class_id = class_for_path(path, self.source, self.config)
        if class_id is None:
            # 只有明确放在空台目录的图片才跳过SAM推理，输出空标签。
            return []
        predictions = self.model.predict(image, self.definitions[class_id]["prompts"])
        objects = process_masks(predictions, self.config["export"])
        if not objects:
            return []  # 记入零检出待复核，不代表这张图真的没有物体。

        width, height = image.size
        target_x, target_y = self.target * [width, height]

        def rank(obj):
            mask = obj["mask"]
            ys, xs = np.nonzero(mask)
            distance = ((xs.mean() - target_x) / width) ** 2 + ((ys.mean() - target_y) / height) ** 2
            return (not bool(mask[int(target_y), int(target_x)]), distance, -obj["score"])

        # 旧分类图每张只对应一个形状目录，不能把旁边杂物也全部赋成该形状。
        selected = dict(min(objects, key=rank))
        selected.update(class_id=class_id, class_name=self.names[class_id])
        if len(objects) > 1:
            selected["review_reason"] = "multiple_candidates"
        return [selected]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="SAM 3 侧面12类 -> YOLO实例分割预标注")
    parser.add_argument("--config", type=Path, default=BASE / "sam3_side_config.yaml")
    parser.add_argument("--source", type=Path, help="本次侧面数据根目录，内含train/val等目录")
    parser.add_argument("--limit", type=int, help="本次每个split的图片上限，0为全部")
    parser.add_argument("--dry-run", action="store_true", help="只检查类别、目录和图片数，不加载模型")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config["dataset"]
    source = args.source.resolve() if args.source else dataset["source_dir"]
    limit = dataset["limit"] if args.limit is None else args.limit
    names = class_names(config)
    files = scan_dataset(source, dataset["splits"], limit, allow_duplicate_stems=True)
    counts = Counter()
    for split, paths in files.items():
        print(f"[{split}] {len(paths)} 张")
        for path in paths:
            counts[class_for_path(path, source, config)] += 1
    if not any(files.values()):
        raise FileNotFoundError(f"没有找到待标注图片：{source}")
    print(f"[INPUT] {source}")
    for class_id, name in enumerate(names):
        print(f"  {class_id:2d}: {name}  {counts[class_id]} 张")
    print(f"[EMPTY] 空背景目录：{counts[None]} 张")
    if args.dry_run:
        return

    output = dataset["output_dir"] / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    print("[MODEL] 加载本地 SAM 3；类别继承目录，完整轮廓仍需复核……", flush=True)
    model = Sam3Interface(config["model"])
    report = export_dataset(
        source, output, dataset["splits"], SidePredictor(model, config, source),
        config["export"], limit=limit, names=names, predict_with_path=True,
    )
    print(f"[DONE] {output}")
    print(f"图片={report['images']}，实例={report['instances']}，零检出={report['zero_detection']}")
    print("空背景目录的空标签是正常结果；形状目录中的零检出必须补标，不能直接当负样本训练。")
    print("自动预标注请在 CVAT 复核；点选标注使用 run_side_interactive.cmd。")


if __name__ == "__main__":
    main()
