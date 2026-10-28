"""官方 SAM 3 批量预标注入口，保留原有 train/val 与图片字节。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from sam3_interface import BASE, Sam3Interface, load_config
from sam3_dataset import export_dataset, scan_dataset


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="SAM 3 -> YOLO-seg / CVAT 预标注")
    parser.add_argument("--config", type=Path, default=BASE / "sam3_config.yaml")
    parser.add_argument("--source", type=Path, help="本次输入根目录，内部为 train/val")
    parser.add_argument("--limit", type=int, help="本次每个 split 最多处理张数，0 为全部")
    parser.add_argument("--dry-run", action="store_true", help="只检查图片目录，不加载模型")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config["dataset"]
    source = args.source.resolve() if args.source else dataset["source_dir"]
    limit = dataset["limit"] if args.limit is None else args.limit
    files = scan_dataset(source, dataset["splits"], limit=limit)
    print(f"[INPUT] {source}")
    for split, images in files.items():
        print(f"  {split}: {len(images)} 张")
    if not any(files.values()):
        raise FileNotFoundError(f"没有找到图片：{source}")
    if args.dry_run:
        return

    # 每次使用独立目录，避免把人工修改过的标签覆盖掉。
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = dataset["output_dir"] / stamp
    print("[MODEL] 加载官方 SAM 3 本地权重，请稍候……", flush=True)
    model = Sam3Interface(config["model"])
    report = export_dataset(
        source_dir=source,
        output_dir=output,
        splits=dataset["splits"],
        predict=lambda image: model.predict(image, config["inference"]["prompts"]),
        config=config["export"],
        limit=limit,
    )
    print(f"[DONE] {output}")
    print(
        f"图片={report['images']}，实例={report['instances']}，"
        f"零检出={report['zero_detection']}，需复核={report['review_count']}"
    )
    print("CVAT: Project -> Import dataset -> Ultralytics YOLO Segmentation 1.0")
    print("预标注完成后需复核：零检出、粘连、遮挡和误选托盘的图片。")


if __name__ == "__main__":
    main()
