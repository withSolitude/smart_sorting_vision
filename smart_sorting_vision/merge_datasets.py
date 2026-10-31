"""合并两个已标注的 YOLO 检测/实例分割数据集；参数见同目录 merge_datasets.yaml。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import yaml


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _find_yaml(source: Path) -> Path:
    source = source.expanduser().resolve()
    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"}:
        return source
    for folder in (source, source / "dataset"):
        for name in ("data.yaml", "data.yml", "dataset.yaml"):
            candidate = folder / name
            if candidate.is_file():
                return candidate
    raise ValueError(f"找不到数据集 YAML：{source}\n请选择已标注的导出目录，不是仅采集照片的目录。")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _class_names(value, source: Path) -> list[str]:
    if isinstance(value, dict):
        indexed = {int(key): str(name).strip() for key, name in value.items()}
        if set(indexed) != set(range(len(indexed))):
            raise ValueError(f"{source}：names 的编号必须从0开始连续排列")
        names = [indexed[i] for i in range(len(indexed))]
    elif isinstance(value, list):
        names = [str(name).strip() for name in value]
    else:
        raise ValueError(f"{source}：需要 names 类别列表或字典")
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{source}：类别名称为空或重复")
    return names


def _split_images(root: Path, entry) -> list[Path]:
    """兼容 split 指向目录、图片清单TXT、图片路径或上述路径的列表。"""
    if entry is None:
        return []
    entries = entry if isinstance(entry, list) else [entry]
    result = []
    for value in entries:
        if not isinstance(value, str):
            raise ValueError("train/val/test 必须填写路径或路径列表")
        path = _resolve(root, value)
        if path.is_dir():
            images = sorted(p.resolve() for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
        elif path.is_file() and path.suffix.lower() == ".txt":
            images = [_resolve(path.parent, line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images = [path]
        else:
            raise ValueError(f"图片目录或清单不存在：{path}")
        for image in images:
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"清单中的图片不存在或格式不支持：{image}")
            result.append(image)
    return result


def _label_path(image: Path) -> Path:
    # 同时支持 images/train/... -> labels/train/... 和 train/images/... -> train/labels/...
    parts = list(image.parts)
    indices = [i for i, part in enumerate(parts[:-1]) if part.lower() == "images"]
    if indices:
        parts[indices[-1]] = "labels"
        label = Path(*parts).with_suffix(".txt")
    else:
        # 也允许图片与标签同目录；不猜测其他目录中的同名标签。
        label = image.with_suffix(".txt")
    if not label.is_file():
        raise ValueError(f"缺少图片对应的标签：{image}\n预期：{label}\n未标注图片不能自动当作空背景；确认无物体时应有空TXT标签。")
    return label


def _read_label(path: Path, mapping: dict[int, int]) -> tuple[str, set[str]]:
    rows = []
    tasks = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        try:
            class_id = int(tokens[0])
            coordinates = [float(value) for value in tokens[1:]]
            if class_id not in mapping:
                raise ValueError("类别编号未在 names 中定义")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in coordinates):
                raise ValueError("坐标必须为0..1之间的有限数值")
            if len(coordinates) == 4:
                if coordinates[2] <= 0 or coordinates[3] <= 0:
                    raise ValueError("检测框宽高必须大于0")
                tasks.add("detect")
            elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                tasks.add("segment")
            else:
                raise ValueError("应为检测框4个坐标，或分割多边形至少3组坐标")
        except (ValueError, IndexError) as exc:
            raise ValueError(f"标签无效：{path}，第{number}行：{exc}") from exc
        rows.append(str(mapping[class_id]) + " " + " ".join(format(v, ".12g") for v in coordinates))
    # 实例顺序不影响标签含义；排序后可比较同图标签是否一致。
    return "".join(row + "\n" for row in sorted(rows)), tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_datasets(dataset_a: Path, dataset_b: Path, output_dir: Path) -> dict:
    """先检查两个数据集，再复制到新目录；不修改源数据或覆盖已有输出。"""
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请选择新目录：{output}")
    sources = [_find_yaml(Path(dataset_a)), _find_yaml(Path(dataset_b))]
    if sources[0] == sources[1]:
        raise ValueError("两个输入指向同一个数据集")

    classes = []
    plans = []
    manifest = []
    seen = {}
    tasks = set()
    duplicates = 0
    for source_index, source in enumerate(sources):
        if output.is_relative_to(source.parent):
            raise ValueError(f"输出目录不能位于源数据集内部：{source.parent}")
        config = yaml.safe_load(source.read_text(encoding="utf-8-sig")) or {}
        if not isinstance(config, dict):
            raise ValueError(f"数据集 YAML 必须是配置对象：{source}")
        names = _class_names(config.get("names"), source)
        mapping = {}
        for i, name in enumerate(names):
            if name not in classes:
                classes.append(name)
            mapping[i] = classes.index(name)
        root = _resolve(source.parent, config.get("path") or ".")
        source_count = 0
        for split in SPLITS:
            for image in _split_images(root, config.get(split)):
                source_count += 1
                label = _label_path(image)
                text, kinds = _read_label(label, mapping)
                tasks.update(kinds)
                if len(tasks) > 1:
                    raise ValueError("输入中同时存在检测框与分割多边形，不能合并为同一任务数据集")
                digest = _sha256(image)
                row = {"source_dataset": str(source), "source_image": str(image),
                       "source_label": str(label), "split": split, "sha256": digest}
                if digest in seen:
                    previous = seen[digest]
                    if previous["split"] != split:
                        raise ValueError(f"相同图片跨 train/val/test，可能导致验证泄漏：\n{previous['source']}\n{image}\n请先确定这张图所属的唯一集合。")
                    if previous["text"] != text:
                        raise ValueError(f"相同图片有不同标注，需先确认使用哪份：\n{previous['source']}\n{image}")
                    duplicates += 1
                    row.update(image=previous["image"], label=previous["label"], status="duplicate")
                else:
                    stem = f"{'a' if source_index == 0 else 'b'}_{len(plans) + 1:07d}"
                    relative_image = f"images/{split}/{stem}{image.suffix.lower()}"
                    relative_label = f"labels/{split}/{stem}.txt"
                    seen[digest] = {"split": split, "text": text, "source": str(image),
                                    "image": relative_image, "label": relative_label}
                    plans.append((image, relative_image, relative_label, text, split))
                    row.update(image=relative_image, label=relative_label, status="included")
                manifest.append(row)
        if not source_count:
            raise ValueError(f"数据集没有任何图片：{source}")

    # 所有类别、标签、清单和完全重复图检查完成后，才开始创建输出。
    output.mkdir(parents=True, exist_ok=False)
    split_lists = {split: [] for split in SPLITS}
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for original, image, label, text, split in plans:
        shutil.copy2(original, output / image)
        (output / label).write_text(text, encoding="utf-8")
        split_lists[split].append("./" + image)
    for split, lines in split_lists.items():
        (output / f"{split}.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    data = {split: f"{split}.txt" for split in SPLITS}
    data["names"] = dict(enumerate(classes))
    (output / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "classes.txt").write_text("".join(name + "\n" for name in classes), encoding="utf-8")
    with (output / "merge_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = {"sources": [str(p) for p in sources], "output_dir": str(output),
               "task": next(iter(tasks)) if tasks else None, "names": classes,
               "images": {split: len(lines) for split, lines in split_lists.items()},
               "total_images": len(plans), "duplicates_skipped": duplicates}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _dataset_name(value: str) -> str:
    path = Path(value)
    if path.suffix.lower() in {".yaml", ".yml"}:
        path = path.parent
    if path.name == "dataset":
        path = path.parent
    return re.sub(r"[^\w.-]+", "_", path.name)[:40] or "dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "merge_datasets.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict) or any(not isinstance(cfg.get(key), str) or not cfg[key].strip()
                                       for key in ("dataset_root", "dataset_a", "dataset_b")):
        raise ValueError("配置中需要填写 dataset_root、dataset_a、dataset_b 三个非空字符串")
    root = _resolve(ROOT, cfg["dataset_root"])
    first, second = cfg["dataset_a"], cfg["dataset_b"]
    name = f"{_dataset_name(first)}_plus_{_dataset_name(second)}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output = root / "merged" / name
    summary = merge_datasets(_resolve(root, first), _resolve(root, second), output)
    print(f"合并完成：{summary['output_dir']}")
    print(f"图片：{summary['images']}；相同图片去重：{summary['duplicates_skipped']}")
    print(f"类别：{summary['names']}")
    print(f"训练使用：{output / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        raise SystemExit(f"合并未完成：{exc}") from None
