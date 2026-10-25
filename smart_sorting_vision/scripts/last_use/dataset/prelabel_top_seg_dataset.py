#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为顶部相机 YOLO-Seg 数据生成 CVAT 预标注包。

针对当前 smart_sorting_vision 工程：
- 原始数据已经在采集阶段分为 train / val；本脚本严格保留该划分。
- 递归读取 train、val 下的场景子目录，但输出到 CVAT/YOLO 标准目录时按 split 展平。
- 不重新编码图片，只复制原图，避免 JPEG 二次压缩。
- 用当前顶部 YOLO-Seg 模型生成 polygon 预标注。
- 输出 CVAT 官方 Ultralytics YOLO Segmentation 1.0 可导入的数据集。
- data.yaml 的 train 和 val 指向不同列表，避免重复导入。

推荐 CVAT 用法：
1. 新建一个 Project（可不预建 label）。
2. Project -> Actions -> Import dataset。
3. Format 选择 "Ultralytics YOLO Segmentation 1.0"。
4. 上传生成的 v1_cvat_project.zip。
5. CVAT 会按 train / val subset 创建独立 Task，原划分不会被打乱。

运行：
    cd ~/smart_sorting_vision
    source .venv/bin/activate
    python scripts/last_use/dataset/prelabel_top_seg_dataset.py
"""

from __future__ import annotations

import csv
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# =============================================================================
# 用户配置区
# =============================================================================

DATASET_VERSION = "v1"

# 当前明确保留 train / val，不重新划分。
SPLITS = ("train", "val")

CLASS_ID = 0
CLASS_NAME = "object"

# None 表示从 config/runtime_config.yaml -> model.top 读取。
MODEL_PATH_OVERRIDE: str | None = None
CONF_OVERRIDE: float | None = None
IMGSZ_OVERRIDE: int | None = None
DEVICE_OVERRIDE = None

# 为了让 CVAT 中 polygon 不出现几百个几乎共线的点，做很轻微的轮廓简化。
# 单位：原图像素。设为 0.0 可完全关闭。
POLYGON_SIMPLIFY_EPSILON_PX = 1.0

# 小于该面积的 polygon 会被丢弃。默认 0 表示不按面积过滤。
MIN_POLYGON_AREA_PX = 0.0

# 每次重新生成时清空旧的 CVAT 输出目录，防止旧图片/旧标签残留。
CLEAN_OUTPUT = True

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =============================================================================
# 工程路径
# scripts/last_use/dataset/prelabel_top_seg_dataset.py -> parents[3] = 工程根目录
# =============================================================================

ROOT = Path(__file__).resolve().parents[3]
RUNTIME_CONFIG = ROOT / "config" / "runtime_config.yaml"

SOURCE_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / DATASET_VERSION
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / f"{DATASET_VERSION}_cvat"
)

BUILD_ROOT = OUTPUT_ROOT / "project_dataset"
MANIFEST_PATH = OUTPUT_ROOT / f"{DATASET_VERSION}_cvat_manifest.csv"
PROJECT_ZIP_PATH = OUTPUT_ROOT / f"{DATASET_VERSION}_cvat_project.zip"


# =============================================================================
# 基础工具
# =============================================================================


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _load_model_config() -> dict:
    if not RUNTIME_CONFIG.exists():
        raise FileNotFoundError(
            f"找不到运行配置：{RUNTIME_CONFIG}"
        )

    with RUNTIME_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    try:
        top_cfg = dict(cfg["model"]["top"])
    except KeyError as exc:
        raise RuntimeError(
            "runtime_config.yaml 缺少 model.top 配置"
        ) from exc

    model_path_value = (
        MODEL_PATH_OVERRIDE
        if MODEL_PATH_OVERRIDE is not None
        else top_cfg.get("path")
    )

    if not model_path_value:
        raise RuntimeError("没有配置顶部模型路径")

    model_path = _resolve_project_path(model_path_value)

    conf = float(
        CONF_OVERRIDE
        if CONF_OVERRIDE is not None
        else top_cfg.get("conf", 0.35)
    )

    imgsz = int(
        IMGSZ_OVERRIDE
        if IMGSZ_OVERRIDE is not None
        else top_cfg.get("imgsz", 832)
    )

    device = (
        DEVICE_OVERRIDE
        if DEVICE_OVERRIDE is not None
        else top_cfg.get("device", 0)
    )

    return {
        "path": model_path,
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
    }


def _find_images(split_root: Path) -> list[Path]:
    images = [
        path
        for path in split_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        )
    ]

    images.sort(
        key=lambda path: path.relative_to(split_root).as_posix()
    )

    return images


def _validate_source_layout() -> dict[str, list[Path]]:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            "找不到顶部数据集目录：\n"
            f"  {SOURCE_ROOT}"
        )

    split_images: dict[str, list[Path]] = {}

    for split in SPLITS:
        split_root = SOURCE_ROOT / split

        if not split_root.exists():
            raise FileNotFoundError(
                f"找不到 {split} 目录：\n"
                f"  {split_root}"
            )

        images = _find_images(split_root)

        if not images:
            raise RuntimeError(
                f"{split} 中没有找到图片：{split_root}"
            )

        split_images[split] = images

    # capture_top_seg_dataset.py 使用全局编号，理论上 stem 应全局唯一。
    # 这里强制检查，避免 CVAT 匹配到错误图片。
    stems: dict[str, Path] = {}
    duplicates: list[tuple[str, Path, Path]] = []

    for split in SPLITS:
        for path in split_images[split]:
            if path.stem in stems:
                duplicates.append((path.stem, stems[path.stem], path))
            else:
                stems[path.stem] = path

    if duplicates:
        lines = [
            "检测到重复图片 stem。为防止 CVAT/YOLO 标注错配，已停止："
        ]
        for stem, p1, p2 in duplicates[:50]:
            lines.append(f"  {stem}: {p1} <-> {p2}")
        raise RuntimeError("\n".join(lines))

    return split_images


def _prepare_output() -> None:
    if OUTPUT_ROOT.resolve() == SOURCE_ROOT.resolve():
        raise RuntimeError("OUTPUT_ROOT 不能与 SOURCE_ROOT 相同")

    if CLEAN_OUTPUT and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (BUILD_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (BUILD_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Polygon 处理
# =============================================================================


def _remove_consecutive_duplicates(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points

    keep = [0]
    for i in range(1, len(points)):
        if not np.allclose(points[i], points[keep[-1]], atol=1e-6):
            keep.append(i)

    result = points[keep]

    if (
        len(result) >= 2
        and np.allclose(result[0], result[-1], atol=1e-6)
    ):
        result = result[:-1]

    return result


def _prepare_polygon(
    polygon_xy: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    points = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)

    if len(points) < 3:
        return None

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    if len(points) < 3:
        return None

    points[:, 0] = np.clip(points[:, 0], 0.0, max(0.0, width - 1.0))
    points[:, 1] = np.clip(points[:, 1], 0.0, max(0.0, height - 1.0))

    points = _remove_consecutive_duplicates(points)

    if len(points) < 3:
        return None

    area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2))))
    if area < MIN_POLYGON_AREA_PX:
        return None

    if POLYGON_SIMPLIFY_EPSILON_PX > 0.0 and len(points) > 3:
        approx = cv2.approxPolyDP(
            points.reshape(-1, 1, 2),
            epsilon=float(POLYGON_SIMPLIFY_EPSILON_PX),
            closed=True,
        ).reshape(-1, 2)

        approx = _remove_consecutive_duplicates(approx)

        # 简化过度时退回原轮廓。
        if len(approx) >= 3:
            points = approx

    normalized = points.copy()
    normalized[:, 0] /= float(width)
    normalized[:, 1] /= float(height)
    normalized = np.clip(normalized, 0.0, 1.0)

    normalized = _remove_consecutive_duplicates(normalized)

    if len(normalized) < 3:
        return None

    return normalized


def _result_to_yolo_seg(
    result,
    width: int,
    height: int,
) -> tuple[str, int, list[float]]:
    if result.masks is None or result.masks.xy is None:
        return "", 0, []

    polygons_xy = list(result.masks.xy)

    confidences: list[float] = []
    if result.boxes is not None and result.boxes.conf is not None:
        confidences = [
            float(v)
            for v in result.boxes.conf.detach().cpu().tolist()
        ]

    lines: list[str] = []
    kept_confidences: list[float] = []

    for index, polygon_xy in enumerate(polygons_xy):
        polygon = _prepare_polygon(
            polygon_xy,
            width=width,
            height=height,
        )

        if polygon is None:
            continue

        values: list[str] = []
        for x, y in polygon:
            values.append(f"{float(x):.6f}")
            values.append(f"{float(y):.6f}")

        lines.append(
            f"{CLASS_ID} " + " ".join(values)
        )

        if index < len(confidences):
            kept_confidences.append(confidences[index])

    if not lines:
        return "", 0, kept_confidences

    return "\n".join(lines) + "\n", len(lines), kept_confidences


# =============================================================================
# CVAT / Ultralytics 数据集输出
# =============================================================================


def _write_subset_list(split: str, image_names: Iterable[str]) -> None:
    lines = [
        f"images/{split}/{name}"
        for name in image_names
    ]

    (BUILD_ROOT / f"{split}.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_data_yaml() -> None:
    lines = [
        "path: ./",
    ]

    for split in SPLITS:
        lines.append(f"{split}: {split}.txt")

    lines.extend(
        [
            "names:",
            f"  {CLASS_ID}: {CLASS_NAME}",
            "",
        ]
    )

    (BUILD_ROOT / "data.yaml").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _zip_directory(source_dir: Path, output_zip: Path) -> None:
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue

            arcname = path.relative_to(source_dir).as_posix()
            archive.write(path, arcname=arcname)


# =============================================================================
# 主程序
# =============================================================================


def main() -> None:
    print("=" * 78)
    print(" TOP SEG -> CVAT PRELABEL (KEEP ORIGINAL TRAIN / VAL SPLIT)")
    print("=" * 78)
    print(f"Project root : {ROOT}")
    print(f"Dataset      : {SOURCE_ROOT}")
    print(f"Output       : {OUTPUT_ROOT}")
    print(f"Splits       : {', '.join(SPLITS)}")

    split_images = _validate_source_layout()
    model_cfg = _load_model_config()

    if not model_cfg["path"].exists():
        raise FileNotFoundError(
            "找不到顶部 Seg 模型：\n"
            f"  {model_cfg['path']}"
        )

    print(f"Model        : {model_cfg['path']}")
    print(f"imgsz/conf   : {model_cfg['imgsz']} / {model_cfg['conf']}")
    print(f"device       : {model_cfg['device']}")
    print(f"poly epsilon : {POLYGON_SIMPLIFY_EPSILON_PX:.2f}px")
    print("=" * 78)

    for split in SPLITS:
        print(f"[INPUT] {split:<5}: {len(split_images[split])} images")

    _prepare_output()

    print("[MODEL] loading...")
    model = YOLO(str(model_cfg["path"]))
    print(f"[MODEL] names={model.names}")

    if isinstance(model.names, dict) and len(model.names) != 1:
        print(
            "[WARN] 顶部模型不是单类别模型。"
            "本脚本仍会把所有预测统一写为 class 0: object。"
        )

    manifest_rows: list[dict] = []
    total_images = sum(len(split_images[s]) for s in SPLITS)
    global_done = 0

    for split in SPLITS:
        image_names: list[str] = []
        split_instances = 0
        empty_count = 0

        image_out_dir = BUILD_ROOT / "images" / split
        label_out_dir = BUILD_ROOT / "labels" / split
        split_root = SOURCE_ROOT / split

        print()
        print("-" * 78)
        print(f"[SPLIT] {split}")
        print("-" * 78)

        for local_index, source_image in enumerate(split_images[split], start=1):
            global_done += 1

            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image is None:
                # Linux/Jetson 一般不会有中文路径问题，但保留一次 imdecode fallback。
                raw = np.fromfile(str(source_image), dtype=np.uint8)
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None

            if image is None:
                raise RuntimeError(f"无法读取图片：{source_image}")

            height, width = image.shape[:2]

            result = model.predict(
                source=image,
                imgsz=model_cfg["imgsz"],
                conf=model_cfg["conf"],
                device=model_cfg["device"],
                verbose=False,
            )[0]

            label_text, instance_count, confidences = _result_to_yolo_seg(
                result,
                width=width,
                height=height,
            )

            # 不重新编码图片，直接复制原始文件。
            destination_image = image_out_dir / source_image.name
            destination_label = label_out_dir / f"{source_image.stem}.txt"

            shutil.copy2(source_image, destination_image)
            destination_label.write_text(label_text, encoding="utf-8")

            image_names.append(source_image.name)
            split_instances += instance_count

            if instance_count == 0:
                empty_count += 1

            source_relative = source_image.relative_to(SOURCE_ROOT).as_posix()

            manifest_rows.append(
                {
                    "split": split,
                    "source_relative_path": source_relative,
                    "cvat_image_path": f"images/{split}/{source_image.name}",
                    "cvat_label_path": f"labels/{split}/{source_image.stem}.txt",
                    "width": width,
                    "height": height,
                    "instances_auto": instance_count,
                    "auto_label_empty": int(instance_count == 0),
                    "confidences": ";".join(f"{c:.4f}" for c in confidences),
                }
            )

            print(
                f"[{global_done:4d}/{total_images:4d}] "
                f"{split:<5} "
                f"{source_image.name:<24} "
                f"objects={instance_count}"
            )

        _write_subset_list(split, image_names)

        print(
            f"[SUMMARY] {split}: "
            f"images={len(image_names)}, "
            f"instances={split_instances}, "
            f"empty_labels={empty_count}"
        )

    _write_data_yaml()

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "split",
            "source_relative_path",
            "cvat_image_path",
            "cvat_label_path",
            "width",
            "height",
            "instances_auto",
            "auto_label_empty",
            "confidences",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    _zip_directory(BUILD_ROOT, PROJECT_ZIP_PATH)

    zip_mb = PROJECT_ZIP_PATH.stat().st_size / 1024 / 1024

    print()
    print("=" * 78)
    print(" DONE")
    print("=" * 78)
    print(f"CVAT dataset : {PROJECT_ZIP_PATH}")
    print(f"Manifest     : {MANIFEST_PATH}")
    print(f"Build dir    : {BUILD_ROOT}")
    print(f"ZIP size     : {zip_mb:.2f} MB")
    print()
    print("CVAT 推荐导入方式：")
    print("  Project -> Actions -> Import dataset")
    print("  Format  -> Ultralytics YOLO Segmentation 1.0")
    print(f"  Upload  -> {PROJECT_ZIP_PATH.name}")
    print()
    print("导入后应得到两个独立 subset/task：train 与 val。")
    print("不要再把 val 指向 images/train，也不要重新随机划分。")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] 用户中断。")
        sys.exit(130)
