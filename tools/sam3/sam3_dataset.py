"""把 SAM 3 预测结果整理成 YOLO-seg 数据集和 CVAT 导入包。"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def scan_dataset(
    source_dir: str | Path,
    splits: Iterable[str],
    limit: int = 0,
    *,
    allow_duplicate_stems: bool = False,
) -> dict[str, list[Path]]:
    """递归寻找已有 train/val/test 图片；limit 为每个 split 上限，0 表示全部。"""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"源图片目录不存在：{source}")
    names = list(splits)
    if not names or len(set(names)) != len(names):
        raise ValueError("splits 不能为空，也不能重复")
    if int(limit) < 0:
        raise ValueError("limit 必须大于等于0")

    result: dict[str, list[Path]] = {}
    seen_stems: dict[str, Path] = {}
    for split in names:
        if not split or Path(split).name != split or split in {".", ".."}:
            raise ValueError(f"split 必须是一级目录名：{split!r}")
        split_root = source / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"没有找到 split 目录：{split_root}")
        files = sorted(
            (
                path for path in split_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.relative_to(split_root).as_posix().casefold(),
        )
        # 输出按 split 展平。相同 stem 会对应同一个标签文件，因此必须先检查。
        for path in files:
            key = path.stem.casefold()
            if key in seen_stems and not allow_duplicate_stems:
                raise ValueError(
                    "发现重名图片，展平后可能造成标签错配，请先调整文件名：\n"
                    f"  {seen_stems[key]}\n  {path}"
                )
            seen_stems[key] = path
        result[split] = files

    if int(limit):
        for split in names:
            result[split] = result[split][:int(limit)]
    return result


def export_image_name(path: Path, source: Path, *, unique: bool = False) -> str:
    """侧面采集目录经常重名，按原相对路径生成稳定后缀，不修改源图片。"""
    if not unique:
        return path.name
    relative = path.resolve().relative_to(source.resolve()).as_posix()
    suffix = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}__{suffix}{path.suffix}"


def process_masks(predictions: list[dict], config: dict) -> list[dict]:
    """按真实 mask 去重，再把每个实例的最大连通区域转换成 polygon。

    一个 SAM 实例始终最多导出一条 YOLO 标签。被遮挡而断开的其他区域和内孔
    无法用一条普通 YOLO polygon 无损表示，因此保留记录，供人工复核。
    """
    min_area = max(0.0, float(config["min_area_px"]))
    epsilon = max(0.0, float(config["simplify_epsilon_px"]))
    iou_threshold = float(config["mask_iou_threshold"])
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("mask_iou_threshold 必须在 (0, 1] 范围内")

    objects: list[dict] = []
    for prediction in sorted(predictions, key=lambda item: float(item["score"]), reverse=True):
        mask = np.asarray(prediction["mask"], dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"SAM mask 必须是 HxW，实际尺寸为：{mask.shape}")
        area = int(np.count_nonzero(mask))
        if area < min_area or area == 0:
            continue

        duplicate = False
        for existing in objects:
            if mask.shape != existing["mask"].shape:
                raise ValueError("同一张图的 SAM mask 尺寸不一致")
            intersection = int(np.count_nonzero(mask & existing["mask"]))
            union = area + existing["mask_area_px"] - intersection
            if union and intersection / union >= iou_threshold:
                duplicate = True
                break
        if duplicate:
            continue

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8,
        )
        if count <= 1:
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_area = int(stats[largest, cv2.CC_STAT_AREA])
        if largest_area < min_area:
            continue
        main_mask = (labels == largest).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            main_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour_index = max(range(len(contours)), key=lambda index: cv2.contourArea(contours[index]))
        contour = contours[contour_index]
        polygon = cv2.approxPolyDP(contour, epsilon, True) if epsilon else contour
        polygon = polygon.reshape(-1, 2).astype(np.float32)
        if len(polygon) < 3 or abs(float(cv2.contourArea(polygon))) < 1.0:
            continue

        has_holes = bool(hierarchy is not None and np.any(hierarchy[0, :, 3] >= 0))
        objects.append({
            **prediction,
            "mask": mask,
            "score": float(prediction["score"]),
            "polygon": polygon,
            "mask_area_px": area,
            "disconnected_pixels": area - largest_area,
            "has_holes": has_holes,
        })
    return objects


def to_yolo_text(objects: list[dict], width: int, height: int, class_count: int | None = None) -> str:
    """写入各实例的类别 ID 和归一化多边形；旧实例没有 ID 时仍为类别0。"""
    if width <= 0 or height <= 0:
        raise ValueError("图片宽高必须大于0")
    lines: list[str] = []
    for obj in objects:
        class_id = int(obj.get("class_id", 0))
        if class_id < 0 or (class_count is not None and class_id >= class_count):
            raise ValueError(f"实例类别 ID 超出 names 范围：{class_id}")
        points = np.asarray(obj["polygon"], dtype=np.float64).reshape(-1, 2).copy()
        points[:, 0] = np.clip(points[:, 0], 0, width - 1) / width
        points[:, 1] = np.clip(points[:, 1], 0, height - 1) / height
        if len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("待导出的 polygon 无效")
        lines.append(f"{class_id} " + " ".join(f"{coordinate:.6f}" for coordinate in points.ravel()))
    return "\n".join(lines) + ("\n" if lines else "")


def draw_preview(image_bgr: np.ndarray, objects: list[dict]) -> np.ndarray:
    """绘制实际导出的轮廓；REVIEW 表示断开区域或内孔需要人工检查。"""
    palette = [(80, 220, 80), (255, 170, 50), (70, 170, 255), (220, 100, 220)]
    overlay = image_bgr.copy()
    view = image_bgr.copy()
    for index, obj in enumerate(objects):
        polygon = np.rint(obj["polygon"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [polygon], palette[index % len(palette)])
    view = cv2.addWeighted(overlay, 0.25, view, 0.75, 0)
    for index, obj in enumerate(objects):
        color = palette[index % len(palette)]
        polygon = np.rint(obj["polygon"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [polygon], True, color, 2)
        x, y = polygon.reshape(-1, 2).min(axis=0)
        review = obj["disconnected_pixels"] > 0 or obj["has_holes"] or bool(obj.get("review_reason"))
        label = f"class={obj['class_id']}" if "class_id" in obj else "object"
        text = f"{index + 1}: {label} {obj['score']:.2f}" + (" REVIEW" if review else "")
        cv2.putText(view, text, (int(x), max(18, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if not objects:
        cv2.putText(view, "NO DETECTION - REVIEW", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA)
    return view


def export_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    splits: Iterable[str],
    predict: Callable[..., list[dict]],
    config: dict,
    limit: int = 0,
    *,
    names: list[str] | None = None,
    predict_with_path: bool = False,
) -> dict:
    """保留 split 生成预标注；侧面入口可传类别表和接收图片路径的预测函数。"""
    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source == output or source in output.parents:
        raise ValueError("输出目录必须位于源数据目录之外，避免生成文件混入采集数据")
    dataset = output / "dataset"
    if dataset.exists():
        raise FileExistsError(f"输出数据集已存在，请换一个输出目录，避免覆盖人工标签：{dataset}")
    unique_names = names is not None
    names = list(names) if names is not None else ["object"]
    split_images = scan_dataset(source, splits, limit=limit, allow_duplicate_stems=unique_names)
    if not any(split_images.values()):
        raise FileNotFoundError(f"没有找到待标注图片：{source}")

    save_previews = bool(config["save_previews"])
    output.mkdir(parents=True, exist_ok=True)
    dataset.mkdir(exist_ok=False)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source),
        "dataset_dir": str(dataset),
        "images": 0,
        "instances": 0,
        "zero_detection": 0,
        "review_count": 0,
        "disconnected_mask_images": 0,
        "hole_mask_images": 0,
        "splits": {},
        "names": names,
    }
    fieldnames = [
        "split", "source_relative_path", "image", "label", "width", "height",
        "raw_predictions", "instances_auto", "zero_detection", "review_needed",
        "review_reasons", "disconnected_pixels", "hole_instances", "scores", "prompts",
        "class_ids", "class_names",
    ]
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        manifest = csv.DictWriter(stream, fieldnames=fieldnames)
        manifest.writeheader()
        for split, files in split_images.items():
            image_dir = dataset / "images" / split
            label_dir = dataset / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            preview_dir = output / "previews" / split
            if save_previews:
                preview_dir.mkdir(parents=True, exist_ok=True)
            split_stats = {"images": 0, "instances": 0, "zero_detection": 0, "review_count": 0}
            image_list: list[str] = []

            for path in files:
                # PIL 支持中文路径；不做缩放、旋转或二次JPEG编码。
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                width, height = image.size
                predictions = predict(image, path) if predict_with_path else predict(image)
                objects = process_masks(predictions, config)
                if any(obj["mask"].shape != (height, width) for obj in objects):
                    raise ValueError(f"SAM mask 尺寸与原图不一致：{path}")

                exported_name = export_image_name(path, source, unique=unique_names)
                label_name = Path(exported_name).stem + ".txt"
                image_relative = f"images/{split}/{exported_name}"
                label_relative = f"labels/{split}/{label_name}"
                shutil.copy2(path, image_dir / exported_name)
                (label_dir / label_name).write_text(
                    to_yolo_text(objects, width, height, class_count=len(names)), encoding="utf-8",
                )
                # ./ 前缀让 YOLO 按清单所在目录解析图片，而不是按运行目录解析。
                image_list.append("./" + image_relative)

                disconnected = sum(obj["disconnected_pixels"] for obj in objects)
                hole_instances = sum(bool(obj["has_holes"]) for obj in objects)
                reasons: list[str] = []
                if not objects:
                    reasons.append("zero_detection")
                if disconnected:
                    reasons.append("disconnected_mask")
                if hole_instances:
                    reasons.append("mask_has_holes")
                reasons.extend(dict.fromkeys(
                    str(obj["review_reason"]) for obj in objects if obj.get("review_reason")
                ))
                needs_review = bool(reasons)
                manifest.writerow({
                    "split": split,
                    "source_relative_path": path.relative_to(source).as_posix(),
                    "image": image_relative,
                    "label": label_relative,
                    "width": width,
                    "height": height,
                    "raw_predictions": len(predictions),
                    "instances_auto": len(objects),
                    "zero_detection": int(not objects),
                    "review_needed": int(needs_review),
                    "review_reasons": ";".join(reasons),
                    "disconnected_pixels": disconnected,
                    "hole_instances": hole_instances,
                    "scores": ";".join(f"{obj['score']:.4f}" for obj in objects),
                    "prompts": ";".join(dict.fromkeys(str(obj.get("prompt", "")) for obj in objects)),
                    "class_ids": ";".join(str(obj.get("class_id", 0)) for obj in objects),
                    "class_names": ";".join(names[int(obj.get("class_id", 0))] for obj in objects),
                })
                stream.flush()

                if save_previews:
                    preview = draw_preview(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), objects)
                    encoded, data = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if not encoded:
                        raise RuntimeError(f"无法编码标注预览：{path}")
                    data.tofile(str(preview_dir / f"{Path(exported_name).stem}.jpg"))

                split_stats["images"] += 1
                split_stats["instances"] += len(objects)
                split_stats["zero_detection"] += int(not objects)
                split_stats["review_count"] += int(needs_review)
                summary["disconnected_mask_images"] += int(disconnected > 0)
                summary["hole_mask_images"] += int(hole_instances > 0)
                print(f"[{split}] {path.name}: {len(objects)} 个实例" + (f"，需复核：{','.join(reasons)}" if reasons else ""))

            (dataset / f"{split}.txt").write_text(
                "\n".join(image_list) + ("\n" if image_list else ""), encoding="utf-8",
            )
            summary["splits"][split] = split_stats
            for name in ("images", "instances", "zero_detection", "review_count"):
                summary[name] += split_stats[name]

    # 省略 path 字段，使训练端以 data.yaml 所在目录为数据集根目录。
    data_config = {split: f"{split}.txt" for split in split_images}
    data_config["names"] = dict(enumerate(names))
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )
    (dataset / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    if bool(config["make_zip"]):
        zip_path = output / "sam3_cvat_project.zip"
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in sorted(dataset.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(dataset).as_posix())
        summary["cvat_zip"] = str(zip_path)

    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary
