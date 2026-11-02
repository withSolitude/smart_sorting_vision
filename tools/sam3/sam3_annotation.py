"""交互标注的保存、断点继续和 YOLO/CVAT 导出，不修改采集原图。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Iterable
import zipfile

import numpy as np
from PIL import Image

from sam3_classes import class_names
from sam3_dataset import export_image_name, scan_dataset, to_yolo_text
from sam3_prelabels import PrelabelSource


def _write_text(path: Path, text: str) -> None:
    """同目录临时文件替换，避免退出时留下半个标签或记录文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class AnnotationSession:
    """一个输出目录对应一个源数据集，可多次打开继续标注。"""

    def __init__(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        splits: Iterable[str],
        export_config: dict,
        limit: int = 0,
        *,
        annotation_config: dict | None = None,
    ):
        self.source = Path(source_dir).expanduser().resolve()
        self.output = Path(output_dir).expanduser().resolve()
        if self.source == self.output or self.source in self.output.parents:
            raise ValueError("交互标注输出目录必须在采集数据目录之外")
        self.dataset = self.output / "dataset"
        self.manifest_path = self.output / "annotations.json"
        self.export_config = export_config
        self.names = class_names(annotation_config or {})
        annotation = (annotation_config or {}).get("annotation", {})
        self.unique_names = bool(annotation.get("class_from_folder", False))
        selected_splits = list(splits)
        files = scan_dataset(
            self.source, selected_splits, limit,
            allow_duplicate_stems=self.unique_names,
        )
        self.entries = [(split, path) for split, paths in files.items() for path in paths]
        if not self.entries:
            raise FileNotFoundError(f"没有找到待标注图片：{self.source}")

        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            # 不让切换输入路径后，旧标签悄悄关联到另一批同名图片。
            recorded_source = Path(self.manifest["source_dir"]).resolve()
            if recorded_source != self.source:
                raise ValueError(
                    f"该输出目录已属于另一数据集：{recorded_source}；请修改 interactive.output_dir"
                )
            # 老会话只有 object；修改类别顺序会改变已有数字标签，必须换输出目录。
            if self.manifest.get("names", ["object"]) != self.names:
                raise ValueError("类别名称或顺序与已保存标注不一致；请恢复原配置或换一个输出目录")
        else:
            if self.dataset.exists():
                raise FileExistsError("输出已有 dataset，但没有交互标注记录；请换一个输出目录")
            self.manifest = {"version": 2, "source_dir": str(self.source), "splits": [], "records": {}}
        self.manifest["names"] = self.names
        self.records = self.manifest["records"]
        # 后来只打开一个 split 时，也保留之前已经确认过的其他 split。
        self.splits = list(dict.fromkeys(self.manifest["splits"] + selected_splits))
        self.manifest["splits"] = self.splits
        self.prelabels = None
        self.prelabels_dir: Path | None = None
        previous_batch = self.manifest.get("prelabels_dir")
        if previous_batch and (Path(previous_batch) / "manifest.csv").is_file():
            self.load_prelabels(previous_batch)

    def _key(self, index: int) -> str:
        return self.entries[index][1].relative_to(self.source).as_posix()

    @property
    def saved_count(self) -> int:
        """整个会话已经人工确认的图片数，包括确认无目标的空标签。"""
        return len(self.records)

    def is_saved(self, index: int) -> bool:
        return self._key(index) in self.records

    def load_prelabels(self, path: str | Path) -> int:
        """接入批量自动标签作为草稿；已保存的人工结果始终优先，不被覆盖。"""
        imported = PrelabelSource(path, self.source, self.names)
        matched = sum(self._key(index) in imported.rows for index in range(len(self.entries)))
        if not matched:
            raise ValueError("该预标注批次没有当前选中图片的记录")
        self.prelabels, self.prelabels_dir = imported, imported.root
        self.manifest["prelabels_dir"] = str(imported.root)
        _write_text(self.manifest_path, json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n")
        return matched

    def configure_prelabels(self, value, batch_output_dir: str | Path) -> int:
        """latest 查找当前数据集最近完成的批次；没有批次时仍可从零交互标注。"""
        if value is None or value == "":
            return 0
        if str(value).lower() != "latest":
            return self.load_prelabels(value)
        batch_root = Path(batch_output_dir)
        candidates = sorted(
            (path for path in batch_root.glob("*") if path.is_dir()
             and (path / "summary.json").is_file() and (path / "manifest.csv").is_file()),
            key=lambda path: path.name, reverse=True,
        )
        for path in candidates:
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            if Path(summary["source_dir"]).resolve() == self.source:
                return self.load_prelabels(path)
        return 0

    def prelabel_status(self, index: int) -> str:
        if self.is_saved(index):
            return "人工已保存"
        return self.prelabels.status(self._key(index)) if self.prelabels is not None else "未预标注"

    def load(self, index: int) -> list[dict]:
        """人工标签优先，其次返回待复核自动轮廓；保存状态仍由 is_saved 区分。"""
        record = self.records.get(self._key(index))
        if record is None:
            if self.prelabels is None:
                return []
            with Image.open(self.entries[index][1]) as image:
                return self.prelabels.load(self._key(index), *image.size)
        with Image.open(self.entries[index][1]) as image:
            if image.size != (record["width"], record["height"]):
                raise ValueError("原图尺寸已改变，不能复用旧标注；请使用新的输出目录")
        return [
            {**obj, "class_id": int(obj.get("class_id", 0)),
             "class_name": self.names[int(obj.get("class_id", 0))],
             "polygon": np.asarray(obj["polygon"], dtype=np.float32)}
            for obj in record["objects"]
        ]

    def save(self, index: int, objects: list[dict]) -> None:
        """只保存用户明确确认的实例；objects=[] 表示人工确认没有目标。"""
        split, source_image = self.entries[index]
        with Image.open(source_image) as image:
            width, height = image.size
        cleaned = []
        for obj in objects:
            class_id = int(obj.get("class_id", 0))
            if not 0 <= class_id < len(self.names):
                raise ValueError(f"实例类别编号超出配置范围：{class_id}")
            if obj.get("class_name", self.names[class_id]) != self.names[class_id]:
                raise ValueError("实例类别名称与编号不匹配，请重新选择类别")
            polygon = np.asarray(obj["polygon"], dtype=np.float64).reshape(-1, 2).copy()
            if len(polygon) < 3 or not np.isfinite(polygon).all():
                raise ValueError("标注多边形至少需要三个有限坐标点")
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            cleaned.append({
                "class_id": class_id,
                "class_name": self.names[class_id],
                "polygon": polygon.tolist(),
                "score": float(obj.get("score", 1.0)),
                "prompt": str(obj.get("prompt", "interactive")),
                "disconnected_pixels": int(obj.get("disconnected_pixels", 0)),
                "has_holes": bool(obj.get("has_holes", False)),
            })

        key = self._key(index)
        old_record = self.records.get(key)
        output_name = export_image_name(source_image, self.source, unique=self.unique_names)
        # 继续旧会话时沿用已有文件名；新多类别会话按相对路径生成稳定唯一名。
        image_relative = old_record["image"] if old_record else f"images/{split}/{output_name}"
        label_relative = old_record["label"] if old_record else f"labels/{split}/{Path(output_name).stem}.txt"
        destination_image = self.dataset / image_relative
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        # 原图字节直接复制，不缩放、不重新编码。
        temporary_image = destination_image.with_name(destination_image.name + ".tmp")
        shutil.copy2(source_image, temporary_image)
        temporary_image.replace(destination_image)
        _write_text(self.dataset / label_relative, to_yolo_text(cleaned, width, height))
        self.records[key] = {
            "source_relative_path": key,
            "split": split,
            "image": image_relative,
            "label": label_relative,
            "width": width,
            "height": height,
            "objects": cleaned,
        }
        _write_text(
            self.manifest_path,
            json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n",
        )
        # 每次保存即更新可训练的数据清单；耗时的 ZIP 只在导出时生成。
        self._write_dataset_lists()

    def _write_dataset_lists(self) -> None:
        for split in self.splits:
            self.dataset.joinpath("images", split).mkdir(parents=True, exist_ok=True)
            self.dataset.joinpath("labels", split).mkdir(parents=True, exist_ok=True)
            images = [
                "./" + record["image"]
                for _, record in sorted(self.records.items()) if record["split"] == split
            ]
            _write_text(self.dataset / f"{split}.txt", "\n".join(images) + ("\n" if images else ""))
        # 未保存的图片不复制，也不会作为空背景混进训练集。
        lines = [f"{split}: {split}.txt" for split in self.splits]
        # 名称使用 JSON 字符串写入 YAML，中文、空格与冒号均可正确解析。
        names = [f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(self.names)]
        _write_text(self.dataset / "data.yaml", "\n".join(lines + ["names:"] + names + [""]))
        _write_text(self.dataset / "classes.txt", "\n".join(self.names) + "\n")

    def export(self) -> dict:
        """导出当前已经确认的图片；不把还没标注的图片视为空图。"""
        if not self.records:
            raise ValueError("还没有保存任何标注，请先确认并保存至少一张图片")
        self._write_dataset_lists()
        summary = {
            "source_dir": str(self.source),
            "dataset_dir": str(self.dataset),
            "images": self.saved_count,
            "instances": sum(len(record["objects"]) for record in self.records.values()),
        }
        if bool(self.export_config["make_zip"]):
            zip_path = self.output / "sam3_interactive_cvat.zip"
            temporary_zip = zip_path.with_name(zip_path.name + ".tmp")
            # 只打包记录中确认过的图和标签，不收录遗留的临时文件。
            paths = ["data.yaml", "classes.txt"] + [f"{split}.txt" for split in self.splits]
            for record in self.records.values():
                paths.extend([record["image"], record["label"]])
            with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for relative in sorted(set(paths)):
                    archive.write(self.dataset / relative, relative)
            temporary_zip.replace(zip_path)
            summary["cvat_zip"] = str(zip_path)
        _write_text(self.output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        return summary

    def finish(self) -> dict:
        return self.export()
