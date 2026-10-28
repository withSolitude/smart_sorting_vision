"""交互标注的保存、断点继续和 YOLO/CVAT 导出，不修改采集原图。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Iterable
import zipfile

import numpy as np
from PIL import Image

from sam3_dataset import scan_dataset, to_yolo_text


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
    ):
        self.source = Path(source_dir).expanduser().resolve()
        self.output = Path(output_dir).expanduser().resolve()
        if self.source == self.output or self.source in self.output.parents:
            raise ValueError("交互标注输出目录必须在采集数据目录之外")
        self.dataset = self.output / "dataset"
        self.manifest_path = self.output / "annotations.json"
        self.export_config = export_config
        selected_splits = list(splits)
        files = scan_dataset(self.source, selected_splits, limit)
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
        else:
            if self.dataset.exists():
                raise FileExistsError("输出已有 dataset，但没有交互标注记录；请换一个输出目录")
            self.manifest = {"version": 1, "source_dir": str(self.source), "splits": [], "records": {}}
        self.records = self.manifest["records"]
        # 后来只打开一个 split 时，也保留之前已经确认过的其他 split。
        self.splits = list(dict.fromkeys(self.manifest["splits"] + selected_splits))
        self.manifest["splits"] = self.splits

    def _key(self, index: int) -> str:
        return self.entries[index][1].relative_to(self.source).as_posix()

    @property
    def saved_count(self) -> int:
        """整个会话已经人工确认的图片数，包括确认无目标的空标签。"""
        return len(self.records)

    def is_saved(self, index: int) -> bool:
        return self._key(index) in self.records

    def load(self, index: int) -> list[dict]:
        """返回已确认的像素多边形；未标注与确认空图由 is_saved 区分。"""
        record = self.records.get(self._key(index))
        if record is None:
            return []
        with Image.open(self.entries[index][1]) as image:
            if image.size != (record["width"], record["height"]):
                raise ValueError("原图尺寸已改变，不能复用旧标注；请使用新的输出目录")
        return [
            {**obj, "polygon": np.asarray(obj["polygon"], dtype=np.float32)}
            for obj in record["objects"]
        ]

    def save(self, index: int, objects: list[dict]) -> None:
        """只保存用户明确确认的实例；objects=[] 表示人工确认没有目标。"""
        split, source_image = self.entries[index]
        with Image.open(source_image) as image:
            width, height = image.size
        cleaned = []
        for obj in objects:
            polygon = np.asarray(obj["polygon"], dtype=np.float64).reshape(-1, 2).copy()
            if len(polygon) < 3 or not np.isfinite(polygon).all():
                raise ValueError("标注多边形至少需要三个有限坐标点")
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            cleaned.append({
                "polygon": polygon.tolist(),
                "score": float(obj.get("score", 1.0)),
                "prompt": str(obj.get("prompt", "interactive")),
                "disconnected_pixels": int(obj.get("disconnected_pixels", 0)),
                "has_holes": bool(obj.get("has_holes", False)),
            })

        image_relative = f"images/{split}/{source_image.name}"
        label_relative = f"labels/{split}/{source_image.stem}.txt"
        destination_image = self.dataset / image_relative
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        # 原图字节直接复制，不缩放、不重新编码。
        temporary_image = destination_image.with_name(destination_image.name + ".tmp")
        shutil.copy2(source_image, temporary_image)
        temporary_image.replace(destination_image)
        _write_text(self.dataset / label_relative, to_yolo_text(cleaned, width, height))
        key = self._key(index)
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
        _write_text(self.dataset / "data.yaml", "\n".join(lines + ["names:", "  0: object", ""]))
        _write_text(self.dataset / "classes.txt", "object\n")

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
