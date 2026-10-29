"""读取批量预标注的 YOLO 多边形，按源相对路径交给交互窗口复核。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml


class PrelabelSource:
    """自动结果只是草稿；读取此对象不会把图片标记成人工确认。"""

    def __init__(self, path: str | Path, source: Path, names: list[str]):
        self.root = Path(path).expanduser().resolve()
        # 文件选择器选到批次内的 dataset 子目录时，也能找到上一级清单。
        if self.root.name == "dataset" and (self.root.parent / "manifest.csv").is_file():
            self.root = self.root.parent
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        if Path(summary["source_dir"]).resolve() != source.resolve():
            raise ValueError("预标注批次对应另一份原图目录，请选择当前数据集生成的批次")
        self.dataset = self.root / "dataset"
        data = yaml.safe_load((self.dataset / "data.yaml").read_text(encoding="utf-8"))
        declared = data["names"]
        declared = [declared[key] for key in sorted(declared, key=int)] if isinstance(declared, dict) else declared
        if declared != names:
            raise ValueError("预标注的类别名称或顺序与当前 YAML 不一致")
        self.names = list(names)
        with (self.root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.rows = {row["source_relative_path"]: row for row in csv.DictReader(stream)}

    def status(self, key: str) -> str:
        row = self.rows.get(key)
        if row is None:
            return "未预标注"
        if int(row["instances_auto"]) == 0:
            return "自动零检出，待检查或补标"
        suffix = f"；{row['review_reasons']}" if row.get("review_reasons") else ""
        return "自动候选，待复核" + suffix

    def load(self, key: str, width: int, height: int) -> list[dict]:
        row = self.rows.get(key)
        if row is None:
            return []
        if (int(row["width"]), int(row["height"])) != (width, height):
            raise ValueError("当前原图尺寸与自动预标注不一致，请为这批原图重新生成候选")
        scores = row.get("scores", "").split(";")
        objects = []
        for index, line in enumerate((self.dataset / row["label"]).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 7 or len(fields) % 2 != 1:
                raise ValueError(f"不是有效的 YOLO 分割多边形：{row['label']}")
            class_id = int(fields[0])
            polygon = np.asarray(fields[1:], dtype=np.float64).reshape(-1, 2)
            if not 0 <= class_id < len(self.names) or not np.isfinite(polygon).all() or (
                (polygon < 0).any() or (polygon > 1).any()
            ):
                raise ValueError(f"类别或归一化轮廓坐标无效：{row['label']}")
            polygon *= [width, height]
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            objects.append({
                "class_id": class_id, "class_name": self.names[class_id],
                "polygon": polygon.astype(np.float32),
                "score": float(scores[index]) if index < len(scores) and scores[index] else 1.0,
                "prompt": "imported prelabel",
                "disconnected_pixels": 0, "has_holes": False,
            })
        return objects
