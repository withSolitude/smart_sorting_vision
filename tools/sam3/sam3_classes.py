"""从 YAML 和采集目录读取标注类别；SAM 3 只负责轮廓，不猜细分形状。"""
from __future__ import annotations

from pathlib import Path


def class_names(config: dict) -> list[str]:
    """列表顺序就是 YOLO 类别 ID；旧顶部配置默认保留 object 单类别。"""
    definitions = config.get("annotation", {}).get("classes")
    if definitions is None:
        return ["object"]
    names = [str(item["name"]).strip() for item in definitions]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("annotation.classes 的类别名不能为空或重复")
    return names


def class_for_path(path: Path, source: Path, config: dict) -> int | None:
    """目录格式为 split/具体形状/颜色/图片；空背景目录返回 None。"""
    annotation = config.get("annotation", {})
    if not annotation.get("class_from_folder", False):
        return 0
    parts = path.resolve().relative_to(source.resolve()).parts
    if len(parts) < 3:
        raise ValueError(f"图片必须放在 split/形状/图片 或 split/形状/颜色/图片：{path}")
    name = parts[1]
    empty_names = {str(item).casefold() for item in annotation.get("empty_folders", [])}
    if name.casefold() in empty_names:
        return None
    names = class_names(config)
    if name not in names:
        raise ValueError(f"未配置形状目录 {name!r}，请先补充 annotation.classes：{path}")
    return names.index(name)
