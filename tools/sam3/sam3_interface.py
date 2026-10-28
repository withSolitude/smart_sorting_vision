"""官方 SAM 3 图片接口：本地权重、文字提示、框提示和交互式正负点。

输入路径、PIL 图片或 OpenCV BGR 图片；输出原图大小的实例 mask。
所有模型导入延迟到创建接口时，导出和检查数据集时不需要先加载 GPU。
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml


BASE = Path(__file__).resolve().parent


def load_config(path: str | Path = BASE / "sam3_config.yaml") -> dict:
    """相对路径统一以配置文件所在目录为起点。"""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, keys in (
        ("model", ("source_dir", "checkpoint")),
        ("dataset", ("source_dir", "output_dir")),
    ):
        for key in keys:
            value = Path(config[section][key]).expanduser()
            if not value.is_absolute():
                value = path.parent / value
            config[section][key] = value.resolve()
    return config


def as_rgb_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """数组按 OpenCV 的 BGR 解释；路径直接用 PIL，支持中文文件名。"""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("图片数组必须为 H×W×3 的 BGR 图像")
    return Image.fromarray(np.ascontiguousarray(array[:, :, ::-1]))


class Sam3Interface:
    """只加载一次模型；同一图片的多次提示共享图像编码结果。"""

    def __init__(self, model_config: dict, *, enable_interactivity: bool = False):
        source = Path(model_config["source_dir"]).resolve()
        checkpoint = Path(model_config["checkpoint"]).resolve()
        if not (source / "sam3" / "model_builder.py").is_file():
            raise FileNotFoundError(f"官方源码目录不正确：{source}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到本地权重：{checkpoint}")
        bpe = source / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe.is_file():
            raise FileNotFoundError(f"源码缺少分词词表：{bpe}")
        # 优先使用用户提供的官方源码，不加载其他环境里的同名包。
        sys.path.insert(0, str(source))
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        self.device = torch.device(model_config["device"])
        self.precision = str(model_config["precision"])
        # 官方融合算子内部固定使用 BF16，本接口只开放对应的 CUDA 路径。
        if self.device.type != "cuda" or self.precision != "bfloat16":
            raise ValueError("当前官方图片接口使用 CUDA GPU，precision 请保持 bfloat16")
        if not torch.cuda.is_available():
            raise RuntimeError("当前 Python 的 PyTorch 无法使用 CUDA")
        torch.cuda.set_device(self.device.index if self.device.index is not None else 0)

        # 显式关闭 Hugging Face 下载，只使用 sam3.pt 和源码中的词表。
        # 官方构造器只识别字符串 cuda，因此 cuda:0 也先按 cuda 构造。
        self.model = build_sam3_image_model(
            bpe_path=str(bpe),
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=self.device.type,
            eval_mode=True,
            # 只有交互入口加载点提示分支，原来的文字批处理保持原配置。
            enable_inst_interactivity=enable_interactivity,
            compile=False,
        ).to(self.device).eval()
        self.processor = Sam3Processor(
            self.model,
            device=str(self.device),
            confidence_threshold=float(model_config["confidence"]),
        )
        self.state = None
        self.image_size = None

    def _autocast(self):
        dtype = getattr(self.torch, self.precision)
        return self.torch.autocast(device_type="cuda", dtype=dtype)

    def set_image(self, image: str | Path | Image.Image | np.ndarray) -> None:
        """载入一张新图片，并缓存编码；不会改变图片尺寸和坐标。"""
        image = as_rgb_image(image)
        self.image_size = image.size
        self.state = None  # 切换图片时释放上一张的 GPU 特征。
        with self.torch.inference_mode(), self._autocast():
            self.state = self.processor.set_image(image)

    def _results(self, output: dict, prompt: str) -> list[dict]:
        masks = output["masks"].detach().cpu().numpy()
        boxes = output["boxes"].detach().float().cpu().numpy()
        scores = output["scores"].detach().float().cpu().numpy()
        return [
            {
                "mask": mask.reshape(mask.shape[-2:]).astype(bool),
                "score": float(score),
                "box": box.tolist(),
                "prompt": prompt,
            }
            for mask, box, score in zip(masks, boxes, scores)
        ]

    def predict_text(self, prompts: str | list[str]) -> list[dict]:
        """执行一个或多个文字提示；此处保留所有结果，由导出层做 mask 去重。"""
        if self.state is None:
            raise RuntimeError("先调用 set_image(image)")
        if isinstance(prompts, str):
            prompts = [prompts]
        results = []
        with self.torch.inference_mode(), self._autocast():
            for prompt in prompts:
                # 不把上一条文字/框提示带入下一次独立预测。
                self.processor.reset_all_prompts(self.state)
                output = self.processor.set_text_prompt(
                    prompt=prompt, state=self.state
                )
                results.extend(self._results(output, prompt))
        return results

    def predict_boxes(
        self,
        positive_boxes: list[list[float]],
        negative_boxes: list[list[float]] | None = None,
        text: str | None = None,
    ) -> list[dict]:
        """框坐标为原图像素 [x1,y1,x2,y2]，自动转换为官方归一化格式。

        框是“示例提示”，SAM 3 可能同时找到画面内相似物体，
        并不保证只返回框内的一个实例。
        """
        if self.state is None:
            raise RuntimeError("先调用 set_image(image)")
        if not positive_boxes:
            raise ValueError("至少提供一个正框")
        width, height = self.image_size
        boxes = [(box, True) for box in positive_boxes]
        boxes.extend((box, False) for box in (negative_boxes or []))
        with self.torch.inference_mode(), self._autocast():
            self.processor.reset_all_prompts(self.state)
            if text:
                self.processor.set_text_prompt(prompt=text, state=self.state)
            for (x1, y1, x2, y2), label in boxes:
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError("提示框必须在原图内，且 x1<x2、y1<y2")
                box = [
                    (x1 + x2) / (2 * width),
                    (y1 + y2) / (2 * height),
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ]
                output = self.processor.add_geometric_prompt(
                    box=box, label=label, state=self.state
                )
            return self._results(output, text or "box prompt")

    def predict_points(
        self,
        point_coords: list[list[float]],
        point_labels: list[int],
        box: list[float] | None = None,
    ) -> list[dict]:
        """交互分割一个目标，返回同一目标的多个备选 mask。

        点坐标使用原图像素 [x, y]，1 为保留目标的正点，0 为排除区域的负点；
        可同时给出原图 [x1, y1, x2, y2] 框，也支持仅给框、点列表为空。
        返回按质量分数排序的候选，不能将它们当成多个物体一起加入标签。
        """
        if self.state is None:
            raise RuntimeError("先调用 set_image(image)")
        if self.model.inst_interactive_predictor is None:
            raise RuntimeError("点交互需使用 enable_interactivity=True 创建接口")

        width, height = self.image_size
        coords = np.asarray(point_coords, dtype=np.float32)
        labels = np.asarray(point_labels)
        if coords.size == 0:
            coords = np.empty((0, 2), dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("点坐标必须为 N×2 的 [x, y] 列表")
        if labels.ndim != 1 or len(labels) != len(coords):
            raise ValueError("每个点必须对应一个标签，正点为1、负点为0")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("点标签只能为1（正点）或0（负点）")
        if not np.isfinite(coords).all() or (
            len(coords) and (
                (coords[:, 0] < 0).any() or (coords[:, 0] >= width).any()
                or (coords[:, 1] < 0).any() or (coords[:, 1] >= height).any()
            )
        ):
            raise ValueError("提示点必须位于原图范围内")

        bounds = None
        if box is not None:
            bounds = np.asarray(box, dtype=np.float32)
            if bounds.shape != (4,) or not np.isfinite(bounds).all():
                raise ValueError("提示框必须为 [x1, y1, x2, y2]")
            x1, y1, x2, y2 = bounds
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("提示框必须在原图内，且 x1<x2、y1<y2")
        if bounds is None and not np.any(labels == 1):
            raise ValueError("至少添加一个正点，或先框选目标")

        with self.torch.inference_mode(), self._autocast():
            masks, scores, _ = self.model.predict_inst(
                self.state,
                point_coords=coords if len(coords) else None,
                point_labels=labels.astype(np.int32) if len(coords) else None,
                box=bounds,
                multimask_output=True,
                return_logits=False,
                normalize_coords=True,
            )

        candidates = []
        # 这里是候选质量分数，不用文字检测的 confidence 阈值过滤。
        for mask, score in zip(masks, scores):
            mask = np.asarray(mask, dtype=bool)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            candidates.append({
                "mask": mask,
                "score": float(score),
                "box": [float(xs.min()), float(ys.min()),
                        float(xs.max() + 1), float(ys.max() + 1)],
                "prompt": "interactive points",
            })
        return sorted(candidates, key=lambda item: item["score"], reverse=True)

    def predict(self, image, prompts: str | list[str]) -> list[dict]:
        """批量预标注入口：设置图片后执行文字提示。"""
        self.set_image(image)
        return self.predict_text(prompts)
