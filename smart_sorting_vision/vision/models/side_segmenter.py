"""侧面 YOLO-seg 推理封装。

和顶部的 `vision/models/top_segmenter.py` 是两个独立模型，不共用权重：
侧视模型用的是 `data/datasets/side_seg/...` 那套数据集（12 个形状类 + 空托盘）。

对外只暴露一件事：

    detect(frame) -> {
        "found": bool,          # 有没有实例
        "reason": str,          # "ok" / "no_object" / "empty_mask" / "inference_error"
        "score": float,
        "class_id": int,
        "class_name": str,      # 形状类别，直接给 SideRecognizer 当 shape_result
        "mask": np.uint8 HxW,   # 0/255，与输入同尺寸
        "bbox": (x1, y1, x2, y2),
    }

注意
----
- `masks.xy` 已经是**原图坐标**（ultralytics 内部做过 letterbox 逆变换），
  所以现场 ROI 尺寸和训练图尺寸不完全一致时也能正确落回原图。
- 判"有没有物体"不看类别，只看有没有实例；空托盘在训练集里是"无标注"，
  模型不会给它出框。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class SideSegmenter:
    def __init__(
        self,
        model_path: str | Path,
        *,
        imgsz: int = 768,
        device: int | str = 0,
        confidence_threshold: float = 0.50,
        max_instances: int = 1,
        mask_threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.imgsz = int(imgsz)
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.max_instances = max(1, int(max_instances))
        self.mask_threshold = float(mask_threshold)

        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        # 触发一次后端初始化，避免第一件货时才编译 TensorRT context
        self.model.predict(
            np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8),
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        print(
            "[SIDE SEG] "
            f"model={self.model_path} imgsz={self.imgsz} "
            f"conf={self.confidence_threshold} max_instances={self.max_instances}"
        )

    # ------------------------------------------------------------------
    def _empty(self, reason: str) -> dict[str, Any]:
        return {
            "found": False,
            "reason": reason,
            "score": 0.0,
            "class_id": -1,
            "class_name": "no_object",
            "mask": None,
            "bbox": None,
        }

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        if frame is None or frame.size == 0:
            return self._empty("empty_frame")

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
                device=self.device,
                max_det=self.max_instances,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - 推理异常要能被上层看到原因
            print(f"[SIDE SEG] 推理失败: {type(exc).__name__}: {exc}")
            return self._empty("inference_error")

        if not results:
            return self._empty("no_object")

        result = results[0]
        boxes = result.boxes
        masks = result.masks
        if boxes is None or len(boxes) == 0:
            return self._empty("no_object")

        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        best = int(np.argmax(scores))

        mask = self._build_mask(result, best, frame.shape[:2])
        if mask is None or not mask.any():
            return self._empty("empty_mask")

        rows = np.flatnonzero(np.any(mask > 0, axis=1))
        cols = np.flatnonzero(np.any(mask > 0, axis=0))
        bbox = (
            int(cols[0]),
            int(rows[0]),
            int(cols[-1]) + 1,
            int(rows[-1]) + 1,
        )

        names = result.names or {}
        class_id = int(classes[best])
        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_id))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            class_name = str(names[class_id])
        else:
            class_name = str(class_id)

        return {
            "found": True,
            "reason": "ok",
            "score": float(scores[best]),
            "class_id": class_id,
            "class_name": class_name,
            "mask": mask,
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    def _build_mask(
        self,
        result: Any,
        index: int,
        frame_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """优先用 polygon（原图坐标），退化时用二值 mask 上采样。"""
        masks = getattr(result, "masks", None)
        if masks is None:
            return None

        height, width = frame_shape
        mask = np.zeros((height, width), dtype=np.uint8)

        polygons = getattr(masks, "xy", None)
        if polygons is not None and index < len(polygons):
            polygon = np.asarray(polygons[index], dtype=np.float64)
            if polygon.size >= 6:
                cv2.fillPoly(
                    mask,
                    [np.round(polygon).astype(np.int32)],
                    255,
                )
                return mask

        data = getattr(masks, "data", None)
        if data is None:
            return None
        binary = (data[index].detach().cpu().numpy() > self.mask_threshold).astype(np.uint8)
        if binary.shape != (height, width):
            binary = cv2.resize(
                binary,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        mask[binary > 0] = 255
        return mask

    # ------------------------------------------------------------------
    def warmup(self, frame: np.ndarray | None = None, runs: int = 2) -> dict[str, float]:
        if frame is None:
            frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        first_ms = 0.0
        times: list[float] = []
        for index in range(max(1, int(runs))):
            start = time.perf_counter()
            self.detect(frame)
            elapsed = (time.perf_counter() - start) * 1000.0
            if index == 0:
                first_ms = elapsed
            times.append(elapsed)
        return {
            "first_ms": first_ms,
            "steady_avg_ms": float(np.mean(times[1:])) if len(times) > 1 else first_ms,
        }
