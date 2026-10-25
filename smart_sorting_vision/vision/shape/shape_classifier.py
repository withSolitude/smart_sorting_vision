from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np
from ultralytics import YOLO


class ShapeClassifier:
    """
    侧面形状分类器。

    职责只做一件事：

        object_result.crop
            ↓
        YOLO Classification
            ↓
        shape + confidence

    不负责：
        - 打开相机
        - 背景采集
        - 背景差分
        - 物体裁剪

    这样与颜色、二维码模块保持同一套上层接口：

        SideCamera
            ↓
        SideObjectExtractor
            ↓
        object_result.crop
            ├── ColorClassifier
            ├── QRDetector
            └── ShapeClassifier
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        imgsz: int = 224,
        device: str | int | None = None,
        confidence_threshold: float = 0.50,
        topk: int = 3,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"形状分类模型不存在：{self.model_path}"
            )

        self.imgsz = int(
            imgsz
        )

        self.device = device

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.topk = max(
            1,
            int(
                topk
            ),
        )

        self.model = YOLO(
            str(
                self.model_path
            )
        )

    def warmup(
        self,
        image: np.ndarray | None = None,
        runs: int = 2,
    ) -> dict[str, float | int]:
        """
        在正式侧面识别前预热分类后端。

        默认使用与 imgsz 一致的虚拟 BGR 图，不依赖识别台当前是否放有物体；
        classify() 会完整经过 TensorRT/PyTorch 推理、probs 解析和 NumPy Top-K。
        """
        if image is None:
            image = np.zeros(
                (self.imgsz, self.imgsz, 3),
                dtype=np.uint8,
            )

        self._validate_image(image)
        runs = max(1, int(runs))
        timings_ms: list[float] = []

        for index in range(runs):
            start = time.perf_counter()
            self.classify(image)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)
            print(
                f"[WARMUP][SIDE] {index + 1}/{runs} "
                f"{elapsed_ms:.1f} ms"
            )

        steady = timings_ms[1:] if len(timings_ms) > 1 else timings_ms
        return {
            "runs": runs,
            "first_ms": float(timings_ms[0]),
            "steady_avg_ms": float(sum(steady) / len(steady)),
            "max_ms": float(max(timings_ms)),
        }

    @staticmethod
    def _validate_image(
        image: np.ndarray,
    ) -> None:

        if image is None:

            raise ValueError(
                "ShapeClassifier 输入图像为空"
            )

        if not isinstance(
            image,
            np.ndarray,
        ):

            raise TypeError(
                "ShapeClassifier 输入必须为 numpy.ndarray"
            )

        if (
            image.ndim != 3
            or
            image.shape[2] != 3
        ):

            raise ValueError(
                "ShapeClassifier 输入必须是 BGR 三通道图像"
            )

        if image.size == 0:

            raise ValueError(
                "ShapeClassifier 输入图像尺寸为 0"
            )

    @staticmethod
    def _class_name(
        names,
        class_id: int,
    ) -> str:

        if isinstance(
            names,
            dict,
        ):

            return str(
                names.get(
                    int(
                        class_id
                    ),
                    str(
                        class_id
                    ),
                )
            )

        if isinstance(
            names,
            (
                list,
                tuple,
            ),
        ):

            if (
                0
                <=
                int(
                    class_id
                )
                <
                len(
                    names
                )
            ):

                return str(
                    names[
                        int(
                            class_id
                        )
                    ]
                )

        return str(
            class_id
        )

    def classify(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:

        self._validate_image(
            image
        )

        predict_kwargs = {
            "source":
                image,

            "imgsz":
                self.imgsz,

            "verbose":
                False,
        }

        if self.device is not None:

            predict_kwargs[
                "device"
            ] = self.device

        results = self.model.predict(
            **predict_kwargs
        )

        if (
            not results
            or
            results[0].probs
            is None
        ):

            return {
                "found":
                    False,

                "shape":
                    "unknown",

                "confidence":
                    0.0,

                "class_id":
                    None,

                "topk":
                    [],

                "reason":
                    "no_classification_result",
            }

        result = results[0]

        probs = result.probs

        class_id = int(
            probs.top1
        )

        confidence = float(
            probs.top1conf.item()
        )

        shape = self._class_name(
            result.names,
            class_id,
        )

        # --------------------------------------------------------
        # Top-K
        # --------------------------------------------------------

        probability_array = (
            probs.data
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        top_indices = (
            np.argsort(
                probability_array
            )[::-1][
                :self.topk
            ]
        )

        topk_result = []

        for index in top_indices:

            topk_result.append(
                {
                    "class_id":
                        int(
                            index
                        ),

                    "shape":
                        self._class_name(
                            result.names,
                            int(
                                index
                            ),
                        ),

                    "confidence":
                        float(
                            probability_array[
                                int(
                                    index
                                )
                            ]
                        ),
                }
            )

        found = (
            confidence
            >=
            self.confidence_threshold
        )

        return {
            "found":
                bool(
                    found
                ),

            "shape":
                (
                    shape
                    if found
                    else "unknown"
                ),

            "raw_shape":
                shape,

            "confidence":
                confidence,

            "class_id":
                class_id,

            "topk":
                topk_result,

            "reason":
                (
                    "ok"
                    if found
                    else "low_confidence"
                ),
        }
