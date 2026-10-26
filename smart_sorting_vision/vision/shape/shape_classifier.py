from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np
from ultralytics import YOLO


class ShapeClassifier:
    """
    侧面形状/空背景分类器。

    职责只做一件事：

        SideCamera 完整 mechanical ROI
            ↓
        YOLO Classification
            ↓
        shape / no_object + confidence

    不负责：
        - 打开相机
        - 背景采集
        - 背景差分
        - 物体裁剪

    正式流程中，完整机械 ROI 先进入本分类器；只有确认有物体后，
    上层才会用裁剪图运行颜色和二维码模块。
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        imgsz: int = 224,
        device: str | int | None = None,
        confidence_threshold: float = 0.50,
        topk: int = 3,
        no_object_labels: tuple[str, ...] | list[str] = (
            "no_object",
            "empty",
            "background",
        ),
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

        # 这些类别来自训练集的空托盘/空背景样本。
        # 形状分类器先用它们判断“是否有物体”，再决定是否运行颜色和二维码。
        self.no_object_labels = {
            str(label).strip().casefold()
            for label in no_object_labels
            if str(label).strip()
        }

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

                "object_present":
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

        is_no_object = (
            shape.strip().casefold()
            in self.no_object_labels
        )
        # found 表示模型给出了足够可信的 top1；object_present 才表示
        # 该 top1 是真实物体。这样 no_object 可以被上层明确区分。
        object_present = bool(found and not is_no_object)

        return {
            "found":
                bool(
                    found
                ),

            "object_present":
                object_present,

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
                    "no_object"
                    if is_no_object and found
                    else "ok"
                    if object_present
                    else "low_confidence"
                ),
        }
