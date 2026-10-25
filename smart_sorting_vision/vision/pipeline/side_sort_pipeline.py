from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vision.processors.side_object_extractor import (
    SideObjectExtractor,
)
from vision.color.color_classifier import (
    ColorClassifier,
)
from vision.code_reader.qr_detector import (
    QRDetector,
)
from vision.shape.shape_classifier import (
    ShapeClassifier,
)
from vision.pipeline.side_recognizer import (
    SideRecognizer,
)
from vision.pipeline.side_recognition_consensus import (
    SideRecognitionConsensus,
)


@dataclass
class SideSortPipelineResult:
    """
    侧面识别流水线统一输出。

    注意：
        这里只输出稳定的识别属性：
            color
            qr_present
            shape

        暂时不在这里硬编码 Box_ID 映射。
        Box_ID 应由后续 BoxManager / 规则配置决定。
    """

    found: bool

    ready: bool

    votes: int
    window: int
    required_hits: int

    color: str | None
    qr_present: bool | None
    shape: str | None

    color_confidence: float
    shape_confidence: float

    reason: str

    raw_result: dict[str, Any] | None
    final_result: dict[str, Any] | None

    crop: np.ndarray | None = None
    crop_mask: np.ndarray | None = None
    crop_bbox: tuple[int, int, int, int] | None = None


class SideSortPipeline:
    """
    侧面识别正式流水线。

    输入：
        SideCamera.read() 返回的 mechanical ROI

    内部：
        SideObjectExtractor
            ↓
        crop + crop_mask
            ↓
        SideRecognizer
            ├─ ColorClassifier
            ├─ QRDetector
            └─ ShapeClassifier
            ↓
        SideRecognitionConsensus
            ↓
        READY

    当前沿用现有侧面识别策略：
        5 帧窗口
        至少 4 帧完整结果一致

    与顶部抓取不同：
        顶部为了响应速度已经改为 2/2；
        侧面这里仍保持现有 4/5，因为颜色/QR/形状组合识别
        比单纯位置确认更容易出现单帧波动。
    """

    def __init__(
        self,
        *,
        background_path: str | Path,
        shape_model_path: str | Path,
        shape_device=0,
        shape_imgsz: int = 224,
        shape_confidence_threshold: float = 0.50,
        shape_topk: int = 3,
        padding_ratio: float = 0.15,
        consensus_window: int = 5,
        consensus_required: int = 4,
        presence_window: int = 4,
        presence_required: int = 3,
        extractor_config: dict[str, Any] | None = None,
        enable_color: bool = True,
        enable_qr: bool = True,
        enable_shape: bool = True,
    ) -> None:
        self.background_path = Path(
            background_path
        )

        self.enable_color = bool(enable_color)
        self.enable_qr = bool(enable_qr)
        self.enable_shape = bool(enable_shape)

        print(
            "[SIDE CONFIG] "
            f"color={self.enable_color} "
            f"qr={self.enable_qr} "
            f"shape={self.enable_shape}"
        )

        self.shape_model_path = Path(
            shape_model_path
        )

        if not self.background_path.exists():
            raise FileNotFoundError(
                "侧面固定背景不存在：\n"
                f"{self.background_path}"
            )

        if not self.shape_model_path.exists():
            raise FileNotFoundError(
                "侧面形状模型不存在：\n"
                f"{self.shape_model_path}"
            )

        extractor_cfg = dict(
            extractor_config or {}
        )
        extractor_cfg.setdefault(
            "padding_ratio",
            padding_ratio,
        )

        self.extractor = (
            SideObjectExtractor(
                **extractor_cfg
            )
        )

        self.presence_window = max(
            1,
            int(presence_window),
        )
        self.presence_required = max(
            1,
            int(presence_required),
        )
        if self.presence_required > self.presence_window:
            raise ValueError(
                "presence_required 不能大于 presence_window"
            )
        self._presence_history = deque(
            maxlen=self.presence_window
        )

        self.color_classifier = ColorClassifier() if self.enable_color else None

        self.qr_detector = QRDetector() if self.enable_qr else None

        self.shape_classifier = (
            ShapeClassifier(
                model_path=
                    self.shape_model_path,
                imgsz=
                    shape_imgsz,
                device=
                    shape_device,
                confidence_threshold=
                    shape_confidence_threshold,
                topk=
                    shape_topk,
            )
            if self.enable_shape else None
        )

        self.recognizer = (
            SideRecognizer(
                color_classifier=
                    self.color_classifier,
                qr_detector=
                    self.qr_detector,
                shape_classifier=
                    self.shape_classifier,
                enable_color=self.enable_color,
                enable_qr=self.enable_qr,
                enable_shape=self.enable_shape,
            )
        )

        self.consensus = (
            SideRecognitionConsensus(
                window_size=
                    consensus_window,
                required_hits=
                    consensus_required,
            )
        )

        self.required_hits = int(
            consensus_required
        )

        self.window_size = int(
            consensus_window
        )

        self.background = (
            self.extractor.load_background(
                self.background_path
            )
        )

    def reset(
        self,
    ) -> None:
        self.consensus.reset()
        self._presence_history.clear()

    def _check_frame(
        self,
        frame: np.ndarray,
    ) -> None:
        if frame is None:
            raise ValueError(
                "Side frame 不能为 None"
            )

        if (
            frame.shape
            !=
            self.background.shape
        ):
            raise RuntimeError(
                "当前 SideCamera ROI 与固定背景尺寸不一致："
                f"\ncurrent    = {frame.shape}"
                f"\nbackground = {self.background.shape}"
            )

    @staticmethod
    def _empty_result(
        *,
        reason: str,
        required_hits: int,
        found: bool = False,
        votes: int = 0,
        window: int = 0,
    ) -> SideSortPipelineResult:
        return SideSortPipelineResult(
            found=bool(found),
            ready=False,
            votes=int(votes),
            window=int(window),
            required_hits=
                required_hits,
            color=None,
            qr_present=None,
            shape=None,
            color_confidence=0.0,
            shape_confidence=0.0,
            reason=
                str(reason),
            raw_result=None,
            final_result=None,
            crop=None,
            crop_mask=None,
            crop_bbox=None,
        )

    def process(
        self,
        frame: np.ndarray,
    ) -> SideSortPipelineResult:
        self._check_frame(
            frame
        )

        object_result = (
            self.extractor.extract(
                frame
            )
        )

        object_candidate = bool(
            object_result.found
            and object_result.crop is not None
            and object_result.crop_mask is not None
        )
        self._presence_history.append(
            object_candidate
        )

        presence_votes = int(
            sum(self._presence_history)
        )
        presence_window_now = int(
            len(self._presence_history)
        )
        presence_confirmed = bool(
            object_candidate
            and presence_votes >= self.presence_required
        )

        if not object_candidate:
            # 当前帧没有提取到物体：识别属性共识必须清空，避免把上一件
            # 或上一段抖动结果带到后续帧。是否真的“掉落”由 Runtime 的
            # 时间+帧数双门槛继续判断。
            self.consensus.reset()

            return self._empty_result(
                reason=getattr(
                    object_result,
                    "reason",
                    "no_object",
                ),
                required_hits=self.required_hits,
                found=False,
            )

        if not presence_confirmed:
            # 找到单帧候选，但还没有通过 3/4（配置值）存在确认。
            # 这里故意返回 found=True，避免 Runtime 把“正在确认的真实物体”
            # 误判为掉落；同时不进入颜色/QR/形状识别。
            self.consensus.reset()
            return self._empty_result(
                reason=(
                    "presence_pending:"
                    f"{presence_votes}/{presence_window_now},"
                    f"required={self.presence_required}/"
                    f"{self.presence_window}"
                ),
                required_hits=self.required_hits,
                found=True,
                votes=presence_votes,
                window=presence_window_now,
            )

        raw_result = (
            self.recognizer.recognize(
                object_result.crop,
                object_result.crop_mask,
            )
        )

        stable_state = (
            self.consensus.update(
                raw_result
            )
        )

        ready = bool(
            stable_state.get(
                "ready",
                False,
            )
        )

        votes = int(
            stable_state.get(
                "votes",
                0,
            )
        )

        window = int(
            stable_state.get(
                "window",
                0,
            )
        )

        final_result = (
            stable_state.get(
                "result"
            )
            if ready
            else None
        )

        if (
            ready
            and
            final_result is not None
        ):
            return SideSortPipelineResult(
                found=True,
                ready=True,
                votes=votes,
                window=window,
                required_hits=
                    self.required_hits,
                color=str(
                    final_result.get(
                        "color",
                        "unknown",
                    )
                ),
                qr_present=bool(
                    final_result.get(
                        "qr_present",
                        False,
                    )
                ),
                shape=str(
                    final_result.get(
                        "shape",
                        "unknown",
                    )
                ),
                color_confidence=float(
                    final_result.get(
                        "color_confidence",
                        0.0,
                    )
                ),
                shape_confidence=float(
                    final_result.get(
                        "shape_confidence",
                        0.0,
                    )
                ),
                reason="ready",
                raw_result=
                    raw_result,
                final_result=
                    final_result,
                crop=
                    object_result.crop,
                crop_mask=
                    object_result.crop_mask,
                crop_bbox=getattr(
                    object_result,
                    "crop_bbox",
                    None,
                ),
            )

        return SideSortPipelineResult(
            found=True,
            ready=False,
            votes=votes,
            window=window,
            required_hits=
                self.required_hits,
            color=str(
                raw_result.get(
                    "color",
                    "unknown",
                )
            ),
            qr_present=bool(
                raw_result.get(
                    "qr_present",
                    False,
                )
            ),
            shape=str(
                raw_result.get(
                    "shape",
                    "unknown",
                )
            ),
            color_confidence=float(
                raw_result.get(
                    "color_confidence",
                    0.0,
                )
            ),
            shape_confidence=float(
                raw_result.get(
                    "shape_confidence",
                    0.0,
                )
            ),
            reason="waiting_consensus",
            raw_result=
                raw_result,
            final_result=None,
            crop=
                object_result.crop,
            crop_mask=
                object_result.crop_mask,
            crop_bbox=getattr(
                object_result,
                "crop_bbox",
                None,
            ),
        )
