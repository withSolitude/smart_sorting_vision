from __future__ import annotations

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
            qr_content（仅启用内容共识时向上层暴露）
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
    qr_content: str | None
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
        SideCamera.read() 返回的完整 mechanical ROI

    内部：
        ShapeClassifier（先判断物体/空背景）
            ↓ no_object：直接结束本周期
        SideObjectExtractor（只用于已有物体的颜色/二维码裁剪）
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
        extractor_config: dict[str, Any] | None = None,
        color_config: dict[str, Any] | None = None,
        qr_config: dict[str, Any] | None = None,
        qr_check_content: bool = False,
        no_object_labels: tuple[str, ...] | list[str] = (
            "no_object",
            "empty",
            "background",
        ),
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
        self.qr_check_content = bool(qr_check_content)

        print(
            "[SIDE CONFIG] "
            f"color={self.enable_color} "
            f"qr={self.enable_qr} "
            f"qr_content_check={self.qr_check_content} "
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

        self.color_classifier = (
            ColorClassifier(**dict(color_config or {}))
            if self.enable_color else None
        )

        self.qr_detector = (
            QRDetector(**dict(qr_config or {}))
            if self.enable_qr else None
        )

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
                no_object_labels=no_object_labels,
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
                qr_check_content=self.qr_check_content,
            )
        )

        self.consensus = (
            SideRecognitionConsensus(
                window_size=
                    consensus_window,
                required_hits=
                    consensus_required,
                check_qr_content=self.qr_check_content,
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
        shape: str | None = None,
        raw_result: dict[str, Any] | None = None,
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
            qr_content=None,
            shape=shape,
            color_confidence=0.0,
            shape_confidence=0.0,
            reason=
                str(reason),
            raw_result=raw_result,
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

        # 第一步始终对完整侧面 ROI 做形状分类。空托盘必须作为训练集中的
        # no_object/empty/background 类别，不再通过背景差分决定有无物体。
        if self.enable_shape and self.shape_classifier is not None:
            shape_probe = self.shape_classifier.classify(frame)
            object_present = bool(shape_probe.get("object_present", False))
            if not object_present:
                self.consensus.reset()
                probe_reason = str(shape_probe.get("reason", "unknown"))
                # 只有模型明确判为 no_object 才立即触发回待机；低置信度
                # 留在当前识别周期等待后续帧，避免一帧抖动误触发重抓。
                if probe_reason == "no_object":
                    return self._empty_result(
                        reason="no_object",
                        required_hits=self.required_hits,
                        found=False,
                        shape="no_object",
                        raw_result={"shape_probe": shape_probe},
                    )
                return self._empty_result(
                    reason=f"shape_pending:{probe_reason}",
                    required_hits=self.required_hits,
                    found=True,
                    shape="unknown",
                    raw_result={"shape_probe": shape_probe},
                )
        else:
            shape_probe = {
                "found": True,
                "object_present": True,
                "shape": "disabled",
                "confidence": 0.0,
                "reason": "shape_disabled",
            }

        # 形状已经确认有物体后，背景差分只作为颜色/二维码的裁剪辅助。
        # 裁剪失败时退回完整 ROI，绝不能把裁剪失败伪装成 no_object。
        object_result = self.extractor.extract(frame)
        if (
            object_result.crop is not None
            and object_result.crop_mask is not None
        ):
            crop = object_result.crop
            crop_mask = object_result.crop_mask
            crop_bbox = getattr(object_result, "crop_bbox", None)
        else:
            crop = frame
            crop_mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
            crop_bbox = None

        raw_result = self.recognizer.recognize(
            crop,
            crop_mask,
            shape_result=shape_probe,
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
                qr_content=(
                    str(final_result.get("qr_content") or "")
                    if self.qr_check_content
                    and bool(final_result.get("qr_present", False))
                    else None
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
                crop=crop,
                crop_mask=crop_mask,
                crop_bbox=crop_bbox,
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
            qr_content=(
                str(raw_result.get("qr_content") or "")
                if self.qr_check_content and bool(raw_result.get("qr_present", False))
                else None
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
            crop=crop,
            crop_mask=crop_mask,
            crop_bbox=crop_bbox,
        )
