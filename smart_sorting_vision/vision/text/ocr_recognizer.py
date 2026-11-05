"""侧面文字识别（OCR）：与颜色、二维码同级，可在 YAML 里单独开关。

默认 `enabled: false`，不加载任何模型；只有开启后第一次识别时才惰性
初始化引擎，没装任何 OCR 依赖也不会影响分拣主流程。

`engine: auto` 依次尝试：

    1. rapidocr_onnxruntime —— onnxruntime，轻量，中英文混合，Jetson 首选；
    2. easyocr             —— torch，需要下载模型；
    3. pytesseract         —— 需要系统安装 tesseract 可执行文件。

也可以显式指定 `engine: rapidocr | easyocr | tesseract`。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TextResult:
    found: bool
    text: str = ""
    confidence: float = 0.0
    engine: str = ""
    reason: str = ""
    box: tuple[int, int, int, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": bool(self.found),
            "text": self.text,
            "confidence": float(self.confidence),
            "engine": self.engine,
            "reason": self.reason,
            "box": self.box,
        }


def _box_xyxy(points) -> tuple[int, int, int, int] | None:
    try:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    except Exception:
        return None
    if pts.size == 0 or not np.all(np.isfinite(pts)):
        return None
    return (
        int(np.floor(pts[:, 0].min())),
        int(np.floor(pts[:, 1].min())),
        int(np.ceil(pts[:, 0].max())),
        int(np.ceil(pts[:, 1].max())),
    )


class TextRecognizer:
    """在侧面物体 crop + mask 上做文字识别。"""

    # 语言名映射：easyocr 用 ch_sim，tesseract 用 chi_sim。
    TESSERACT_LANG = {
        "ch_sim": "chi_sim",
        "ch_tra": "chi_tra",
        "en": "eng",
        "ja": "jpn",
        "ko": "kor",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.engine = str(cfg.get("engine", "auto")).strip().lower()
        raw_languages = cfg.get("languages", ("ch_sim", "en"))
        if isinstance(raw_languages, str):
            raw_languages = [raw_languages]
        self.languages = tuple(
            str(item).strip() for item in raw_languages if str(item).strip()
        ) or ("ch_sim", "en")
        self.min_confidence = float(cfg.get("min_confidence", 0.35))
        self.mask_erode_px = max(0, int(cfg.get("mask_erode_px", 2)))
        self.background = str(cfg.get("mask_background", "white")).strip().lower()
        self.upscale = max(1.0, float(cfg.get("upscale", 2.0)))
        self.psm = int(cfg.get("psm", 7))
        self.max_chars = max(1, int(cfg.get("max_chars", 64)))

        self._impl: Any = None
        self._engine_name = ""
        self._load_error = ""
        self._load_attempted = False

    # ------------------------------------------------------------------
    # 引擎
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        if self._load_attempted:
            return self._impl is not None
        self._load_attempted = True

        if self.engine in ("", "auto"):
            candidates = ("rapidocr", "easyocr", "tesseract")
        else:
            candidates = (self.engine,)

        for name in candidates:
            try:
                if name in ("rapidocr", "rapidocr_onnxruntime"):
                    from rapidocr_onnxruntime import RapidOCR

                    self._impl = ("rapidocr", RapidOCR())
                elif name == "easyocr":
                    import easyocr

                    self._impl = (
                        "easyocr",
                        easyocr.Reader(list(self.languages), gpu=True, verbose=False),
                    )
                elif name in ("tesseract", "pytesseract"):
                    import pytesseract

                    self._impl = ("tesseract", pytesseract)
                else:
                    self._load_error = f"unknown_engine:{name}"
                    continue
                self._engine_name = self._impl[0]
                self._load_error = ""
                return True
            except Exception as exc:  # 缺依赖/模型下载失败都只记录，不抛出
                self._load_error = f"{name}:{exc}"
                self._impl = None
        return False

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    def _prepare(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        if (
            crop is None
            or mask is None
            or crop.ndim != 3
            or crop.shape[2] != 3
            or mask.ndim != 2
            or crop.shape[:2] != mask.shape[:2]
        ):
            return None

        working = mask
        if self.mask_erode_px > 0:
            kernel_size = self.mask_erode_px * 2 + 1
            working = cv2.erode(
                mask,
                np.ones((kernel_size, kernel_size), dtype=np.uint8),
                iterations=1,
            )

        ys, xs = np.nonzero(working)
        if xs.size == 0:
            return None

        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None

        roi = crop[y0:y1, x0:x1].copy()
        roi_mask = working[y0:y1, x0:x1]
        fill = 0 if self.background == "black" else 255
        cleaned = np.full_like(roi, fill)
        cleaned[roi_mask > 0] = roi[roi_mask > 0]

        if self.upscale > 1.01:
            cleaned = cv2.resize(
                cleaned,
                None,
                fx=self.upscale,
                fy=self.upscale,
                interpolation=cv2.INTER_CUBIC,
            )
        return cleaned

    # ------------------------------------------------------------------
    # 各引擎调用：统一返回 [(text, confidence, box_xyxy)]
    # ------------------------------------------------------------------

    def _run(
        self,
        name: str,
        impl: Any,
        image: np.ndarray,
    ) -> list[tuple[str, float, tuple[int, int, int, int] | None]]:
        items: list[tuple[str, float, tuple[int, int, int, int] | None]] = []

        if name == "rapidocr":
            result, _ = impl(image)
            for entry in result or []:
                if not entry or len(entry) < 3:
                    continue
                items.append(
                    (str(entry[1]), float(entry[2]), _box_xyxy(entry[0]))
                )
            return items

        if name == "easyocr":
            for box, text, score in impl.readtext(image, detail=1):
                items.append((str(text), float(score), _box_xyxy(box)))
            return items

        # tesseract
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lang = "+".join(
            self.TESSERACT_LANG.get(item, item) for item in self.languages
        )
        config = f"--psm {self.psm}"
        if lang:
            config += f" -l {lang}"
        data = impl.image_to_data(
            gray,
            output_type=impl.Output.DICT,
            config=config,
        )
        for index, text in enumerate(data.get("text", [])):
            word = str(text).strip()
            if not word:
                continue
            try:
                score = float(data["conf"][index]) / 100.0
            except (KeyError, TypeError, ValueError):
                score = 0.0
            box = (
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["left"][index]) + int(data["width"][index]),
                int(data["top"][index]) + int(data["height"][index]),
            )
            items.append((word, score, box))
        return items

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def recognize(self, crop: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        if crop is None or mask is None:
            return TextResult(False, reason="empty_input", engine=self.engine).as_dict()

        image = self._prepare(crop, mask)
        if image is None:
            return TextResult(False, reason="empty_mask", engine=self.engine).as_dict()

        if not self._load():
            return TextResult(
                False,
                reason=f"engine_missing:{self._load_error}"
                if self._load_error
                else "engine_missing",
                engine=self.engine,
            ).as_dict()

        name, impl = self._impl
        try:
            items = self._run(name, impl, image)
        except Exception as exc:
            return TextResult(
                False,
                reason=f"engine_error:{exc}",
                engine=name,
            ).as_dict()

        kept = [
            (word.strip(), score, box)
            for word, score, box in items
            if word.strip() and score >= self.min_confidence
        ]
        if not kept:
            return TextResult(False, reason="no_text", engine=name).as_dict()

        text = " ".join(word for word, _, _ in kept)[: self.max_chars]
        best_score = max(score for _, score, _ in kept)
        return TextResult(
            True,
            text=text,
            confidence=float(best_score),
            engine=name,
            reason="ok",
            box=kept[0][2],
        ).as_dict()
