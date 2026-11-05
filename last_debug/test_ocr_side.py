"""侧面 OCR 文字识别自检脚本。

用法（在仓库任意位置都能跑）：

    python last_debug/test_ocr_side.py
    python last_debug/test_ocr_side.py --engine rapidocr
    python last_debug/test_ocr_side.py --image crop.jpg          # 用真实侧面 crop
    python last_debug/test_ocr_side.py --image crop.jpg --mask mask.png

检查内容：
1. 指定不存在的引擎时返回 engine_missing，不抛异常；
2. 空 mask 返回 empty_mask；
3. auto 模式在没装任何 OCR 依赖时也能优雅返回（engine_missing）；
4. 合成一张写有 ABC123 的图，装了引擎时真实识别一遍；
5. 顺带检查侧面识别链路（SideRecognizer）能否把 text 带进结果。

退出码：0 = 所有“必须通过”的项都通过；1 = 有失败项。
真实 OCR 的识别结果只作参考，不参与退出码（不同引擎精度不同）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "smart_sorting_vision"
sys.path.insert(0, str(PKG))

from vision.text.ocr_recognizer import TextRecognizer  # noqa: E402


def synth_text_image(text: str = "ABC123", size=240):
    """黑字白底；不用 cv2.putText 也可以，但这里用 cv2 方便调字号。"""
    import cv2

    image = np.full((size, size * 2, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (20, int(size * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (0, 0, 0),
        6,
        cv2.LINE_AA,
    )
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    return image, mask


def run(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="侧面 OCR 自检")
    parser.add_argument("--engine", default="auto", help="auto / rapidocr / easyocr / tesseract")
    parser.add_argument("--image", type=Path, help="真实侧面 crop 图片")
    parser.add_argument("--mask", type=Path, help="对应 mask（缺省用全图）")
    args = parser.parse_args()

    failures = 0

    # 1. 未知引擎：必须优雅失败
    result = TextRecognizer({"engine": "does_not_exist"}).recognize(
        *synth_text_image()
    )
    failures += not run(
        "未知引擎返回 engine_missing",
        (not result["found"]) and "engine_missing" in result["reason"],
        f"reason={result['reason']}",
    )

    # 2. 空 mask
    crop, _ = synth_text_image()
    empty_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    result = TextRecognizer({"engine": args.engine}).recognize(crop, empty_mask)
    failures += not run(
        "空 mask 返回 empty_mask",
        (not result["found"]) and result["reason"] == "empty_mask",
        f"reason={result['reason']}",
    )

    # 3. auto 模式无依赖也要能返回
    crop, mask = synth_text_image()
    auto = TextRecognizer({"engine": args.engine}).recognize(crop, mask)
    graceful = auto["found"] or auto["reason"].startswith(
        ("engine_missing", "no_text", "engine_error")
    )
    failures += not run(
        "auto 模式不抛异常",
        graceful,
        f"found={auto['found']} engine={auto['engine'] or '-'} "
        f"text={auto['text']!r} reason={auto['reason']}",
    )

    # 4. 真实识别（仅供参考）
    if auto["found"]:
        matched = "ABC123" in auto["text"].replace(" ", "").upper()
        print(
            f"[INFO] 合成图识别：text={auto['text']!r} "
            f"confidence={auto['confidence']:.3f} "
            f"C{'≈' if matched else '≠'}ABC123"
        )
    else:
        print(
            "[INFO] 当前环境没有可用 OCR 引擎；"
            "装一个即可真实识别：pip install rapidocr_onnxruntime"
        )

    # 5. 侧面识别链路（可选，缺 ultralytics 时跳过）
    try:
        from vision.pipeline.side_recognizer import SideRecognizer

        class _Color:
            def classify(self, crop, mask):
                return {"color": "red", "confidence": 0.9}

        class _QR:
            def detect(self, crop):
                return {"found": False, "text": ""}

        class _OCR:
            def recognize(self, crop, mask):
                return {
                    "found": True,
                    "text": "ABC123",
                    "confidence": 0.88,
                    "engine": "stub",
                    "reason": "ok",
                    "box": None,
                }

        recognizer = SideRecognizer(
            color_classifier=_Color(),
            qr_detector=_QR(),
            shape_classifier=None,
            ocr_recognizer=_OCR(),
            enable_color=True,
            enable_qr=True,
            enable_shape=False,
            enable_ocr=True,
        )
        out = recognizer.recognize(*synth_text_image())
        failures += not run(
            "SideRecognizer 透传 text",
            out.get("text") == "ABC123" and out.get("text_confidence") == 0.88,
            f"text={out.get('text')!r}",
        )
    except Exception as exc:  # 本机没装 ultralytics 时只提示
        print(f"[SKIP] SideRecognizer 链路检查：{exc}")

    # 6. 真实图片（可选）
    if args.image:
        import cv2

        image = cv2.imread(str(args.image))
        if image is None:
            failures += not run("读取 --image", False, str(args.image))
        else:
            if args.mask:
                mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            result = TextRecognizer({"engine": args.engine}).recognize(image, mask)
            print(
                f"[INFO] 真实图片：size={image.shape[1]}x{image.shape[0]} "
                f"found={result['found']} text={result['text']!r} "
                f"confidence={result['confidence']:.3f} reason={result['reason']}"
            )

    print()
    print(f"结果：{'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
