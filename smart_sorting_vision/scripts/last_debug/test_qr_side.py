"""侧面二维码检测自检脚本。

用法（在仓库任意位置都能跑）：

    python scripts/last_debug/test_qr_side.py
    python scripts/last_debug/test_qr_side.py --image side_crop.jpg

检查内容：
1. 空白图不会误报二维码；
2. 环境里装了 qrcode 时，生成一张 SORT-001 的二维码并真实解码；
3. 可选 --image：用真实侧面 crop 跑一遍，打印结果。

依赖：cv2 / numpy / zxing-cpp（缺少 zxingcpp 时脚本会 SKIP 而不是报错）。
退出码：0 = 必须通过的项都通过（或环境缺失被跳过）；1 = 有失败项。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "smart_sorting_vision"))

from vision.code_reader.qr_detector import QRDetector  # noqa: E402


def run(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def synth_qr_image(text: str = "SORT-001", side: int = 480):
    """用 qrcode 生成二维码并居中贴到白底画布；没装 qrcode 返回 None。"""
    try:
        import qrcode
        import cv2
    except ModuleNotFoundError:
        return None

    try:
        code = qrcode.make(text)
    except Exception:
        return None

    rgb = np.array(code.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    target = int(side * 0.75)
    scale = target / max(bgr.shape[:2])
    bgr = cv2.resize(
        bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    h, w = bgr.shape[:2]
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = bgr
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="侧面二维码检测自检")
    parser.add_argument("--image", type=Path, help="真实侧面 crop 图片")
    args = parser.parse_args()

    try:
        detector = QRDetector()
    except RuntimeError as exc:
        print(f"[SKIP] 无法初始化 QRDetector：{exc}")
        print("结果：跳过（缺 zxingcpp 依赖）")
        return 0

    failures = 0

    blank = np.full((480, 480, 3), 255, dtype=np.uint8)
    result = detector.detect(blank)
    failures += not run(
        "空白图不误报二维码",
        not bool(result.get("found", False)),
        f"found={result.get('found')} method={result.get('method', '-')}",
    )

    qr = synth_qr_image()
    if qr is None:
        print("[SKIP] 未安装 qrcode，跳过真实解码；pip install qrcode[pil]")
    else:
        result = detector.detect(qr)
        text = str(result.get("text", ""))
        failures += not run(
            "合成二维码可解码",
            bool(result.get("found", False)) and "SORT-001" in text,
            f"text={text!r} method={result.get('method', '-')}",
        )

    if args.image:
        import cv2

        image = cv2.imread(str(args.image))
        if image is None:
            failures += not run("读取 --image", False, str(args.image))
        else:
            result = detector.detect(image)
            print(
                f"[INFO] 真实图片：size={image.shape[1]}x{image.shape[0]} "
                f"found={result.get('found')} text={result.get('text')!r} "
                f"method={result.get('method', '-')}"
            )

    print()
    print(f"结果：{'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
