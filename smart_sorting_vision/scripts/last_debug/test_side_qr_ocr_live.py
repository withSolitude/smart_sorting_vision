"""侧面相机实时二维码 + 文字（OCR）检测。

打开侧面相机，逐帧做二维码解码和 OCR，并在画面上叠加结果、终端打印。
用于现场确认“侧面画面里到底有没有二维码/文字”。

用法：

    python smart_sorting_vision/scripts/last_debug/test_side_qr_ocr_live.py
    python smart_sorting_vision/scripts/last_debug/test_side_qr_ocr_live.py --no-window
    python smart_sorting_vision/scripts/last_debug/test_side_qr_ocr_live.py --engine rapidocr
    python smart_sorting_vision/scripts/last_debug/test_side_qr_ocr_live.py --save-dir out/live

按键：q / ESC 退出；s 保存当前帧。
二维码和 OCR 的参数默认读 runtime_config.yaml 的 side.recognition，
缺依赖时对应通道显示 SKIP，不影响另一个通道。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config  # noqa: E402
from vision.cameras.side_camera import SideCamera  # noqa: E402
from vision.code_reader.qr_detector import QRDetector  # noqa: E402
from vision.text.ocr_recognizer import TextRecognizer  # noqa: E402


def build_detectors(cfg: dict, engine: str):
    recognition = cfg["vision"]["side"]["recognition"]

    qr_cfg = dict(recognition.get("qr") or {})
    qr_cfg.pop("enabled", None)
    qr_cfg.pop("check_content_consistency", None)
    try:
        qr_detector = QRDetector(**qr_cfg)
    except Exception as exc:
        print(f"[SKIP] 二维码通道不可用：{exc}")
        qr_detector = None

    ocr_cfg = dict(recognition.get("ocr") or {})
    ocr_cfg.pop("enabled", None)
    ocr_cfg.pop("run_when_qr_found", None)
    if engine:
        ocr_cfg["engine"] = engine
    ocr = TextRecognizer(ocr_cfg)
    return qr_detector, ocr


def object_mask(frame: np.ndarray) -> np.ndarray:
    """ROI 多边形外的像素已被置黑，用“非黑”区域当 OCR mask。"""
    mask = (frame.max(axis=2) > 30).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        mask[:] = 255
    return mask


def draw(frame: np.ndarray, qr: dict, ocr: dict) -> np.ndarray:
    view = frame.copy()
    if qr.get("found"):
        points = qr.get("points")
        if points is not None:
            pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [pts], True, (0, 220, 0), 2)
        cv2.putText(view, f"QR: {qr.get('text', '')}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
    box = ocr.get("box")
    if ocr.get("found") and box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 180, 255), 2)
    status = (
        f"QR={'1' if qr.get('found') else '0'}  "
        f"OCR={'1' if ocr.get('found') else '0'}  {ocr.get('engine') or '-'}"
    )
    cv2.putText(view, status, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    if ocr.get("found"):
        cv2.putText(view, ocr.get("text", "")[:48], (12, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2, cv2.LINE_AA)
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description="侧面相机实时二维码 + OCR")
    parser.add_argument("--config", default="config/runtime_config.yaml")
    parser.add_argument("--engine", default="", help="覆盖 OCR 引擎：auto/rapidocr/easyocr/tesseract")
    parser.add_argument("--no-window", action="store_true", help="不弹预览窗口，只打印")
    parser.add_argument("--interval", type=float, default=0.5, help="终端打印间隔（秒）")
    parser.add_argument("--save-dir", type=Path, help="检测到结果时保存帧和 txt")
    parser.add_argument("--duration", type=float, default=0.0, help="运行秒数，0=一直跑")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_runtime_config(ROOT, config_path)
    SideCamera.configure(cfg["camera"]["side"])

    qr_detector, ocr = build_detectors(cfg, args.engine)
    print(
        f"[LIVE] side={cfg['camera']['side']['device']} "
        f"qr={'on' if qr_detector else 'SKIP'} "
        f"ocr_engine={ocr.engine}"
    )

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    camera = SideCamera()
    window = "side QR/OCR (q=quit)"
    last_print = 0.0
    started = time.monotonic()
    last_qr = {"found": False, "text": ""}
    last_ocr = {"found": False, "text": "", "engine": ""}

    try:
        camera.open()
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            qr = (
                qr_detector.detect(frame)
                if qr_detector is not None
                else {"found": False, "text": "", "reason": "missing_zxingcpp"}
            )
            ocr_result = ocr.recognize(frame, object_mask(frame))
            last_qr, last_ocr = qr, ocr_result

            now = time.monotonic()
            if now - last_print >= max(0.05, args.interval):
                last_print = now
                print(
                    f"[LIVE] qr={'1' if qr.get('found') else '0'} "
                    f"text={qr.get('text', '')!r} | "
                    f"ocr={'1' if ocr_result.get('found') else '0'} "
                    f"text={ocr_result.get('text', '')!r} "
                    f"conf={ocr_result.get('confidence', 0.0):.3f} "
                    f"engine={ocr_result.get('engine') or '-'} "
                    f"reason={ocr_result.get('reason', '')}"
                )

            if args.save_dir and (qr.get("found") or ocr_result.get("found")):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(args.save_dir / f"side_{stamp}.jpg"), frame)
                with open(args.save_dir / f"side_{stamp}.txt", "w", encoding="utf-8") as fh:
                    fh.write(f"qr_found={qr.get('found')} qr_text={qr.get('text', '')}\n")
                    fh.write(f"ocr_found={ocr_result.get('found')} ocr_text={ocr_result.get('text', '')}\n")

            if not args.no_window:
                cv2.imshow(window, draw(frame, qr, ocr_result))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s") and args.save_dir:
                    cv2.imwrite(str(args.save_dir / f"side_manual_{int(time.time())}.jpg"), frame)

            if args.duration > 0 and now - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    print(
        f"[LIVE] 结束：最后一次 qr={last_qr.get('found')} {last_qr.get('text', '')!r} | "
        f"ocr={last_ocr.get('found')} {last_ocr.get('text', '')!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
