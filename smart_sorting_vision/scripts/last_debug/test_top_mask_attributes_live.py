"""实时验证顶部 YOLO-seg mask、HSV/Lab 颜色和 YOLO-cls 形状。

本脚本只打开顶部相机和视觉模型，不打开 STM32，不发送抓取或储物命令。

按键：
    q / Esc 退出
    s        保存当前画面
    p        暂停/继续
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config
from vision.cameras.top_camera import TopCamera
from vision.color.color_classifier import ColorClassifier
from vision.models.top_segmenter import TopSegmenter
from vision.pipeline.top_attribute_recognizer import TopAttributeRecognizer
from vision.shape.shape_classifier import ShapeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/runtime_config.yaml")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=ROOT / "outputs" / "debug" / "top_mask_attributes",
    )
    return parser.parse_args()


def _polygon_mask(frame: np.ndarray, polygon) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def _draw_object(frame: np.ndarray, polygon, result, index: int) -> None:
    mask = _polygon_mask(frame, polygon)
    overlay = frame.copy()
    overlay[mask > 0] = (40, 180, 40)
    frame[:] = cv2.addWeighted(frame, 0.72, overlay, 0.28, 0.0)

    points = np.round(np.asarray(polygon)).astype(np.int32).reshape(-1, 1, 2)
    if len(points) >= 3:
        cv2.polylines(frame, [points], True, (0, 255, 0), 2, cv2.LINE_AA)

    center = np.mean(np.asarray(polygon, dtype=np.float32), axis=0)
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    label = (
        f"#{index} {result.color}/{result.shape} "
        f"c={result.color_confidence:.2f} s={result.shape_confidence:.2f}"
    )
    cv2.putText(
        frame,
        label,
        (max(5, x - 160), max(24, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (x, y), 4, (255, 0, 255), -1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_runtime_config(ROOT, config_path.resolve())

    TopCamera.configure(
        cfg["camera"]["top"],
        calibration_file=cfg["calibration"]["top_intrinsics"],
    )

    top_model = cfg["model"]["top"]
    segmenter = TopSegmenter(
        model_path=top_model["path"],
        conf=float(top_model["conf"]),
        imgsz=int(top_model["imgsz"]),
        device=top_model["device"],
    )

    side_recognition = cfg["vision"]["side"]["recognition"]
    color_cfg = dict(side_recognition["color"])
    color_cfg.pop("enabled", None)
    color_classifier = ColorClassifier(**color_cfg)

    side_shape_cfg = dict(cfg["model"]["side"])
    top_shape_cfg = dict(cfg.get("model", {}).get("top_shape") or {})
    shape_path = top_shape_cfg.get("path") or side_shape_cfg["path"]
    shape_classifier = ShapeClassifier(
        model_path=shape_path,
        imgsz=int(top_shape_cfg.get("imgsz", side_shape_cfg["imgsz"])),
        device=top_shape_cfg.get("device", side_shape_cfg["device"]),
        confidence_threshold=float(
            top_shape_cfg.get(
                "confidence_threshold", side_shape_cfg["confidence_threshold"]
            )
        ),
        topk=int(top_shape_cfg.get("topk", side_shape_cfg["topk"])),
        no_object_labels=tuple(
            str(label)
            for label in side_recognition["shape"].get(
                "no_object_labels", ["no_object", "empty", "background"]
            )
        ),
    )

    top_attr_cfg = cfg["vision"]["top"].get("top_attribute_recognition") or {}
    recognizer = TopAttributeRecognizer(
        color_classifier=color_classifier,
        shape_classifier=shape_classifier,
        padding_ratio=float(top_attr_cfg.get("mask_padding_ratio", 0.12)),
        shape_background_bgr=tuple(top_attr_cfg.get("shape_background_bgr", [0, 0, 0])),
        min_mask_pixels=int(top_attr_cfg.get("min_mask_pixels", 150)),
    )

    camera = TopCamera()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    window = "TOP MASK ATTRIBUTES"
    paused = False
    last_frame: np.ndarray | None = None
    last_signature = None
    last_print_at = 0.0
    save_index = 0

    print(f"[CONFIG] config={config_path.resolve()}")
    print(f"[MODEL] top_seg={top_model['path']}")
    print(f"[MODEL] top_shape={shape_path}")
    print("[INFO] q/Esc退出, s保存, p暂停/继续")

    try:
        camera.open()
        while True:
            if not paused:
                frame = camera.read()
                if frame is None:
                    continue
                last_frame = frame.copy()
                result = segmenter.predict(frame)
                objects = segmenter.get_objects(result)
                labels = []
                for index, obj in enumerate(objects):
                    polygon = obj.get("polygon")
                    if polygon is None:
                        labels.append(("unknown", "unknown", 0.0, 0.0))
                        continue
                    attr = recognizer.classify(frame, polygon)
                    labels.append(
                        (
                            attr.color,
                            attr.shape,
                            round(attr.color_confidence, 3),
                            round(attr.shape_confidence, 3),
                        )
                    )
                    _draw_object(frame, polygon, attr, index)

                now = time.monotonic()
                signature = tuple(labels)
                if signature != last_signature or now - last_print_at >= 1.0:
                    print(f"[TOP ATTR] objects={len(objects)} labels={labels}")
                    last_signature = signature
                    last_print_at = now
                last_frame = frame.copy()

            if last_frame is not None:
                display = last_frame.copy()
                cv2.putText(
                    display,
                    "PAUSED" if paused else "LIVE | q quit, s save, p pause",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
            elif key == ord("s") and last_frame is not None:
                save_index += 1
                path = args.save_dir / (
                    f"top_attr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{save_index:03d}.jpg"
                )
                cv2.imwrite(str(path), last_frame)
                print(f"[SAVE] {path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
