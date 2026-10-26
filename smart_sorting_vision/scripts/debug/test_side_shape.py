"""实时检查侧面形状/空背景分类。

该脚本与正式流程保持同一个输入：完整的侧面 mechanical ROI。
它不做背景差分，也不运行颜色和二维码，适合先确认 no_object 类别是否可靠。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config
from vision.cameras.side_camera import SideCamera
from vision.shape.shape_classifier import ShapeClassifier


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    config_path = ROOT / "config" / "runtime_config.yaml"
    cfg = load_runtime_config(ROOT, config_path)
    SideCamera.configure(cfg["camera"]["side"])

    model_cfg = cfg["model"]["side"]
    shape_cfg = cfg["vision"]["side"]["recognition"]["shape"]
    classifier = ShapeClassifier(
        model_path=model_cfg["path"],
        imgsz=int(model_cfg["imgsz"]),
        device=model_cfg.get("device"),
        confidence_threshold=float(model_cfg.get("confidence_threshold", 0.50)),
        topk=int(model_cfg.get("topk", 3)),
        no_object_labels=tuple(shape_cfg.get("no_object_labels", ("no_object",))),
    )

    save_dir = _project_path(cfg["debug"]["frames_dir"]) / "side_shape"
    save_dir.mkdir(parents=True, exist_ok=True)
    camera = SideCamera()
    last_frame = None
    last_time = time.perf_counter()
    fps = 0.0

    print("完整侧面 ROI 形状分类调试")
    print("S：保存 ROI 和分类结果；P：打印结果；Q/ESC：退出")
    print(f"模型：{model_cfg['path']}")
    print(f"空背景类别：{shape_cfg.get('no_object_labels', [])}")

    try:
        camera.open()
        while True:
            frame = camera.read()
            if frame is None:
                continue

            start = time.perf_counter()
            result = classifier.classify(frame)
            inference_ms = (time.perf_counter() - start) * 1000.0
            last_frame = frame.copy()

            now = time.perf_counter()
            current_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps = current_fps if fps == 0.0 else 0.9 * fps + 0.1 * current_fps

            object_present = bool(result.get("object_present", False))
            shape = str(result.get("shape", "unknown"))
            confidence = float(result.get("confidence", 0.0))
            reason = str(result.get("reason", ""))
            view = frame.copy()
            color = (0, 255, 0) if object_present else (0, 165, 255)
            cv2.putText(
                view,
                f"{shape}  object={int(object_present)}  conf={confidence:.2f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                view,
                f"reason={reason}  infer={inference_ms:.1f}ms  FPS={fps:.1f}",
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                view,
                "S=save  P=print  Q/ESC=quit",
                (16, view.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Side Shape Classifier", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("p"), ord("P")):
                print(json.dumps(result, ensure_ascii=False, indent=2))
            if key in (ord("s"), ord("S")) and last_frame is not None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_path = save_dir / f"side_shape_{stamp}_{shape}.jpg"
                json_path = image_path.with_suffix(".json")
                cv2.imwrite(str(image_path), last_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                json_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[SAVE] {image_path}")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
