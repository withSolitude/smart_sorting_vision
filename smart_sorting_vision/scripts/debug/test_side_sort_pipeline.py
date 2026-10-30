"""实时检查完整的侧面识别流水线。

输入先经过完整 ROI 的形状/空背景分类；确认有物体后才运行颜色和二维码，
最后用 YAML 中的多帧共识参数输出 READY。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.app_config import load_runtime_config
from vision.cameras.side_camera import SideCamera
from vision.pipeline.side_sort_pipeline import SideSortPipeline


def main() -> None:
    cfg = load_runtime_config(ROOT, ROOT / "config" / "runtime_config.yaml")
    SideCamera.configure(cfg["camera"]["side"])

    model_cfg = cfg["model"]["side"]
    side_cfg = cfg["vision"]["side"]
    recognition_cfg = side_cfg["recognition"]
    color_cfg = dict(recognition_cfg["color"])
    qr_cfg = dict(recognition_cfg["qr"])
    shape_cfg = dict(recognition_cfg["shape"])
    enable_color = bool(color_cfg.pop("enabled"))
    enable_qr = bool(qr_cfg.pop("enabled"))
    qr_check_content = bool(qr_cfg.pop("check_content_consistency", False))
    enable_shape = bool(shape_cfg.pop("enabled"))
    no_object_labels = tuple(shape_cfg.pop("no_object_labels", ("no_object",)))

    pipeline = SideSortPipeline(
        background_path=cfg["calibration"]["side_background"],
        shape_model_path=model_cfg["path"],
        shape_device=model_cfg.get("device"),
        shape_imgsz=int(model_cfg.get("imgsz", 224)),
        shape_confidence_threshold=float(model_cfg.get("confidence_threshold", 0.50)),
        shape_topk=int(model_cfg.get("topk", 3)),
        padding_ratio=float(side_cfg.get("padding_ratio", 0.15)),
        consensus_window=int(side_cfg["consensus"]["window"]),
        consensus_required=int(side_cfg["consensus"]["required"]),
        extractor_config=side_cfg["crop_extractor"]["extractor"],
        color_config=color_cfg,
        qr_config=qr_cfg,
        qr_check_content=qr_check_content,
        no_object_labels=no_object_labels,
        enable_color=enable_color,
        enable_qr=enable_qr,
        enable_shape=enable_shape,
    )

    camera = SideCamera()
    print("侧面完整识别流水线调试")
    print("R：清空多帧共识；Q/ESC：退出")
    try:
        camera.open()
        while True:
            frame = camera.read()
            if frame is None:
                continue
            result = pipeline.process(frame)
            view = frame.copy()

            if not result.found:
                line1 = f"NO OBJECT  ({result.reason})"
                line2 = "返回 Disable 后重新开始顶部识别"
                color = (0, 165, 255)
            elif result.ready:
                line1 = f"READY {result.votes}/{result.window}"
                line2 = (
                    f"color={result.color}  QR={'Y' if result.qr_present else 'N'} "
                    f"shape={result.shape}"
                )
                color = (0, 255, 0)
            else:
                line1 = f"WAIT {result.votes}/{result.window} need={result.required_hits}"
                line2 = (
                    f"color={result.color}  QR={'Y' if result.qr_present else 'N'} "
                    f"shape={result.shape}  {result.reason}"
                )
                color = (0, 165, 255)

            if result.crop_bbox is not None:
                x1, y1, x2, y2 = result.crop_bbox
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(view, line1, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
            cv2.putText(view, line2, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(view, "R=reset  Q/ESC=quit", (16, view.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Side Sort Pipeline", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                pipeline.reset()
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
