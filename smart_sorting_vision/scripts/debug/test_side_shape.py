from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ================================================================
# Project root
#
# 当前文件建议放：
#
# smart_sorting_vision/
# └── scripts/
#     └── test_side_shape.py
#
# 所以 parents[1] 就是项目根目录。
# ================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(
            ROOT
        ),
    )


from vision.cameras.side_camera import SideCamera
from vision.processors.side_object_extractor import (
    SideObjectExtractor,
)
from vision.shape.shape_classifier import (
    ShapeClassifier,
)


# ================================================================
# Paths
# ================================================================

BACKGROUND_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "side_camera"
    / "side_mechanical_roi_background.png"
)


MODEL_PATH = (
    ROOT
    / "outputs"
    / "training"
    / "side_cls"
    / "v2_yolo26n_cls"
    / "weights"
    / "best.pt"
)


SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "side_shape_classifier"
    / "parameter_tuning"
)


# ================================================================
# Settings
# ================================================================

# Windows + NVIDIA GPU:
DEVICE = 0

# 如果在 NUC CPU 上测试，改成：
# DEVICE = "cpu"

IMGSZ = 224

CONFIDENCE_THRESHOLD = 0.50


# ================================================================
# Helpers
# ================================================================

def json_safe(
    value: Any,
) -> Any:

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):

        return float(
            value
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            str(
                key
            ):
                json_safe(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return [
            json_safe(
                item
            )
            for item in value
        ]

    return value


def load_fixed_background(
    extractor: SideObjectExtractor,
) -> np.ndarray:

    if not BACKGROUND_PATH.exists():

        raise FileNotFoundError(
            "\n固定背景图不存在："
            f"\n{BACKGROUND_PATH}"
            "\n\n请先运行固定背景采集程序。"
        )

    background = (
        extractor.load_background(
            BACKGROUND_PATH
        )
    )

    print(
        f"[BACKGROUND] loaded: "
        f"{BACKGROUND_PATH}"
    )

    print(
        f"[BACKGROUND] shape : "
        f"{background.shape}"
    )

    return background


def warm_up_camera(
    camera: SideCamera,
    background: np.ndarray,
    *,
    frame_count: int = 20,
) -> None:

    frame = None

    for _ in range(
        frame_count
    ):

        frame = camera.read()

    if frame is None:

        raise RuntimeError(
            "相机预热失败"
        )

    if (
        frame.shape
        !=
        background.shape
    ):

        raise RuntimeError(
            "\n当前 SideCamera ROI 与固定背景尺寸不一致："
            f"\ncurrent    = {frame.shape}"
            f"\nbackground = {background.shape}"
            "\n\n请重新采集固定背景。"
        )

    print(
        f"[Camera] warm-up complete: "
        f"{frame_count} frames"
    )


def save_sample(
    *,
    mechanical_roi: np.ndarray,
    display: np.ndarray,
    object_result,
    shape_result: dict[str, Any] | None,
    camera_info: dict[str, Any],
) -> None:

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    predicted_shape = (
        "no_object"
        if shape_result is None
        else str(
            shape_result.get(
                "shape",
                "unknown",
            )
        )
    )

    prefix = (
        SAVE_DIR
        /
        (
            f"side_shape_{timestamp}"
            f"_pred-{predicted_shape}"
        )
    )

    cv2.imwrite(
        str(
            prefix
        )
        +
        "_roi.jpg",
        mechanical_roi,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    cv2.imwrite(
        str(
            prefix
        )
        +
        "_display.jpg",
        display,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if (
        object_result is not None
        and
        object_result.crop is not None
    ):

        cv2.imwrite(
            str(
                prefix
            )
            +
            "_crop.jpg",
            object_result.crop,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

    if (
        object_result is not None
        and
        object_result.crop_mask is not None
    ):

        cv2.imwrite(
            str(
                prefix
            )
            +
            "_crop_mask.png",
            object_result.crop_mask,
        )

    metadata = {
        "background_path":
            str(
                BACKGROUND_PATH
            ),

        "model_path":
            str(
                MODEL_PATH
            ),

        "camera":
            json_safe(
                camera_info
            ),

        "object": {
            "found":
                bool(
                    object_result.found
                )
                if object_result is not None
                else False,

            "reason":
                getattr(
                    object_result,
                    "reason",
                    None,
                ),

            "bbox":
                json_safe(
                    getattr(
                        object_result,
                        "bbox",
                        None,
                    )
                ),

            "crop_bbox":
                json_safe(
                    getattr(
                        object_result,
                        "crop_bbox",
                        None,
                    )
                ),
        },

        "shape_result":
            json_safe(
                shape_result
            )
            if shape_result is not None
            else None,
    }

    Path(
        str(
            prefix
        )
        +
        ".json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"[SAVE] {prefix}"
    )
    print()


# ================================================================
# Main
# ================================================================

def main() -> None:

    camera = SideCamera()

    extractor = SideObjectExtractor(
        padding_ratio=0.15
    )

    classifier = ShapeClassifier(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        device=DEVICE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        topk=3,
    )

    try:

        camera.open()

        camera_info = (
            camera.get_stream_info()
        )

        background = (
            load_fixed_background(
                extractor
            )
        )

        warm_up_camera(
            camera,
            background,
            frame_count=20,
        )

        print()
        print(
            "=" * 72
        )

        print(
            "SIDE SHAPE REALTIME TEST"
        )

        print(
            "=" * 72
        )

        print(
            f"MODEL : {MODEL_PATH}"
        )

        print(
            f"DEVICE: {DEVICE}"
        )

        print(
            f"IMGSZ : {IMGSZ}"
        )

        print(
            f"CONF  : {CONFIDENCE_THRESHOLD}"
        )

        print(
            "S      : 保存当前调试样本"
        )

        print(
            "P      : 打印当前形状结果"
        )

        print(
            "B      : 重新加载固定背景"
        )

        print(
            "Q/ESC  : 退出"
        )

        print(
            "=" * 72
        )

        previous_time = (
            time.perf_counter()
        )

        smooth_fps = 0.0

        while True:

            mechanical_roi = (
                camera.read()
            )

            object_result = (
                extractor.extract(
                    mechanical_roi
                )
            )

            display = (
                mechanical_roi.copy()
            )

            shape_result = None
            inference_ms = 0.0

            if (
                object_result.found
                and
                object_result.crop is not None
            ):

                start_time = (
                    time.perf_counter()
                )

                shape_result = (
                    classifier.classify(
                        object_result.crop
                    )
                )

                inference_ms = (
                    (
                        time.perf_counter()
                        -
                        start_time
                    )
                    *
                    1000.0
                )

                shape = (
                    shape_result.get(
                        "shape",
                        "unknown",
                    )
                )

                confidence = float(
                    shape_result.get(
                        "confidence",
                        0.0,
                    )
                )

                line1 = (
                    f"Shape={shape} "
                    f"Conf={confidence:.2f}"
                )

                line2 = (
                    f"Infer={inference_ms:.1f} ms"
                )

                crop_bbox = getattr(
                    object_result,
                    "crop_bbox",
                    None,
                )

                if crop_bbox is not None:

                    x1, y1, x2, y2 = (
                        crop_bbox
                    )

                    cv2.rectangle(
                        display,
                        (
                            int(
                                x1
                            ),
                            int(
                                y1
                            ),
                        ),
                        (
                            int(
                                x2
                            ),
                            int(
                                y2
                            ),
                        ),
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow(
                    "Side Shape Crop",
                    object_result.crop,
                )

            else:

                line1 = (
                    "No Object"
                )

                line2 = (
                    getattr(
                        object_result,
                        "reason",
                        "",
                    )
                )

            # ----------------------------------------------------
            # FPS
            # ----------------------------------------------------

            current_time = (
                time.perf_counter()
            )

            dt = (
                current_time
                -
                previous_time
            )

            previous_time = (
                current_time
            )

            if dt > 0:

                current_fps = (
                    1.0
                    /
                    dt
                )

                if smooth_fps == 0:

                    smooth_fps = (
                        current_fps
                    )

                else:

                    smooth_fps = (
                        0.90
                        *
                        smooth_fps
                        +
                        0.10
                        *
                        current_fps
                    )

            # ----------------------------------------------------
            # Draw
            # ----------------------------------------------------

            cv2.putText(
                display,
                line1,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.64,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                line2,
                (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"FPS={smooth_fps:.1f}",
                (18, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "S=save P=print B=reload-bg Q=quit",
                (
                    18,
                    display.shape[0]
                    -
                    18,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Side Shape Recognition",
                display,
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):

                break

            if key in (
                ord("b"),
                ord("B"),
            ):

                background = (
                    load_fixed_background(
                        extractor
                    )
                )

                print(
                    "[BACKGROUND] fixed background reloaded"
                )

                continue

            if key in (
                ord("p"),
                ord("P"),
            ):

                print()
                print(
                    "=" * 72
                )

                print(
                    "CURRENT SHAPE RESULT"
                )

                print(
                    "=" * 72
                )

                if shape_result is None:

                    print(
                        "No valid shape result"
                    )

                else:

                    print(
                        json.dumps(
                            json_safe(
                                shape_result
                            ),
                            ensure_ascii=False,
                            indent=2,
                        )
                    )

                print(
                    "=" * 72
                )

            if key in (
                ord("s"),
                ord("S"),
            ):

                camera_info = (
                    camera.get_stream_info()
                )

                save_sample(
                    mechanical_roi=mechanical_roi,
                    display=display,
                    object_result=object_result,
                    shape_result=shape_result,
                    camera_info=camera_info,
                )

    finally:

        camera.close()

        cv2.destroyAllWindows()

        print()
        print(
            "Shape realtime test finished."
        )


if __name__ == "__main__":
    main()
