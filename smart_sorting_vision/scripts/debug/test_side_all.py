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
# 自动寻找项目根目录
#
# 这样这个测试脚本以后放到 scripts/ 下更深的目录也不用再改
# parents[1] / parents[3]。
# ================================================================

def find_project_root() -> Path:

    current = Path(
        __file__
    ).resolve()

    for parent in current.parents:

        if (
            (
                parent
                / "vision"
            ).is_dir()
            and
            (
                parent
                / "data"
            ).is_dir()
        ):

            return parent

    raise RuntimeError(
        "无法自动找到 smart_sorting_vision 项目根目录"
    )


ROOT = find_project_root()

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


# ------------------------------------------------
# 形状模型：
#
# 这里按你现在的新输出路径写。
# 如果你当前实际使用的是 v1，把目录名改成
# v1_yolo26n_cls 即可。
# ------------------------------------------------

SHAPE_MODEL_PATH = (
    ROOT
    / "outputs"
    / "run"
    / "side_cls"
    / "v2_yolo26n_cls"
    / "weights"
    / "best.pt"
)


SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "side_combined"
    / "parameter_tuning"
)


# Windows + NVIDIA:
SHAPE_DEVICE = 0

# NUC CPU 时改成：
# SHAPE_DEVICE = "cpu"


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

    return background


def warm_up_and_validate(
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
        )


def save_sample(
    *,
    mechanical_roi: np.ndarray,
    display: np.ndarray,
    object_result,
    combined_result: dict[str, Any] | None,
    qr_display: np.ndarray | None,
) -> None:

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )[:-3]
    )

    prefix = (
        SAVE_DIR
        /
        f"side_all_{timestamp}"
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

    if qr_display is not None:

        cv2.imwrite(
            str(
                prefix
            )
            +
            "_qr_display.jpg",
            qr_display,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

    Path(
        str(
            prefix
        )
        +
        ".json"
    ).write_text(
        json.dumps(
            json_safe(
                combined_result
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[SAVE] {prefix}"
    )


# ================================================================
# Main
# ================================================================

def main() -> None:

    if not SHAPE_MODEL_PATH.exists():

        raise FileNotFoundError(
            "\n形状模型不存在："
            f"\n{SHAPE_MODEL_PATH}"
            "\n\n请确认 SHAPE_MODEL_PATH。"
        )

    camera = SideCamera()

    extractor = SideObjectExtractor(
        padding_ratio=0.15
    )

    color_classifier = (
        ColorClassifier()
    )

    qr_detector = (
        QRDetector()
    )

    shape_classifier = (
        ShapeClassifier(
            model_path=SHAPE_MODEL_PATH,
            imgsz=224,
            device=SHAPE_DEVICE,
            confidence_threshold=0.50,
            topk=3,
        )
    )

    recognizer = (
        SideRecognizer(
            color_classifier=
                color_classifier,

            qr_detector=
                qr_detector,

            shape_classifier=
                shape_classifier,
        )
    )

    try:

        camera.open()

        background = (
            load_fixed_background(
                extractor
            )
        )

        warm_up_and_validate(
            camera,
            background,
            frame_count=20,
        )

        print()
        print(
            "=" * 72
        )

        print(
            "SIDE COMBINED RECOGNITION TEST"
        )

        print(
            "=" * 72
        )

        print(
            f"Shape model: "
            f"{SHAPE_MODEL_PATH}"
        )

        print(
            "S     : 保存当前样本"
        )

        print(
            "P     : 打印完整结果"
        )

        print(
            "B     : 重新加载固定背景"
        )

        print(
            "Q/ESC : 退出"
        )

        print(
            "=" * 72
        )

        previous_time = (
            time.perf_counter()
        )

        smooth_fps = 0.0

        combined_result = None
        qr_display = None

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

            combined_result = None
            qr_display = None

            if (
                object_result.found
                and
                object_result.crop is not None
                and
                object_result.crop_mask is not None
            ):

                start = (
                    time.perf_counter()
                )

                combined_result = (
                    recognizer.recognize(
                        object_result.crop,
                        object_result.crop_mask,
                    )
                )

                total_ms = (
                    (
                        time.perf_counter()
                        -
                        start
                    )
                    *
                    1000.0
                )

                color = (
                    combined_result[
                        "color"
                    ]
                )

                color_conf = (
                    combined_result[
                        "color_confidence"
                    ]
                )

                qr_present = (
                    combined_result[
                        "qr_present"
                    ]
                )

                shape = (
                    combined_result[
                        "shape"
                    ]
                )

                shape_conf = (
                    combined_result[
                        "shape_confidence"
                    ]
                )

                line1 = (
                    f"Color: {color} "
                    f"({color_conf:.2f})"
                )

                line2 = (
                    f"QR: "
                    f"{'YES' if qr_present else 'NO'}"
                )

                line3 = (
                    f"Shape: {shape} "
                    f"({shape_conf:.2f})"
                )

                line4 = (
                    f"Total inference: "
                    f"{total_ms:.1f} ms"
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

                qr_result = (
                    combined_result[
                        "details"
                    ][
                        "qr"
                    ]
                )

                qr_display = (
                    qr_detector.draw_result(
                        object_result.crop,
                        qr_result,
                    )
                )

                cv2.imshow(
                    "Side Combined Object Crop",
                    qr_display,
                )

            else:

                line1 = (
                    "No Object"
                )

                line2 = ""
                line3 = ""

                line4 = getattr(
                    object_result,
                    "reason",
                    "",
                )

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

            cv2.putText(
                display,
                line1,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if line2:

                cv2.putText(
                    display,
                    line2,
                    (18, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if line3:

                cv2.putText(
                    display,
                    line3,
                    (18, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                line4,
                (18, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"FPS: {smooth_fps:.1f}",
                (18, 144),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
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
                0.41,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Side Combined Recognition",
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
                    "[BACKGROUND] reloaded"
                )

                continue

            if key in (
                ord("p"),
                ord("P"),
            ):

                print()
                print(
                    json.dumps(
                        json_safe(
                            combined_result
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )

            if key in (
                ord("s"),
                ord("S"),
            ):

                save_sample(
                    mechanical_roi=
                        mechanical_roi,

                    display=
                        display,

                    object_result=
                        object_result,

                    combined_result=
                        combined_result,

                    qr_display=
                        qr_display,
                )

    finally:

        camera.close()

        cv2.destroyAllWindows()

        print()
        print(
            "Combined recognition test finished."
        )


if __name__ == "__main__":
    main()
