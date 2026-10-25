from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

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
from vision.pipeline.side_recognition_consensus import (
    SideRecognitionConsensus,
)


BACKGROUND_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "side_camera"
    / "side_mechanical_roi_background.png"
)


# ================================================================
# 按你实际模型位置修改这一项。
# ================================================================

SHAPE_MODEL_PATH = (
    ROOT
    / "outputs"
    / "training"
    / "side_cls"
    / "v2_yolo26n_cls"
    / "weights"
    / "best.pt"
)


SHAPE_DEVICE = 0

# 如果以后在 NUC CPU 上：
# SHAPE_DEVICE = "cpu"
CHINESE_FONT_PATH = Path(
    "/usr/share/fonts/opentype/noto/"
    "NotoSansCJK-Regular.ttc"
)


@lru_cache(maxsize=8)
def get_chinese_font(
    font_size: int,
) -> ImageFont.FreeTypeFont:

    if not CHINESE_FONT_PATH.exists():

        raise FileNotFoundError(
            f"找不到中文字体：{CHINESE_FONT_PATH}"
        )

    return ImageFont.truetype(
        str(CHINESE_FONT_PATH),
        font_size,
    )


def draw_unicode_panel(
    image: np.ndarray,
    items,
    *,
    panel_height: int = 105,
) -> None:
    """
    使用 Pillow 绘制中文。

    只转换画面顶部区域，避免转换整张图片而进一步降低帧率。
    color_bgr 仍然按照 OpenCV 的 BGR 格式填写。
    """

    height = min(
        panel_height,
        image.shape[0],
    )

    panel_rgb = cv2.cvtColor(
        image[:height],
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(
        panel_rgb
    )

    drawer = ImageDraw.Draw(
        pil_image
    )

    for (
        text,
        position,
        font_size,
        color_bgr,
        stroke_width,
    ) in items:

        color_rgb = (
            int(color_bgr[2]),
            int(color_bgr[1]),
            int(color_bgr[0]),
        )

        drawer.text(
            position,
            str(text),
            font=get_chinese_font(
                font_size
            ),
            fill=color_rgb,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0),
        )

    image[:height] = cv2.cvtColor(
        np.asarray(pil_image),
        cv2.COLOR_RGB2BGR,
    )

# ================================================================
# 多帧一致性
#
# 5 帧中至少 4 帧完整结果完全相同。
#
# 如果你想严格 5/5：
# REQUIRED_HITS = 5
# ================================================================

WINDOW_SIZE = 5
REQUIRED_HITS = 4


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
            str(key):
                json_safe(item)
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
            json_safe(item)
            for item in value
        ]

    return value


def load_fixed_background(
    extractor: SideObjectExtractor,
) -> np.ndarray:

    if not BACKGROUND_PATH.exists():

        raise FileNotFoundError(
            f"固定背景不存在：{BACKGROUND_PATH}"
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
            "\n当前 SideCamera ROI 与固定背景不一致："
            f"\ncurrent    = {frame.shape}"
            f"\nbackground = {background.shape}"
        )


def main() -> None:

    if not SHAPE_MODEL_PATH.exists():

        raise FileNotFoundError(
            "\n形状模型不存在："
            f"\n{SHAPE_MODEL_PATH}"
            "\n请修改 SHAPE_MODEL_PATH。"
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
            model_path=
                SHAPE_MODEL_PATH,

            imgsz=224,

            device=
                SHAPE_DEVICE,

            confidence_threshold=
                0.50,

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

    consensus = (
        SideRecognitionConsensus(
            window_size=
                WINDOW_SIZE,

            required_hits=
                REQUIRED_HITS,
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
        )

        print()
        print(
            "=" * 72
        )

        print(
            "SIDE MULTI-FRAME COMBINED TEST"
        )

        print(
            "=" * 72
        )

        print(
            f"Consensus: "
            f"{REQUIRED_HITS}/{WINDOW_SIZE}"
        )

        print(
            "同一个完整结果 "
            "(color, QR, shape) "
            "达到票数后才输出 READY。"
        )

        print(
            "物体拿走后自动 reset。"
        )

        print(
            "P     : 打印当前状态"
        )

        print(
            "R     : 手动 reset"
        )

        print(
            "Q/ESC : 退出"
        )

        print(
            "=" * 72
        )

        last_ready = False

        previous_time = (
            time.perf_counter()
        )

        smooth_fps = 0.0

        raw_result = None
        stable_state = None

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

            # ====================================================
            # No object -> reset complete history
            # ====================================================

            if not (
                object_result.found
                and
                object_result.crop
                is not None
                and
                object_result.crop_mask
                is not None
            ):

                consensus.reset()

                raw_result = None

                stable_state = {
                    "ready":
                        False,

                    "votes":
                        0,

                    "window":
                        0,

                    "required_hits":
                        REQUIRED_HITS,

                    "candidate":
                        None,

                    "result":
                        None,
                }

                last_ready = False

                line1 = (
                    "No Object"
                )

                line2 = (
                    "Consensus: 0/0"
                )

                line3 = ""

                line4 = getattr(
                    object_result,
                    "reason",
                    "",
                )

            else:

                # =================================================
                # Same frame -> Color + QR + Shape
                # =================================================

                start = (
                    time.perf_counter()
                )

                raw_result = (
                    recognizer.recognize(
                        object_result.crop,
                        object_result.crop_mask,
                    )
                )

                stable_state = (
                    consensus.update(
                        raw_result
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

                raw_color = (
                    raw_result[
                        "color"
                    ]
                )

                raw_qr = (
                    raw_result[
                        "qr_present"
                    ]
                )

                raw_shape = (
                    raw_result[
                        "shape"
                    ]
                )

                line1 = (
                    f"RAW: "
                    f"{raw_color} | "
                    f"QR={'Y' if raw_qr else 'N'} | "
                    f"{raw_shape}"
                )

                line2 = (
                    f"Consensus: "
                    f"{stable_state['votes']}"
                    f"/"
                    f"{stable_state['window']} "
                    f"(need {REQUIRED_HITS})"
                )

                if (
                    stable_state[
                        "ready"
                    ]
                    and
                    stable_state[
                        "result"
                    ]
                    is not None
                ):

                    final_result = (
                        stable_state[
                            "result"
                        ]
                    )

                    line3 = (
                        f"READY: "
                        f"{final_result['color']} | "
                        f"QR="
                        f"{'Y' if final_result['qr_present'] else 'N'} | "
                        f"{final_result['shape']}"
                    )

                    line4 = (
                        f"ColorConf="
                        f"{final_result['color_confidence']:.2f} "
                        f"ShapeConf="
                        f"{final_result['shape_confidence']:.2f} "
                        f"{total_ms:.1f}ms"
                    )

                    if not last_ready:

                        print()
                        print(
                            "=" * 72
                        )

                        print(
                            "FINAL RECOGNITION READY"
                        )

                        print(
                            json.dumps(
                                json_safe(
                                    final_result
                                ),
                                ensure_ascii=False,
                                indent=2,
                            )
                        )

                        print(
                            "=" * 72
                        )

                    last_ready = True

                else:

                    line3 = (
                        "WAITING..."
                    )

                    line4 = (
                        f"{total_ms:.1f} ms"
                    )

                    last_ready = False

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
                            int(x1),
                            int(y1),
                        ),
                        (
                            int(x2),
                            int(y2),
                        ),
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow(
                    "Side Multi-frame Crop",
                    object_result.crop,
                )

            # ====================================================
            # FPS
            # ====================================================

            now = (
                time.perf_counter()
            )

            dt = (
                now
                -
                previous_time
            )

            previous_time = (
                now
            )

            if dt > 0:

                fps = (
                    1.0 / dt
                )

                if smooth_fps == 0:

                    smooth_fps = fps

                else:

                    smooth_fps = (
                        0.90
                        *
                        smooth_fps
                        +
                        0.10
                        *
                        fps
                    )

            # ====================================================
            # Draw
            # ====================================================
            cv2.putText(
                display,
                line2,
                (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            ready = bool(
                stable_state
                and
                stable_state.get(
                    "ready",
                    False,
                )
            )

            draw_unicode_panel(
                display,
                [
                    (
                        line1,
                        (16, 7),
                        18,
                        (255, 255, 255),
                        0,
                    ),
                    (
                        line3,
                        (16, 62),
                        20,
                        (
                            (0, 255, 0)
                            if ready
                            else (0, 165, 255)
                        ),
                        1,
                    ),
                ],
            )


            cv2.putText(
                display,
                line4,
                (16, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"FPS={smooth_fps:.1f}",
                (16, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "P=print R=reset Q=quit",
                (
                    16,
                    display.shape[0]
                    -
                    16,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Side Multi-frame Recognition",
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
                ord("r"),
                ord("R"),
            ):

                consensus.reset()

                last_ready = False

                print(
                    "[Consensus] reset"
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
                    "RAW RESULT"
                )

                print(
                    json.dumps(
                        json_safe(
                            raw_result
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )

                print(
                    "STABLE STATE"
                )

                print(
                    json.dumps(
                        json_safe(
                            stable_state
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )

                print(
                    "=" * 72
                )

    finally:

        camera.close()

        cv2.destroyAllWindows()

        print()
        print(
            "Multi-frame combined test finished."
        )


if __name__ == "__main__":
    main()
