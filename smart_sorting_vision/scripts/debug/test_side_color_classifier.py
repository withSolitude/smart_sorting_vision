from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from vision.cameras.side_camera import SideCamera
from vision.processors.side_object_extractor import SideObjectExtractor
from vision.color.color_classifier import ColorClassifier


SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "side_color_classifier"
    / "parameter_tuning"
)


def capture_background(
    camera: SideCamera,
    extractor: SideObjectExtractor,
) -> None:

    input(
        "\n请清空侧面识别台，然后按 Enter 建立空背景..."
    )

    extractor.capture_background(
        camera=camera,
        frame_count=30,
        settle_frames=15,
    )

    print(
        "[BACKGROUND] complete"
    )


def _json_safe(
    value: Any,
) -> Any:
    """
    把 numpy 类型转换为 JSON 可保存类型。
    """

    if isinstance(
        value,
        np.ndarray,
    ):
        return None

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        ),
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
                _json_safe(
                    item
                )
            for key, item
            in value.items()
            if not isinstance(
                item,
                np.ndarray,
            )
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    return value


def _classifier_settings(
    classifier: ColorClassifier,
) -> dict[str, Any]:

    names = [
        "mask_erode_px",
        "achromatic_s_max",
        "achromatic_lab_chroma_max",
        "reliable_v_min",
        "reliable_v_max",
        "chromatic_s_min",
        "chromatic_lab_chroma_min",
        "hue_hist_bins",
        "dominant_hue_half_width",
        "black_v_max",
        "black_lab_l_max",
        "white_lab_l_p75_min",
        "white_lab_l_p90_min",
        "brown_h_min",
        "brown_h_max",
        "brown_v_max",
        "brown_lab_l_max",
        "purple_ambiguous_h_min",
        "purple_ambiguous_h_max",
        "purple_lab_a_min",
        "pastel_pink_h_min",
        "pastel_pink_h_max",
        "pastel_pink_s_max",
        "min_pixels",
    ]

    output = {}

    for name in names:

        if hasattr(
            classifier,
            name,
        ):
            output[
                name
            ] = getattr(
                classifier,
                name,
            )

    output[
        "enabled_colors"
    ] = sorted(
        classifier.enabled_colors
    )

    return output


def _extractor_settings(
    extractor: SideObjectExtractor,
) -> dict[str, Any]:

    names = [
        "padding_ratio",
        "min_padding_px",
        "chroma_threshold",
        "light_luma_threshold",
        "dark_luma_threshold",
        "strong_chroma_threshold",
        "strong_light_luma_threshold",
        "strong_dark_luma_threshold",
        "min_object_area",
        "min_object_area_ratio",
        "max_object_area_ratio",
    ]

    output = {}

    for name in names:

        if hasattr(
            extractor,
            name,
        ):
            output[
                name
            ] = getattr(
                extractor,
                name,
            )

    return output


def save_current_sample(
    *,
    mechanical_roi: np.ndarray,
    display: np.ndarray,
    object_result,
    color_result: dict[str, Any] | None,
    camera_info: dict[str, Any],
    classifier: ColorClassifier,
    extractor: SideObjectExtractor,
) -> None:

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    predicted_color = (
        "no_object"
        if color_result is None
        else str(
            color_result.get(
                "color",
                "unknown",
            )
        )
    )

    prefix = (
        SAVE_DIR
        / (
            f"side_color_{timestamp}"
            f"_pred-{predicted_color}"
        )
    )

    # ============================================================
    # 1. 原始机械 ROI
    # ============================================================

    cv2.imwrite(
        str(
            prefix
        )
        + "_roi.jpg",
        mechanical_roi,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    # ============================================================
    # 2. 带结果文字 / crop 框的显示图
    # ============================================================

    cv2.imwrite(
        str(
            prefix
        )
        + "_display.jpg",
        display,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    # ============================================================
    # 3. SideObjectExtractor 输出
    # ============================================================

    if (
        object_result is not None
        and object_result.crop is not None
    ):

        cv2.imwrite(
            str(
                prefix
            )
            + "_crop.jpg",
            object_result.crop,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

    if (
        object_result is not None
        and object_result.object_mask
        is not None
    ):

        cv2.imwrite(
            str(
                prefix
            )
            + "_object_mask.png",
            object_result.object_mask,
        )

    if (
        object_result is not None
        and object_result.crop_mask
        is not None
    ):

        cv2.imwrite(
            str(
                prefix
            )
            + "_crop_mask.png",
            object_result.crop_mask,
        )

    if (
        object_result is not None
        and getattr(
            object_result,
            "core_mask",
            None,
        )
        is not None
    ):

        cv2.imwrite(
            str(
                prefix
            )
            + "_core_mask.png",
            object_result.core_mask,
        )

    if (
        object_result is not None
        and getattr(
            object_result,
            "difference_score",
            None,
        )
        is not None
    ):

        cv2.imwrite(
            str(
                prefix
            )
            + "_difference_score.png",
            object_result.difference_score,
        )

    # ============================================================
    # 4. ColorClassifier 输出 Mask
    # ============================================================

    if color_result is not None:

        stable_mask = color_result.get(
            "stable_mask"
        )

        if stable_mask is not None:

            cv2.imwrite(
                str(
                    prefix
                )
                + "_stable_mask.png",
                stable_mask,
            )

        reliable_mask = color_result.get(
            "reliable_mask"
        )

        if reliable_mask is not None:

            cv2.imwrite(
                str(
                    prefix
                )
                + "_reliable_mask.png",
                reliable_mask,
            )

    # ============================================================
    # 5. JSON
    #
    # 之后你把这份 JSON 和图片一起发给我，
    # 我就能直接看：
    # H / S / V / Lab / chromatic_ratio / confidence
    # 以及当时相机曝光、白平衡等参数。
    # ============================================================

    metadata = {
        "timestamp":
            timestamp,

        "camera":
            _json_safe(
                camera_info
            ),

        "extractor_settings":
            _json_safe(
                _extractor_settings(
                    extractor
                )
            ),

        "classifier_settings":
            _json_safe(
                _classifier_settings(
                    classifier
                )
            ),

        "object": {
            "found":
                bool(
                    object_result.found
                )
                if object_result
                is not None
                else False,

            "reason":
                getattr(
                    object_result,
                    "reason",
                    None,
                ),

            "bbox":
                _json_safe(
                    getattr(
                        object_result,
                        "bbox",
                        None,
                    )
                ),

            "crop_bbox":
                _json_safe(
                    getattr(
                        object_result,
                        "crop_bbox",
                        None,
                    )
                ),

            "area":
                _json_safe(
                    getattr(
                        object_result,
                        "area",
                        None,
                    )
                ),

            "area_ratio":
                _json_safe(
                    getattr(
                        object_result,
                        "area_ratio",
                        None,
                    )
                ),

            "crop_shape":
                (
                    list(
                        object_result.crop.shape
                    )
                    if (
                        object_result
                        is not None
                        and object_result.crop
                        is not None
                    )
                    else None
                ),
        },

        "color_result":
            (
                _json_safe(
                    color_result
                )
                if color_result
                is not None
                else None
            ),
    }

    Path(
        str(
            prefix
        )
        + ".json"
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
        "=" * 72
    )

    print(
        "[SAVE] Color tuning sample"
    )

    print(
        "=" * 72
    )

    print(
        f"Prediction : {predicted_color}"
    )

    if color_result is not None:

        print(
            f"Confidence : "
            f"{float(color_result.get('confidence', 0.0)):.3f}"
        )

        print(
            f"Kind       : "
            f"{color_result.get('kind', '')}"
        )

        print(
            f"Reason     : "
            f"{color_result.get('reason', '')}"
        )

        print(
            f"Features   : "
            f"{color_result.get('features', {})}"
        )

    print(
        f"Saved to   : {SAVE_DIR}"
    )

    print(
        f"Prefix     : {prefix.name}"
    )

    print(
        "=" * 72
    )

    print()


def main() -> None:

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera = SideCamera()

    extractor = SideObjectExtractor(
        padding_ratio=0.15
    )

    # ============================================================
    # 通用颜色识别参数
    #
    # 现在先不要继续改。
    # 你先用 S 保存识别错误 / 边界颜色样本，
    # 再根据真实样本统一调参数。
    # ============================================================

    classifier = ColorClassifier(
        enabled_colors=[
            "red",
            "orange",
            "yellow",
            "green",
            "cyan",
            "blue",
            "purple",
            "pink",
            "brown",
            "black",
            "white",
            "gray",
        ]
    )

    try:

        camera.open()

        camera_info = (
            camera.get_stream_info()
        )

        print()
        print(
            "=" * 72
        )

        print(
            "UNIVERSAL SIDE COLOR TEST"
        )

        print(
            "=" * 72
        )

        for key, value in (
            camera_info.items()
        ):
            print(
                f"{key:<28}: {value}"
            )

        print(
            "=" * 72
        )

        print(
            "Q / ESC : 退出"
        )

        print(
            "B       : 重新建立空背景"
        )

        print(
            "S       : 保存当前调参样本"
        )

        print(
            "P       : 打印当前颜色详细结果"
        )

        print(
            f"SAVE DIR: {SAVE_DIR}"
        )

        print(
            "=" * 72
        )

        capture_background(
            camera,
            extractor,
        )

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

            color_result = None

            if (
                object_result.found
                and
                object_result.crop is not None
                and
                object_result.crop_mask
                is not None
            ):

                color_result = (
                    classifier.classify(
                        object_result.crop,
                        object_result.crop_mask,
                    )
                )

                color = color_result.get(
                    "color",
                    "unknown",
                )

                confidence = float(
                    color_result.get(
                        "confidence",
                        0.0,
                    )
                )

                kind = color_result.get(
                    "kind",
                    ""
                )

                reason = color_result.get(
                    "reason",
                    ""
                )

                chromatic_ratio = float(
                    color_result.get(
                        "chromatic_ratio",
                        0.0,
                    )
                )

                line1 = (
                    f"Color={color} "
                    f"Conf={confidence:.2f}"
                )

                line2 = (
                    f"Kind={kind} "
                    f"Chromatic={chromatic_ratio:.2f}"
                )

                line3 = (
                    f"{reason}"
                )

                if (
                    object_result.crop_bbox
                    is not None
                ):

                    x1, y1, x2, y2 = (
                        object_result.crop_bbox
                    )

                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow(
                    "Universal Color Crop",
                    object_result.crop,
                )

                reliable_mask = (
                    color_result.get(
                        "reliable_mask"
                    )
                )

                if reliable_mask is not None:

                    cv2.imshow(
                        "Universal Color Reliable Mask",
                        reliable_mask,
                    )

                stable_mask = (
                    color_result.get(
                        "stable_mask"
                    )
                )

                if stable_mask is not None:

                    cv2.imshow(
                        "Universal Color Stable Mask",
                        stable_mask,
                    )

            else:

                line1 = (
                    "No Object"
                )

                line2 = ""

                line3 = (
                    object_result.reason
                )

            cv2.putText(
                display,
                line1,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
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
                    0.48,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                line3,
                (18, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "S=save  P=print  B=background  Q=quit",
                (
                    18,
                    display.shape[0]
                    - 18,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Universal Side Color Recognition",
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
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

                capture_background(
                    camera,
                    extractor,
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
                    "CURRENT COLOR RESULT"
                )

                print(
                    "=" * 72
                )

                if color_result is None:

                    print(
                        "No valid color result"
                    )

                else:

                    printable = (
                        _json_safe(
                            color_result
                        )
                    )

                    print(
                        json.dumps(
                            printable,
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

                # 每次保存时重新读一次 camera info，
                # 如果你之后改曝光测试，不会把参数记错。
                camera_info = (
                    camera.get_stream_info()
                )

                save_current_sample(
                    mechanical_roi=mechanical_roi,
                    display=display,
                    object_result=object_result,
                    color_result=color_result,
                    camera_info=camera_info,
                    classifier=classifier,
                    extractor=extractor,
                )

    finally:

        camera.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
