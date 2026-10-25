from __future__ import annotations

import json
import sys
import time
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
from vision.code_reader.qr_detector import (
    QRDetector,
    QRTemporalFilter,
)


BACKGROUND_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "side_camera"
    / "side_mechanical_roi_background.png"
)


SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "side_qr"
    / "parameter_tuning"
)


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
) -> object:
    """
    正式/测试运行时只从磁盘加载固定背景。

    不在这里重新拍背景。
    如果背景不存在，直接报错，避免把现场物体误拍进背景。
    """

    if not BACKGROUND_PATH.exists():

        raise FileNotFoundError(
            "\n固定背景图不存在："
            f"\n{BACKGROUND_PATH}"
            "\n\n请先运行："
            "\npython scripts/capture_side_fixed_background.py"
            "\n生成固定背景后再启动检测程序。"
        )

    background = (
        extractor.load_background(
            BACKGROUND_PATH
        )
    )

    print(
        f"[BACKGROUND] loaded: {BACKGROUND_PATH}"
    )

    print(
        f"[BACKGROUND] shape : {background.shape}"
    )

    return background


def warm_up_and_validate(
    camera: SideCamera,
    background,
    *,
    warmup_frames: int = 20,
) -> None:
    """
    相机打开后先丢弃若干帧，让相机输出稳定。

    同时检查当前 SideCamera 机械 ROI 尺寸是否与固定背景一致。
    如果尺寸不一致，说明 ROI / 分辨率已经改过，必须重新拍背景。
    """

    last_frame = None

    for _ in range(
        max(
            1,
            int(
                warmup_frames
            ),
        )
    ):

        last_frame = (
            camera.read()
        )

    if last_frame is None:

        raise RuntimeError(
            "相机预热失败：没有读取到图像"
        )

    if (
        last_frame.shape
        !=
        background.shape
    ):

        raise RuntimeError(
            "\n固定背景尺寸与当前 SideCamera ROI 不一致："
            f"\nbackground = {background.shape}"
            f"\ncurrent    = {last_frame.shape}"
            "\n\n说明相机 ROI / 分辨率已经发生变化。"
            "\n请重新运行 capture_side_fixed_background.py。"
        )

    print(
        f"[Camera] warm-up complete: "
        f"{warmup_frames} frames"
    )

    print(
        "[BACKGROUND] ROI shape validation passed"
    )



def save_sample(
    *,
    mechanical_roi: np.ndarray,
    mechanical_display: np.ndarray,
    object_result,
    qr_crop: np.ndarray | None,
    qr_display: np.ndarray | None,
    qr_binary: np.ndarray | None,
    raw_result: dict[str, Any],
    stable_result: dict[str, Any],
    camera_info: dict[str, Any],
) -> None:

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    state = (
        "qr"
        if stable_result.get(
            "found",
            False,
        )
        else
        "noqr"
    )

    prefix = (
        SAVE_DIR
        /
        f"side_qr_{timestamp}_{state}"
    )

    cv2.imwrite(
        str(prefix)
        +
        "_roi.jpg",
        mechanical_roi,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    cv2.imwrite(
        str(prefix)
        +
        "_display.jpg",
        mechanical_display,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if qr_crop is not None:

        cv2.imwrite(
            str(prefix)
            +
            "_crop.jpg",
            qr_crop,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

    if qr_display is not None:

        cv2.imwrite(
            str(prefix)
            +
            "_qr_display.jpg",
            qr_display,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

    if qr_binary is not None:

        cv2.imwrite(
            str(prefix)
            +
            "_binary.png",
            qr_binary,
        )

    if (
        object_result is not None
        and
        object_result.crop_mask
        is not None
    ):

        cv2.imwrite(
            str(prefix)
            +
            "_crop_mask.png",
            object_result.crop_mask,
        )

    metadata = {
        "background_path":
            str(
                BACKGROUND_PATH
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

        "raw_qr":
            json_safe(
                raw_result
            ),

        "stable_qr":
            json_safe(
                stable_result
            ),
    }

    Path(
        str(prefix)
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


def main() -> None:

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera = SideCamera()

    # 与颜色/后续形状模块保持一致：
    # 先从机械 ROI 中提取完整 object crop。
    extractor = SideObjectExtractor(
        padding_ratio=0.15
    )

    detector = QRDetector(
        opencv_scales=(
            1.0,
            1.5,
        ),
        use_clahe=True,
        use_finder_fallback=True,
    )

    # 保留原来较保守的 4/5 投票。
    temporal_filter = QRTemporalFilter(
        window_size=5,
        required_hits=4,
        max_normalized_center_shift=0.12,
        max_center_shift=50.0,
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
            "SIDE QR PRESENCE TEST V2"
        )

        print(
            "=" * 72
        )

        print(
            "Q / ESC : 退出"
        )

        print(
            "B       : 重新加载固定背景"
        )

        print(
            "S       : 保存当前调试样本"
        )

        print(
            "P       : 打印当前检测结果"
        )

        print(
            f"SAVE DIR: {SAVE_DIR}"
        )

        print(
            "=" * 72
        )
        background = (
            load_fixed_background(
                extractor
            )
        )

        warm_up_and_validate(
            camera,
            background,
            warmup_frames=20,
        )

        temporal_filter.reset()

        previous_time = (
            time.perf_counter()
        )

        smooth_fps = 0.0

        last_stable_found = None

        while True:

            mechanical_roi = (
                camera.read()
            )

            object_result = (
                extractor.extract(
                    mechanical_roi
                )
            )

            mechanical_display = (
                mechanical_roi.copy()
            )

            qr_crop = None
            qr_display = None
            qr_binary = None

            if (
                object_result.found
                and
                object_result.crop
                is not None
            ):

                qr_crop = (
                    object_result.crop
                )

                raw_result = (
                    detector.detect(
                        qr_crop
                    )
                )

                stable_result = (
                    temporal_filter.update(
                        raw_result
                    )
                )

                qr_display = (
                    detector.draw_result(
                        qr_crop,
                        raw_result,
                        stable_result,
                    )
                )

                qr_binary = (
                    detector.prepare_debug_binary(
                        qr_crop
                    )
                )

                if (
                    object_result.crop_bbox
                    is not None
                ):

                    x1, y1, x2, y2 = (
                        object_result.crop_bbox
                    )

                    cv2.rectangle(
                        mechanical_display,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

            else:

                # 没有物体时必须清空历史。
                # 否则上一件物体的 QR 正票可能残留到下一件物体。
                temporal_filter.reset()

                raw_result = (
                    detector.empty_result()
                )

                stable_result = {
                    "found":
                        False,

                    "raw_found":
                        False,

                    "votes":
                        0,

                    "window":
                        0,

                    "required_hits":
                        temporal_filter.required_hits,

                    "spatial_ok":
                        False,

                    "center":
                        None,

                    "normalized_center":
                        None,

                    "maximum_shift":
                        None,

                    "shift_mode":
                        "",
                }

            stable_found = bool(
                stable_result[
                    "found"
                ]
            )

            if (
                stable_found
                !=
                last_stable_found
            ):

                print()
                print(
                    "=" * 72
                )

                print(
                    "[QR] "
                    +
                    (
                        "存在二维码"
                        if stable_found
                        else
                        "没有二维码"
                    )
                )

                print(
                    f"Raw      : "
                    f"{raw_result.get('found', False)}"
                )

                print(
                    f"Method   : "
                    f"{raw_result.get('method', '')}"
                )

                print(
                    f"Vote     : "
                    f"{stable_result.get('votes', 0)}"
                    f"/"
                    f"{stable_result.get('window', 0)}"
                )

                print(
                    f"Center   : "
                    f"{stable_result.get('center')}"
                )

                print(
                    "=" * 72
                )

                last_stable_found = (
                    stable_found
                )

            # ----------------------------------------------------
            # Mechanical display
            # ----------------------------------------------------

            status = (
                "QR: YES"
                if stable_found
                else "QR: NO"
            )

            status_color = (
                (0, 255, 0)
                if stable_found
                else (0, 0, 255)
            )

            cv2.putText(
                mechanical_display,
                status,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                status_color,
                2,
                cv2.LINE_AA,
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
                    1.0 / dt
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
                mechanical_display,
                f"FPS: {smooth_fps:.1f}",
                (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                mechanical_display,
                "S=save P=print B=reload-bg Q=quit",
                (
                    18,
                    mechanical_display.shape[0]
                    -
                    18,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Side QR Mechanical ROI",
                mechanical_display,
            )

            if qr_display is not None:

                cv2.imshow(
                    "Side QR Object Crop",
                    qr_display,
                )

            if qr_binary is not None:

                cv2.imshow(
                    "Side QR Binary",
                    qr_binary,
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

                # B 只重新从磁盘加载固定背景，不拍新背景。
                temporal_filter.reset()

                last_stable_found = (
                    None
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
                    json.dumps(
                        {
                            "raw":
                                json_safe(
                                    raw_result
                                ),

                            "stable":
                                json_safe(
                                    stable_result
                                ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
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
                    mechanical_display=mechanical_display,
                    object_result=object_result,
                    qr_crop=qr_crop,
                    qr_display=qr_display,
                    qr_binary=qr_binary,
                    raw_result=raw_result,
                    stable_result=stable_result,
                    camera_info=camera_info,
                )

    finally:

        # SideCamera 当前接口使用 close()。
        camera.close()

        cv2.destroyAllWindows()

        print()
        print(
            "QR detector finished"
        )


if __name__ == "__main__":
    main()