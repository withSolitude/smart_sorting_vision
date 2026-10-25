from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# Project root
# ============================================================

def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (
            (parent / "config" / "runtime_config.yaml").is_file()
            and (parent / "runtime" / "app_config.py").is_file()
            and (parent / "vision" / "cameras" / "side_camera.py").is_file()
            and (parent / "vision" / "processors" / "side_object_extractor.py").is_file()
            and (parent / "vision" / "color" / "color_classifier.py").is_file()
        ):
            return parent

    raise RuntimeError(
        "Cannot find smart_sorting_vision project root. "
        "Put this file in scripts/last_use/calibration/."
    )


ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from runtime.app_config import load_runtime_config  # noqa: E402
from vision.cameras.side_camera import SideCamera  # noqa: E402
from vision.processors.side_object_extractor import SideObjectExtractor  # noqa: E402
from vision.color.color_classifier import ColorClassifier, HueBand  # noqa: E402


# ============================================================
# Paths / windows
# ============================================================

CONFIG_PATH = ROOT / "config" / "runtime_config.yaml"

SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "side_color_calibration"
)

WINDOW_MAIN = "SIDE COLOR CALIBRATION"
WINDOW_CROP = "COLOR CROP"
WINDOW_STABLE = "COLOR STABLE MASK"
WINDOW_RELIABLE = "COLOR RELIABLE MASK"
WINDOW_CTRL = "COLOR BOUNDARY CONTROLS"


# ============================================================
# Helpers
# ============================================================

def noop(_: int) -> None:
    pass


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
            if not isinstance(v, np.ndarray)
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(v)
            for v in value
        ]

    return value


def draw_lines(
    image: np.ndarray,
    lines: list[str],
    *,
    x: int = 12,
    y: int = 26,
    line_height: int = 25,
) -> None:
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += line_height


def resolve_project_path(value: Any) -> Path:
    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


# ============================================================
# Read defaults directly from production ColorClassifier
# ============================================================

def get_hue_band(
    classifier: ColorClassifier,
    name: str,
) -> HueBand:
    for band in classifier.HUE_BANDS:
        if band.name == name:
            return band

    raise RuntimeError(
        f"ColorClassifier.HUE_BANDS missing: {name}"
    )


def read_production_defaults(
    classifier: ColorClassifier,
) -> dict[str, Any]:
    cyan = get_hue_band(
        classifier,
        "cyan",
    )

    blue = get_hue_band(
        classifier,
        "blue",
    )

    purple = get_hue_band(
        classifier,
        "purple",
    )

    return {
        # 正式 color_classifier.py 当前的真实参数
        "cyan_blue_split": int(round(cyan.end)),
        "blue_purple_split": int(round(blue.end)),

        "purple_ambiguous_h_min": int(
            classifier.purple_ambiguous_h_min
        ),
        "purple_ambiguous_h_max": int(
            classifier.purple_ambiguous_h_max
        ),
        "purple_lab_a_min": int(
            classifier.purple_lab_a_min
        ),

        "black_v_max": int(
            classifier.black_v_max
        ),

        "black_lab_l_max": int(
            classifier.black_lab_l_max
        ),

        "enable_brown": (
            "brown"
            in classifier.enabled_colors
        ),

        # 用于恢复 HUE_BANDS 时保留其它边界
        "orange_start": float(
            get_hue_band(
                classifier,
                "orange",
            ).start
        ),

        "orange_end": float(
            get_hue_band(
                classifier,
                "orange",
            ).end
        ),

        "yellow_end": float(
            get_hue_band(
                classifier,
                "yellow",
            ).end
        ),

        "green_end": float(
            get_hue_band(
                classifier,
                "green",
            ).end
        ),

        "cyan_start": float(
            cyan.start
        ),

        "purple_end": float(
            purple.end
        ),

        "pink_end": float(
            get_hue_band(
                classifier,
                "pink",
            ).end
        ),
    }


# ============================================================
# OpenCV controls
# ============================================================

def create_controls(
    defaults: dict[str, Any],
) -> None:
    cv2.namedWindow(
        WINDOW_CTRL,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_CTRL,
        640,
        430,
    )

    cv2.createTrackbar(
        "Cyan-Blue H",
        WINDOW_CTRL,
        int(defaults["cyan_blue_split"]),
        140,
        noop,
    )

    cv2.createTrackbar(
        "Blue-Purple H",
        WINDOW_CTRL,
        int(defaults["blue_purple_split"]),
        170,
        noop,
    )

    cv2.createTrackbar(
        "Purple Amb H Min",
        WINDOW_CTRL,
        int(
            defaults[
                "purple_ambiguous_h_min"
            ]
        ),
        170,
        noop,
    )

    cv2.createTrackbar(
        "Purple Amb H Max",
        WINDOW_CTRL,
        int(
            defaults[
                "purple_ambiguous_h_max"
            ]
        ),
        179,
        noop,
    )

    cv2.createTrackbar(
        "Purple Lab-a Min",
        WINDOW_CTRL,
        int(
            defaults[
                "purple_lab_a_min"
            ]
        ),
        255,
        noop,
    )

    cv2.createTrackbar(
        "Black V Max",
        WINDOW_CTRL,
        int(
            defaults[
                "black_v_max"
            ]
        ),
        120,
        noop,
    )

    cv2.createTrackbar(
        "Black Lab-L Max",
        WINDOW_CTRL,
        int(
            defaults[
                "black_lab_l_max"
            ]
        ),
        160,
        noop,
    )

    cv2.createTrackbar(
        "Enable Brown",
        WINDOW_CTRL,
        int(
            bool(
                defaults[
                    "enable_brown"
                ]
            )
        ),
        1,
        noop,
    )


def reset_controls(
    defaults: dict[str, Any],
) -> None:
    cv2.setTrackbarPos(
        "Cyan-Blue H",
        WINDOW_CTRL,
        int(
            defaults[
                "cyan_blue_split"
            ]
        ),
    )

    cv2.setTrackbarPos(
        "Blue-Purple H",
        WINDOW_CTRL,
        int(
            defaults[
                "blue_purple_split"
            ]
        ),
    )

    cv2.setTrackbarPos(
        "Purple Amb H Min",
        WINDOW_CTRL,
        int(
            defaults[
                "purple_ambiguous_h_min"
            ]
        ),
    )

    cv2.setTrackbarPos(
        "Purple Amb H Max",
        WINDOW_CTRL,
        int(
            defaults[
                "purple_ambiguous_h_max"
            ]
        ),
    )

    cv2.setTrackbarPos(
        "Purple Lab-a Min",
        WINDOW_CTRL,
        int(
            defaults[
                "purple_lab_a_min"
            ]
        ),
    )

    cv2.setTrackbarPos(
        "Black V Max",
        WINDOW_CTRL,
        int(
            defaults[
                "black_v_max"
            ]
        ),
    )

    cv2.setTrackbarPos(
        "Black Lab-L Max",
        WINDOW_CTRL,
        int(
            defaults[
                "black_lab_l_max"
            ]
        ),
    )

    cv2.setTrackbarPos(
        "Enable Brown",
        WINDOW_CTRL,
        int(
            bool(
                defaults[
                    "enable_brown"
                ]
            )
        ),
    )


def apply_controls(
    classifier: ColorClassifier,
    production_defaults: dict[str, Any],
) -> dict[str, Any]:
    cyan_blue = cv2.getTrackbarPos(
        "Cyan-Blue H",
        WINDOW_CTRL,
    )

    blue_purple = cv2.getTrackbarPos(
        "Blue-Purple H",
        WINDOW_CTRL,
    )

    purple_h_min = cv2.getTrackbarPos(
        "Purple Amb H Min",
        WINDOW_CTRL,
    )

    purple_h_max = cv2.getTrackbarPos(
        "Purple Amb H Max",
        WINDOW_CTRL,
    )

    purple_lab_a_min = cv2.getTrackbarPos(
        "Purple Lab-a Min",
        WINDOW_CTRL,
    )

    black_v_max = cv2.getTrackbarPos(
        "Black V Max",
        WINDOW_CTRL,
    )

    black_lab_l_max = cv2.getTrackbarPos(
        "Black Lab-L Max",
        WINDOW_CTRL,
    )

    enable_brown = bool(
        cv2.getTrackbarPos(
            "Enable Brown",
            WINDOW_CTRL,
        )
    )

    # --------------------------------------------------------
    # 参数保护
    # --------------------------------------------------------

    cyan_blue = int(
        np.clip(
            cyan_blue,
            int(
                production_defaults[
                    "cyan_start"
                ]
            ) + 1,
            blue_purple - 1,
        )
    )

    blue_purple = int(
        np.clip(
            blue_purple,
            cyan_blue + 1,
            int(
                production_defaults[
                    "purple_end"
                ]
            ) - 1,
        )
    )

    purple_h_min = int(
        min(
            purple_h_min,
            purple_h_max - 1,
        )
    )

    purple_h_max = int(
        max(
            purple_h_max,
            purple_h_min + 1,
        )
    )

    # --------------------------------------------------------
    # 实时修改本次标定用 classifier
    #
    # 注意：
    # 这里只修改内存里的 classifier，不写回正式文件。
    # --------------------------------------------------------

    classifier.HUE_BANDS = (
        HueBand(
            "orange",
            float(
                production_defaults[
                    "orange_start"
                ]
            ),
            float(
                production_defaults[
                    "orange_end"
                ]
            ),
        ),

        HueBand(
            "yellow",
            float(
                production_defaults[
                    "orange_end"
                ]
            ),
            float(
                production_defaults[
                    "yellow_end"
                ]
            ),
        ),

        HueBand(
            "green",
            float(
                production_defaults[
                    "yellow_end"
                ]
            ),
            float(
                production_defaults[
                    "green_end"
                ]
            ),
        ),

        HueBand(
            "cyan",
            float(
                production_defaults[
                    "cyan_start"
                ]
            ),
            float(
                cyan_blue
            ),
        ),

        HueBand(
            "blue",
            float(
                cyan_blue
            ),
            float(
                blue_purple
            ),
        ),

        HueBand(
            "purple",
            float(
                blue_purple
            ),
            float(
                production_defaults[
                    "purple_end"
                ]
            ),
        ),

        HueBand(
            "pink",
            float(
                production_defaults[
                    "purple_end"
                ]
            ),
            float(
                production_defaults[
                    "pink_end"
                ]
            ),
        ),
    )

    classifier.purple_ambiguous_h_min = int(
        purple_h_min
    )

    classifier.purple_ambiguous_h_max = int(
        purple_h_max
    )

    classifier.purple_lab_a_min = int(
        purple_lab_a_min
    )

    classifier.black_v_max = int(
        black_v_max
    )

    classifier.black_lab_l_max = int(
        black_lab_l_max
    )

    if enable_brown:
        classifier.enabled_colors.add(
            "brown"
        )
    else:
        classifier.enabled_colors.discard(
            "brown"
        )

    return {
        "cyan_blue_split": int(
            cyan_blue
        ),

        "blue_purple_split": int(
            blue_purple
        ),

        "purple_ambiguous_h_min": int(
            purple_h_min
        ),

        "purple_ambiguous_h_max": int(
            purple_h_max
        ),

        "purple_lab_a_min": int(
            purple_lab_a_min
        ),

        "black_v_max": int(
            black_v_max
        ),

        "black_lab_l_max": int(
            black_lab_l_max
        ),

        "enable_brown": bool(
            enable_brown
        ),
    }


# ============================================================
# Save sample
# ============================================================

def save_sample(
    *,
    roi: np.ndarray,
    display: np.ndarray,
    object_result: Any,
    color_result: dict[str, Any] | None,
    params: dict[str, Any],
    production_defaults: dict[str, Any],
    camera_info: dict[str, Any],
) -> None:
    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    predicted = (
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
        / f"{stamp}_pred-{predicted}"
    )

    cv2.imwrite(
        str(prefix) + "_roi.jpg",
        roi,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    cv2.imwrite(
        str(prefix) + "_display.jpg",
        display,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if (
        object_result is not None
        and getattr(
            object_result,
            "crop",
            None,
        )
        is not None
    ):
        cv2.imwrite(
            str(prefix) + "_crop.jpg",
            object_result.crop,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

    if (
        object_result is not None
        and getattr(
            object_result,
            "crop_mask",
            None,
        )
        is not None
    ):
        cv2.imwrite(
            str(prefix) + "_crop_mask.png",
            object_result.crop_mask,
        )

    if color_result is not None:
        stable_mask = color_result.get(
            "stable_mask"
        )

        reliable_mask = color_result.get(
            "reliable_mask"
        )

        if stable_mask is not None:
            cv2.imwrite(
                str(prefix)
                + "_stable_mask.png",
                stable_mask,
            )

        if reliable_mask is not None:
            cv2.imwrite(
                str(prefix)
                + "_reliable_mask.png",
                reliable_mask,
            )

    metadata = {
        "timestamp": stamp,

        "camera": json_safe(
            camera_info
        ),

        # 当前滑条参数
        "parameters": json_safe(
            params
        ),

        # 正式 color_classifier.py 启动时读到的参数
        "production_defaults": json_safe(
            production_defaults
        ),

        "object": {
            "found": (
                bool(
                    object_result.found
                )
                if object_result is not None
                else False
            ),

            "reason": (
                getattr(
                    object_result,
                    "reason",
                    None,
                )
                if object_result
                is not None
                else None
            ),

            "crop_bbox": (
                json_safe(
                    getattr(
                        object_result,
                        "crop_bbox",
                        None,
                    )
                )
                if object_result
                is not None
                else None
            ),
        },

        "color_result": (
            json_safe(
                color_result
            )
            if color_result
            is not None
            else None
        ),
    }

    Path(
        str(prefix) + ".json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[SAVE] {prefix}"
    )


# ============================================================
# Main
# ============================================================

def main() -> int:
    # --------------------------------------------------------
    # 1. 使用最终比赛 runtime_config.yaml
    # --------------------------------------------------------

    cfg = load_runtime_config(
        ROOT,
        CONFIG_PATH,
    )

    side_cfg = cfg[
        "camera"
    ][
        "side"
    ]

    SideCamera.configure(
        side_cfg
    )

    background_value = cfg[
        "calibration"
    ][
        "side_background"
    ]

    background_path = resolve_project_path(
        background_value
    )

    if not background_path.is_file():
        raise FileNotFoundError(
            "Side background not found:\n"
            f"{background_path}"
        )

    # --------------------------------------------------------
    # 2. 正式相机 + 正式背景提取器
    # --------------------------------------------------------

    camera = SideCamera()

    extractor = SideObjectExtractor(
        padding_ratio=0.15
    )

    extractor.load_background(
        background_path
    )

    # --------------------------------------------------------
    # 3. 正式比赛 ColorClassifier
    #
    # 重点：
    # 所有滑条默认值直接从这个 classifier 读取。
    # 不再在标定脚本里维护另一套 100/130/... 参数。
    # --------------------------------------------------------

    classifier = ColorClassifier()

    production_defaults = (
        read_production_defaults(
            classifier
        )
    )

    print(
        "=" * 78
    )

    print(
        "SIDE COLOR CALIBRATION "
        "- PRODUCTION ColorClassifier"
    )

    print(
        "=" * 78
    )

    print(
        f"[CONFIG] {CONFIG_PATH}"
    )

    print(
        f"[BACKGROUND] {background_path}"
    )

    print()

    print(
        "[PRODUCTION DEFAULTS]"
    )

    print(
        "  Cyan / Blue H       = "
        f"{production_defaults['cyan_blue_split']}"
    )

    print(
        "  Blue / Purple H     = "
        f"{production_defaults['blue_purple_split']}"
    )

    print(
        "  Purple Amb H        = "
        f"{production_defaults['purple_ambiguous_h_min']}"
        "-"
        f"{production_defaults['purple_ambiguous_h_max']}"
    )

    print(
        "  Purple Lab-a Min    = "
        f"{production_defaults['purple_lab_a_min']}"
    )

    print(
        "  Black V Max         = "
        f"{production_defaults['black_v_max']}"
    )

    print(
        "  Black Lab-L Max     = "
        f"{production_defaults['black_lab_l_max']}"
    )

    print(
        "  Brown Enabled       = "
        f"{production_defaults['enable_brown']}"
    )

    print()

    print(
        "Keys:"
    )

    print(
        "  P : print current "
        "classification + features"
    )

    print(
        "  S : save current "
        "images + JSON"
    )

    print(
        "  R : reset sliders to "
        "CURRENT production ColorClassifier"
    )

    print(
        "  Q / ESC : quit"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # 4. OpenCV UI
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_MAIN,
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        WINDOW_CROP,
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        WINDOW_STABLE,
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        WINDOW_RELIABLE,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_MAIN,
        940,
        760,
    )

    cv2.resizeWindow(
        WINDOW_CROP,
        520,
        420,
    )

    cv2.resizeWindow(
        WINDOW_STABLE,
        520,
        420,
    )

    cv2.resizeWindow(
        WINDOW_RELIABLE,
        520,
        420,
    )

    create_controls(
        production_defaults
    )

    # --------------------------------------------------------
    # Runtime state
    # --------------------------------------------------------

    last_roi: np.ndarray | None = None
    last_display: np.ndarray | None = None
    last_object_result: Any = None
    last_color_result: (
        dict[str, Any] | None
    ) = None
    last_params: (
        dict[str, Any] | None
    ) = None

    try:
        camera.open()

        camera_info = (
            camera.get_stream_info()
        )

        print(
            "[CAMERA] opened with "
            "final competition config"
        )

        print(
            camera_info
        )

        # ----------------------------------------------------
        # Main loop
        # ----------------------------------------------------

        while True:
            params = apply_controls(
                classifier,
                production_defaults,
            )

            roi = camera.read()

            object_result = (
                extractor.extract(
                    roi
                )
            )

            display = roi.copy()

            color_result = None

            lines = [
                (
                    "Production ColorClassifier "
                    "| P print | S save | "
                    "R reset | Q quit"
                ),

                (
                    f"Cyan/Blue="
                    f"{params['cyan_blue_split']}  "
                    f"Blue/Purple="
                    f"{params['blue_purple_split']}  "
                    f"PurpleAmb="
                    f"{params['purple_ambiguous_h_min']}"
                    "-"
                    f"{params['purple_ambiguous_h_max']}  "
                    f"Lab-a>="
                    f"{params['purple_lab_a_min']}"
                ),

                (
                    f"BlackV<="
                    f"{params['black_v_max']}  "
                    f"BlackLabL<="
                    f"{params['black_lab_l_max']}  "
                    f"Brown="
                    f"{'ON' if params['enable_brown'] else 'OFF'}"
                ),
            ]

            if (
                object_result.found
                and object_result.crop
                is not None
                and object_result.crop_mask
                is not None
            ):
                # ------------------------------------------------
                # 正式比赛 ColorClassifier
                # ------------------------------------------------

                color_result = (
                    classifier.classify(
                        object_result.crop,
                        object_result.crop_mask,
                    )
                )

                color = str(
                    color_result.get(
                        "color",
                        "unknown",
                    )
                )

                confidence = float(
                    color_result.get(
                        "confidence",
                        0.0,
                    )
                )

                kind = str(
                    color_result.get(
                        "kind",
                        "",
                    )
                )

                reason = str(
                    color_result.get(
                        "reason",
                        "",
                    )
                )

                features = (
                    color_result.get(
                        "features",
                        {},
                    )
                    or {}
                )

                lines.append(
                    f"Predict={color}  "
                    f"Confidence="
                    f"{confidence:.3f}  "
                    f"Kind={kind}"
                )

                lines.append(
                    f"Reason={reason}"
                )

                hue = features.get(
                    "hue"
                )

                saturation = (
                    features.get(
                        "s"
                    )
                )

                value = features.get(
                    "v"
                )

                lab_l = features.get(
                    "lab_l"
                )

                lab_a = features.get(
                    "lab_a"
                )

                purity = features.get(
                    "dominant_purity"
                )

                if (
                    hue is not None
                    and saturation
                    is not None
                    and value is not None
                ):
                    lines.append(
                        "Formal features: "
                        f"H={float(hue):.2f}  "
                        f"S={float(saturation):.1f}  "
                        f"V={float(value):.1f}"
                    )

                if (
                    lab_l is not None
                    and lab_a is not None
                ):
                    if purity is not None:
                        lines.append(
                            f"Lab-L="
                            f"{float(lab_l):.1f}  "
                            f"Lab-a="
                            f"{float(lab_a):.1f}  "
                            f"Purity="
                            f"{float(purity):.3f}"
                        )
                    else:
                        lines.append(
                            f"Lab-L="
                            f"{float(lab_l):.1f}  "
                            f"Lab-a="
                            f"{float(lab_a):.1f}"
                        )

                # ------------------------------------------------
                # Object bbox
                # ------------------------------------------------

                if (
                    object_result.crop_bbox
                    is not None
                ):
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    ) = (
                        object_result.crop_bbox
                    )

                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                # ------------------------------------------------
                # Crop
                # ------------------------------------------------

                cv2.imshow(
                    WINDOW_CROP,
                    object_result.crop,
                )

                # ------------------------------------------------
                # Masks from production classifier
                # ------------------------------------------------

                stable_mask = (
                    color_result.get(
                        "stable_mask"
                    )
                )

                reliable_mask = (
                    color_result.get(
                        "reliable_mask"
                    )
                )

                if stable_mask is None:
                    stable_show = np.zeros(
                        (
                            object_result.crop.shape[0],
                            object_result.crop.shape[1],
                        ),
                        dtype=np.uint8,
                    )
                else:
                    stable_show = (
                        stable_mask
                    )

                if reliable_mask is None:
                    reliable_show = np.zeros(
                        (
                            object_result.crop.shape[0],
                            object_result.crop.shape[1],
                        ),
                        dtype=np.uint8,
                    )
                else:
                    reliable_show = (
                        reliable_mask
                    )

                cv2.imshow(
                    WINDOW_STABLE,
                    stable_show,
                )

                cv2.imshow(
                    WINDOW_RELIABLE,
                    reliable_show,
                )

            else:
                lines.append(
                    "No object: "
                    + str(
                        getattr(
                            object_result,
                            "reason",
                            "unknown",
                        )
                    )
                )

                blank = np.zeros(
                    (320, 480),
                    dtype=np.uint8,
                )

                cv2.imshow(
                    WINDOW_STABLE,
                    blank,
                )

                cv2.imshow(
                    WINDOW_RELIABLE,
                    blank,
                )

                cv2.imshow(
                    WINDOW_CROP,
                    cv2.cvtColor(
                        blank,
                        cv2.COLOR_GRAY2BGR,
                    ),
                )

            # ------------------------------------------------
            # Main preview
            # ------------------------------------------------

            draw_lines(
                display,
                lines,
            )

            cv2.imshow(
                WINDOW_MAIN,
                display,
            )

            last_roi = roi.copy()
            last_display = (
                display.copy()
            )
            last_object_result = (
                object_result
            )
            last_color_result = (
                color_result
            )
            last_params = dict(
                params
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

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
                ord("p"),
                ord("P"),
            ):
                print()
                print(
                    "=" * 78
                )
                print(
                    "CURRENT COLOR "
                    "CALIBRATION RESULT"
                )
                print(
                    "=" * 78
                )

                print(
                    json.dumps(
                        {
                            "parameters":
                                last_params,

                            "production_defaults":
                                production_defaults,

                            "color_result":
                                json_safe(
                                    last_color_result
                                ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )

                print(
                    "=" * 78
                )

                continue

            if key in (
                ord("s"),
                ord("S"),
            ):
                if (
                    last_roi
                    is not None
                    and last_display
                    is not None
                ):
                    save_sample(
                        roi=last_roi,
                        display=last_display,
                        object_result=last_object_result,
                        color_result=last_color_result,
                        params=(
                            last_params
                            or {}
                        ),
                        production_defaults=production_defaults,
                        camera_info=camera.get_stream_info(),
                    )

                continue

            if key in (
                ord("r"),
                ord("R"),
            ):
                reset_controls(
                    production_defaults
                )

                print(
                    "[RESET] sliders -> "
                    "current production "
                    "ColorClassifier defaults"
                )

                continue

    finally:
        try:
            camera.release()
        except Exception:
            try:
                camera.close()
            except Exception:
                pass

        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
