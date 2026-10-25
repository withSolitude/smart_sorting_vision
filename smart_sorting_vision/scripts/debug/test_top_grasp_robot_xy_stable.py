from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (
            (parent / "vision").is_dir()
            and
            (parent / "data").is_dir()
        ):
            return parent

    raise RuntimeError(
        "无法找到 smart_sorting_vision 工程根目录"
    )


ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from vision.cameras.top_camera import TopCamera
from vision.models.top_segmenter import TopSegmenter
from vision.grasp.collision_aware_grasp_planner import (
    CollisionAwareGraspPlanner,
)
from vision.calibration.pixel_to_robot import (
    PixelToRobotTransform,
)
from vision.pipeline.top_pick_consensus import (
    TopPickConsensus,
)


MODEL_PATH = (
    ROOT
    / "outputs"
    / "training"
    / "top_seg"
    / "v3"
    / "v3_interim_aug"
    / "weights"
    / "best.pt"
)

CALIBRATION_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "robot_mapping"
    / "pixel_to_robot.json"
)

CALIBRATION_POINTS_PATH = (
    ROOT
    / "data"
    / "calibration"
    / "top_camera"
    / "robot_mapping"
    / "tray4_calibration_points.csv"
)

SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "top_grasp_robot_xy_stable"
)


# ================================================================
# YOLO
# ================================================================

CONF = 0.35
IMGSZ = 832

DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# 防碰撞抓取
# ================================================================

OBSTACLE_MARGIN = 12
BORDER_MARGIN = 12
MIN_SAFE_AREA = 50
MIN_CLEARANCE = 3.0


# ================================================================
# 多帧稳定
#
# 5帧窗口，至少4帧为同一个目标。
# ================================================================

CONSENSUS_WINDOW = 2
CONSENSUS_REQUIRED = 2

# 同一物体的图像中心允许跨帧移动范围。
MAX_TRACK_CENTER_SHIFT_PX = 55.0

# 进入 READY 前，历史机械抓取点离中位数的最大距离。
MAX_GRASP_SPREAD_MM = 3.0

# 连续丢失几帧后认为目标被拿走 / 换目标。
RESET_AFTER_MISSES = 3


def safe_release_camera(
    camera,
) -> None:
    if hasattr(
        camera,
        "release",
    ):
        try:
            camera.release()
            return
        except Exception:
            pass

    if hasattr(
        camera,
        "close",
    ):
        try:
            camera.close()
        except Exception:
            pass


def load_calibration_polygon(
    path: Path,
) -> np.ndarray | None:
    """
    P1~P4 只用于画参考线。
    不作为抓取区域限制。
    """

    if not path.exists():
        return None

    points: dict[
        str,
        tuple[
            float,
            float,
        ],
    ] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f
        )

        for row in reader:
            point_id = str(
                row["id"]
            ).strip()

            points[
                point_id
            ] = (
                float(
                    row["u"]
                ),
                float(
                    row["v"]
                ),
            )

    ids = [
        "P1",
        "P2",
        "P3",
        "P4",
    ]

    if not all(
        point_id in points
        for point_id
        in ids
    ):
        return None

    return np.asarray(
        [
            points["P1"],
            points["P2"],
            points["P3"],
            points["P4"],
        ],
        dtype=np.float32,
    )


def point_inside_polygon(
    point: tuple[
        int,
        int,
    ],
    polygon: np.ndarray | None,
) -> bool:
    if polygon is None:
        return False

    value = (
        cv2.pointPolygonTest(
            polygon.reshape(
                -1,
                1,
                2,
            ),
            (
                float(
                    point[0]
                ),
                float(
                    point[1]
                ),
            ),
            False,
        )
    )

    return (
        value
        >=
        0
    )


def draw_calibration_polygon(
    image: np.ndarray,
    polygon: np.ndarray | None,
) -> None:
    if polygon is None:
        return

    contour = (
        np.round(
            polygon
        )
        .astype(
            np.int32
        )
        .reshape(
            -1,
            1,
            2,
        )
    )

    cv2.polylines(
        image,
        [
            contour
        ],
        True,
        (
            255,
            255,
            0,
        ),
        2,
        cv2.LINE_AA,
    )

    for i, point in enumerate(
        polygon,
        start=1,
    ):
        cv2.putText(
            image,
            f"P{i}",
            (
                int(
                    round(
                        point[0]
                    )
                )
                +
                8,
                int(
                    round(
                        point[1]
                    )
                )
                -
                8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                255,
                255,
                0,
            ),
            1,
            cv2.LINE_AA,
        )


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[
        int,
        int,
    ],
    *,
    scale: float = 0.52,
    thickness: int = 1,
) -> None:
    x, y = origin

    (
        width,
        height,
    ), baseline = (
        cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
    )

    cv2.rectangle(
        image,
        (
            x - 3,
            y - height - 5,
        ),
        (
            x + width + 4,
            y + baseline + 4,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    cv2.putText(
        image,
        text,
        (
            x,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (
            255,
            255,
            255,
        ),
        thickness,
        cv2.LINE_AA,
    )


def polygon_center(
    polygon: np.ndarray,
) -> tuple[
    float,
    float,
]:
    moments = cv2.moments(
        polygon.astype(
            np.float32
        )
    )

    if abs(
        moments["m00"]
    ) > 1e-9:
        return (
            float(
                moments["m10"]
                /
                moments["m00"]
            ),
            float(
                moments["m01"]
                /
                moments["m00"]
            ),
        )

    mean = np.mean(
        polygon,
        axis=0,
    )

    return (
        float(
            mean[0]
        ),
        float(
            mean[1]
        ),
    )


def build_candidates(
    *,
    frame: np.ndarray,
    objects: list[dict],
    planner: CollisionAwareGraspPlanner,
    transform: PixelToRobotTransform,
    calibration_polygon: np.ndarray | None,
) -> list[dict]:
    valid_objects: list[
        tuple[
            int,
            np.ndarray,
        ]
    ] = []

    for list_index, obj in enumerate(
        objects
    ):
        polygon = obj.get(
            "polygon"
        )

        if polygon is None:
            continue

        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        if (
            polygon.ndim != 2
            or
            polygon.shape[1] != 2
            or
            len(
                polygon
            )
            <
            3
        ):
            continue

        valid_objects.append(
            (
                list_index,
                polygon,
            )
        )

    candidates = []

    for (
        list_index,
        target_polygon,
    ) in valid_objects:
        obj = objects[
            list_index
        ]

        other_polygons = [
            other_polygon
            for (
                other_index,
                other_polygon,
            )
            in valid_objects
            if (
                other_index
                !=
                list_index
            )
        ]

        grasp_result = (
            planner.plan(
                target_polygon=
                    target_polygon,
                other_polygons=
                    other_polygons,
                image_shape=
                    frame.shape,
            )
        )

        if (
            not
            grasp_result.valid
            or
            grasp_result.grasp_point
            is None
        ):
            continue

        (
            grasp_x,
            grasp_y,
        ) = (
            grasp_result.grasp_point
        )

        (
            x_mm,
            y_mm,
        ) = (
            transform.pixel_to_robot(
                grasp_x,
                grasp_y,
            )
        )

        (
            center_x,
            center_y,
        ) = polygon_center(
            target_polygon
        )

        confidence = float(
            obj.get(
                "confidence",
                0.0,
            )
        )

        clearance = float(
            grasp_result.clearance
        )

        score = (
            clearance
            *
            (
                0.5
                +
                0.5
                *
                max(
                    0.0,
                    min(
                        confidence,
                        1.0,
                    ),
                )
            )
        )

        inside_reference = (
            point_inside_polygon(
                (
                    grasp_x,
                    grasp_y,
                ),
                calibration_polygon,
            )
        )

        candidates.append(
            {
                "list_index":
                    int(
                        list_index
                    ),

                "object_index":
                    obj.get(
                        "index",
                        list_index,
                    ),

                "class_name":
                    str(
                        obj.get(
                            "class_name",
                            "",
                        )
                    ),

                "polygon":
                    target_polygon,

                "center_pixel":
                    (
                        center_x,
                        center_y,
                    ),

                "grasp_x":
                    int(
                        grasp_x
                    ),

                "grasp_y":
                    int(
                        grasp_y
                    ),

                "x_mm":
                    float(
                        x_mm
                    ),

                "y_mm":
                    float(
                        y_mm
                    ),

                "confidence":
                    confidence,

                "clearance":
                    clearance,

                "score":
                    float(
                        score
                    ),

                "inside_reference":
                    bool(
                        inside_reference
                    ),
            }
        )

    return candidates


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "找不到顶部 V3 模型：\n"
            f"{MODEL_PATH}"
        )

    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            "找不到 pixel_to_robot.json：\n"
            f"{CALIBRATION_PATH}"
        )

    transform = (
        PixelToRobotTransform.load(
            CALIBRATION_PATH
        )
    )

    calibration_polygon = (
        load_calibration_polygon(
            CALIBRATION_POINTS_PATH
        )
    )

    segmenter = TopSegmenter(
        model_path=MODEL_PATH,
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
    )

    planner = (
        CollisionAwareGraspPlanner(
            obstacle_margin=
                OBSTACLE_MARGIN,
            border_margin=
                BORDER_MARGIN,
            min_safe_area=
                MIN_SAFE_AREA,
            min_clearance=
                MIN_CLEARANCE,
        )
    )

    consensus = TopPickConsensus(
        window_size=
            CONSENSUS_WINDOW,
        required_hits=
            CONSENSUS_REQUIRED,
        max_track_center_shift_px=
            MAX_TRACK_CENTER_SHIFT_PX,
        max_grasp_spread_mm=
            MAX_GRASP_SPREAD_MM,
        reset_after_misses=
            RESET_AFTER_MISSES,
    )

    camera = TopCamera()

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 78
    )
    print(
        "TOP PICK MULTI-FRAME STABLE TEST"
    )
    print(
        "=" * 78
    )
    print(
        f"Model     : "
        f"{MODEL_PATH}"
    )
    print(
        f"Transform : "
        f"{CALIBRATION_PATH}"
    )
    print(
        f"Device    : "
        f"{DEVICE}"
    )
    print(
        f"Consensus : "
        f"{CONSENSUS_REQUIRED}/"
        f"{CONSENSUS_WINDOW}"
    )
    print(
        "Area      : "
        "整个 TopCamera 画面均允许抓取；"
        "P1~P4 仅作为参考框"
    )
    print(
        "Output    : "
        "只显示 READY 和机械 XY；"
        "不向 STM32 发送任何命令"
    )
    print(
        "=" * 78
    )
    print(
        "Q / ESC : 退出"
    )
    print(
        "R       : 手动 reset 多帧稳定器"
    )
    print(
        "S       : 保存当前画面"
    )
    print(
        "=" * 78
    )

    window = (
        "Top Pick Stable"
    )

    previous_time = (
        time.perf_counter()
    )

    smooth_fps = 0.0

    last_ready_key = None
    last_ready_print_time = 0.0

    try:
        camera.open()

        cv2.namedWindow(
            window,
            cv2.WINDOW_NORMAL,
        )

        while True:
            frame = camera.read()

            transform.check_frame(
                frame
            )

            infer_start = (
                time.perf_counter()
            )

            result = (
                segmenter.predict(
                    frame
                )
            )

            objects = (
                segmenter.get_objects(
                    result
                )
            )

            infer_ms = (
                (
                    time.perf_counter()
                    -
                    infer_start
                )
                *
                1000.0
            )

            candidates = (
                build_candidates(
                    frame=
                        frame,
                    objects=
                        objects,
                    planner=
                        planner,
                    transform=
                        transform,
                    calibration_polygon=
                        calibration_polygon,
                )
            )

            stable = (
                consensus.update(
                    candidates
                )
            )

            display = (
                frame.copy()
            )

            draw_calibration_polygon(
                display,
                calibration_polygon,
            )

            # ----------------------------------------------------
            # 画当前帧全部合法候选
            # ----------------------------------------------------
            for candidate in candidates:
                contour = (
                    np.round(
                        candidate[
                            "polygon"
                        ]
                    )
                    .astype(
                        np.int32
                    )
                    .reshape(
                        -1,
                        1,
                        2,
                    )
                )

                if candidate[
                    "inside_reference"
                ]:
                    color = (
                        0,
                        200,
                        0,
                    )

                    area_tag = (
                        "IN"
                    )

                else:
                    color = (
                        0,
                        165,
                        255,
                    )

                    area_tag = (
                        "OUT"
                    )

                cv2.drawContours(
                    display,
                    [
                        contour
                    ],
                    -1,
                    color,
                    2,
                )

                gx = candidate[
                    "grasp_x"
                ]

                gy = candidate[
                    "grasp_y"
                ]

                cv2.circle(
                    display,
                    (
                        gx,
                        gy,
                    ),
                    5,
                    color,
                    -1,
                )

                draw_text(
                    display,
                    (
                        f"{area_tag} "
                        f"XY="
                        f"({candidate['x_mm']:.1f},"
                        f"{candidate['y_mm']:.1f})"
                    ),
                    (
                        gx + 8,
                        gy - 8,
                    ),
                )

            # ----------------------------------------------------
            # 多帧结果
            # ----------------------------------------------------
            if (
                stable.ready
                and
                stable.grasp_pixel
                is not None
                and
                stable.x_mm
                is not None
                and
                stable.y_mm
                is not None
            ):
                gx, gy = (
                    stable.grasp_pixel
                )

                cv2.circle(
                    display,
                    (
                        gx,
                        gy,
                    ),
                    15,
                    (
                        0,
                        255,
                        255,
                    ),
                    3,
                )

                cv2.drawMarker(
                    display,
                    (
                        gx,
                        gy,
                    ),
                    (
                        0,
                        255,
                        255,
                    ),
                    cv2.MARKER_CROSS,
                    30,
                    2,
                )

                draw_text(
                    display,
                    (
                        "READY "
                        f"{stable.votes}/"
                        f"{stable.window_size}  "
                        f"pixel=({gx},{gy})  "
                        f"robot="
                        f"({stable.x_mm:.2f},"
                        f"{stable.y_mm:.2f})mm  "
                        f"clear="
                        f"{stable.clearance_px:.1f}px"
                    ),
                    (
                        18,
                        80,
                    ),
                    scale=0.60,
                    thickness=2,
                )

                ready_key = (
                    round(
                        stable.x_mm,
                        1,
                    ),
                    round(
                        stable.y_mm,
                        1,
                    ),
                )

                now = (
                    time.perf_counter()
                )

                if (
                    ready_key
                    !=
                    last_ready_key
                    or
                    now
                    -
                    last_ready_print_time
                    >
                    1.0
                ):
                    print(
                        "[READY] "
                        f"votes="
                        f"{stable.votes}/"
                        f"{stable.window_size} "
                        f"class="
                        f"{stable.class_name} "
                        f"pixel="
                        f"{stable.grasp_pixel} "
                        f"robot="
                        f"({stable.x_mm:.3f},"
                        f"{stable.y_mm:.3f}) mm "
                        f"conf="
                        f"{stable.confidence:.3f} "
                        f"clearance="
                        f"{stable.clearance_px:.2f}px"
                    )

                    last_ready_key = (
                        ready_key
                    )

                    last_ready_print_time = (
                        now
                    )

            else:
                draw_text(
                    display,
                    (
                        "WAIT "
                        f"{stable.votes}/"
                        f"{stable.window_size}  "
                        f"{stable.reason}"
                    ),
                    (
                        18,
                        80,
                    ),
                    scale=0.60,
                    thickness=2,
                )

            # ----------------------------------------------------
            # FPS
            # ----------------------------------------------------
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
                    1.0
                    /
                    dt
                )

                if (
                    smooth_fps
                    ==
                    0.0
                ):
                    smooth_fps = (
                        fps
                    )

                else:
                    smooth_fps = (
                        0.9
                        *
                        smooth_fps
                        +
                        0.1
                        *
                        fps
                    )

            draw_text(
                display,
                (
                    f"FPS={smooth_fps:.1f} "
                    f"Infer={infer_ms:.1f}ms "
                    f"Objects={len(objects)} "
                    f"Candidates={len(candidates)} "
                    "ALL AREA | NO STM32"
                ),
                (
                    18,
                    34,
                ),
                scale=0.54,
                thickness=2,
            )

            cv2.imshow(
                window,
                display,
            )

            key = (
                cv2.waitKey(
                    1
                )
                &
                0xFF
            )

            if key in (
                ord(
                    "q"
                ),
                ord(
                    "Q"
                ),
                27,
            ):
                break

            if key in (
                ord(
                    "r"
                ),
                ord(
                    "R"
                ),
            ):
                consensus.reset()

                print(
                    "[RESET] "
                    "TopPickConsensus reset"
                )

            if key in (
                ord(
                    "s"
                ),
                ord(
                    "S"
                ),
            ):
                timestamp = (
                    datetime.now()
                    .strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )
                )

                save_path = (
                    SAVE_DIR
                    /
                    f"top_pick_stable_"
                    f"{timestamp}.jpg"
                )

                cv2.imwrite(
                    str(
                        save_path
                    ),
                    display,
                )

                print(
                    f"[SAVE] "
                    f"{save_path}"
                )

    finally:
        safe_release_camera(
            camera
        )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
