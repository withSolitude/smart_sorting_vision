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
    sys.path.insert(0, str(ROOT))


from vision.cameras.top_camera import TopCamera
from vision.models.top_segmenter import TopSegmenter
from vision.grasp.collision_aware_grasp_planner import (
    CollisionAwareGraspPlanner,
)
from vision.calibration.pixel_to_robot import (
    PixelToRobotTransform,
)


# ================================================================
# 正式 V3 顶部实例分割模型
# ================================================================

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
    / "top_grasp_robot_xy"
)


# ================================================================
# 推理配置
# ================================================================

CONF = 0.35
IMGSZ = 832

DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# 防碰撞抓取参数
# 当前仍然是 pixel 单位
# ================================================================

OBSTACLE_MARGIN = 12
BORDER_MARGIN = 12
MIN_SAFE_AREA = 50
MIN_CLEARANCE = 3.0


# ================================================================
# 重要：
#
# 现在不再限制抓取点必须位于4个标定点围成的四边形内部。
#
# 4点 Homography 将对整个 TopCamera.read() 图像直接进行转换：
#
#     pixel (u,v)
#          ↓
#     Homography
#          ↓
#     robot (X,Y)
#
# 即使抓取点位于 P1~P4 外部，也照常计算机械坐标。
#
# 标定四边形仅用于画面参考，不参与过滤。
# ================================================================

ALLOW_OUTSIDE_CALIBRATION_POLYGON = True


def safe_release_camera(camera) -> None:
    if hasattr(camera, "release"):
        try:
            camera.release()
            return
        except Exception:
            pass

    if hasattr(camera, "close"):
        try:
            camera.close()
        except Exception:
            pass


def load_calibration_polygon(
    path: Path,
) -> np.ndarray | None:
    """
    读取 P1~P4，只用于画面显示参考。
    即使文件不存在，也不影响像素->机械坐标转换。
    """

    if not path.exists():
        print(
            "[WARN] 找不到 tray4_calibration_points.csv，"
            "将不绘制4点标定参考框。"
        )
        return None

    records: dict[
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
        reader = csv.DictReader(f)

        for row in reader:
            point_id = str(
                row["id"]
            ).strip()

            records[
                point_id
            ] = (
                float(
                    row["u"]
                ),
                float(
                    row["v"]
                ),
            )

    expected = [
        "P1",
        "P2",
        "P3",
        "P4",
    ]

    if not all(
        point_id in records
        for point_id
        in expected
    ):
        print(
            "[WARN] 标定点CSV中 P1~P4 不完整，"
            "不绘制参考框。"
        )
        return None

    return np.asarray(
        [
            records["P1"],
            records["P2"],
            records["P3"],
            records["P4"],
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
        return True

    result = cv2.pointPolygonTest(
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

    return result >= 0


def draw_calibration_polygon(
    display: np.ndarray,
    polygon: np.ndarray | None,
) -> None:
    """
    仅作为参考框显示。
    不限制目标选择。
    """

    if polygon is None:
        return

    poly_int = (
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
        display,
        [poly_int],
        True,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    for i, point in enumerate(
        polygon,
        start=1,
    ):
        x = int(
            round(
                point[0]
            )
        )

        y = int(
            round(
                point[1]
            )
        )

        cv2.putText(
            display,
            f"P{i}",
            (
                x + 8,
                y - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


def get_valid_polygons(
    objects: list[dict],
) -> list[
    tuple[
        int,
        np.ndarray,
    ]
]:
    valid = []

    for i, obj in enumerate(
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
            len(polygon) < 3
        ):
            continue

        valid.append(
            (
                i,
                polygon,
            )
        )

    return valid


def draw_text_with_background(
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
        text_w,
        text_h,
    ), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )

    cv2.rectangle(
        image,
        (
            x - 3,
            y - text_h - 5,
        ),
        (
            x + text_w + 4,
            y + baseline + 4,
        ),
        (0, 0, 0),
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
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


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

    camera = TopCamera()

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 76)
    print(
        "TOP GRASP PIXEL -> ROBOT XY TEST"
    )
    print("=" * 76)
    print(
        f"Model       : "
        f"{MODEL_PATH}"
    )
    print(
        f"Calibration : "
        f"{CALIBRATION_PATH}"
    )
    print(
        f"Device      : "
        f"{DEVICE}"
    )
    print(
        "Mode        : "
        "只计算/显示机械坐标，"
        "不向 STM32 发送运动命令"
    )
    print(
        "Area        : "
        "允许整个 TopCamera 图像区域，"
        "P1~P4 外部同样计算 XY"
    )
    print("=" * 76)
    print(
        "Q / ESC : 退出"
    )
    print(
        "S       : 保存当前调试画面"
    )
    print("=" * 76)

    window = (
        "Top Grasp -> Robot XY"
    )

    previous_time = (
        time.perf_counter()
    )

    smooth_fps = 0.0

    last_target_key = None
    last_print_time = 0.0

    try:
        camera.open()

        cv2.namedWindow(
            window,
            cv2.WINDOW_NORMAL,
        )

        while True:
            frame = camera.read()

            # ----------------------------------------------------
            # 图像尺寸必须与生成 Homography 时完全一致。
            # ----------------------------------------------------
            transform.check_frame(
                frame
            )

            start_inference = (
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

            inference_ms = (
                (
                    time.perf_counter()
                    -
                    start_inference
                )
                *
                1000.0
            )

            display = (
                frame.copy()
            )

            # 标定四边形只作为参考线显示
            draw_calibration_polygon(
                display,
                calibration_polygon,
            )

            valid_polygons = (
                get_valid_polygons(
                    objects
                )
            )

            candidates = []

            # ====================================================
            # 对每一个物体分别做防碰撞抓取规划
            # ====================================================
            for (
                object_list_index,
                target_polygon,
            ) in valid_polygons:

                obj = objects[
                    object_list_index
                ]

                other_polygons = [
                    polygon
                    for (
                        other_index,
                        polygon,
                    )
                    in valid_polygons
                    if (
                        other_index
                        !=
                        object_list_index
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

                contour = (
                    np.round(
                        target_polygon
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

                if (
                    not
                    grasp_result.valid
                    or
                    grasp_result.grasp_point
                    is None
                ):
                    cv2.drawContours(
                        display,
                        [contour],
                        -1,
                        (
                            0,
                            0,
                            255,
                        ),
                        2,
                    )

                    continue

                (
                    grasp_x,
                    grasp_y,
                ) = (
                    grasp_result.grasp_point
                )

                # ------------------------------------------------
                # 只标记该点是否在4点标定参考框内部。
                # 无论 inside/outside 都继续做坐标转换。
                # ------------------------------------------------
                inside_calibration = (
                    point_inside_polygon(
                        (
                            grasp_x,
                            grasp_y,
                        ),
                        calibration_polygon,
                    )
                )

                try:
                    (
                        x_mm,
                        y_mm,
                    ) = (
                        transform.pixel_to_robot(
                            grasp_x,
                            grasp_y,
                            apply_tcp_offset=True,
                        )
                    )

                except Exception as exc:
                    cv2.drawContours(
                        display,
                        [contour],
                        -1,
                        (
                            0,
                            0,
                            255,
                        ),
                        2,
                    )

                    print(
                        "[WARN] 坐标转换失败：",
                        repr(
                            exc
                        ),
                    )

                    continue

                confidence = float(
                    obj.get(
                        "confidence",
                        0.0,
                    )
                )

                score = (
                    float(
                        grasp_result.clearance
                    )
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

                candidates.append(
                    {
                        "obj":
                            obj,

                        "polygon":
                            target_polygon,

                        "grasp_result":
                            grasp_result,

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

                        "score":
                            float(
                                score
                            ),

                        "inside_calibration":
                            bool(
                                inside_calibration
                            ),
                    }
                )

                # ------------------------------------------------
                # 绿色 = 在4点参考框内部
                # 橙色 = 在4点参考框外部
                #
                # 两者都有效，都参与最终目标选择。
                # ------------------------------------------------
                if inside_calibration:
                    contour_color = (
                        0,
                        200,
                        0,
                    )

                    point_color = (
                        0,
                        255,
                        0,
                    )

                    area_text = (
                        "IN"
                    )

                else:
                    contour_color = (
                        0,
                        165,
                        255,
                    )

                    point_color = (
                        0,
                        165,
                        255,
                    )

                    area_text = (
                        "OUT"
                    )

                cv2.drawContours(
                    display,
                    [contour],
                    -1,
                    contour_color,
                    2,
                )

                cv2.circle(
                    display,
                    (
                        grasp_x,
                        grasp_y,
                    ),
                    6,
                    point_color,
                    -1,
                )

                text = (
                    f"{area_text} "
                    f"ID="
                    f"{obj.get('index', object_list_index)} "
                    f"XY="
                    f"({x_mm:.1f},{y_mm:.1f})"
                )

                draw_text_with_background(
                    display,
                    text,
                    (
                        grasp_x + 10,
                        grasp_y - 10,
                    ),
                )

            # ====================================================
            # 选择当前最适合抓取的目标
            #
            # inside/outside 不参与过滤。
            # 所有合法抓取点统一竞争。
            # ====================================================
            selected = None

            if candidates:
                selected = max(
                    candidates,
                    key=
                        lambda item:
                            item["score"],
                )

                grasp_x = (
                    selected[
                        "grasp_x"
                    ]
                )

                grasp_y = (
                    selected[
                        "grasp_y"
                    ]
                )

                x_mm = (
                    selected[
                        "x_mm"
                    ]
                )

                y_mm = (
                    selected[
                        "y_mm"
                    ]
                )

                obj = (
                    selected[
                        "obj"
                    ]
                )

                grasp_result = (
                    selected[
                        "grasp_result"
                    ]
                )

                inside_calibration = (
                    selected[
                        "inside_calibration"
                    ]
                )

                cv2.circle(
                    display,
                    (
                        grasp_x,
                        grasp_y,
                    ),
                    13,
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
                        grasp_x,
                        grasp_y,
                    ),
                    (
                        0,
                        255,
                        255,
                    ),
                    cv2.MARKER_CROSS,
                    26,
                    2,
                )

                area_text = (
                    "IN"
                    if inside_calibration
                    else
                    "OUT"
                )

                selected_text = (
                    f"SELECTED[{area_text}]  "
                    f"pixel=({grasp_x},{grasp_y})  "
                    f"robot="
                    f"({x_mm:.2f},{y_mm:.2f}) mm  "
                    f"clear="
                    f"{grasp_result.clearance:.1f}px"
                )

                draw_text_with_background(
                    display,
                    selected_text,
                    (
                        18,
                        78,
                    ),
                    scale=0.58,
                    thickness=2,
                )

                target_key = (
                    int(
                        obj.get(
                            "index",
                            -1,
                        )
                    ),
                    round(
                        x_mm,
                        1,
                    ),
                    round(
                        y_mm,
                        1,
                    ),
                )

                now = (
                    time.perf_counter()
                )

                if (
                    target_key
                    !=
                    last_target_key
                    or
                    now
                    -
                    last_print_time
                    >
                    1.0
                ):
                    print(
                        "[TARGET] "
                        f"ID="
                        f"{obj.get('index', '?')} "
                        f"class="
                        f"{obj.get('class_name', '?')} "
                        f"conf="
                        f"{selected['confidence']:.3f} "
                        f"area={area_text} "
                        f"pixel="
                        f"({grasp_x},{grasp_y}) "
                        f"robot="
                        f"({x_mm:.3f},"
                        f"{y_mm:.3f}) mm "
                        f"clearance="
                        f"{grasp_result.clearance:.2f}px"
                    )

                    last_target_key = (
                        target_key
                    )

                    last_print_time = (
                        now
                    )

            else:
                draw_text_with_background(
                    display,
                    "NO SAFE TARGET",
                    (
                        18,
                        78,
                    ),
                    scale=0.58,
                    thickness=2,
                )

            # ====================================================
            # FPS / 状态
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
                current_fps = (
                    1.0
                    /
                    dt
                )

                if smooth_fps == 0.0:
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

            status = (
                f"FPS={smooth_fps:.1f}  "
                f"Infer={inference_ms:.1f}ms  "
                f"Objects={len(objects)}  "
                f"Safe={len(candidates)}  "
                "ALL AREA | NO STM32 OUTPUT"
            )

            draw_text_with_background(
                display,
                status,
                (
                    18,
                    34,
                ),
                scale=0.56,
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

                path = (
                    SAVE_DIR
                    /
                    f"top_grasp_robot_xy_{timestamp}.jpg"
                )

                cv2.imwrite(
                    str(
                        path
                    ),
                    display,
                )

                print(
                    f"[SAVE] "
                    f"{path}"
                )

    finally:
        safe_release_camera(
            camera
        )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
