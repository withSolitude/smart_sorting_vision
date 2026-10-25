from pathlib import Path
import math
import time

import cv2
import numpy as np

from vision.cameras.top_camera import TopCamera
from vision.models.top_segmenter import TopSegmenter

from vision.grasp.mask_processor import MaskProcessor

from vision.grasp.collision_aware_grasp_planner import (
    CollisionAwareGraspPlanner,
)


# ============================================================
# 工程根目录
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# 模型
#
# 如果你当前真正使用的是自己训练的 best.pt，
# 后面只改这里。
# ============================================================

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


# ============================================================
# 推理配置
# ============================================================

CONF = 0.25

IMGSZ = 640


# ============================================================
# Jetson GPU
#
# 正常Jetson用：
#     "0"
#
# 如果CUDA环境暂时有问题：
#     改成 "cpu"
# ============================================================

DEVICE = "0"


# ============================================================
# 防碰撞抓取参数
#
# 当前全部还是像素单位。
# ============================================================

OBSTACLE_MARGIN = 12

BORDER_MARGIN = 12

MIN_SAFE_AREA = 50

MIN_CLEARANCE = 3.0


# ============================================================
# 显示安全区域
# ============================================================

SHOW_SAFE_AREA = True


# ============================================================
# 保存目录
# ============================================================

SAVE_DIR = (
    ROOT
    / "outputs"
    / "debug"
    / "top_grasp_live"
)


# ============================================================
# Safe Mask 半透明显示
# ============================================================

def draw_safe_mask(
    image: np.ndarray,
    safe_mask: np.ndarray | None,
) -> None:

    if safe_mask is None:
        return

    if not np.any(
        safe_mask
    ):
        return

    # --------------------------------------------------------
    # 为了避免整张画面反复addWeighted，
    # 这里只绘制safe mask轮廓。
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        safe_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        image,
        contours,
        -1,
        (
            80,
            180,
            80,
        ),
        1,
    )


# ============================================================
# 绘制单个物体
# ============================================================

def draw_object(
    display: np.ndarray,
    obj: dict,
    polygon: np.ndarray,
    geometry: dict,
    grasp_result,
) -> None:

    # ========================================================
    # Polygon
    # ========================================================

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


    # --------------------------------------------------------
    # 有效目标绿色
    # 无效目标橙色
    # --------------------------------------------------------

    if grasp_result.valid:

        object_color = (
            0,
            255,
            0,
        )

    else:

        object_color = (
            0,
            165,
            255,
        )


    cv2.drawContours(
        display,
        [
            contour
        ],
        -1,
        object_color,
        2,
    )


    # ========================================================
    # 几何中心
    #
    # 红点
    # ========================================================

    center_x, center_y = (
        geometry[
            "center"
        ]
    )


    cx = int(
        round(
            center_x
        )
    )

    cy = int(
        round(
            center_y
        )
    )


    cv2.circle(
        display,
        (
            cx,
            cy,
        ),
        5,
        (
            0,
            0,
            255,
        ),
        -1,
    )


    # ========================================================
    # 物体方向
    #
    # 黄色线
    # ========================================================

    angle = float(
        geometry[
            "angle"
        ]
    )


    angle_rad = math.radians(
        angle
    )


    direction_length = 40


    dx = int(
        direction_length
        *
        math.cos(
            angle_rad
        )
    )


    dy = int(
        direction_length
        *
        math.sin(
            angle_rad
        )
    )


    cv2.line(
        display,
        (
            cx - dx,
            cy - dy,
        ),
        (
            cx + dx,
            cy + dy,
        ),
        (
            0,
            255,
            255,
        ),
        2,
    )


    # ========================================================
    # Safe area
    # ========================================================

    if SHOW_SAFE_AREA:

        draw_safe_mask(
            display,
            grasp_result.safe_mask,
        )


    # ========================================================
    # 没有抓取点
    # ========================================================

    if (
        grasp_result.grasp_point
        is None
    ):

        text = (
            f"ID={obj['index']} "
            f"INVALID "
            f"{grasp_result.reason}"
        )


        cv2.putText(
            display,
            text,
            (
                cx + 10,
                cy - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                0,
                0,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        return


    # ========================================================
    # 抓取点
    # ========================================================

    grasp_x, grasp_y = (
        grasp_result.grasp_point
    )


    # --------------------------------------------------------
    # 有效：
    # 蓝色抓取点
    #
    # 无效：
    # 橙色
    # --------------------------------------------------------

    if grasp_result.valid:

        grasp_color = (
            255,
            0,
            0,
        )

    else:

        grasp_color = (
            0,
            165,
            255,
        )


    cv2.circle(
        display,
        (
            grasp_x,
            grasp_y,
        ),
        7,
        grasp_color,
        -1,
    )


    # ========================================================
    # 安全半径
    # ========================================================

    radius = int(
        round(
            grasp_result.clearance
        )
    )


    if radius > 0:

        cv2.circle(
            display,
            (
                grasp_x,
                grasp_y,
            ),
            radius,
            grasp_color,
            2,
        )


    # ========================================================
    # 中心到抓取点
    # ========================================================

    cv2.line(
        display,
        (
            cx,
            cy,
        ),
        (
            grasp_x,
            grasp_y,
        ),
        (
            255,
            255,
            0,
        ),
        1,
    )


    # ========================================================
    # 信息文字
    # ========================================================

    x1 = int(
        obj[
            "bbox"
        ][0]
    )

    y1 = int(
        obj[
            "bbox"
        ][1]
    )


    text_y = max(
        20,
        y1 - 8,
    )


    text = (
        f"ID={obj['index']} "
        f"Conf={obj['confidence']:.2f} "
        f"G=({grasp_x},{grasp_y}) "
        f"C={grasp_result.clearance:.1f}"
    )


    cv2.putText(
        display,
        text,
        (
            x1,
            text_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        object_color,
        1,
        cv2.LINE_AA,
    )


# ============================================================
# 主函数
# ============================================================

def main():

    # ========================================================
    # 模型检查
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\n顶部实例分割模型不存在："
            f"\n{MODEL_PATH}"
            "\n"
            "\n请修改 MODEL_PATH。"
        )


    # ========================================================
    # 创建顶部相机
    #
    # TopCamera参数已经统一写在top_camera.py里，
    # 这里不要再次设置曝光、分辨率等参数。
    # ========================================================

    camera = TopCamera()


    # ========================================================
    # YOLO实例分割
    # ========================================================

    segmenter = TopSegmenter(
        model_path=
            MODEL_PATH,

        conf=
            CONF,

        imgsz=
            IMGSZ,

        device=
            DEVICE,
    )


    # ========================================================
    # Mask几何信息
    # ========================================================

    mask_processor = MaskProcessor(
        edge_margin=
            BORDER_MARGIN,
    )


    # ========================================================
    # 多物体防碰撞抓取规划
    # ========================================================

    grasp_planner = (
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


    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # 打开相机
    # ========================================================

    camera.open()


    print()
    print(
        "=" * 72
    )

    print(
        "TOP CAMERA REALTIME COLLISION-AWARE GRASP TEST"
    )

    print(
        "=" * 72
    )

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print()

    print(
        "绿色轮廓 : 有效物体"
    )

    print(
        "红点     : Mask几何中心"
    )

    print(
        "黄色线   : 物体方向"
    )

    print(
        "蓝点     : 安全抓取点"
    )

    print(
        "蓝圈     : 抓取点安全半径"
    )

    print()

    print(
        "Q / ESC : 退出"
    )

    print(
        "S       : 保存当前结果"
    )

    print(
        "P       : 打印当前抓取点"
    )

    print(
        "=" * 72
    )


    # ========================================================
    # FPS
    # ========================================================

    previous_time = (
        time.perf_counter()
    )

    smooth_fps = 0.0


    save_index = 0


    # 保存上一帧结果，供P键打印
    latest_grasp_results = []


    try:

        while True:

            # ====================================================
            # 1. 读取顶部相机
            #
            # read()输出：
            #
            # 去畸变
            # +
            # SAFE ROI
            #
            # 当前应约：
            #
            # width  = 870
            # height = 840
            # ====================================================

            frame = (
                camera.read()
            )


            # ====================================================
            # 2. YOLO实例分割
            # ====================================================

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


            # ====================================================
            # 3. 过滤有效Mask
            # ====================================================

            valid_objects = []


            for obj in objects:

                polygon = (
                    obj.get(
                        "polygon"
                    )
                )


                if polygon is None:

                    continue


                polygon = np.asarray(
                    polygon,
                    dtype=np.float32,
                )


                if (
                    polygon.ndim
                    != 2
                ):

                    continue


                if (
                    polygon.shape[1]
                    != 2
                ):

                    continue


                if (
                    len(
                        polygon
                    )
                    <
                    3
                ):

                    continue


                valid_objects.append(
                    obj
                )


            # ====================================================
            # 4. 显示图
            # ====================================================

            display = (
                frame.copy()
            )


            latest_grasp_results = []


            valid_grasp_count = 0


            # ====================================================
            # 5. 每一个物体分别规划抓取点
            #
            # 当前目标之外的所有Mask：
            #
            #     都作为障碍物
            # ====================================================

            for (
                target_index,
                obj,
            ) in enumerate(
                valid_objects
            ):

                target_polygon = (
                    np.asarray(
                        obj[
                            "polygon"
                        ],
                        dtype=np.float32,
                    )
                )


                # ================================================
                # 其他物体
                # ================================================

                other_polygons = []


                for (
                    other_index,
                    other_obj,
                ) in enumerate(
                    valid_objects
                ):

                    if (
                        other_index
                        ==
                        target_index
                    ):

                        continue


                    other_polygon = (
                        other_obj.get(
                            "polygon"
                        )
                    )


                    if (
                        other_polygon
                        is None
                    ):

                        continue


                    other_polygons.append(
                        np.asarray(
                            other_polygon,
                            dtype=np.float32,
                        )
                    )


                # ================================================
                # 几何信息
                # ================================================

                try:

                    geometry = (
                        mask_processor.analyze(
                            polygon=
                                target_polygon,

                            image_shape=
                                frame.shape,
                        )
                    )

                except ValueError:

                    continue


                # ================================================
                # 防碰撞抓取规划
                # ================================================

                grasp_result = (
                    grasp_planner.plan(
                        target_polygon=
                            target_polygon,

                        other_polygons=
                            other_polygons,

                        image_shape=
                            frame.shape,
                    )
                )


                if grasp_result.valid:

                    valid_grasp_count += 1


                # ================================================
                # 保存当前数据
                # ================================================

                latest_grasp_results.append(
                    {
                        "id":
                            obj[
                                "index"
                            ],

                        "class_name":
                            obj[
                                "class_name"
                            ],

                        "confidence":
                            obj[
                                "confidence"
                            ],

                        "center":
                            geometry[
                                "center"
                            ],

                        "angle":
                            geometry[
                                "angle"
                            ],

                        "valid":
                            grasp_result.valid,

                        "grasp_point":
                            grasp_result.grasp_point,

                        "clearance":
                            grasp_result.clearance,

                        "self_clearance":
                            grasp_result.self_clearance,

                        "obstacle_clearance":
                            grasp_result.obstacle_clearance,

                        "safe_area":
                            grasp_result.safe_area,

                        "reason":
                            grasp_result.reason,
                    }
                )


                # ================================================
                # 绘制
                # ================================================

                draw_object(
                    display=
                        display,

                    obj=
                        obj,

                    polygon=
                        target_polygon,

                    geometry=
                        geometry,

                    grasp_result=
                        grasp_result,
                )


            # ====================================================
            # 6. FPS
            # ====================================================

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


            # ====================================================
            # 7. 左上角状态
            # ====================================================

            status_1 = (
                f"FPS: "
                f"{smooth_fps:.1f}"
            )


            status_2 = (
                f"Objects: "
                f"{len(valid_objects)}  "
                f"Valid grasp: "
                f"{valid_grasp_count}"
            )


            cv2.putText(
                display,
                status_1,
                (
                    20,
                    30,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )


            cv2.putText(
                display,
                status_2,
                (
                    20,
                    60,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )


            # ====================================================
            # 8. 显示
            # ====================================================

            cv2.imshow(
                "Top Camera - Collision Grasp",
                display,
            )


            # ====================================================
            # 9. 键盘
            # ====================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )


            # ----------------------------------------------------
            # Q / ESC退出
            # ----------------------------------------------------

            if (
                key
                ==
                ord(
                    "q"
                )
                or
                key
                ==
                27
            ):

                break


            # ----------------------------------------------------
            # S保存
            # ----------------------------------------------------

            if (
                key
                ==
                ord(
                    "s"
                )
            ):

                save_path = (
                    SAVE_DIR
                    /
                    (
                        f"grasp_"
                        f"{save_index:04d}"
                        f".jpg"
                    )
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


                save_index += 1


            # ----------------------------------------------------
            # P打印抓取点
            # ----------------------------------------------------

            if (
                key
                ==
                ord(
                    "p"
                )
            ):

                print()
                print(
                    "=" * 72
                )

                print(
                    "CURRENT GRASP RESULTS"
                )

                print(
                    "=" * 72
                )


                if not latest_grasp_results:

                    print(
                        "当前没有有效Mask"
                    )


                for item in (
                    latest_grasp_results
                ):

                    print()

                    print(
                        f"ID: "
                        f"{item['id']}"
                    )

                    print(
                        f"Class: "
                        f"{item['class_name']}"
                    )

                    print(
                        f"Confidence: "
                        f"{item['confidence']:.3f}"
                    )

                    print(
                        f"Center: "
                        f"{item['center']}"
                    )

                    print(
                        f"Angle: "
                        f"{item['angle']:.2f} deg"
                    )

                    print(
                        f"Valid: "
                        f"{item['valid']}"
                    )

                    print(
                        f"Grasp Point: "
                        f"{item['grasp_point']}"
                    )

                    print(
                        f"Clearance: "
                        f"{item['clearance']:.2f} px"
                    )

                    print(
                        f"Self Clearance: "
                        f"{item['self_clearance']:.2f} px"
                    )

                    print(
                        f"Obstacle Clearance: "
                        f"{item['obstacle_clearance']:.2f} px"
                    )

                    print(
                        f"Safe Area: "
                        f"{item['safe_area']}"
                    )

                    print(
                        f"Reason: "
                        f"{item['reason']}"
                    )


                print(
                    "=" * 72
                )


    finally:

        # ========================================================
        # 关闭顶部相机
        # ========================================================

        camera.release()


        cv2.destroyAllWindows()


    print()
    print(
        "顶部相机实时抓取点测试结束"
    )


if __name__ == "__main__":

    main()