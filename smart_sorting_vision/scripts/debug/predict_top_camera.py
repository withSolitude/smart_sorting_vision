import time
import math
from pathlib import Path

import cv2

from vision.cameras.top_camera import TopCamera
from vision.models.top_segmenter import TopSegmenter
from vision.grasp.mask_processor import MaskProcessor
from vision.grasp.grasp_planner import GraspPlanner


# ============================================================
# 工程根目录
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 模型路径
# ============================================================

MODEL_PATH = (
    ROOT
    / "weights"
    / "top"
    / "best_model.pt"
)


# ============================================================
# YOLO参数
# ============================================================

CONF = 0.10
IMGSZ = 640
DEVICE = "cpu"


def main():

    print(
        "=================================="
    )
    print(
        "顶部视觉实时抓取点测试"
    )
    print(
        "=================================="
    )

    # ========================================================
    # 1. 检查模型
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"找不到模型：\n"
            f"{MODEL_PATH}"
        )

    print(
        f"模型：{MODEL_PATH}"
    )

    # ========================================================
    # 2. 加载YOLO实例分割模型
    # ========================================================

    segmenter = TopSegmenter(
        model_path=MODEL_PATH,
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
    )

    # ========================================================
    # 3. Mask几何处理器
    # ========================================================

    mask_processor = MaskProcessor(
        edge_margin=10
    )

    # ========================================================
    # 4. 抓取规划器
    # ========================================================

    grasp_planner = GraspPlanner()

    # ========================================================
    # 5. 顶部相机
    # ========================================================

    camera = TopCamera(
        device="/dev/video0",
        width=1280,
        height=720,
        fps=30,
    )

    camera.open()

    print()
    print(
        "绿色线：物体Mask轮廓"
    )
    print(
        "红点：物体几何中心"
    )
    print(
        "蓝点：安全抓取点"
    )
    print(
        "蓝圈：抓取安全半径"
    )
    print(
        "Q：退出"
    )
    print()

    # ========================================================
    # FPS统计
    # ========================================================

    previous_time = time.perf_counter()

    smooth_fps = 0.0

    try:

        while True:

            # =================================================
            # 1. 读取实时画面
            # =================================================

            frame = camera.read()

            # =================================================
            # 2. YOLO实例分割
            # =================================================

            result = segmenter.predict(
                frame
            )

            # =================================================
            # 3. 获取所有物体
            # =================================================

            objects = (
                segmenter.get_objects(
                    result
                )
            )

            # =================================================
            # 4. 不使用result.plot()
            #
            # 从原始图像开始自己绘制
            # =================================================

            display = frame.copy()

            valid_object_count = 0

            # =================================================
            # 5. 遍历每个实例
            # =================================================

            for obj in objects:

                polygon = obj[
                    "polygon"
                ]

                # 没有Mask就不能计算抓取点
                if polygon is None:
                    continue

                if len(polygon) < 3:
                    continue

                valid_object_count += 1

                # =============================================
                # 6. Mask几何分析
                # =============================================

                geometry = (
                    mask_processor.analyze(
                        polygon=polygon,
                        image_shape=frame.shape,
                    )
                )

                # =============================================
                # 7. 计算安全抓取点
                # =============================================

                grasp_result = (
                    grasp_planner.find_safe_grasp_point(
                        polygon=polygon,
                        image_shape=frame.shape,
                    )
                )

                # =============================================
                # 8. 获取结果
                # =============================================

                center_x, center_y = (
                    geometry["center"]
                )

                angle = geometry[
                    "angle"
                ]

                grasp_x, grasp_y = (
                    grasp_result[
                        "grasp_point"
                    ]
                )

                clearance = (
                    grasp_result[
                        "clearance"
                    ]
                )

                # =============================================
                # 9. YOLO Polygon → OpenCV轮廓
                # =============================================

                contour = (
                    polygon
                    .astype("int32")
                    .reshape(
                        -1,
                        1,
                        2,
                    )
                )

                # =============================================
                # 10. 画Mask边缘
                #
                # 这就是你想要的：
                # 沿着物体轮廓画线
                # =============================================

                cv2.drawContours(
                    display,
                    [contour],
                    -1,
                    (0, 255, 0),
                    3,
                )

                # =============================================
                # 11. 几何中心
                # =============================================

                cx = int(
                    center_x
                )

                cy = int(
                    center_y
                )

                cv2.circle(
                    display,
                    (cx, cy),
                    6,
                    (0, 0, 255),
                    -1,
                )

                # =============================================
                # 12. 安全抓取点
                # =============================================

                cv2.circle(
                    display,
                    (
                        grasp_x,
                        grasp_y,
                    ),
                    8,
                    (255, 0, 0),
                    -1,
                )

                # =============================================
                # 13. 安全抓取范围
                # =============================================

                cv2.circle(
                    display,
                    (
                        grasp_x,
                        grasp_y,
                    ),
                    max(
                        1,
                        int(clearance),
                    ),
                    (255, 0, 0),
                    2,
                )

                # =============================================
                # 14. 几何中心 → 抓取点
                # =============================================

                cv2.line(
                    display,
                    (cx, cy),
                    (
                        grasp_x,
                        grasp_y,
                    ),
                    (255, 255, 0),
                    2,
                )

                # =============================================
                # 15. 画物体方向
                #
                # 根据angle画一条方向线
                # =============================================

                direction_length = 60

                angle_rad = (
                    math.radians(
                        angle
                    )
                )

                dx = int(
                    direction_length
                    * math.cos(
                        angle_rad
                    )
                )

                dy = int(
                    direction_length
                    * math.sin(
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
                    (0, 255, 255),
                    2,
                )

                # =============================================
                # 16. 显示目标信息
                # =============================================

                text = (
                    f"ID={obj['index']} "
                    f"Conf={obj['confidence']:.2f} "
                    f"A={angle:.1f}"
                )

                # 用bbox左上角作为文字位置
                x1 = int(
                    obj["bbox"][0]
                )

                y1 = int(
                    obj["bbox"][1]
                )

                text_y = max(
                    25,
                    y1 - 10,
                )

                cv2.putText(
                    display,
                    text,
                    (
                        x1,
                        text_y,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # =================================================
            # 17. FPS
            # =================================================

            current_time = (
                time.perf_counter()
            )

            elapsed = (
                current_time
                - previous_time
            )

            previous_time = (
                current_time
            )

            if elapsed > 0:

                current_fps = (
                    1.0 / elapsed
                )

                if smooth_fps == 0:

                    smooth_fps = (
                        current_fps
                    )

                else:

                    smooth_fps = (
                        0.9
                        * smooth_fps
                        + 0.1
                        * current_fps
                    )

            # =================================================
            # 18. 左上角状态
            # =================================================

            status = (
                f"FPS: {smooth_fps:.1f}  "
                f"Objects: {valid_object_count}"
            )

            cv2.putText(
                display,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # =================================================
            # 19. 实时显示
            # =================================================

            cv2.imshow(
                "Top Vision - Grasp",
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

    print(
        "顶部视觉程序结束"
    )


if __name__ == "__main__":
    main()