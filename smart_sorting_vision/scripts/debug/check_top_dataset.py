from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 工程根目录
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# YOLO数据集目录
# ============================================================

DATASET_DIR = (
    ROOT
    / "datasets"
    / "top"
    / "yolo"
)

IMAGE_DIR = (
    DATASET_DIR
    / "images"
    / "train"
)

LABEL_DIR = (
    DATASET_DIR
    / "labels"
    / "train"
)


def read_yolo_seg_label(
    label_path: Path,
    image_width: int,
    image_height: int,
):
    """
    读取YOLO实例分割标签。

    YOLO格式：

    class_id x1 y1 x2 y2 ... xn yn

    其中坐标已经归一化到0~1。

    本函数将其重新转换成原图像素坐标。
    """

    objects = []

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as file:

        lines = file.readlines()

    for line_index, line in enumerate(lines):

        line = line.strip()

        if not line:
            continue

        values = line.split()

        # 至少：
        # class + 3个点
        # 1 + 6 = 7个数字
        if len(values) < 7:

            print(
                f"[警告] {label_path.name} "
                f"第{line_index + 1}行数据不足"
            )

            continue

        # ============================================
        # 类别
        # ============================================

        class_id = int(
            float(values[0])
        )

        # ============================================
        # 后面全部是polygon坐标
        # ============================================

        coordinates = [
            float(value)
            for value in values[1:]
        ]

        # x y必须成对
        if len(coordinates) % 2 != 0:

            print(
                f"[警告] {label_path.name} "
                f"第{line_index + 1}行坐标数量错误"
            )

            continue

        polygon = []

        for i in range(
            0,
            len(coordinates),
            2,
        ):

            x_normalized = coordinates[i]
            y_normalized = coordinates[i + 1]

            # ========================================
            # 归一化坐标 → 像素坐标
            # ========================================

            x = int(
                round(
                    x_normalized
                    * image_width
                )
            )

            y = int(
                round(
                    y_normalized
                    * image_height
                )
            )

            # 防止坐标越界
            x = max(
                0,
                min(
                    x,
                    image_width - 1,
                ),
            )

            y = max(
                0,
                min(
                    y,
                    image_height - 1,
                ),
            )

            polygon.append(
                [x, y]
            )

        polygon = np.array(
            polygon,
            dtype=np.int32,
        )

        objects.append(
            {
                "class_id": class_id,
                "polygon": polygon,
            }
        )

    return objects


def main():

    print(
        "=================================="
    )

    print(
        "顶部YOLO-Seg数据集检查程序"
    )

    print(
        "=================================="
    )

    # ========================================================
    # 找所有训练图片
    # ========================================================

    image_files = sorted(
        list(
            IMAGE_DIR.glob("*.jpg")
        )
        + list(
            IMAGE_DIR.glob("*.jpeg")
        )
        + list(
            IMAGE_DIR.glob("*.png")
        )
    )

    if len(image_files) == 0:

        raise RuntimeError(
            f"没有找到训练图片：{IMAGE_DIR}"
        )

    print(
        f"训练图片数量："
        f"{len(image_files)}"
    )

    print()
    print("操作方式：")
    print("N / 空格：下一张")
    print("P：上一张")
    print("Q：退出")
    print()

    image_index = 0

    # ========================================================
    # 主循环
    # ========================================================

    while True:

        image_path = (
            image_files[
                image_index
            ]
        )

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        # ====================================================
        # 读取图片
        # ====================================================

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"[错误] 无法读取："
                f"{image_path}"
            )

            image_index += 1

            if image_index >= len(
                image_files
            ):
                image_index = 0

            continue

        image_height, image_width = (
            image.shape[:2]
        )

        display = image.copy()

        # ====================================================
        # 检查标签
        # ====================================================

        if not label_path.exists():

            print(
                f"[警告] 没有标签："
                f"{label_path.name}"
            )

            objects = []

        else:

            objects = (
                read_yolo_seg_label(
                    label_path=label_path,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        # ====================================================
        # 绘制每个实例
        # ====================================================

        for object_index, obj in enumerate(
            objects
        ):

            polygon = obj[
                "polygon"
            ]

            # --------------------------------------------
            # 画polygon轮廓
            # --------------------------------------------

            contour = polygon.reshape(
                -1,
                1,
                2,
            )

            cv2.drawContours(
                display,
                [contour],
                -1,
                (0, 255, 0),
                2,
            )

            # --------------------------------------------
            # 给polygon加半透明填充
            # --------------------------------------------

            overlay = display.copy()

            cv2.fillPoly(
                overlay,
                [polygon],
                (0, 255, 0),
            )

            display = cv2.addWeighted(
                overlay,
                0.20,
                display,
                0.80,
                0,
            )

            # --------------------------------------------
            # 计算大致显示位置
            # --------------------------------------------

            moments = cv2.moments(
                contour
            )

            if abs(
                moments["m00"]
            ) > 1e-6:

                center_x = int(
                    moments["m10"]
                    / moments["m00"]
                )

                center_y = int(
                    moments["m01"]
                    / moments["m00"]
                )

            else:

                center_x = int(
                    polygon[:, 0].mean()
                )

                center_y = int(
                    polygon[:, 1].mean()
                )

            # --------------------------------------------
            # 显示实例编号
            # --------------------------------------------

            text = (
                f"ID={object_index} "
                f"class={obj['class_id']}"
            )

            cv2.putText(
                display,
                text,
                (
                    center_x,
                    center_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # ====================================================
        # 显示图片信息
        # ====================================================

        info_text = (
            f"{image_index + 1}"
            f"/"
            f"{len(image_files)} "
            f"{image_path.name} "
            f"objects={len(objects)}"
        )

        cv2.putText(
            display,
            info_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ====================================================
        # 显示
        # ====================================================

        cv2.imshow(
            "YOLO Seg Dataset Check",
            display,
        )

        print(
            f"[检查] "
            f"{image_path.name} "
            f"实例数量={len(objects)}"
        )

        key = (
            cv2.waitKey(0)
            & 0xFF
        )

        # ====================================================
        # Q退出
        # ====================================================

        if key == ord("q"):

            break

        # ====================================================
        # P上一张
        # ====================================================

        elif key == ord("p"):

            image_index -= 1

            if image_index < 0:
                image_index = (
                    len(image_files)
                    - 1
                )

        # ====================================================
        # 其他按键默认下一张
        # ====================================================

        else:

            image_index += 1

            if image_index >= len(
                image_files
            ):

                image_index = 0

    cv2.destroyAllWindows()

    print(
        "数据集检查结束"
    )


if __name__ == "__main__":
    main()