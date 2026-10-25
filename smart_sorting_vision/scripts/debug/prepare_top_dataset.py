import json
import random
import shutil
from pathlib import Path

import cv2


# ============================================================
# 工程路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT
    / "datasets"
    / "top"
    / "raw"
)

YOLO_DIR = (
    ROOT
    / "datasets"
    / "top"
    / "yolo"
)


# ============================================================
# 数据集划分参数
# ============================================================

TRAIN_RATIO = 0.8

# 固定随机种子
# 保证每次运行时train/val划分一致
RANDOM_SEED = 42


# ============================================================
# 当前顶部模型只有一个类别
# ============================================================

CLASS_NAMES = {
    "object": 0,
}


def find_image(json_path: Path):
    """
    根据Labelme JSON找到对应图片。

    支持：
        .jpg
        .jpeg
        .png
    """

    for suffix in [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]:

        image_path = (
            json_path.parent
            / f"{json_path.stem}{suffix}"
        )

        if image_path.exists():
            return image_path

    return None


def load_image_size(
    json_data,
    image_path: Path,
):
    """
    获取图片宽高。

    优先使用Labelme JSON中的：
        imageWidth
        imageHeight

    如果JSON中没有，
    再使用OpenCV读取图片。
    """

    image_width = json_data.get(
        "imageWidth"
    )

    image_height = json_data.get(
        "imageHeight"
    )

    if (
        image_width is not None
        and image_height is not None
    ):

        return (
            int(image_width),
            int(image_height),
        )

    # JSON没有宽高时读取原图片
    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"无法读取图片：{image_path}"
        )

    image_height, image_width = (
        image.shape[:2]
    )

    return (
        image_width,
        image_height,
    )


def convert_polygon_to_yolo(
    points,
    image_width,
    image_height,
):
    """
    Labelme Polygon坐标
    转成YOLO Seg归一化坐标。

    Labelme:
        x, y 单位是像素

    YOLO:
        x / image_width
        y / image_height

    最终所有坐标位于 [0, 1]
    """

    if len(points) < 3:
        return None

    normalized_points = []

    for point in points:

        if len(point) < 2:
            continue

        x = float(point[0])
        y = float(point[1])

        # 防止标注点稍微超过图片边界
        x = max(
            0.0,
            min(
                x,
                image_width - 1,
            ),
        )

        y = max(
            0.0,
            min(
                y,
                image_height - 1,
            ),
        )

        x_normalized = (
            x / image_width
        )

        y_normalized = (
            y / image_height
        )

        normalized_points.extend(
            [
                x_normalized,
                y_normalized,
            ]
        )

    # 至少3个点
    if len(normalized_points) < 6:
        return None

    return normalized_points


def convert_labelme_json(
    json_path: Path,
):
    """
    将一个Labelme JSON转换成YOLO Seg标签。

    Returns:
        image_path
        yolo_lines
    """

    # --------------------------------------------------------
    # 1. 读取JSON
    # --------------------------------------------------------

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:

        json_data = json.load(
            file
        )

    # --------------------------------------------------------
    # 2. 找对应图片
    # --------------------------------------------------------

    image_path = find_image(
        json_path
    )

    if image_path is None:

        print(
            f"[跳过] 找不到对应图片："
            f"{json_path.name}"
        )

        return None, []

    # --------------------------------------------------------
    # 3. 获取图片尺寸
    # --------------------------------------------------------

    image_width, image_height = (
        load_image_size(
            json_data=json_data,
            image_path=image_path,
        )
    )

    # --------------------------------------------------------
    # 4. 读取所有Shapes
    # --------------------------------------------------------

    shapes = json_data.get(
        "shapes",
        []
    )

    yolo_lines = []

    for shape_index, shape in enumerate(
        shapes
    ):

        label = shape.get(
            "label"
        )

        shape_type = shape.get(
            "shape_type",
            "polygon",
        )

        points = shape.get(
            "points",
            []
        )

        # ----------------------------------------------------
        # 检查类别
        # ----------------------------------------------------

        if label not in CLASS_NAMES:

            print(
                f"[警告] "
                f"{json_path.name} "
                f"发现未知标签：{label}，跳过"
            )

            continue

        # ----------------------------------------------------
        # 必须是Polygon
        # ----------------------------------------------------

        if shape_type != "polygon":

            print(
                f"[警告] "
                f"{json_path.name} "
                f"第{shape_index}个标注不是polygon，"
                f"类型={shape_type}，跳过"
            )

            continue

        # ----------------------------------------------------
        # Polygon → YOLO坐标
        # ----------------------------------------------------

        normalized_points = (
            convert_polygon_to_yolo(
                points=points,
                image_width=image_width,
                image_height=image_height,
            )
        )

        if normalized_points is None:

            print(
                f"[警告] "
                f"{json_path.name} "
                f"存在无效Polygon，跳过"
            )

            continue

        # ----------------------------------------------------
        # 类别ID
        # ----------------------------------------------------

        class_id = (
            CLASS_NAMES[label]
        )

        # ----------------------------------------------------
        # 生成YOLO一行标签
        #
        # 格式：
        #
        # class x1 y1 x2 y2 ...
        # ----------------------------------------------------

        coordinate_text = " ".join(
            f"{value:.6f}"
            for value
            in normalized_points
        )

        line = (
            f"{class_id} "
            f"{coordinate_text}"
        )

        yolo_lines.append(
            line
        )

    return (
        image_path,
        yolo_lines,
    )


def prepare_directories():
    """
    创建干净的YOLO数据集目录。

    每次重新生成数据集时，
    删除旧的yolo目录，
    防止残留旧图片和旧标签。
    """

    if YOLO_DIR.exists():

        print(
            f"[清理] 删除旧数据集："
            f"{YOLO_DIR}"
        )

        shutil.rmtree(
            YOLO_DIR
        )

    directories = [

        YOLO_DIR
        / "images"
        / "train",

        YOLO_DIR
        / "images"
        / "val",

        YOLO_DIR
        / "labels"
        / "train",

        YOLO_DIR
        / "labels"
        / "val",
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_sample(
    image_path: Path,
    yolo_lines,
    split_name: str,
):
    """
    将图片和YOLO标签保存到
    train或者val目录。
    """

    # --------------------------------------------------------
    # 图片目标路径
    # --------------------------------------------------------

    image_output = (
        YOLO_DIR
        / "images"
        / split_name
        / image_path.name
    )

    # --------------------------------------------------------
    # 标签目标路径
    # --------------------------------------------------------

    label_output = (
        YOLO_DIR
        / "labels"
        / split_name
        / f"{image_path.stem}.txt"
    )

    # --------------------------------------------------------
    # 复制图片
    # --------------------------------------------------------

    shutil.copy2(
        image_path,
        image_output,
    )

    # --------------------------------------------------------
    # 写YOLO标签
    # --------------------------------------------------------

    with open(
        label_output,
        "w",
        encoding="utf-8",
    ) as file:

        for line in yolo_lines:

            file.write(
                line + "\n"
            )


def create_yaml():
    """
    创建Ultralytics训练所需top.yaml。
    """

    yaml_path = (
        YOLO_DIR
        / "top.yaml"
    )

    yaml_text = f"""path: {YOLO_DIR}

train: images/train
val: images/val

names:
  0: object
"""

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            yaml_text
        )

    print(
        f"[生成] {yaml_path}"
    )


def main():

    print(
        "=================================="
    )

    print(
        "Labelme → YOLO26-Seg 数据集转换"
    )

    print(
        "=================================="
    )

    print(
        f"Labelme目录：{RAW_DIR}"
    )

    print(
        f"YOLO目录：{YOLO_DIR}"
    )

    # ========================================================
    # 1. 检查raw目录
    # ========================================================

    if not RAW_DIR.exists():

        raise RuntimeError(
            f"数据目录不存在：{RAW_DIR}"
        )

    # ========================================================
    # 2. 找到所有JSON
    # ========================================================

    json_files = sorted(
        RAW_DIR.glob(
            "*.json"
        )
    )

    print(
        f"找到Labelme标注："
        f"{len(json_files)} 个"
    )

    if len(json_files) == 0:

        raise RuntimeError(
            "没有找到Labelme JSON标注文件"
        )

    # ========================================================
    # 3. 转换所有JSON
    # ========================================================

    samples = []

    total_objects = 0

    for json_path in json_files:

        image_path, yolo_lines = (
            convert_labelme_json(
                json_path
            )
        )

        if image_path is None:
            continue

        # 当前阶段：
        # 没有有效Polygon的图片先不加入训练集

        if len(yolo_lines) == 0:

            print(
                f"[跳过] "
                f"{json_path.name} "
                f"没有有效object Polygon"
            )

            continue

        samples.append(
            (
                image_path,
                yolo_lines,
            )
        )

        total_objects += len(
            yolo_lines
        )

        print(
            f"[转换成功] "
            f"{json_path.name} "
            f"→ {len(yolo_lines)} 个实例"
        )

    # ========================================================
    # 4. 检查有效样本
    # ========================================================

    if len(samples) < 2:

        raise RuntimeError(
            "有效标注图片不足2张，"
            "无法划分train/val"
        )

    # ========================================================
    # 5. 打乱
    # ========================================================

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        samples
    )

    # ========================================================
    # 6. 计算train数量
    # ========================================================

    train_count = int(
        len(samples)
        * TRAIN_RATIO
    )

    # 防止val为空
    train_count = max(
        1,
        min(
            train_count,
            len(samples) - 1,
        ),
    )

    train_samples = (
        samples[:train_count]
    )

    val_samples = (
        samples[train_count:]
    )

    # ========================================================
    # 7. 创建数据集目录
    # ========================================================

    prepare_directories()

    # ========================================================
    # 8. 保存train
    # ========================================================

    for image_path, yolo_lines in train_samples:

        save_sample(
            image_path=image_path,
            yolo_lines=yolo_lines,
            split_name="train",
        )

    # ========================================================
    # 9. 保存val
    # ========================================================

    for image_path, yolo_lines in val_samples:

        save_sample(
            image_path=image_path,
            yolo_lines=yolo_lines,
            split_name="val",
        )

    # ========================================================
    # 10. 创建top.yaml
    # ========================================================

    create_yaml()

    # ========================================================
    # 11. 输出统计信息
    # ========================================================

    print()
    print(
        "=================================="
    )

    print(
        "数据集转换完成"
    )

    print(
        "=================================="
    )

    print(
        f"有效图片总数："
        f"{len(samples)}"
    )

    print(
        f"物体实例总数："
        f"{total_objects}"
    )

    print(
        f"训练集："
        f"{len(train_samples)} 张"
    )

    print(
        f"验证集："
        f"{len(val_samples)} 张"
    )

    print(
        f"数据集目录："
        f"{YOLO_DIR}"
    )

    print(
        f"配置文件："
        f"{YOLO_DIR / 'top.yaml'}"
    )

    print(
        "=================================="
    )


if __name__ == "__main__":
    main()