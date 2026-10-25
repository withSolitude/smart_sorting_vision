from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 工程根目录
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# V2新采图片
# ============================================================

DEFAULT_IMAGE_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "new_images"
)


# ============================================================
# V1模型自动生成的预标注
# ============================================================

DEFAULT_LABEL_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "auto_labels"
)


# ============================================================
# CVAT打包输出目录
# ============================================================

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "cvat_imports"
)


# ============================================================
# 正式图片尺寸
#
# TopCamera.read()
# ->
# width  = 870
# height = 840
# ============================================================

EXPECTED_WIDTH = 870
EXPECTED_HEIGHT = 840


# ============================================================
# YOLO类别
# ============================================================

CLASS_ID = 0
CLASS_NAME = "object"


# ============================================================
# 支持的图片格式
# ============================================================

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# ============================================================
# Windows中文路径安全读取
#
# 不直接使用：
#
# cv2.imread(str(path))
#
# 而使用：
#
# np.fromfile()
#       ↓
# cv2.imdecode()
#
# 这样对Windows中文路径更加稳定。
# ============================================================

def imread_unicode(
    path: Path,
) -> np.ndarray:

    # ========================================================
    # 文件存在检查
    # ========================================================

    if not path.exists():

        raise FileNotFoundError(
            "图片不存在：\n"
            f"{path}"
        )

    if not path.is_file():

        raise RuntimeError(
            "路径不是文件：\n"
            f"{path}"
        )

    # ========================================================
    # 文件大小检查
    # ========================================================

    file_size = path.stat().st_size

    if file_size <= 0:

        raise RuntimeError(
            "图片文件为空：\n"
            f"{path}"
        )

    # ========================================================
    # 读取二进制字节
    # ========================================================

    try:

        data = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

    except Exception as exc:

        raise RuntimeError(
            "无法读取图片文件字节：\n"
            f"{path}"
        ) from exc

    if data.size == 0:

        raise RuntimeError(
            "图片文件读取结果为空：\n"
            f"{path}"
        )

    # ========================================================
    # OpenCV解码
    # ========================================================

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise RuntimeError(
            "图片文件存在，但OpenCV无法解码。\n"
            "该图片可能已经损坏：\n"
            f"{path}"
        )

    return image


# ============================================================
# 递归查找所有图片
# ============================================================

def find_images(
    image_root: Path,
) -> list[Path]:

    images: list[Path] = []

    for path in image_root.rglob("*"):

        if not path.is_file():

            continue

        if (
            path.suffix.lower()
            not in IMAGE_SUFFIXES
        ):

            continue

        images.append(
            path
        )

    # ========================================================
    # 为了每次生成结果顺序稳定
    # ========================================================

    images.sort(
        key=lambda path:
            path.as_posix()
    )

    return images


# ============================================================
# 获取图片对应的自动标签路径
#
# 例如：
#
# new_images/
# single/black/圆锥/top_v2_000078.jpg
#
# ->
#
# auto_labels/
# single/black/圆锥/top_v2_000078.txt
# ============================================================

def get_label_path(
    image_path: Path,
    image_root: Path,
    label_root: Path,
) -> Path:

    relative_path = (
        image_path.relative_to(
            image_root
        )
    )

    label_path = (
        label_root
        / relative_path
    )

    return label_path.with_suffix(
        ".txt"
    )


# ============================================================
# 图片检查
# ============================================================

def validate_image(
    image_path: Path,
) -> None:

    image = imread_unicode(
        image_path
    )

    height, width = (
        image.shape[:2]
    )

    # ========================================================
    # 尺寸必须和训练时正式输入一致
    # ========================================================

    if (
        width != EXPECTED_WIDTH
        or
        height != EXPECTED_HEIGHT
    ):

        raise RuntimeError(
            "图片尺寸错误：\n"
            f"{image_path}\n"
            f"实际：{width}x{height}\n"
            f"期望："
            f"{EXPECTED_WIDTH}x"
            f"{EXPECTED_HEIGHT}"
        )


# ============================================================
# 验证一个YOLO segmentation标签
#
# YOLO segmentation格式：
#
# 0 x1 y1 x2 y2 x3 y3 ...
#
# 空txt也是合法的：
#
# - empty场景
# - 或者V1模型完全漏检
#
# 后者后面必须在CVAT人工补。
#
# 返回：
#
# 实例数量
# ============================================================

def validate_label(
    label_path: Path,
) -> int:

    # ========================================================
    # 标签必须存在
    #
    # 即使没有检测到物体，
    # auto_label程序也应该生成空txt。
    # ========================================================

    if not label_path.exists():

        raise FileNotFoundError(
            "缺少自动标签：\n"
            f"{label_path}"
        )

    if not label_path.is_file():

        raise RuntimeError(
            "Label路径不是文件：\n"
            f"{label_path}"
        )

    # ========================================================
    # 读取标签
    # ========================================================

    try:

        text = label_path.read_text(
            encoding="utf-8"
        )

    except Exception as exc:

        raise RuntimeError(
            "无法读取Label：\n"
            f"{label_path}"
        ) from exc

    text = text.strip()

    # ========================================================
    # 空标签
    # ========================================================

    if not text:

        return 0

    instance_count = 0

    # ========================================================
    # 一行一个实例
    # ========================================================

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        line = line.strip()

        if not line:

            continue

        tokens = line.split()

        # ====================================================
        # 至少：
        #
        # class
        # +
        # 3个二维点
        #
        # =
        # 1 + 6 = 7个token
        # ====================================================

        if len(tokens) < 7:

            raise RuntimeError(
                f"Label格式错误：\n"
                f"{label_path}\n"
                f"第 {line_number} 行\n"
                "Polygon至少需要3个点。\n"
                f"内容：{line}"
            )

        # ====================================================
        # class后面的坐标数量必须是偶数
        # ====================================================

        coordinate_count = (
            len(tokens)
            -
            1
        )

        if coordinate_count % 2 != 0:

            raise RuntimeError(
                f"Label格式错误：\n"
                f"{label_path}\n"
                f"第 {line_number} 行\n"
                "Polygon坐标数量不是偶数。"
            )

        # ====================================================
        # 类别检查
        # ====================================================

        try:

            class_id = int(
                tokens[0]
            )

        except ValueError as exc:

            raise RuntimeError(
                f"Label类别无法解析：\n"
                f"{label_path}\n"
                f"第 {line_number} 行\n"
                f"class={tokens[0]}"
            ) from exc

        if class_id != CLASS_ID:

            raise RuntimeError(
                f"发现非法类别：\n"
                f"{label_path}\n"
                f"第 {line_number} 行\n"
                f"class={class_id}\n"
                f"当前数据集只允许 class=0 ({CLASS_NAME})"
            )

        # ====================================================
        # 坐标检查
        # ====================================================

        try:

            coordinates = [
                float(value)
                for value
                in tokens[1:]
            ]

        except ValueError as exc:

            raise RuntimeError(
                f"Label存在无法解析的坐标：\n"
                f"{label_path}\n"
                f"第 {line_number} 行"
            ) from exc

        # ====================================================
        # NaN / Inf / 越界检查
        # ====================================================

        for coordinate in coordinates:

            if not math.isfinite(
                coordinate
            ):

                raise RuntimeError(
                    "Label存在NaN或Inf：\n"
                    f"{label_path}\n"
                    f"第 {line_number} 行"
                )

            if not (
                0.0
                <=
                coordinate
                <=
                1.0
            ):

                raise RuntimeError(
                    "Label坐标超出[0,1]：\n"
                    f"{label_path}\n"
                    f"第 {line_number} 行\n"
                    f"value={coordinate}"
                )

        # ====================================================
        # Polygon点数
        # ====================================================

        point_count = (
            len(coordinates)
            //
            2
        )

        if point_count < 3:

            raise RuntimeError(
                "Polygon点数不足：\n"
                f"{label_path}\n"
                f"第 {line_number} 行\n"
                f"points={point_count}"
            )

        instance_count += 1

    return instance_count


# ============================================================
# 检查图片文件名是否全局唯一
#
# 后面要把所有中文目录中的图片扁平化：
#
# single/black/圆柱/top_v2_000001.jpg
# contact/top_v2_000002.jpg
#
# ->
#
# images/train/
#
# 所以不能出现两个同名文件。
# ============================================================

def verify_unique_image_names(
    image_paths: list[Path],
) -> None:

    name_to_paths: dict[
        str,
        list[Path],
    ] = {}

    for image_path in image_paths:

        name_to_paths.setdefault(
            image_path.name,
            [],
        ).append(
            image_path
        )

    duplicates = {

        name: paths

        for (
            name,
            paths
        ) in name_to_paths.items()

        if len(paths) > 1
    }

    if not duplicates:

        return

    print()

    print(
        "============================================================"
    )

    print(
        "发现重复图片文件名"
    )

    print(
        "============================================================"
    )

    for name, paths in (
        duplicates.items()
    ):

        print()

        print(
            f"文件名：{name}"
        )

        for path in paths:

            print(
                f"  {path}"
            )

    raise RuntimeError(
        "存在重复文件名，"
        "不能安全扁平化到CVAT。"
    )


# ============================================================
# 检查Label是否对应唯一图片
# ============================================================

def verify_label_names(
    image_paths: list[Path],
    image_root: Path,
    label_root: Path,
) -> None:

    seen_label_paths = set()

    for image_path in image_paths:

        label_path = (
            get_label_path(
                image_path=
                    image_path,

                image_root=
                    image_root,

                label_root=
                    label_root,
            )
        )

        resolved = (
            label_path.resolve()
        )

        if resolved in seen_label_paths:

            raise RuntimeError(
                "发现不同图片映射到同一个Label：\n"
                f"{label_path}"
            )

        seen_label_paths.add(
            resolved
        )


# ============================================================
# 创建完整数据集ZIP
# ============================================================

def zip_directory(
    source_dir: Path,
    output_zip: Path,
) -> None:

    output_zip.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for path in source_dir.rglob("*"):

            if not path.is_file():

                continue

            archive_name = (
                path.relative_to(
                    source_dir
                )
            )

            archive.write(
                filename=path,
                arcname=archive_name.as_posix(),
            )


# ============================================================
# 单独建立上传图片使用的ZIP
#
# CVAT创建Task时上传。
#
# ZIP结构：
#
# top_v2_000001.jpg
# top_v2_000002.jpg
# ...
#
# 不保留中文目录。
# ============================================================

def build_flat_image_zip(
    image_paths: list[Path],
    output_zip: Path,
) -> None:

    output_zip.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for image_path in image_paths:

            archive.write(
                filename=image_path,
                arcname=image_path.name,
            )


# ============================================================
# 获取场景类型
#
# 例如：
#
# single/black/圆柱/top_v2_000001.jpg
# ->
# single
#
# contact/top_v2_000500.jpg
# ->
# contact
# ============================================================

def get_scene_type(
    image_path: Path,
    image_root: Path,
) -> str:

    relative_path = (
        image_path.relative_to(
            image_root
        )
    )

    parts = (
        relative_path.parts
    )

    if not parts:

        return "unknown"

    return parts[0]


# ============================================================
# 判断是否属于真正empty目录
# ============================================================

def is_empty_scene(
    image_path: Path,
    image_root: Path,
) -> bool:

    return (
        get_scene_type(
            image_path=
                image_path,

            image_root=
                image_root,
        )
        ==
        "empty"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build CVAT Ultralytics YOLO Segmentation "
            "pre-annotation package for Top Seg V2."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="V2 new_images根目录",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABEL_ROOT,
        help="V1自动预标注auto_labels根目录",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="CVAT打包输出目录",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已经存在的打包结果",
    )

    args = parser.parse_args()

    # ========================================================
    # 路径
    # ========================================================

    image_root = (
        args.images
        .expanduser()
        .resolve()
    )

    label_root = (
        args.labels
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output
        .expanduser()
        .resolve()
    )

    # ========================================================
    # 输入目录检查
    # ========================================================

    if not image_root.exists():

        raise FileNotFoundError(
            "new_images不存在：\n"
            f"{image_root}"
        )

    if not image_root.is_dir():

        raise RuntimeError(
            "new_images不是目录：\n"
            f"{image_root}"
        )

    if not label_root.exists():

        raise FileNotFoundError(
            "auto_labels不存在：\n"
            f"{label_root}"
        )

    if not label_root.is_dir():

        raise RuntimeError(
            "auto_labels不是目录：\n"
            f"{label_root}"
        )

    # ========================================================
    # 查找所有图片
    # ========================================================

    image_paths = (
        find_images(
            image_root
        )
    )

    if not image_paths:

        raise RuntimeError(
            "new_images中没有找到任何图片：\n"
            f"{image_root}"
        )

    # ========================================================
    # 图片文件名唯一检查
    # ========================================================

    verify_unique_image_names(
        image_paths
    )

    # ========================================================
    # Label映射唯一性
    # ========================================================

    verify_label_names(
        image_paths=
            image_paths,

        image_root=
            image_root,

        label_root=
            label_root,
    )

    # ========================================================
    # 输出文件
    # ========================================================

    stage_root = (
        output_root
        /
        "_top_seg_v2_stage"
    )

    dataset_zip = (
        output_root
        /
        "top_seg_v2_preannotated.zip"
    )

    images_zip = (
        output_root
        /
        "top_seg_v2_images_for_cvat.zip"
    )

    summary_path = (
        output_root
        /
        "top_seg_v2_cvat_summary.json"
    )

    # ========================================================
    # 防误覆盖
    # ========================================================

    existing_paths = [

        path

        for path in (
            stage_root,
            dataset_zip,
            images_zip,
            summary_path,
        )

        if path.exists()
    ]

    if (
        existing_paths
        and
        not args.force
    ):

        print()

        print(
            "以下输出已经存在："
        )

        for path in existing_paths:

            print(
                f"  {path}"
            )

        raise RuntimeError(
            "\n如果确定重新生成，请运行：\n"
            "python -m "
            "scripts.dataset."
            "build_cvat_v2_preannotation_package "
            "--force"
        )

    # ========================================================
    # 清理旧结果
    # ========================================================

    if stage_root.exists():

        shutil.rmtree(
            stage_root
        )

    if args.force:

        for path in (
            dataset_zip,
            images_zip,
            summary_path,
        ):

            if path.exists():

                path.unlink()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # CVAT / Ultralytics打包目录
    #
    # stage/
    #
    # ├── data.yaml
    # ├── train.txt
    #
    # ├── images/
    # │   └── train/
    #
    # └── labels/
    #     └── train/
    # ========================================================

    image_stage = (
        stage_root
        /
        "images"
        /
        "train"
    )

    label_stage = (
        stage_root
        /
        "labels"
        /
        "train"
    )

    image_stage.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_stage.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 统计
    # ========================================================

    total_instances = 0

    empty_label_count = 0

    empty_scene_count = 0

    non_empty_scene_empty_label_count = 0

    scene_counter: dict[
        str,
        int,
    ] = {}

    empty_prediction_images: list[str] = []

    non_empty_scene_empty_predictions: list[str] = []

    train_lines: list[str] = []

    # ========================================================
    # 开始
    # ========================================================

    print()

    print(
        "============================================================"
    )

    print(
        "BUILD TOP SEG V2 CVAT PACKAGE"
    )

    print(
        "============================================================"
    )

    print(
        f"Image root:"
    )

    print(
        image_root
    )

    print()

    print(
        "Label root:"
    )

    print(
        label_root
    )

    print()

    print(
        f"Total images: "
        f"{len(image_paths)}"
    )

    print(
        "============================================================"
    )

    try:

        # ====================================================
        # 逐张检查和复制
        # ====================================================

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):

            relative_path = (
                image_path.relative_to(
                    image_root
                )
            )

            label_path = (
                get_label_path(
                    image_path=
                        image_path,

                    image_root=
                        image_root,

                    label_root=
                        label_root,
                )
            )

            scene_type = (
                get_scene_type(
                    image_path=
                        image_path,

                    image_root=
                        image_root,
                )
            )

            # =================================================
            # Scene统计
            # =================================================

            scene_counter[
                scene_type
            ] = (
                scene_counter.get(
                    scene_type,
                    0,
                )
                +
                1
            )

            # =================================================
            # 图片检查
            #
            # 这里使用中文路径安全读取。
            # =================================================

            try:

                validate_image(
                    image_path
                )

            except Exception as exc:

                print()

                print(
                    "============================================================"
                )

                print(
                    "[IMAGE ERROR]"
                )

                print(
                    "============================================================"
                )

                print(
                    f"图片："
                    f"{relative_path}"
                )

                print()

                print(
                    f"完整路径："
                    f"{image_path}"
                )

                print()

                print(
                    f"文件大小："
                    f"{image_path.stat().st_size if image_path.exists() else 'N/A'} bytes"
                )

                print(
                    "============================================================"
                )

                raise exc

            # =================================================
            # Label检查
            # =================================================

            try:

                instance_count = (
                    validate_label(
                        label_path
                    )
                )

            except Exception as exc:

                print()

                print(
                    "============================================================"
                )

                print(
                    "[LABEL ERROR]"
                )

                print(
                    "============================================================"
                )

                print(
                    f"Image:"
                )

                print(
                    image_path
                )

                print()

                print(
                    "Expected label:"
                )

                print(
                    label_path
                )

                print(
                    "============================================================"
                )

                raise exc

            # =================================================
            # 实例统计
            # =================================================

            total_instances += (
                instance_count
            )

            if instance_count == 0:

                empty_label_count += 1

                empty_prediction_images.append(
                    relative_path.as_posix()
                )

                if is_empty_scene(
                    image_path=
                        image_path,

                    image_root=
                        image_root,
                ):

                    empty_scene_count += 1

                else:

                    non_empty_scene_empty_label_count += 1

                    non_empty_scene_empty_predictions.append(
                        relative_path.as_posix()
                    )

            # =================================================
            # 扁平化目标
            # =================================================

            destination_image = (
                image_stage
                /
                image_path.name
            )

            destination_label = (
                label_stage
                /
                (
                    image_path.stem
                    +
                    ".txt"
                )
            )

            # =================================================
            # 复制图片
            #
            # shutil.copy2对中文Windows路径没有
            # cv2.imread那类问题。
            # =================================================

            shutil.copy2(
                image_path,
                destination_image,
            )

            # =================================================
            # 复制Label
            # =================================================

            shutil.copy2(
                label_path,
                destination_label,
            )

            # =================================================
            # train.txt
            # =================================================

            train_lines.append(
                "images/train/"
                +
                image_path.name
            )

            # =================================================
            # 打印
            # =================================================

            marker = ""

            if instance_count == 0:

                if scene_type == "empty":

                    marker = (
                        " [EMPTY OK]"
                    )

                else:

                    marker = (
                        " [CHECK IN CVAT]"
                    )

            print(
                f"[{index:4d}/"
                f"{len(image_paths):4d}] "
                f"{relative_path} "
                f"-> "
                f"{instance_count} object(s)"
                f"{marker}"
            )

        # ====================================================
        # train.txt
        # ====================================================

        train_txt_content = (
            "\n".join(
                train_lines
            )
            +
            "\n"
        )

        (
            stage_root
            /
            "train.txt"
        ).write_text(
            train_txt_content,
            encoding="utf-8",
        )

        # ====================================================
        # data.yaml
        # ====================================================

        data_yaml_content = (
            "path: ./\n"
            "train: train.txt\n"
            "\n"
            "names:\n"
            f"  0: {CLASS_NAME}\n"
        )

        (
            stage_root
            /
            "data.yaml"
        ).write_text(
            data_yaml_content,
            encoding="utf-8",
        )

        # ====================================================
        # 生成预标注数据ZIP
        # ====================================================

        print()

        print(
            "Creating pre-annotation ZIP..."
        )

        zip_directory(
            source_dir=
                stage_root,

            output_zip=
                dataset_zip,
        )

        # ====================================================
        # 生成CVAT创建Task使用的图片ZIP
        # ====================================================

        print(
            "Creating image ZIP..."
        )

        build_flat_image_zip(
            image_paths=
                image_paths,

            output_zip=
                images_zip,
        )

        # ====================================================
        # Summary
        # ====================================================

        summary = {

            "image_root":
                str(
                    image_root
                ),

            "label_root":
                str(
                    label_root
                ),

            "total_images":
                len(
                    image_paths
                ),

            "total_instances":
                total_instances,

            "empty_label_count":
                empty_label_count,

            "empty_scene_count":
                empty_scene_count,

            "non_empty_scene_empty_label_count":
                non_empty_scene_empty_label_count,

            "scene_counts":
                scene_counter,

            "empty_prediction_images":
                empty_prediction_images,

            "non_empty_scene_empty_predictions":
                non_empty_scene_empty_predictions,

            "class_names": {

                "0":
                    CLASS_NAME,
            },

            "images_zip":
                str(
                    images_zip
                ),

            "preannotation_zip":
                str(
                    dataset_zip
                ),
        }

        summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=4,
            ),
            encoding="utf-8",
        )

    except Exception:

        # ====================================================
        # 构建失败时删除半成品Stage
        # ====================================================

        if stage_root.exists():

            shutil.rmtree(
                stage_root
            )

        raise

    # ========================================================
    # Stage完成后删除
    # ========================================================

    if stage_root.exists():

        shutil.rmtree(
            stage_root
        )

    # ========================================================
    # 最终报告
    # ========================================================

    print()

    print(
        "============================================================"
    )

    print(
        "CVAT PACKAGE READY"
    )

    print(
        "============================================================"
    )

    print(
        f"Images             : "
        f"{len(image_paths)}"
    )

    print(
        f"Predicted instances: "
        f"{total_instances}"
    )

    print(
        f"Empty labels       : "
        f"{empty_label_count}"
    )

    print(
        f"  real empty scene : "
        f"{empty_scene_count}"
    )

    print(
        f"  possible misses  : "
        f"{non_empty_scene_empty_label_count}"
    )

    print()

    print(
        "Scene counts:"
    )

    for scene, count in sorted(
        scene_counter.items()
    ):

        print(
            f"  "
            f"{scene:<12}"
            f": "
            f"{count}"
        )

    # ========================================================
    # 漏检提醒
    # ========================================================

    if (
        non_empty_scene_empty_label_count
        >
        0
    ):

        print()

        print(
            "------------------------------------------------------------"
        )

        print(
            "以下非empty图片的V1预测完全为空，"
            "导入CVAT后必须优先人工检查："
        )

        print(
            "------------------------------------------------------------"
        )

        for path in (
            non_empty_scene_empty_predictions[
                :30
            ]
        ):

            print(
                f"[CHECK] {path}"
            )

        if (
            len(
                non_empty_scene_empty_predictions
            )
            >
            30
        ):

            print(
                "..."
            )

            print(
                "完整列表请查看："
            )

            print(
                summary_path
            )

    print()

    print(
        "------------------------------------------------------------"
    )

    print(
        "1. 创建CVAT Task时上传："
    )

    print(
        images_zip
    )

    print()

    print(
        "2. Task创建完成后，"
        "Upload annotations时上传："
    )

    print(
        dataset_zip
    )

    print()

    print(
        "3. Summary："
    )

    print(
        summary_path
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":

    main()