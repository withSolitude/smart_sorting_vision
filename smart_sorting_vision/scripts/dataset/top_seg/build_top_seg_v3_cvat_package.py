from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


# ============================================================
# 工程根目录
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# V3 原始图片
#
# 与 V2 内部格式完全一致，只是根目录改为 new_images_v3。
# 本程序不会读取 V2 数据目录。
# ============================================================

DEFAULT_IMAGE_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "new_images_v3"
)


# ============================================================
# V3 自动预标注
#
# 相对目录与 new_images_v3 镜像。
# ============================================================

DEFAULT_LABEL_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "auto_labels_v3"
)


# ============================================================
# 输出目录
# ============================================================

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "cvat_imports_v3"
)


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 查找图片
# ============================================================

def find_images(
    image_root: Path,
) -> list[Path]:

    images = [
        path
        for path in image_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    ]

    images.sort(
        key=lambda path:
        path.relative_to(
            image_root
        ).as_posix()
    )

    return images


# ============================================================
# 查找标签
# ============================================================

def find_labels(
    label_root: Path,
) -> list[Path]:

    labels = [
        path
        for path in label_root.rglob("*.txt")
        if path.is_file()
    ]

    labels.sort(
        key=lambda path:
        path.relative_to(
            label_root
        ).as_posix()
    )

    return labels


# ============================================================
# 建立 stem -> Path
#
# V3 图片使用全局唯一文件名：
#
# top_v3_000001.jpg
# top_v3_000002.jpg
# ...
#
# 所以 CVAT 打包时可以安全展平。
# ============================================================

def build_unique_stem_map(
    paths: list[Path],
    kind: str,
) -> dict[str, Path]:

    result: dict[str, Path] = {}

    duplicates: dict[
        str,
        list[Path],
    ] = {}

    for path in paths:

        stem = path.stem

        if stem in result:

            duplicates.setdefault(
                stem,
                [
                    result[stem]
                ],
            ).append(
                path
            )

        else:

            result[
                stem
            ] = path

    if duplicates:

        lines = [
            f"{kind}存在重复文件 stem，不能安全展平："
        ]

        for stem, duplicate_paths in sorted(
            duplicates.items()
        ):

            lines.append(
                f"\n[{stem}]"
            )

            for path in duplicate_paths:

                lines.append(
                    f"  {path}"
                )

        raise RuntimeError(
            "\n".join(
                lines
            )
        )

    return result


# ============================================================
# 验证一个 YOLO Segmentation 标签文件
#
# 格式：
#
# class x1 y1 x2 y2 x3 y3 ...
#
# 当前项目只有：
#
# 0: object
# ============================================================

def validate_seg_label(
    label_path: Path,
) -> tuple[
    int,
    list[str],
]:

    text = label_path.read_text(
        encoding="utf-8-sig"
    )

    stripped = text.strip()

    # --------------------------------------------------------
    # 空标签合法：
    #
    # empty 场景应当为空；
    # 非 empty 场景也可能因为旧模型漏检暂时为空，
    # 这种情况后面会输出 warning，让 CVAT 人工补标。
    # --------------------------------------------------------

    if not stripped:

        return (
            0,
            [],
        )

    errors: list[str] = []

    instance_count = 0

    for line_number, raw_line in enumerate(
        text.splitlines(),
        start=1,
    ):

        line = raw_line.strip()

        if not line:

            continue

        tokens = line.split()

        # class + 至少3个点 = 1 + 6
        if len(tokens) < 7:

            errors.append(
                f"line {line_number}: "
                f"字段数量不足 ({len(tokens)})"
            )

            continue

        if (
            len(tokens)
            -
            1
        ) % 2 != 0:

            errors.append(
                f"line {line_number}: "
                "坐标数量不是偶数"
            )

            continue

        try:

            class_id = int(
                tokens[0]
            )

        except ValueError:

            errors.append(
                f"line {line_number}: "
                f"class id不是整数: "
                f"{tokens[0]}"
            )

            continue

        if class_id != 0:

            errors.append(
                f"line {line_number}: "
                f"当前项目只允许class 0，"
                f"实际={class_id}"
            )

        try:

            coordinates = [
                float(value)
                for value
                in tokens[1:]
            ]

        except ValueError:

            errors.append(
                f"line {line_number}: "
                "存在无法解析的坐标"
            )

            continue

        for coordinate_index, value in enumerate(
            coordinates,
            start=1,
        ):

            if not (
                0.0
                <=
                value
                <=
                1.0
            ):

                errors.append(
                    f"line {line_number}: "
                    f"第{coordinate_index}个坐标"
                    f"越界={value}"
                )

        point_count = (
            len(coordinates)
            //
            2
        )

        if point_count < 3:

            errors.append(
                f"line {line_number}: "
                f"Polygon点数不足={point_count}"
            )

        instance_count += 1

    return (
        instance_count,
        errors,
    )


# ============================================================
# 场景
#
# relative path:
#
# contact/top_v3_xxx.jpg
#
# single/red/圆柱/top_v3_xxx.jpg
#
# 第一级目录始终是 scene。
# ============================================================

def get_scene(
    image_path: Path,
    image_root: Path,
) -> str:

    relative = image_path.relative_to(
        image_root
    )

    if len(
        relative.parts
    ) < 2:

        raise RuntimeError(
            "图片没有位于scene目录下："
            f"{relative}"
        )

    return relative.parts[0]


# ============================================================
# ZIP目录
# ============================================================

def zip_directory(
    source_dir: Path,
    output_zip: Path,
) -> None:

    if output_zip.exists():

        output_zip.unlink()

    with ZipFile(
        output_zip,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for path in sorted(
            source_dir.rglob("*")
        ):

            if not path.is_file():

                continue

            arcname = (
                path.relative_to(
                    source_dir
                )
                .as_posix()
            )

            archive.write(
                path,
                arcname=arcname,
            )


# ============================================================
# 生成只含图片的 CVAT Task 数据 ZIP
#
# 图片在ZIP根目录展平：
#
# top_v3_000001.jpg
# top_v3_000002.jpg
# ...
#
# 由于 V3 文件名全局唯一，可以安全这样做。
# ============================================================

def build_images_zip(
    records: list[dict],
    images_zip_path: Path,
) -> None:

    if images_zip_path.exists():

        images_zip_path.unlink()

    with ZipFile(
        images_zip_path,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for record in records:

            image_path: Path = (
                record[
                    "image_path"
                ]
            )

            archive.write(
                image_path,
                arcname=
                    image_path.name,
            )


# ============================================================
# 主构建
# ============================================================

def build_package(
    image_root: Path,
    label_root: Path,
    output_dir: Path,
    clean: bool,
) -> None:

    print(
        "============================================================"
    )

    print(
        "V3 -> CVAT ULTRALYTICS YOLO SEGMENTATION PACKAGER"
    )

    print(
        "============================================================"
    )

    print(
        f"Image root : {image_root}"
    )

    print(
        f"Label root : {label_root}"
    )

    print(
        f"Output dir : {output_dir}"
    )

    print(
        "============================================================"
    )

    # ========================================================
    # 1. 输入检查
    # ========================================================

    if not image_root.exists():

        raise FileNotFoundError(
            "找不到V3图片目录：\n"
            f"{image_root}"
        )

    if not label_root.exists():

        raise FileNotFoundError(
            "找不到V3自动标签目录：\n"
            f"{label_root}"
        )

    images = find_images(
        image_root
    )

    labels = find_labels(
        label_root
    )

    if not images:

        raise RuntimeError(
            "new_images_v3中没有图片"
        )

    if not labels:

        raise RuntimeError(
            "auto_labels_v3中没有标签"
        )

    print()
    print(
        f"Images found : {len(images)}"
    )

    print(
        f"Labels found : {len(labels)}"
    )

    # ========================================================
    # 2. 全局唯一文件名检查
    # ========================================================

    image_map = build_unique_stem_map(
        images,
        "图片",
    )

    label_map = build_unique_stem_map(
        labels,
        "标签",
    )

    image_stems = set(
        image_map
    )

    label_stems = set(
        label_map
    )

    missing_labels = sorted(
        image_stems
        -
        label_stems
    )

    orphan_labels = sorted(
        label_stems
        -
        image_stems
    )

    if missing_labels:

        print()
        print(
            "以下图片没有对应自动标签："
        )

        for stem in missing_labels:

            print(
                f"  {image_map[stem]}"
            )

    if orphan_labels:

        print()
        print(
            "以下标签找不到对应图片："
        )

        for stem in orphan_labels:

            print(
                f"  {label_map[stem]}"
            )

    # --------------------------------------------------------
    # 对于自动标注阶段：
    #
    # 图片无标签文件不能静默放过。
    #
    # 即使模型完全漏检，也应当存在一个空txt，
    # 这样才能区分：
    #
    # 1. 模型确实预测为空
    # 2. 标注程序漏跑/文件丢失
    # --------------------------------------------------------

    if missing_labels:

        raise RuntimeError(
            "存在图片缺失标签文件。"
            "请先修复auto_labels_v3。"
        )

    if orphan_labels:

        raise RuntimeError(
            "存在无对应图片的标签文件。"
        )

    # ========================================================
    # 3. 标签格式验证 + 统计
    # ========================================================

    records: list[dict] = []

    label_errors: list[str] = []

    scene_image_counter = Counter()

    scene_instance_counter = Counter()

    empty_label_nonempty_scene = []

    nonempty_label_empty_scene = []

    total_instances = 0

    for image_path in images:

        stem = (
            image_path.stem
        )

        label_path = (
            label_map[
                stem
            ]
        )

        scene = get_scene(
            image_path,
            image_root,
        )

        (
            instance_count,
            errors,
        ) = validate_seg_label(
            label_path
        )

        if errors:

            for error in errors:

                label_errors.append(
                    f"{label_path}: {error}"
                )

        is_empty_label = (
            instance_count == 0
        )

        scene_image_counter[
            scene
        ] += 1

        scene_instance_counter[
            scene
        ] += instance_count

        total_instances += (
            instance_count
        )

        if (
            scene != "empty"
            and
            is_empty_label
        ):

            empty_label_nonempty_scene.append(
                image_path
            )

        if (
            scene == "empty"
            and
            not is_empty_label
        ):

            nonempty_label_empty_scene.append(
                image_path
            )

        records.append(
            {
                "image_path":
                    image_path,

                "label_path":
                    label_path,

                "scene":
                    scene,

                "image_name":
                    image_path.name,

                "stem":
                    stem,

                "original_relative_path":
                    image_path.relative_to(
                        image_root
                    ).as_posix(),

                "label_relative_path":
                    label_path.relative_to(
                        label_root
                    ).as_posix(),

                "instances":
                    instance_count,

                "label_is_empty":
                    is_empty_label,
            }
        )

    if label_errors:

        print()
        print(
            "LABEL VALIDATION ERRORS"
        )

        for error in label_errors:

            print(
                f"  {error}"
            )

        raise RuntimeError(
            "YOLO Segmentation标签格式检查失败"
        )

    # ========================================================
    # 4. 数据统计
    # ========================================================

    print()
    print(
        "------------------------------------------------------------"
    )

    print(
        "V3 statistics"
    )

    print(
        "------------------------------------------------------------"
    )

    for scene in sorted(
        scene_image_counter
    ):

        print(
            f"{scene:<12} "
            f"images={scene_image_counter[scene]:4d} "
            f"instances={scene_instance_counter[scene]:4d}"
        )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"Total images    : {len(records)}"
    )

    print(
        f"Total instances : {total_instances}"
    )

    # ========================================================
    # 5. 重要Warning
    # ========================================================

    if empty_label_nonempty_scene:

        print()
        print(
            "[WARNING] 非empty场景存在空标签。"
        )

        print(
            "这些通常代表旧模型漏检，"
            "导入CVAT后必须人工补画："
        )

        for path in (
            empty_label_nonempty_scene
        ):

            print(
                "  "
                +
                path.relative_to(
                    image_root
                ).as_posix()
            )

    if nonempty_label_empty_scene:

        print()
        print(
            "[WARNING] empty场景存在非空标签。"
        )

        print(
            "这通常代表新硬件背景假阳性，"
            "CVAT中应删除这些对象："
        )

        for path in (
            nonempty_label_empty_scene
        ):

            print(
                "  "
                +
                path.relative_to(
                    image_root
                ).as_posix()
            )

    # ========================================================
    # 6. 输出目录
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    package_dir = (
        output_dir
        / "v3_cvat_ultralytics_seg"
    )

    if (
        clean
        and
        package_dir.exists()
    ):

        shutil.rmtree(
            package_dir
        )

    package_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_train_dir = (
        package_dir
        / "images"
        / "train"
    )

    labels_train_dir = (
        package_dir
        / "labels"
        / "train"
    )

    images_train_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_train_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 7. 展平复制图片和标签
    # ========================================================

    for record in records:

        image_path = (
            record[
                "image_path"
            ]
        )

        label_path = (
            record[
                "label_path"
            ]
        )

        destination_image = (
            images_train_dir
            /
            image_path.name
        )

        destination_label = (
            labels_train_dir
            /
            f"{image_path.stem}.txt"
        )

        shutil.copy2(
            image_path,
            destination_image,
        )

        shutil.copy2(
            label_path,
            destination_label,
        )

    # ========================================================
    # 8. train.txt
    #
    # CVAT官方Ultralytics YOLO格式：
    #
    # images/train/image1.jpg
    # images/train/image2.jpg
    # ...
    # ========================================================

    train_txt_path = (
        package_dir
        / "train.txt"
    )

    train_lines = [
        (
            "images/train/"
            +
            record[
                "image_name"
            ]
        )
        for record in records
    ]

    train_txt_path.write_text(
        "\n".join(
            train_lines
        )
        +
        "\n",
        encoding="utf-8",
    )

    # ========================================================
    # 9. data.yaml
    #
    # 单类别：
    #
    # 0: object
    # ========================================================

    data_yaml_path = (
        package_dir
        / "data.yaml"
    )

    data_yaml_text = (
        "path: ./\n"
        "train: train.txt\n"
        "names:\n"
        "  0: object\n"
    )

    data_yaml_path.write_text(
        data_yaml_text,
        encoding="utf-8",
    )

    # ========================================================
    # 10. Scene Manifest
    #
    # 这个文件不放进CVAT标准ZIP。
    #
    # 它用来在CVAT导出后，
    # 恢复图片原本属于：
    #
    # contact / mixed / stacked / empty / single/... 等
    # ========================================================

    manifest_path = (
        output_dir
        / "v3_cvat_scene_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "image_name",
                "stem",
                "scene",
                "original_relative_path",
                "label_relative_path",
                "instances_auto",
                "auto_label_empty",
            ]
        )

        for record in records:

            writer.writerow(
                [
                    record[
                        "image_name"
                    ],

                    record[
                        "stem"
                    ],

                    record[
                        "scene"
                    ],

                    record[
                        "original_relative_path"
                    ],

                    record[
                        "label_relative_path"
                    ],

                    record[
                        "instances"
                    ],

                    int(
                        record[
                            "label_is_empty"
                        ]
                    ),
                ]
            )

    # ========================================================
    # 11. CVAT标准完整数据集ZIP
    #
    # ZIP根目录：
    #
    # data.yaml
    # train.txt
    # images/train/*.jpg
    # labels/train/*.txt
    #
    # 用于：
    # Ultralytics YOLO Segmentation 1.0
    # ========================================================

    dataset_zip_path = (
        output_dir
        / "v3_cvat_ultralytics_seg.zip"
    )

    zip_directory(
        package_dir,
        dataset_zip_path,
    )

    # ========================================================
    # 12. CVAT建任务图片ZIP
    #
    # 先用这个ZIP创建Task。
    #
    # 然后：
    #
    # Upload annotations
    # -> Ultralytics YOLO Segmentation 1.0
    # -> 选择 v3_cvat_ultralytics_seg.zip
    # ========================================================

    images_zip_path = (
        output_dir
        / "v3_cvat_images.zip"
    )

    build_images_zip(
        records,
        images_zip_path,
    )

    # ========================================================
    # 13. Summary
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "CVAT PACKAGE GENERATED SUCCESSFULLY"
    )

    print(
        "============================================================"
    )

    print(
        f"Images ZIP : "
        f"{images_zip_path}"
    )

    print(
        f"Dataset ZIP: "
        f"{dataset_zip_path}"
    )

    print(
        f"Manifest   : "
        f"{manifest_path}"
    )

    print(
        f"Build dir  : "
        f"{package_dir}"
    )

    print()
    print(
        "CVAT使用顺序："
    )

    print(
        "1. 创建Task，标签只建一个：object"
    )

    print(
        "2. 上传 v3_cvat_images.zip 作为Task图片"
    )

    print(
        "3. Task创建完成后选择 Upload annotations"
    )

    print(
        "4. 格式选择："
        "Ultralytics YOLO Segmentation 1.0"
    )

    print(
        "5. 上传："
        "v3_cvat_ultralytics_seg.zip"
    )

    if empty_label_nonempty_scene:

        print()
        print(
            "特别注意："
            f"{len(empty_label_nonempty_scene)} 张"
            "非empty图片当前自动标签为空，"
            "导入CVAT后必须人工补标。"
        )

    print(
        "============================================================"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build CVAT Ultralytics YOLO "
            "Segmentation package for V3."
        )
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
    )

    parser.add_argument(
        "--label-root",
        type=Path,
        default=DEFAULT_LABEL_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--no-clean",
        action="store_true",
        help=(
            "Do not delete previous build directory."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    build_package(
        image_root=
            args.image_root.resolve(),

        label_root=
            args.label_root.resolve(),

        output_dir=
            args.output_dir.resolve(),

        clean=
            not args.no_clean,
    )


if __name__ == "__main__":

    main()
