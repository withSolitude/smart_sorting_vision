from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import zipfile

from collections import Counter
from collections import defaultdict

from pathlib import Path
from pathlib import PurePosixPath


# ============================================================
# 项目根目录
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# 顶部实例分割数据集目录
# ============================================================

TOP_SEG_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
)


# ============================================================
# 原始数据
# ============================================================

RAW_ROOT = (
    TOP_SEG_ROOT
    / "raw"
)

RAW_IMAGE_DIR = (
    RAW_ROOT
    / "images"
)

IMAGES_CSV = (
    RAW_ROOT
    / "images.csv"
)

LAYOUTS_CSV = (
    RAW_ROOT
    / "layouts.csv"
)

OBJECTS_CSV = (
    RAW_ROOT
    / "objects.csv"
)


# ============================================================
# CVAT导出
# ============================================================

DEFAULT_CVAT_ZIP = (
    TOP_SEG_ROOT
    / "cvat_exports"
    / "top_seg_v1_batch001.zip"
)


# ============================================================
# 正式数据集输出
# ============================================================

PREPARED_ROOT = (
    TOP_SEG_ROOT
    / "prepared"
)


# ============================================================
# 当前相机正式训练图尺寸
#
# TopCamera.read()
# ->
# 870 x 840
#
# OpenCV shape:
# (840, 870, 3)
# ============================================================

EXPECTED_WIDTH = 870
EXPECTED_HEIGHT = 840


# ============================================================
# 当前任务只有一个类别
# ============================================================

CLASS_ID = 0
CLASS_NAME = "object"


# ============================================================
# 默认数据集划分
#
# 注意：
# 不是按图片随机划分。
#
# 是按 layout_id 分组划分。
# ============================================================

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

DEFAULT_RANDOM_SEED = 2026


# ============================================================
# 读取CSV
# ============================================================

def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:

    if not path.exists():

        raise FileNotFoundError(
            f"CSV不存在：\n{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


# ============================================================
# 按某个字段创建唯一索引
#
# 例如：
#
# image -> images.csv row
# layout_id -> layouts.csv row
# ============================================================

def build_unique_index(
    rows: list[dict[str, str]],
    key: str,
    source_name: str,
) -> dict[str, dict[str, str]]:

    result = {}

    for row in rows:

        value = (
            row.get(
                key,
                ""
            )
            .strip()
        )

        if not value:

            raise RuntimeError(
                f"{source_name} 中存在空 {key}"
            )

        if value in result:

            raise RuntimeError(
                f"{source_name} 中存在重复 {key}："
                f"{value}"
            )

        result[
            value
        ] = row

    return result


# ============================================================
# 读取CVAT导出ZIP
#
# 返回：
#
# image_names:
# [
#     top_000000.jpg,
#     ...
# ]
#
# labels:
# {
#     "top_000000": "...",
#     ...
# }
# ============================================================

def load_cvat_export(
    zip_path: Path,
) -> tuple[
    list[str],
    dict[str, str],
]:

    if not zip_path.exists():

        raise FileNotFoundError(
            "找不到CVAT导出ZIP：\n"
            f"{zip_path}"
        )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        members = (
            archive.namelist()
        )

        # ====================================================
        # 必须有 train.txt
        #
        # 你的当前CVAT包就是这个结构。
        # ====================================================

        if "train.txt" not in members:

            raise RuntimeError(
                "CVAT ZIP 中找不到 train.txt。\n"
                "请确认导出格式是 "
                "Ultralytics YOLO Segmentation。"
            )

        train_text = (
            archive
            .read(
                "train.txt"
            )
            .decode(
                "utf-8"
            )
        )

        image_names = []

        seen_images = set()

        for line in (
            train_text
            .splitlines()
        ):

            line = (
                line.strip()
            )

            if not line:

                continue

            # ------------------------------------------------
            # CVAT中类似：
            #
            # data/images/train/top_000000.jpg
            #
            # 这里只取最后文件名。
            # ------------------------------------------------

            image_name = (
                PurePosixPath(
                    line
                )
                .name
            )

            if image_name in seen_images:

                raise RuntimeError(
                    "CVAT train.txt 中出现重复图片："
                    f"{image_name}"
                )

            seen_images.add(
                image_name
            )

            image_names.append(
                image_name
            )

        if not image_names:

            raise RuntimeError(
                "CVAT train.txt 为空。"
            )

        # ====================================================
        # 读取labels
        # ====================================================

        labels = {}

        for member in members:

            member_path = (
                PurePosixPath(
                    member
                )
            )

            if (
                member_path.suffix.lower()
                !=
                ".txt"
            ):

                continue

            # ------------------------------------------------
            # 忽略train.txt
            # ------------------------------------------------

            if (
                member_path.name
                ==
                "train.txt"
            ):

                continue

            # ------------------------------------------------
            # 只处理labels目录
            # ------------------------------------------------

            if (
                "labels"
                not in member_path.parts
            ):

                continue

            stem = (
                member_path.stem
            )

            if stem in labels:

                raise RuntimeError(
                    "CVAT ZIP 中同一个图片存在多个label："
                    f"{stem}"
                )

            labels[
                stem
            ] = (
                archive
                .read(
                    member
                )
                .decode(
                    "utf-8"
                )
            )

    return (
        image_names,
        labels,
    )


# ============================================================
# 验证YOLO Segmentation label
#
# 返回：
#
# instance_count
# warnings
# ============================================================

def validate_label_text(
    image_name: str,
    label_text: str,
    expected_empty: bool,
) -> tuple[
    int,
    list[str],
]:

    warnings = []

    stripped = (
        label_text.strip()
    )

    # ========================================================
    # 空标签
    # ========================================================

    if not stripped:

        if not expected_empty:

            raise RuntimeError(
                f"{image_name}: "
                "非empty布局没有任何实例标注。"
            )

        return (
            0,
            warnings,
        )

    # ========================================================
    # empty布局却有标注
    # ========================================================

    if expected_empty:

        raise RuntimeError(
            f"{image_name}: "
            "布局被记录为empty，"
            "但CVAT中存在实例标注。"
        )

    instance_count = 0

    for line_number, line in enumerate(
        stripped.splitlines(),
        start=1,
    ):

        line = (
            line.strip()
        )

        if not line:

            continue

        tokens = (
            line.split()
        )

        # ----------------------------------------------------
        # 至少：
        #
        # class
        # +
        # 3个(x,y)
        #
        # = 7个token
        # ----------------------------------------------------

        if len(tokens) < 7:

            raise RuntimeError(
                f"{image_name}: "
                f"第{line_number}行 "
                "polygon点数不足。\n"
                f"{line}"
            )

        if (
            (
                len(tokens)
                -
                1
            )
            %
            2
            !=
            0
        ):

            raise RuntimeError(
                f"{image_name}: "
                f"第{line_number}行 "
                "坐标数量不是偶数。"
            )

        # ====================================================
        # 类别
        # ====================================================

        try:

            class_id = int(
                tokens[0]
            )

        except ValueError as exc:

            raise RuntimeError(
                f"{image_name}: "
                f"类别无法解析："
                f"{tokens[0]}"
            ) from exc

        if (
            class_id
            !=
            CLASS_ID
        ):

            raise RuntimeError(
                f"{image_name}: "
                f"发现非法类别 {class_id}，"
                f"当前任务只允许类别 {CLASS_ID}。"
            )

        # ====================================================
        # Polygon坐标
        # ====================================================

        try:

            coordinates = [
                float(value)
                for value
                in tokens[1:]
            ]

        except ValueError as exc:

            raise RuntimeError(
                f"{image_name}: "
                f"第{line_number}行存在非法坐标。"
            ) from exc

        for coordinate in coordinates:

            if not math.isfinite(
                coordinate
            ):

                raise RuntimeError(
                    f"{image_name}: "
                    "存在 NaN / Inf 坐标。"
                )

            if not (
                0.0
                <= coordinate
                <= 1.0
            ):

                raise RuntimeError(
                    f"{image_name}: "
                    "存在超出[0,1]的坐标："
                    f"{coordinate}"
                )

        number_of_points = (
            len(
                coordinates
            )
            //
            2
        )

        if number_of_points > 200:

            warnings.append(
                f"{image_name}: "
                f"第{line_number}个实例有 "
                f"{number_of_points} 个polygon点，"
                "非常密集，但暂不作为错误。"
            )

        instance_count += 1

    return (
        instance_count,
        warnings,
    )


# ============================================================
# 将字符串安全转换成int
# ============================================================

def safe_int(
    value: str,
    default: int = 0,
) -> int:

    try:

        return int(
            value
        )

    except (
        ValueError,
        TypeError,
    ):

        return default


# ============================================================
# 生成完整数据记录
# ============================================================

def build_records(
    image_names: list[str],
    cvat_labels: dict[str, str],
    image_index: dict[str, dict[str, str]],
    layout_index: dict[str, dict[str, str]],
) -> tuple[
    list[dict],
    list[str],
]:

    records = []

    warnings = []

    for image_name in image_names:

        # ====================================================
        # 原图存在检查
        # ====================================================

        image_path = (
            RAW_IMAGE_DIR
            /
            image_name
        )

        if not image_path.exists():

            raise FileNotFoundError(
                "CVAT中的图片在raw/images中不存在：\n"
                f"{image_name}\n"
                f"Expected:\n{image_path}"
            )

        # ====================================================
        # images.csv中必须存在
        # ====================================================

        if image_name not in image_index:

            raise RuntimeError(
                "CVAT图片没有出现在images.csv中："
                f"{image_name}"
            )

        image_row = (
            image_index[
                image_name
            ]
        )

        layout_id = (
            image_row.get(
                "layout_id",
                ""
            )
            .strip()
        )

        if not layout_id:

            raise RuntimeError(
                f"{image_name}: "
                "images.csv 中 layout_id 为空。"
            )

        # ====================================================
        # layout必须存在
        # ====================================================

        if layout_id not in layout_index:

            raise RuntimeError(
                f"{image_name}: "
                f"引用了不存在的layout："
                f"{layout_id}"
            )

        layout_row = (
            layout_index[
                layout_id
            ]
        )

        scene_type = (
            layout_row.get(
                "scene_type",
                ""
            )
            .strip()
        )

        object_count = (
            safe_int(
                layout_row.get(
                    "object_count",
                    "0"
                )
            )
        )

        expected_empty = (
            scene_type
            ==
            "empty"
            or
            object_count
            ==
            0
        )

        # ====================================================
        # 获取label
        #
        # empty场景允许CVAT不产生label文件。
        # ====================================================

        stem = (
            Path(
                image_name
            )
            .stem
        )

        label_text = (
            cvat_labels.get(
                stem,
                ""
            )
        )

        if (
            stem
            not in cvat_labels
            and
            not expected_empty
        ):

            raise RuntimeError(
                f"{image_name}: "
                "CVAT中没有对应label文件，"
                "但该layout并不是empty。"
            )

        (
            instance_count,
            label_warnings,
        ) = validate_label_text(
            image_name=
                image_name,

            label_text=
                label_text,

            expected_empty=
                expected_empty,
        )

        warnings.extend(
            label_warnings
        )

        # ====================================================
        # 与采集时登记的object_count比较
        #
        # 这里不作为硬错误。
        #
        # 因为极端情况下一个真实物体可能完全被遮挡，
        # object_count 和可见实例数可能不同。
        # ====================================================

        if (
            not expected_empty
            and
            object_count
            !=
            instance_count
        ):

            warnings.append(
                f"{image_name}: "
                f"layout记录 object_count="
                f"{object_count}，"
                f"CVAT可见实例={instance_count}。"
            )

        record = {

            "image":
                image_name,

            "stem":
                stem,

            "image_path":
                image_path,

            "label_text":
                label_text,

            "layout_id":
                layout_id,

            "session_id":
                image_row.get(
                    "session_id",
                    ""
                ),

            "scene_type":
                scene_type,

            "object_count":
                object_count,

            "occlusion":
                layout_row.get(
                    "occlusion",
                    ""
                ),

            "edge":
                layout_row.get(
                    "edge",
                    ""
                ),

            "zone":
                layout_row.get(
                    "zone",
                    ""
                ),

            "instances":
                instance_count,
        }

        records.append(
            record
        )

    return (
        records,
        warnings,
    )


# ============================================================
# Group Split
#
# 核心：
#
# 同一个 layout_id 的所有图片
# 永远进入同一个split。
#
# 同时尽量按照图片数量保持：
#
# train / val / test
# =
# 70 / 15 / 15
#
# 并辅助考虑scene_type分布。
# ============================================================

def split_by_layout(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:

    ratios = {

        "train":
            train_ratio,

        "val":
            val_ratio,

        "test":
            test_ratio,
    }

    ratio_sum = sum(
        ratios.values()
    )

    if not math.isclose(
        ratio_sum,
        1.0,
        abs_tol=1e-8,
    ):

        raise ValueError(
            "train/val/test比例之和必须等于1。"
        )

    # ========================================================
    # 按layout分组
    # ========================================================

    layout_records = (
        defaultdict(
            list
        )
    )

    for record in records:

        layout_records[
            record[
                "layout_id"
            ]
        ].append(
            record
        )

    if len(
        layout_records
    ) < 3:

        raise RuntimeError(
            "独立layout少于3个，"
            "无法建立train/val/test。"
        )

    groups = []

    for (
        layout_id,
        items,
    ) in layout_records.items():

        scene_type = (
            items[0][
                "scene_type"
            ]
        )

        groups.append(
            {
                "layout_id":
                    layout_id,

                "n_images":
                    len(
                        items
                    ),

                "scene_type":
                    scene_type,
            }
        )

    rng = random.Random(
        seed
    )

    # --------------------------------------------------------
    # 先随机，之后stable sort。
    #
    # 相同图片数量的layout保持随机顺序。
    # --------------------------------------------------------

    rng.shuffle(
        groups
    )

    groups.sort(
        key=lambda item:
            item[
                "n_images"
            ],
        reverse=True,
    )

    total_images = len(
        records
    )

    target_images = {

        split:
            total_images
            *
            ratio

        for (
            split,
            ratio
        ) in ratios.items()
    }

    # ========================================================
    # scene分布目标
    # ========================================================

    scene_total_layouts = Counter(
        group[
            "scene_type"
        ]
        for group
        in groups
    )

    target_scene = {

        split: {
            scene:
                count
                *
                ratios[
                    split
                ]

            for (
                scene,
                count
            ) in (
                scene_total_layouts.items()
            )
        }

        for split
        in ratios
    }

    current_images = {

        "train": 0,
        "val": 0,
        "test": 0,
    }

    current_scene = {

        split:
            Counter()

        for split
        in ratios
    }

    assignment = {}

    split_names = [
        "train",
        "val",
        "test",
    ]

    # ========================================================
    # 贪心分配
    #
    # 优先满足：
    #
    # 1. 当前split缺多少图片
    # 2. 当前scene在该split中缺多少layout
    # ========================================================

    for group in groups:

        scene = (
            group[
                "scene_type"
            ]
        )

        number_of_images = (
            group[
                "n_images"
            ]
        )

        candidate_scores = []

        for split in split_names:

            target_image_count = max(
                target_images[
                    split
                ],
                1.0,
            )

            image_need = (
                target_images[
                    split
                ]
                -
                current_images[
                    split
                ]
            ) / target_image_count

            target_scene_count = max(
                target_scene[
                    split
                ].get(
                    scene,
                    0.0,
                ),
                1.0,
            )

            scene_need = (
                target_scene[
                    split
                ].get(
                    scene,
                    0.0,
                )
                -
                current_scene[
                    split
                ][
                    scene
                ]
            ) / target_scene_count

            projected_images = (
                current_images[
                    split
                ]
                +
                number_of_images
            )

            over_ratio = max(
                0.0,
                (
                    projected_images
                    -
                    target_images[
                        split
                    ]
                )
                /
                target_image_count,
            )

            score = (
                1.0
                *
                image_need

                +
                0.25
                *
                scene_need

                -
                0.50
                *
                over_ratio

                +
                rng.random()
                *
                1e-9
            )

            candidate_scores.append(
                (
                    score,
                    split,
                )
            )

        candidate_scores.sort(
            reverse=True
        )

        chosen_split = (
            candidate_scores[
                0
            ][
                1
            ]
        )

        assignment[
            group[
                "layout_id"
            ]
        ] = (
            chosen_split
        )

        current_images[
            chosen_split
        ] += (
            number_of_images
        )

        current_scene[
            chosen_split
        ][
            scene
        ] += 1

    # ========================================================
    # 极端情况下保证三个集合都有layout
    # ========================================================

    layouts_per_split = {

        split: [
            layout_id

            for (
                layout_id,
                assigned_split
            ) in (
                assignment.items()
            )

            if (
                assigned_split
                ==
                split
            )
        ]

        for split
        in split_names
    }

    for empty_split in split_names:

        if layouts_per_split[
            empty_split
        ]:

            continue

        donor_split = max(
            split_names,
            key=lambda name:
                len(
                    layouts_per_split[
                        name
                    ]
                ),
        )

        if len(
            layouts_per_split[
                donor_split
            ]
        ) <= 1:

            raise RuntimeError(
                "无法保证三个split都包含layout。"
            )

        # ----------------------------------------------------
        # 找donor中图片最少的layout移动过去
        # ----------------------------------------------------

        donor_layout_ids = (
            layouts_per_split[
                donor_split
            ]
        )

        donor_layout_ids.sort(
            key=lambda layout_id:
                len(
                    layout_records[
                        layout_id
                    ]
                )
        )

        moved_layout = (
            donor_layout_ids[
                0
            ]
        )

        assignment[
            moved_layout
        ] = (
            empty_split
        )

        layouts_per_split[
            donor_split
        ].remove(
            moved_layout
        )

        layouts_per_split[
            empty_split
        ].append(
            moved_layout
        )

    return assignment


# ============================================================
# 创建目录
# ============================================================

def create_output_structure(
    output_root: Path,
) -> None:

    for split in (
        "train",
        "val",
        "test",
    ):

        (
            output_root
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output_root
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 生成正式数据集
# ============================================================

def materialize_dataset(
    records: list[dict],
    assignment: dict[str, str],
    output_root: Path,
) -> list[dict]:

    manifest_rows = []

    for record in records:

        split = (
            assignment[
                record[
                    "layout_id"
                ]
            ]
        )

        source_image = (
            record[
                "image_path"
            ]
        )

        destination_image = (
            output_root
            / "images"
            / split
            / record[
                "image"
            ]
        )

        destination_label = (
            output_root
            / "labels"
            / split
            / (
                record[
                    "stem"
                ]
                +
                ".txt"
            )
        )

        # ====================================================
        # 图片
        # ====================================================

        shutil.copy2(
            source_image,
            destination_image,
        )

        # ====================================================
        # 标签
        #
        # empty图片写空txt。
        # ====================================================

        destination_label.write_text(
            record[
                "label_text"
            ],
            encoding="utf-8",
        )

        manifest_rows.append(
            {

                "image":
                    record[
                        "image"
                    ],

                "split":
                    split,

                "layout_id":
                    record[
                        "layout_id"
                    ],

                "session_id":
                    record[
                        "session_id"
                    ],

                "scene_type":
                    record[
                        "scene_type"
                    ],

                "object_count":
                    record[
                        "object_count"
                    ],

                "instances":
                    record[
                        "instances"
                    ],

                "occlusion":
                    record[
                        "occlusion"
                    ],

                "edge":
                    record[
                        "edge"
                    ],

                "zone":
                    record[
                        "zone"
                    ],
            }
        )

    return manifest_rows


# ============================================================
# 保存 split_manifest.csv
# ============================================================

def write_manifest(
    output_root: Path,
    rows: list[dict],
) -> None:

    path = (
        output_root
        / "split_manifest.csv"
    )

    fieldnames = (
        "image",
        "split",
        "layout_id",
        "session_id",
        "scene_type",
        "object_count",
        "instances",
        "occlusion",
        "edge",
        "zone",
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# data.yaml
# ============================================================

def write_data_yaml(
    output_root: Path,
) -> None:

    # --------------------------------------------------------
    # 使用绝对路径。
    #
    # 当前在NUC上最稳定。
    #
    # 如果以后完整复制到Windows训练，
    # 再重新执行prepare脚本或修改这里的path即可。
    # --------------------------------------------------------

    dataset_path = (
        output_root
        .resolve()
        .as_posix()
    )

    content = (
        f'path: "{dataset_path}"\n'
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: {CLASS_NAME}\n"
    )

    (
        output_root
        / "data.yaml"
    ).write_text(
        content,
        encoding="utf-8",
    )


# ============================================================
# Dataset Summary
# ============================================================

def build_summary(
    manifest_rows: list[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    cvat_zip: Path,
    warnings: list[str],
) -> dict:

    summary = {

        "source_cvat_zip":
            str(
                cvat_zip
                .resolve()
            ),

        "random_seed":
            seed,

        "requested_ratios": {

            "train":
                train_ratio,

            "val":
                val_ratio,

            "test":
                test_ratio,
        },

        "total_images":
            len(
                manifest_rows
            ),

        "total_instances":
            sum(
                int(
                    row[
                        "instances"
                    ]
                )
                for row
                in manifest_rows
            ),

        "total_layouts":
            len(
                {
                    row[
                        "layout_id"
                    ]
                    for row
                    in manifest_rows
                }
            ),

        "warning_count":
            len(
                warnings
            ),

        "warnings":
            warnings,

        "splits": {},
    }

    for split in (
        "train",
        "val",
        "test",
    ):

        split_rows = [
            row
            for row
            in manifest_rows
            if (
                row[
                    "split"
                ]
                ==
                split
            )
        ]

        layouts = {
            row[
                "layout_id"
            ]
            for row
            in split_rows
        }

        scene_counter = Counter(
            row[
                "scene_type"
            ]
            for row
            in split_rows
        )

        occlusion_counter = Counter(
            row[
                "occlusion"
            ]
            for row
            in split_rows
        )

        edge_counter = Counter(
            row[
                "edge"
            ]
            for row
            in split_rows
        )

        summary[
            "splits"
        ][
            split
        ] = {

            "images":
                len(
                    split_rows
                ),

            "layouts":
                len(
                    layouts
                ),

            "instances":
                sum(
                    int(
                        row[
                            "instances"
                        ]
                    )
                    for row
                    in split_rows
                ),

            "scene_type":
                dict(
                    scene_counter
                ),

            "occlusion":
                dict(
                    occlusion_counter
                ),

            "edge":
                dict(
                    edge_counter
                ),
        }

    return summary


# ============================================================
# Layout泄漏检查
# ============================================================

def verify_no_layout_leakage(
    manifest_rows: list[dict],
) -> None:

    layout_sets = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        layout_sets[
            split
        ] = {
            row[
                "layout_id"
            ]

            for row
            in manifest_rows

            if (
                row[
                    "split"
                ]
                ==
                split
            )
        }

    train_val = (
        layout_sets[
            "train"
        ]
        &
        layout_sets[
            "val"
        ]
    )

    train_test = (
        layout_sets[
            "train"
        ]
        &
        layout_sets[
            "test"
        ]
    )

    val_test = (
        layout_sets[
            "val"
        ]
        &
        layout_sets[
            "test"
        ]
    )

    if (
        train_val
        or
        train_test
        or
        val_test
    ):

        raise RuntimeError(
            "检测到layout_id泄漏！\n"
            f"train-val: {sorted(train_val)}\n"
            f"train-test: {sorted(train_test)}\n"
            f"val-test: {sorted(val_test)}"
        )


# ============================================================
# 打印摘要
# ============================================================

def print_summary(
    summary: dict,
) -> None:

    print()

    print(
        "============================================================"
    )

    print(
        "TOP SEG DATASET PREPARED"
    )

    print(
        "============================================================"
    )

    print(
        f"Total images    : "
        f"{summary['total_images']}"
    )

    print(
        f"Total layouts   : "
        f"{summary['total_layouts']}"
    )

    print(
        f"Total instances : "
        f"{summary['total_instances']}"
    )

    print()

    for split in (
        "train",
        "val",
        "test",
    ):

        info = (
            summary[
                "splits"
            ][
                split
            ]
        )

        print(
            f"{split.upper():<6} "
            f"images={info['images']:<4} "
            f"layouts={info['layouts']:<4} "
            f"instances={info['instances']:<4}"
        )

    print()

    print(
        f"Warnings: "
        f"{summary['warning_count']}"
    )

    print(
        "============================================================"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare top-camera YOLO segmentation dataset "
            "using layout-level split."
        )
    )

    parser.add_argument(
        "--cvat-zip",
        type=Path,
        default=DEFAULT_CVAT_ZIP,
        help="CVAT Ultralytics YOLO Segmentation export ZIP",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Prepared dataset version, default: v1",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing prepared version and rebuild.",
    )

    args = (
        parser.parse_args()
    )

    cvat_zip = (
        args.cvat_zip
        .expanduser()
        .resolve()
    )

    output_root = (
        PREPARED_ROOT
        /
        args.version
    )

    temporary_root = (
        PREPARED_ROOT
        /
        (
            "."
            +
            args.version
            +
            "_building"
        )
    )

    # ========================================================
    # 基础文件检查
    # ========================================================

    required_paths = (

        RAW_IMAGE_DIR,
        IMAGES_CSV,
        LAYOUTS_CSV,
        cvat_zip,
    )

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"缺少必要文件/目录：\n{path}"
            )

    # ========================================================
    # 已存在版本保护
    # ========================================================

    if output_root.exists():

        if not args.force:

            raise RuntimeError(
                f"输出目录已经存在：\n"
                f"{output_root}\n\n"
                "如果确定要重新生成，运行时增加：\n"
                "--force"
            )

    if temporary_root.exists():

        shutil.rmtree(
            temporary_root
        )

    temporary_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        print(
            "============================================================"
        )

        print(
            "PREPARE TOP SEGMENTATION DATASET"
        )

        print(
            "============================================================"
        )

        print(
            f"CVAT ZIP:\n{cvat_zip}"
        )

        print()

        print(
            f"Output:\n{output_root}"
        )

        print()

        # ====================================================
        # CSV
        # ====================================================

        image_rows = (
            read_csv_rows(
                IMAGES_CSV
            )
        )

        layout_rows = (
            read_csv_rows(
                LAYOUTS_CSV
            )
        )

        image_index = (
            build_unique_index(
                image_rows,
                key="image",
                source_name="images.csv",
            )
        )

        layout_index = (
            build_unique_index(
                layout_rows,
                key="layout_id",
                source_name="layouts.csv",
            )
        )

        # ====================================================
        # CVAT
        # ====================================================

        (
            cvat_image_names,
            cvat_labels,
        ) = load_cvat_export(
            cvat_zip
        )

        print(
            f"CVAT images : "
            f"{len(cvat_image_names)}"
        )

        print(
            f"CVAT labels : "
            f"{len(cvat_labels)}"
        )

        # ====================================================
        # 建立记录并检查label
        # ====================================================

        (
            records,
            warnings,
        ) = build_records(
            image_names=
                cvat_image_names,

            cvat_labels=
                cvat_labels,

            image_index=
                image_index,

            layout_index=
                layout_index,
        )

        print(
            f"Validated   : "
            f"{len(records)} images"
        )

        # ====================================================
        # Split
        # ====================================================

        assignment = (
            split_by_layout(
                records=
                    records,

                train_ratio=
                    args.train_ratio,

                val_ratio=
                    args.val_ratio,

                test_ratio=
                    args.test_ratio,

                seed=
                    args.seed,
            )
        )

        # ====================================================
        # 建立临时输出
        # ====================================================

        create_output_structure(
            temporary_root
        )

        manifest_rows = (
            materialize_dataset(
                records=
                    records,

                assignment=
                    assignment,

                output_root=
                    temporary_root,
            )
        )

        # ====================================================
        # 泄漏检查
        # ====================================================

        verify_no_layout_leakage(
            manifest_rows
        )

        # ====================================================
        # Manifest
        # ====================================================

        write_manifest(
            temporary_root,
            manifest_rows,
        )

        # ====================================================
        # Summary
        # ====================================================

        summary = (
            build_summary(
                manifest_rows=
                    manifest_rows,

                seed=
                    args.seed,

                train_ratio=
                    args.train_ratio,

                val_ratio=
                    args.val_ratio,

                test_ratio=
                    args.test_ratio,

                cvat_zip=
                    cvat_zip,

                warnings=
                    warnings,
            )
        )

        (
            temporary_root
            /
            "summary.json"
        ).write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=4,
            ),
            encoding="utf-8",
        )

        # ====================================================
        # data.yaml
        #
        # 注意这里要等最终目录确定后生成。
        # ====================================================

        # 先删除旧版本
        if output_root.exists():

            shutil.rmtree(
                output_root
            )

        # 原子式替换
        temporary_root.rename(
            output_root
        )

        write_data_yaml(
            output_root
        )

        print_summary(
            summary
        )

        print()

        print(
            "Dataset saved:"
        )

        print(
            output_root
        )

        print()

        print(
            "Next:"
        )

        print(
            "python -m "
            "scripts.dataset.validate_top_seg_dataset"
        )

    except Exception:

        # ----------------------------------------------------
        # 构建失败时不留下半成品
        # ----------------------------------------------------

        if temporary_root.exists():

            shutil.rmtree(
                temporary_root
            )

        raise


if __name__ == "__main__":

    main()