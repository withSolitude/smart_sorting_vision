from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics

from collections import Counter
from collections import defaultdict

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 项目根目录
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


TOP_SEG_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
)


PREPARED_ROOT = (
    TOP_SEG_ROOT
    / "prepared"
)


# ============================================================
# 正式训练图尺寸
# ============================================================

EXPECTED_WIDTH = 870
EXPECTED_HEIGHT = 840


# ============================================================
# 当前单类别
# ============================================================

EXPECTED_CLASS_ID = 0


# ============================================================
# 极小Polygon面积提醒阈值
#
# 归一化面积。
#
# 这里只WARNING，不自动删除。
# ============================================================

MIN_POLYGON_AREA_NORMALIZED = 1e-5


# ============================================================
# 读取Manifest
# ============================================================

def read_manifest(
    path: Path,
) -> list[dict[str, str]]:

    if not path.exists():

        raise FileNotFoundError(
            f"找不到split_manifest.csv：\n{path}"
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
# SHA256
#
# 检查不同split是否出现完全相同图片。
# ============================================================

def calculate_sha256(
    path: Path,
) -> str:

    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024
                *
                1024
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return (
        digest.hexdigest()
    )


# ============================================================
# Polygon面积
#
# 输入：
#
# points:
# Nx2
#
# 坐标是归一化0~1
# ============================================================

def polygon_area(
    points: np.ndarray,
) -> float:

    x = points[
        :,
        0
    ]

    y = points[
        :,
        1
    ]

    area = (
        0.5
        *
        abs(
            np.dot(
                x,
                np.roll(
                    y,
                    1,
                )
            )
            -
            np.dot(
                y,
                np.roll(
                    x,
                    1,
                )
            )
        )
    )

    return float(
        area
    )


# ============================================================
# 解析单个Label
#
# 返回：
#
# [
#     {
#         class_id,
#         points,
#     }
# ]
# ============================================================

def parse_label(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Label不存在：\n{path}"
        )

    text = (
        path
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    if not text:

        return []

    objects = []

    for line_number, line in enumerate(
        text.splitlines(),
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

        if len(tokens) < 7:

            raise RuntimeError(
                f"{path.name}: "
                f"第{line_number}行 "
                "polygon至少需要3个点。"
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
                f"{path.name}: "
                f"第{line_number}行 "
                "坐标数量错误。"
            )

        try:

            class_id = int(
                tokens[0]
            )

        except ValueError as exc:

            raise RuntimeError(
                f"{path.name}: "
                f"第{line_number}行 "
                "class id无法解析。"
            ) from exc

        if (
            class_id
            !=
            EXPECTED_CLASS_ID
        ):

            raise RuntimeError(
                f"{path.name}: "
                f"发现非法类别 "
                f"{class_id}。"
            )

        try:

            values = np.asarray(
                [
                    float(
                        value
                    )
                    for value
                    in tokens[
                        1:
                    ]
                ],
                dtype=np.float64,
            )

        except ValueError as exc:

            raise RuntimeError(
                f"{path.name}: "
                f"第{line_number}行 "
                "坐标无法解析。"
            ) from exc

        if not np.all(
            np.isfinite(
                values
            )
        ):

            raise RuntimeError(
                f"{path.name}: "
                "存在NaN或Inf。"
            )

        if (
            np.any(
                values
                <
                0.0
            )
            or
            np.any(
                values
                >
                1.0
            )
        ):

            raise RuntimeError(
                f"{path.name}: "
                "存在超出[0,1]范围的坐标。"
            )

        points = (
            values
            .reshape(
                -1,
                2,
            )
        )

        if len(
            points
        ) < 3:

            raise RuntimeError(
                f"{path.name}: "
                "polygon点数少于3。"
            )

        objects.append(
            {

                "class_id":
                    class_id,

                "points":
                    points,
            }
        )

    return objects


# ============================================================
# 生成实例显示颜色
#
# 只是debug显示。
# ============================================================

def instance_color(
    index: int,
) -> tuple[int, int, int]:

    # BGR
    palette = (

        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 255, 0),
        (255, 128, 0),
    )

    return (
        palette[
            index
            %
            len(
                palette
            )
        ]
    )


# ============================================================
# 绘制Ground Truth
# ============================================================

def draw_ground_truth(
    image: np.ndarray,
    objects: list[dict],
) -> np.ndarray:

    result = (
        image.copy()
    )

    overlay = (
        image.copy()
    )

    height, width = (
        image.shape[
            :2
        ]
    )

    if not objects:

        cv2.putText(
            result,
            "EMPTY / NEGATIVE",
            (
                20,
                40,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (
                0,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        return result

    for index, obj in enumerate(
        objects
    ):

        normalized = (
            obj[
                "points"
            ]
        )

        points = (
            normalized.copy()
        )

        points[
            :,
            0
        ] *= (
            width
        )

        points[
            :,
            1
        ] *= (
            height
        )

        points = (
            np.round(
                points
            )
            .astype(
                np.int32
            )
        )

        color = (
            instance_color(
                index
            )
        )

        polygon = (
            points
            .reshape(
                -1,
                1,
                2,
            )
        )

        cv2.fillPoly(
            overlay,
            [
                polygon
            ],
            color,
        )

        cv2.polylines(
            result,
            [
                polygon
            ],
            isClosed=True,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        center = np.mean(
            points,
            axis=0,
        )

        center_x = int(
            center[
                0
            ]
        )

        center_y = int(
            center[
                1
            ]
        )

        cv2.putText(
            result,
            f"#{index + 1}",
            (
                center_x,
                center_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # 半透明Mask
    # ========================================================

    result = cv2.addWeighted(
        overlay,
        0.25,
        result,
        0.75,
        0.0,
    )

    return result


# ============================================================
# Main Validator
# ============================================================

def validate_dataset(
    dataset_root: Path,
    visualize_per_split: int,
    seed: int,
) -> dict:

    errors = []

    warnings = []

    manifest_path = (
        dataset_root
        /
        "split_manifest.csv"
    )

    data_yaml = (
        dataset_root
        /
        "data.yaml"
    )

    if not data_yaml.exists():

        errors.append(
            f"缺少data.yaml: {data_yaml}"
        )

    manifest = (
        read_manifest(
            manifest_path
        )
    )

    if not manifest:

        raise RuntimeError(
            "split_manifest.csv为空。"
        )

    # ========================================================
    # image唯一性
    # ========================================================

    image_names = [
        row[
            "image"
        ]
        for row
        in manifest
    ]

    duplicate_images = [
        image
        for (
            image,
            count
        ) in Counter(
            image_names
        ).items()
        if count > 1
    ]

    if duplicate_images:

        errors.append(
            "Manifest中图片重复："
            +
            str(
                duplicate_images
            )
        )

    # ========================================================
    # Layout泄漏检查
    # ========================================================

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
            in manifest

            if (
                row[
                    "split"
                ]
                ==
                split
            )
        }

    intersections = {

        "train-val":
            (
                layout_sets[
                    "train"
                ]
                &
                layout_sets[
                    "val"
                ]
            ),

        "train-test":
            (
                layout_sets[
                    "train"
                ]
                &
                layout_sets[
                    "test"
                ]
            ),

        "val-test":
            (
                layout_sets[
                    "val"
                ]
                &
                layout_sets[
                    "test"
                ]
            ),
    }

    for name, intersection in (
        intersections.items()
    ):

        if intersection:

            errors.append(
                f"{name} 存在layout泄漏："
                f"{sorted(intersection)}"
            )

    # ========================================================
    # Debug output
    # ========================================================

    debug_root = (
        ROOT
        / "outputs"
        / "debug"
        / "top_seg_ground_truth"
        / dataset_root.name
    )

    debug_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(
        seed
    )

    # ========================================================
    # 统计
    # ========================================================

    total_instances = 0

    polygon_point_counts = []

    polygon_areas = []

    split_statistics = {}

    hashes = defaultdict(
        list
    )

    instance_count_mismatches = []

    for split in (
        "train",
        "val",
        "test",
    ):

        split_rows = [
            row
            for row
            in manifest
            if (
                row[
                    "split"
                ]
                ==
                split
            )
        ]

        split_instance_count = 0

        split_layouts = {
            row[
                "layout_id"
            ]
            for row
            in split_rows
        }

        # ====================================================
        # 为可视化随机选样本
        # ====================================================

        visualization_names = set()

        if (
            visualize_per_split
            >
            0
            and
            split_rows
        ):

            selected = rng.sample(
                split_rows,
                k=min(
                    visualize_per_split,
                    len(
                        split_rows
                    ),
                ),
            )

            visualization_names = {
                row[
                    "image"
                ]
                for row
                in selected
            }

        split_debug_dir = (
            debug_root
            /
            split
        )

        split_debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for row in split_rows:

            image_name = (
                row[
                    "image"
                ]
            )

            stem = (
                Path(
                    image_name
                )
                .stem
            )

            image_path = (
                dataset_root
                / "images"
                / split
                / image_name
            )

            label_path = (
                dataset_root
                / "labels"
                / split
                / (
                    stem
                    +
                    ".txt"
                )
            )

            # ================================================
            # 图片存在
            # ================================================

            if not image_path.exists():

                errors.append(
                    f"缺少图片：{image_path}"
                )

                continue

            # ================================================
            # Label存在
            # ================================================

            if not label_path.exists():

                errors.append(
                    f"缺少Label：{label_path}"
                )

                continue

            # ================================================
            # OpenCV读取
            # ================================================

            image = cv2.imread(
                str(
                    image_path
                ),
                cv2.IMREAD_COLOR,
            )

            if image is None:

                errors.append(
                    f"OpenCV无法读取："
                    f"{image_path}"
                )

                continue

            height, width = (
                image.shape[
                    :2
                ]
            )

            if (
                width
                !=
                EXPECTED_WIDTH
                or
                height
                !=
                EXPECTED_HEIGHT
            ):

                errors.append(
                    f"{image_name}: "
                    f"尺寸错误 "
                    f"{width}x{height}, "
                    f"expected "
                    f"{EXPECTED_WIDTH}x"
                    f"{EXPECTED_HEIGHT}"
                )

            # ================================================
            # 完全重复图片检查
            # ================================================

            image_hash = (
                calculate_sha256(
                    image_path
                )
            )

            hashes[
                image_hash
            ].append(
                (
                    split,
                    image_name,
                )
            )

            # ================================================
            # Label解析
            # ================================================

            try:

                objects = (
                    parse_label(
                        label_path
                    )
                )

            except Exception as exc:

                errors.append(
                    f"{image_name}: "
                    f"{exc}"
                )

                continue

            instance_count = (
                len(
                    objects
                )
            )

            total_instances += (
                instance_count
            )

            split_instance_count += (
                instance_count
            )

            # ================================================
            # 与采集记录object_count比较
            # ================================================

            try:

                recorded_object_count = int(
                    row.get(
                        "object_count",
                        "0"
                    )
                )

            except ValueError:

                recorded_object_count = 0

            scene_type = (
                row.get(
                    "scene_type",
                    ""
                )
            )

            expected_empty = (
                scene_type
                ==
                "empty"
                or
                recorded_object_count
                ==
                0
            )

            if (
                expected_empty
                and
                instance_count
                !=
                0
            ):

                errors.append(
                    f"{image_name}: "
                    "empty布局却存在实例标注。"
                )

            if (
                not expected_empty
                and
                instance_count
                !=
                recorded_object_count
            ):

                instance_count_mismatches.append(
                    {
                        "image":
                            image_name,

                        "layout_id":
                            row[
                                "layout_id"
                            ],

                        "recorded_object_count":
                            recorded_object_count,

                        "label_instances":
                            instance_count,
                    }
                )

            # ================================================
            # Polygon质量
            # ================================================

            for object_index, obj in enumerate(
                objects,
                start=1,
            ):

                points = (
                    obj[
                        "points"
                    ]
                )

                number_of_points = (
                    len(
                        points
                    )
                )

                polygon_point_counts.append(
                    number_of_points
                )

                area = (
                    polygon_area(
                        points
                    )
                )

                polygon_areas.append(
                    area
                )

                if (
                    area
                    <
                    MIN_POLYGON_AREA_NORMALIZED
                ):

                    warnings.append(
                        f"{image_name} "
                        f"instance#{object_index}: "
                        f"polygon面积过小 "
                        f"{area:.8f}"
                    )

                if number_of_points > 200:

                    warnings.append(
                        f"{image_name} "
                        f"instance#{object_index}: "
                        f"{number_of_points}个polygon点，"
                        "轮廓非常密。"
                    )

                # ============================================
                # 连续重复点检查
                # ============================================

                for index in range(
                    1,
                    number_of_points,
                ):

                    if np.allclose(
                        points[
                            index
                        ],
                        points[
                            index
                            -
                            1
                        ],
                        atol=1e-12,
                    ):

                        warnings.append(
                            f"{image_name} "
                            f"instance#{object_index}: "
                            "存在连续重复polygon点。"
                        )

                        break

            # ================================================
            # Ground Truth可视化
            # ================================================

            if (
                image_name
                in
                visualization_names
            ):

                visualization = (
                    draw_ground_truth(
                        image,
                        objects,
                    )
                )

                output_file = (
                    split_debug_dir
                    /
                    (
                        stem
                        +
                        "_gt.jpg"
                    )
                )

                cv2.imwrite(
                    str(
                        output_file
                    ),
                    visualization,
                )

        split_statistics[
            split
        ] = {

            "images":
                len(
                    split_rows
                ),

            "layouts":
                len(
                    split_layouts
                ),

            "instances":
                split_instance_count,
        }

    # ========================================================
    # 跨split完全相同图片
    # ========================================================

    for image_hash, locations in (
        hashes.items()
    ):

        if len(
            locations
        ) <= 1:

            continue

        split_names = {
            split
            for (
                split,
                _
            )
            in locations
        }

        if len(
            split_names
        ) > 1:

            errors.append(
                "发现完全相同图片跨split："
                f"{locations}"
            )

    # ========================================================
    # 检查目录里是否存在Manifest之外的图片/标签
    # ========================================================

    manifest_image_set = {
        (
            row[
                "split"
            ],
            row[
                "image"
            ],
        )
        for row
        in manifest
    }

    for split in (
        "train",
        "val",
        "test",
    ):

        image_dir = (
            dataset_root
            / "images"
            / split
        )

        label_dir = (
            dataset_root
            / "labels"
            / split
        )

        if not image_dir.exists():

            errors.append(
                f"目录不存在：{image_dir}"
            )

            continue

        if not label_dir.exists():

            errors.append(
                f"目录不存在：{label_dir}"
            )

            continue

        for image_path in image_dir.iterdir():

            if not image_path.is_file():

                continue

            if (
                (
                    split,
                    image_path.name,
                )
                not in
                manifest_image_set
            ):

                warnings.append(
                    "存在Manifest之外的图片："
                    f"{image_path}"
                )

    # ========================================================
    # Polygon统计
    # ========================================================

    polygon_statistics = {}

    if polygon_point_counts:

        polygon_statistics = {

            "minimum_points":
                min(
                    polygon_point_counts
                ),

            "median_points":
                float(
                    statistics.median(
                        polygon_point_counts
                    )
                ),

            "maximum_points":
                max(
                    polygon_point_counts
                ),

            "minimum_normalized_area":
                min(
                    polygon_areas
                ),

            "median_normalized_area":
                float(
                    statistics.median(
                        polygon_areas
                    )
                ),

            "maximum_normalized_area":
                max(
                    polygon_areas
                ),
        }

    report = {

        "dataset_root":
            str(
                dataset_root
                .resolve()
            ),

        "total_images":
            len(
                manifest
            ),

        "total_layouts":
            len(
                {
                    row[
                        "layout_id"
                    ]
                    for row
                    in manifest
                }
            ),

        "total_instances":
            total_instances,

        "splits":
            split_statistics,

        "polygon_statistics":
            polygon_statistics,

        "instance_count_mismatches":
            instance_count_mismatches,

        "errors":
            errors,

        "warnings":
            warnings,

        "status":
            (
                "FAILED"
                if errors
                else
                "PASSED"
            ),
    }

    # ========================================================
    # 保存报告
    # ========================================================

    report_path = (
        dataset_root
        /
        "validation_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    return report


# ============================================================
# 打印报告
# ============================================================

def print_report(
    report: dict,
) -> None:

    print()

    print(
        "============================================================"
    )

    print(
        "TOP SEG DATASET VALIDATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Status          : "
        f"{report['status']}"
    )

    print(
        f"Images          : "
        f"{report['total_images']}"
    )

    print(
        f"Layouts         : "
        f"{report['total_layouts']}"
    )

    print(
        f"Instances       : "
        f"{report['total_instances']}"
    )

    print()

    for split in (
        "train",
        "val",
        "test",
    ):

        info = (
            report[
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

    if report[
        "polygon_statistics"
    ]:

        stats = (
            report[
                "polygon_statistics"
            ]
        )

        print()

        print(
            "Polygon points:"
        )

        print(
            f"min    = "
            f"{stats['minimum_points']}"
        )

        print(
            f"median = "
            f"{stats['median_points']}"
        )

        print(
            f"max    = "
            f"{stats['maximum_points']}"
        )

    print()

    print(
        f"Instance count mismatches: "
        f"{len(report['instance_count_mismatches'])}"
    )

    print(
        f"Warnings                : "
        f"{len(report['warnings'])}"
    )

    print(
        f"Errors                  : "
        f"{len(report['errors'])}"
    )

    # ========================================================
    # Errors
    # ========================================================

    if report[
        "errors"
    ]:

        print()

        print(
            "------------------------------------------------------------"
        )

        print(
            "ERRORS"
        )

        print(
            "------------------------------------------------------------"
        )

        for error in (
            report[
                "errors"
            ]
        ):

            print(
                f"[ERROR] {error}"
            )

    # ========================================================
    # Warnings只显示前20条
    # ========================================================

    if report[
        "warnings"
    ]:

        print()

        print(
            "------------------------------------------------------------"
        )

        print(
            "WARNINGS (first 20)"
        )

        print(
            "------------------------------------------------------------"
        )

        for warning in (
            report[
                "warnings"
            ][
                :20
            ]
        ):

            print(
                f"[WARNING] {warning}"
            )

        if len(
            report[
                "warnings"
            ]
        ) > 20:

            print(
                "..."
            )

            print(
                "完整warning请查看 "
                "validation_report.json"
            )

    print()

    print(
        "============================================================"
    )

    print(
        "Ground Truth preview:"
    )

    print(
        ROOT
        / "outputs"
        / "debug"
        / "top_seg_ground_truth"
        / Path(
            report[
                "dataset_root"
            ]
        ).name
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
            "Strict validation for "
            "top-camera YOLO segmentation dataset."
        )
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1",
    )

    parser.add_argument(
        "--visualize-per-split",
        type=int,
        default=5,
        help=(
            "Number of random GT previews "
            "to generate for each split."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = (
        parser.parse_args()
    )

    dataset_root = (
        PREPARED_ROOT
        /
        args.version
    )

    if not dataset_root.exists():

        raise FileNotFoundError(
            "Prepared dataset不存在：\n"
            f"{dataset_root}\n\n"
            "请先运行：\n"
            "python -m "
            "scripts.dataset.prepare_top_seg_dataset"
        )

    report = (
        validate_dataset(
            dataset_root=
                dataset_root,

            visualize_per_split=
                args.visualize_per_split,

            seed=
                args.seed,
        )
    )

    print_report(
        report
    )

    if report[
        "errors"
    ]:

        raise SystemExit(
            1
        )


if __name__ == "__main__":

    main()