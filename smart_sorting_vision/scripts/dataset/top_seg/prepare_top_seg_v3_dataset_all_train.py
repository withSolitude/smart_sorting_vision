from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


# ============================================================
# 工程根目录
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 输入
# ============================================================

# 已经冻结的 V2 prepared 数据
DEFAULT_V2_PREPARED_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "prepared"
    / "v2"
)

# V3 原始图片
DEFAULT_V3_IMAGE_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "new_images_v3"
)

# 从 CVAT 导出的 V3 人工修正标注 ZIP
DEFAULT_V3_CVAT_ZIP = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "cvat_exports_v3"
    / "top_seg_v3.zip"
)

# 输出
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "prepared"
    / "v3"
)


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

KNOWN_SCENES = {
    "single",
    "mixed",
    "contact",
    "occlusion",
    "stacked",
    "edge",
    "empty",
}


# ============================================================
# SHA256
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# 查找图片
# ============================================================

def list_images(
    root: Path,
) -> list[Path]:

    images = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    ]

    images.sort(
        key=lambda path:
        path.relative_to(
            root
        ).as_posix()
    )

    return images


# ============================================================
# 检查 stem 唯一
# ============================================================

def ensure_unique_stems(
    paths: list[Path],
    description: str,
) -> dict[str, Path]:

    mapping: dict[str, Path] = {}

    duplicates: dict[
        str,
        list[Path],
    ] = defaultdict(list)

    for path in paths:

        stem = path.stem

        if stem in mapping:

            if not duplicates[
                stem
            ]:

                duplicates[
                    stem
                ].append(
                    mapping[
                        stem
                    ]
                )

            duplicates[
                stem
            ].append(
                path
            )

        else:

            mapping[
                stem
            ] = path

    if duplicates:

        lines = [
            f"{description}存在重复 stem，不能继续："
        ]

        for stem, items in sorted(
            duplicates.items()
        ):

            lines.append(
                f"\n[{stem}]"
            )

            for item in items:

                lines.append(
                    f"  {item}"
                )

        raise RuntimeError(
            "\n".join(
                lines
            )
        )

    return mapping


# ============================================================
# V3 scene
# ============================================================

def get_v3_scene(
    image_path: Path,
    image_root: Path,
) -> str:

    relative = (
        image_path.relative_to(
            image_root
        )
    )

    if len(
        relative.parts
    ) < 2:

        raise RuntimeError(
            "V3图片没有位于scene目录：\n"
            f"{relative}"
        )

    scene = (
        relative.parts[0]
    )

    if scene not in KNOWN_SCENES:

        raise RuntimeError(
            "未知V3场景目录：\n"
            f"{scene}\n"
            f"允许："
            f"{sorted(KNOWN_SCENES)}"
        )

    return scene


# ============================================================
# YOLO Seg 标签检查
# ============================================================

def validate_seg_text(
    text: str,
    source: str,
) -> int:

    stripped = (
        text.strip()
    )

    if not stripped:

        return 0

    instance_count = 0

    for line_number, raw_line in enumerate(
        text.splitlines(),
        start=1,
    ):

        line = (
            raw_line.strip()
        )

        if not line:
            continue

        tokens = (
            line.split()
        )

        # class + 至少3个点
        if len(tokens) < 7:

            raise RuntimeError(
                f"{source} "
                f"line {line_number}: "
                "字段数量不足"
            )

        if (
            len(tokens) - 1
        ) % 2 != 0:

            raise RuntimeError(
                f"{source} "
                f"line {line_number}: "
                "Polygon坐标数量不是偶数"
            )

        try:

            class_id = int(
                tokens[0]
            )

        except ValueError as exc:

            raise RuntimeError(
                f"{source} "
                f"line {line_number}: "
                "class id不是整数"
            ) from exc

        if class_id != 0:

            raise RuntimeError(
                f"{source} "
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

        except ValueError as exc:

            raise RuntimeError(
                f"{source} "
                f"line {line_number}: "
                "坐标解析失败"
            ) from exc

        if len(
            coordinates
        ) < 6:

            raise RuntimeError(
                f"{source} "
                f"line {line_number}: "
                "Polygon少于3个点"
            )

        for value in coordinates:

            if not (
                0.0
                <= value
                <= 1.0
            ):

                raise RuntimeError(
                    f"{source} "
                    f"line {line_number}: "
                    "存在超出[0,1]的坐标"
                )

        instance_count += 1

    return instance_count


# ============================================================
# 读取 CVAT V3 标签
#
# CVAT Ultralytics YOLO Segmentation 导出通常为：
#
# labels/train/*.txt
# train.txt
#
# empty 图片可能没有空txt，
# 这种情况后面根据 new_images_v3/empty 自动补空标签。
# ============================================================

def load_cvat_labels(
    cvat_zip: Path,
) -> tuple[
    dict[str, str],
    set[str],
]:

    labels: dict[
        str,
        str,
    ] = {}

    train_stems: set[
        str
    ] = set()

    with ZipFile(
        cvat_zip,
        "r",
    ) as archive:

        names = (
            archive.namelist()
        )

        for name in names:

            path = Path(
                name
            )

            if (
                len(path.parts) >= 3
                and
                path.parts[-3:-1]
                ==
                ("labels", "train")
                and
                path.suffix.lower()
                ==
                ".txt"
            ):

                stem = (
                    path.stem
                )

                if stem in labels:

                    raise RuntimeError(
                        "CVAT ZIP中重复标签stem："
                        f"{stem}"
                    )

                text = (
                    archive.read(
                        name
                    )
                    .decode(
                        "utf-8-sig"
                    )
                )

                validate_seg_text(
                    text,
                    f"{cvat_zip.name}:{name}",
                )

                labels[
                    stem
                ] = text

        if "train.txt" in names:

            text = (
                archive.read(
                    "train.txt"
                )
                .decode(
                    "utf-8-sig"
                )
            )

            for raw_line in (
                text.splitlines()
            ):

                line = (
                    raw_line.strip()
                )

                if not line:
                    continue

                train_stems.add(
                    Path(
                        line
                    ).stem
                )

    if not labels:

        raise RuntimeError(
            "CVAT ZIP中没有找到"
            " labels/train/*.txt"
        )

    return (
        labels,
        train_stems,
    )


# ============================================================
# 写一个 image + label
# ============================================================

def copy_pair(
    image_path: Path,
    label_text: str,
    output_root: Path,
    split: str,
) -> tuple[
    Path,
    Path,
]:

    image_dir = (
        output_root
        / "images"
        / split
    )

    label_dir = (
        output_root
        / "labels"
        / split
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_image = (
        image_dir
        / image_path.name
    )

    destination_label = (
        label_dir
        / f"{image_path.stem}.txt"
    )

    if (
        destination_image.exists()
        or
        destination_label.exists()
    ):

        raise RuntimeError(
            "输出文件冲突：\n"
            f"{destination_image}\n"
            f"{destination_label}"
        )

    shutil.copy2(
        image_path,
        destination_image,
    )

    destination_label.write_text(
        label_text,
        encoding="utf-8",
    )

    return (
        destination_image,
        destination_label,
    )


# ============================================================
# 复制 prepared/v2 的固定 split
# ============================================================

def copy_v2_split(
    v2_root: Path,
    source_split: str,
    output_root: Path,
    output_split: str,
) -> list[dict]:

    source_image_dir = (
        v2_root
        / "images"
        / source_split
    )

    source_label_dir = (
        v2_root
        / "labels"
        / source_split
    )

    if not source_image_dir.exists():

        raise FileNotFoundError(
            f"找不到V2图片split："
            f"{source_image_dir}"
        )

    if not source_label_dir.exists():

        raise FileNotFoundError(
            f"找不到V2标签split："
            f"{source_label_dir}"
        )

    images = [
        path
        for path
        in source_image_dir.iterdir()
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    ]

    images.sort(
        key=lambda path:
        path.name
    )

    records = []

    for image_path in images:

        label_path = (
            source_label_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            raise RuntimeError(
                "V2 prepared图片缺少标签：\n"
                f"{image_path}"
            )

        label_text = (
            label_path.read_text(
                encoding="utf-8-sig"
            )
        )

        instances = (
            validate_seg_text(
                label_text,
                str(label_path),
            )
        )

        destination_image, destination_label = (
            copy_pair(
                image_path=
                    image_path,

                label_text=
                    label_text,

                output_root=
                    output_root,

                split=
                    output_split,
            )
        )

        records.append(
            {
                "image_name":
                    image_path.name,

                "split":
                    output_split,

                "source_generation":
                    "v2_prepared",

                "source_split":
                    source_split,

                "scene":
                    "",

                "instances":
                    instances,

                "sha256":
                    sha256_file(
                        image_path
                    ),

                "output_image":
                    destination_image,

                "output_label":
                    destination_label,
            }
        )

    return records


# ============================================================
# YAML
# ============================================================

def write_yaml(
    output_path: Path,
    dataset_root: Path,
    test_split: str,
) -> None:

    text = (
        f"path: "
        f"{dataset_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"test: images/{test_split}\n"
        "names:\n"
        "  0: object\n"
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--v2-prepared-root",
        type=Path,
        default=
            DEFAULT_V2_PREPARED_ROOT,
    )

    parser.add_argument(
        "--v3-image-root",
        type=Path,
        default=
            DEFAULT_V3_IMAGE_ROOT,
    )

    parser.add_argument(
        "--cvat-zip",
        type=Path,
        default=
            DEFAULT_V3_CVAT_ZIP,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=
            DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    v2_root = (
        args.v2_prepared_root
        .resolve()
    )

    v3_root = (
        args.v3_image_root
        .resolve()
    )

    cvat_zip = (
        args.cvat_zip
        .resolve()
    )

    output_root = (
        args.output_root
        .resolve()
    )

    print(
        "=" * 70
    )

    print(
        "PREPARE TOP-SEG V3"
    )

    print(
        "POLICY:"
    )

    print(
        "  train = V2 train + ALL V3"
    )

    print(
        "  val = V2 val"
    )

    print(
        "  test = V2 test"
    )

    print(
        "  legacy_test = V1 legacy test"
    )

    print(
        "=" * 70
    )

    print(
        f"V2 prepared : {v2_root}"
    )

    print(
        f"V3 images   : {v3_root}"
    )

    print(
        f"CVAT ZIP    : {cvat_zip}"
    )

    print(
        f"Output      : {output_root}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 基础检查
    # ========================================================

    if not v2_root.exists():

        raise FileNotFoundError(
            f"找不到prepared/v2："
            f"{v2_root}"
        )

    if not v3_root.exists():

        raise FileNotFoundError(
            f"找不到new_images_v3："
            f"{v3_root}"
        )

    if not cvat_zip.exists():

        raise FileNotFoundError(
            "找不到CVAT导出的V3 ZIP：\n"
            f"{cvat_zip}"
        )

    if output_root.exists():

        if not args.overwrite:

            raise RuntimeError(
                "输出目录已经存在：\n"
                f"{output_root}\n"
                "如果确认要重新生成，"
                "运行时加 --overwrite"
            )

        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 读取V3图片
    # ========================================================

    v3_images = (
        list_images(
            v3_root
        )
    )

    if not v3_images:

        raise RuntimeError(
            "new_images_v3中没有图片"
        )

    v3_image_map = (
        ensure_unique_stems(
            v3_images,
            "V3图片",
        )
    )

    # ========================================================
    # 2. 读取CVAT人工修正标签
    # ========================================================

    (
        cvat_labels,
        train_stems,
    ) = load_cvat_labels(
        cvat_zip
    )

    # train.txt存在时严格检查
    if train_stems:

        missing = sorted(
            set(
                v3_image_map
            )
            -
            train_stems
        )

        extra = sorted(
            train_stems
            -
            set(
                v3_image_map
            )
        )

        if missing:

            raise RuntimeError(
                "以下V3图片没有出现在"
                " CVAT train.txt：\n"
                +
                "\n".join(
                    missing
                )
            )

        if extra:

            raise RuntimeError(
                "CVAT train.txt存在"
                " new_images_v3中没有的图片：\n"
                +
                "\n".join(
                    extra
                )
            )

    # ========================================================
    # 3. 建立V3人工GT
    #
    # V3全部进入train。
    # ========================================================

    print()
    print(
        "LOAD V3 CHECKED GT"
    )

    v3_records = []

    empty_labels_created = []

    scene_counter = (
        Counter()
    )

    scene_instance_counter = (
        Counter()
    )

    for image_path in v3_images:

        stem = (
            image_path.stem
        )

        scene = (
            get_v3_scene(
                image_path,
                v3_root,
            )
        )

        if stem in cvat_labels:

            label_text = (
                cvat_labels[
                    stem
                ]
            )

        else:

            # CVAT可能不为empty导出空txt
            if scene == "empty":

                label_text = ""

                empty_labels_created.append(
                    image_path
                )

            else:

                raise RuntimeError(
                    "非empty V3图片缺少"
                    " CVAT人工标签：\n"
                    f"{image_path.relative_to(v3_root)}"
                )

        instances = (
            validate_seg_text(
                label_text,
                f"V3:{image_path.name}",
            )
        )

        if (
            scene == "empty"
            and
            instances != 0
        ):

            raise RuntimeError(
                "empty场景存在非空人工标签：\n"
                f"{image_path.relative_to(v3_root)}"
            )

        scene_counter[
            scene
        ] += 1

        scene_instance_counter[
            scene
        ] += instances

        v3_records.append(
            {
                "image_path":
                    image_path,

                "label_text":
                    label_text,

                "scene":
                    scene,

                "instances":
                    instances,

                "sha256":
                    sha256_file(
                        image_path
                    ),
            }
        )

    print(
        f"V3 images          : "
        f"{len(v3_records)}"
    )

    print(
        f"V3 instances       : "
        f"{sum(record['instances'] for record in v3_records)}"
    )

    print(
        f"Empty labels added : "
        f"{len(empty_labels_created)}"
    )

    print()

    for scene in sorted(
        scene_counter
    ):

        print(
            f"{scene:<12} "
            f"images={scene_counter[scene]:4d} "
            f"instances={scene_instance_counter[scene]:4d}"
        )

    # ========================================================
    # 4. SHA256检查
    #
    # 检查V3是否和V2已有数据完全重复。
    # ========================================================

    print()
    print(
        "SHA256 DUPLICATE CHECK"
    )

    history_hashes: dict[
        str,
        Path,
    ] = {}

    for split in (
        "train",
        "val",
        "test",
        "legacy_test",
    ):

        directory = (
            v2_root
            / "images"
            / split
        )

        if not directory.exists():

            raise FileNotFoundError(
                "prepared/v2缺少split：\n"
                f"{directory}"
            )

        for path in directory.iterdir():

            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_SUFFIXES
            ):

                value = (
                    sha256_file(
                        path
                    )
                )

                if value in history_hashes:

                    raise RuntimeError(
                        "prepared/v2内部存在"
                        "完全重复图像：\n"
                        f"{history_hashes[value]}\n"
                        f"{path}"
                    )

                history_hashes[
                    value
                ] = path

    for record in v3_records:

        value = (
            record[
                "sha256"
            ]
        )

        if value in history_hashes:

            raise RuntimeError(
                "V3与历史数据存在"
                "完全重复图像：\n"
                f"历史："
                f"{history_hashes[value]}\n"
                f"V3："
                f"{record['image_path']}"
            )

    print(
        "SHA256 duplicate check: PASSED"
    )

    # ========================================================
    # 5. 构建 prepared/v3
    # ========================================================

    manifest_records = []

    # --------------------------------------------------------
    # V2 train -> train
    # --------------------------------------------------------

    print()
    print(
        "COPY V2 TRAIN"
    )

    manifest_records.extend(
        copy_v2_split(
            v2_root=
                v2_root,

            source_split=
                "train",

            output_root=
                output_root,

            output_split=
                "train",
        )
    )

    # --------------------------------------------------------
    # V3全部 -> train
    # --------------------------------------------------------

    print(
        "COPY ALL V3 -> TRAIN"
    )

    for record in v3_records:

        image_path = (
            record[
                "image_path"
            ]
        )

        destination_image, destination_label = (
            copy_pair(
                image_path=
                    image_path,

                label_text=
                    record[
                        "label_text"
                    ],

                output_root=
                    output_root,

                split=
                    "train",
            )
        )

        manifest_records.append(
            {
                "image_name":
                    image_path.name,

                "split":
                    "train",

                "source_generation":
                    "v3",

                "source_split":
                    "all_to_train",

                "scene":
                    record[
                        "scene"
                    ],

                "instances":
                    record[
                        "instances"
                    ],

                "sha256":
                    record[
                        "sha256"
                    ],

                "output_image":
                    destination_image,

                "output_label":
                    destination_label,
            }
        )

    # --------------------------------------------------------
    # V2 val -> val
    # --------------------------------------------------------

    print(
        "COPY V2 VAL"
    )

    manifest_records.extend(
        copy_v2_split(
            v2_root=
                v2_root,

            source_split=
                "val",

            output_root=
                output_root,

            output_split=
                "val",
        )
    )

    # --------------------------------------------------------
    # V2 test -> test
    # --------------------------------------------------------

    print(
        "COPY V2 TEST"
    )

    manifest_records.extend(
        copy_v2_split(
            v2_root=
                v2_root,

            source_split=
                "test",

            output_root=
                output_root,

            output_split=
                "test",
        )
    )

    # --------------------------------------------------------
    # V1 legacy test保持
    # --------------------------------------------------------

    print(
        "COPY LEGACY TEST"
    )

    manifest_records.extend(
        copy_v2_split(
            v2_root=
                v2_root,

            source_split=
                "legacy_test",

            output_root=
                output_root,

            output_split=
                "legacy_test",
        )
    )

    # ========================================================
    # 6. 最终一致性检查
    # ========================================================

    print()
    print(
        "FINAL CHECK"
    )

    final_counts = {}

    for split in (
        "train",
        "val",
        "test",
        "legacy_test",
    ):

        image_dir = (
            output_root
            / "images"
            / split
        )

        label_dir = (
            output_root
            / "labels"
            / split
        )

        images = [
            path
            for path
            in image_dir.iterdir()
            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_SUFFIXES
            )
        ]

        labels = list(
            label_dir.glob(
                "*.txt"
            )
        )

        image_stems = {
            path.stem
            for path in images
        }

        label_stems = {
            path.stem
            for path in labels
        }

        if (
            image_stems
            !=
            label_stems
        ):

            raise RuntimeError(
                f"{split} "
                "images/labels不一致"
            )

        final_counts[
            split
        ] = len(
            images
        )

        print(
            f"{split:<12} "
            f"images={len(images):4d} "
            f"labels={len(labels):4d}"
        )

    # ========================================================
    # 7. YAML
    # ========================================================

    write_yaml(
        output_path=
            output_root
            / "data.yaml",

        dataset_root=
            output_root,

        test_split=
            "test",
    )

    write_yaml(
        output_path=
            output_root
            / "data_legacy_test.yaml",

        dataset_root=
            output_root,

        test_split=
            "legacy_test",
    )

    # ========================================================
    # 8. Manifest
    # ========================================================

    manifest_path = (
        output_root
        / "split_manifest.csv"
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
                "split",
                "source_generation",
                "source_split",
                "scene",
                "instances",
                "sha256",
            ]
        )

        for record in sorted(
            manifest_records,
            key=lambda item: (
                item[
                    "split"
                ],
                item[
                    "image_name"
                ],
            ),
        ):

            writer.writerow(
                [
                    record[
                        "image_name"
                    ],

                    record[
                        "split"
                    ],

                    record[
                        "source_generation"
                    ],

                    record[
                        "source_split"
                    ],

                    record[
                        "scene"
                    ],

                    record[
                        "instances"
                    ],

                    record[
                        "sha256"
                    ],
                ]
            )

    # ========================================================
    # 9. Summary
    # ========================================================

    summary = {
        "policy": {
            "train":
                "prepared/v2 train + ALL V3 checked images",

            "val":
                "prepared/v2 val",

            "test":
                "prepared/v2 original test",

            "legacy_test":
                "prepared/v2 legacy_test",

            "v3_split":
                "none; all V3 samples are used for training",
        },

        "v3": {
            "images":
                len(
                    v3_records
                ),

            "instances":
                sum(
                    record[
                        "instances"
                    ]
                    for record
                    in v3_records
                ),

            "empty_labels_created":
                len(
                    empty_labels_created
                ),

            "scene_images":
                dict(
                    sorted(
                        scene_counter.items()
                    )
                ),

            "scene_instances":
                dict(
                    sorted(
                        scene_instance_counter.items()
                    )
                ),
        },

        "final_split_images":
            final_counts,
    }

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Done
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "PREPARED/V3 GENERATED"
    )

    print(
        "=" * 70
    )

    print(
        f"data.yaml       : "
        f"{output_root / 'data.yaml'}"
    )

    print(
        f"legacy yaml     : "
        f"{output_root / 'data_legacy_test.yaml'}"
    )

    print(
        f"manifest        : "
        f"{manifest_path}"
    )

    print(
        f"summary         : "
        f"{output_root / 'summary.json'}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()
