from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import zipfile

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 工程目录
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# 输入
# ============================================================

V1_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "prepared"
    / "v1"
)


V2_IMAGE_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "new_images"
)


V2_CHECKED_ZIP = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "cvat_exports"
    / "top_seg_v2.zip"
)


# ============================================================
# 输出
# ============================================================

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "prepared"
    / "v2"
)


# ============================================================
# 图像参数
# ============================================================

EXPECTED_WIDTH = 870
EXPECTED_HEIGHT = 840


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# ============================================================
# Split
#
# 只针对V2新增数据：
#
# train 70%
# val   15%
# test  15%
#
# V1原有split不改变。
# ============================================================

VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 2026


# ============================================================
# 类别
# ============================================================

CLASS_ID = 0
CLASS_NAME = "object"


# ============================================================
# 数据项
# ============================================================

@dataclass
class DataItem:

    name: str

    image_path: Path

    label_text: str

    source_version: str

    original_split: str

    scene: str

    color: str

    shape: str

    source_relative_path: str

    instance_count: int

    sha256: str = ""


# ============================================================
# Windows中文路径安全读取
# ============================================================

def imread_unicode(
    path: Path,
) -> np.ndarray:

    if not path.exists():

        raise FileNotFoundError(
            f"图片不存在：\n{path}"
        )

    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    if data.size == 0:

        raise RuntimeError(
            f"图片文件为空：\n{path}"
        )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise RuntimeError(
            "图片无法解码，可能已损坏：\n"
            f"{path}"
        )

    return image


# ============================================================
# SHA256
# ============================================================

def calculate_sha256(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            hasher.update(
                block
            )

    return hasher.hexdigest()


# ============================================================
# 图片验证
# ============================================================

def validate_image(
    path: Path,
) -> None:

    image = imread_unicode(
        path
    )

    height, width = image.shape[:2]

    if (
        width != EXPECTED_WIDTH
        or
        height != EXPECTED_HEIGHT
    ):

        raise RuntimeError(
            "图片尺寸错误：\n"
            f"{path}\n"
            f"实际：{width}x{height}\n"
            f"期望："
            f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
        )


# ============================================================
# YOLO Segmentation Label验证
#
# 返回实例数量
# ============================================================

def validate_label_text(
    text: str,
    source_name: str,
) -> int:

    text = text.strip()

    if not text:

        return 0

    instance_count = 0

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        tokens = line.split()

        # class + 至少3个二维点
        if len(tokens) < 7:

            raise RuntimeError(
                f"{source_name}\n"
                f"第{line_number}行Polygon不足3点"
            )

        if (
            len(tokens) - 1
        ) % 2 != 0:

            raise RuntimeError(
                f"{source_name}\n"
                f"第{line_number}行坐标数不是偶数"
            )

        try:

            class_id = int(
                tokens[0]
            )

        except ValueError as exc:

            raise RuntimeError(
                f"{source_name}\n"
                f"第{line_number}行class非法"
            ) from exc

        if class_id != CLASS_ID:

            raise RuntimeError(
                f"{source_name}\n"
                f"发现class={class_id}，"
                "当前只能为class=0"
            )

        try:

            coordinates = [
                float(value)
                for value
                in tokens[1:]
            ]

        except ValueError as exc:

            raise RuntimeError(
                f"{source_name}\n"
                f"第{line_number}行存在非法数字"
            ) from exc

        point_count = (
            len(coordinates)
            //
            2
        )

        if point_count < 3:

            raise RuntimeError(
                f"{source_name}\n"
                f"Polygon只有{point_count}个点"
            )

        for value in coordinates:

            if not math.isfinite(
                value
            ):

                raise RuntimeError(
                    f"{source_name}\n"
                    "存在NaN/Inf"
                )

            if not (
                0.0
                <=
                value
                <=
                1.0
            ):

                raise RuntimeError(
                    f"{source_name}\n"
                    f"坐标超出[0,1]：{value}"
                )

        instance_count += 1

    return instance_count


# ============================================================
# 查找图片
# ============================================================

def find_images(
    root: Path,
) -> list[Path]:

    paths = []

    for path in root.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_SUFFIXES
        ):

            paths.append(
                path
            )

    paths.sort(
        key=lambda path:
        path.as_posix()
    )

    return paths


# ============================================================
# 解析V2目录信息
#
# single/black/圆柱/top_v2_xxx.jpg
#
# ->
#
# scene = single
# color = black
# shape = 圆柱
#
#
# contact/top_v2_xxx.jpg
#
# ->
#
# scene = contact
# ============================================================

def parse_v2_metadata(
    image_path: Path,
) -> tuple[
    str,
    str,
    str,
    str,
]:

    relative = image_path.relative_to(
        V2_IMAGE_ROOT
    )

    parts = relative.parts

    if not parts:

        raise RuntimeError(
            f"无法解析V2路径：{image_path}"
        )

    scene = parts[0]

    color = ""

    shape = ""

    if scene == "single":

        if len(parts) < 4:

            raise RuntimeError(
                "single图片目录结构错误：\n"
                f"{relative}\n\n"
                "正确格式应为：\n"
                "single/颜色/形状/图片.jpg"
            )

        color = parts[1]

        shape = parts[2]

    return (
        scene,
        color,
        shape,
        relative.as_posix(),
    )


# ============================================================
# 扫描V1
# ============================================================

def load_v1_split(
    split: str,
) -> list[DataItem]:

    image_root = (
        V1_ROOT
        / "images"
        / split
    )

    label_root = (
        V1_ROOT
        / "labels"
        / split
    )

    if not image_root.exists():

        raise FileNotFoundError(
            f"V1 images/{split}不存在：\n"
            f"{image_root}"
        )

    if not label_root.exists():

        raise FileNotFoundError(
            f"V1 labels/{split}不存在：\n"
            f"{label_root}"
        )

    image_paths = find_images(
        image_root
    )

    items = []

    for image_path in image_paths:

        label_path = (
            label_root
            /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            raise FileNotFoundError(
                "V1图片缺少Label：\n"
                f"Image: {image_path}\n"
                f"Label: {label_path}"
            )

        validate_image(
            image_path
        )

        label_text = label_path.read_text(
            encoding="utf-8"
        )

        instance_count = validate_label_text(
            label_text,
            str(label_path),
        )

        items.append(
            DataItem(
                name=image_path.name,
                image_path=image_path,
                label_text=label_text,
                source_version="v1",
                original_split=split,
                scene="legacy_v1",
                color="",
                shape="",
                source_relative_path=(
                    image_path
                    .relative_to(ROOT)
                    .as_posix()
                ),
                instance_count=instance_count,
            )
        )

    return items


# ============================================================
# new_images建立全局文件名索引
# ============================================================

def build_v2_image_index() -> dict[
    str,
    Path,
]:

    image_paths = find_images(
        V2_IMAGE_ROOT
    )

    result = {}

    for image_path in image_paths:

        if image_path.name in result:

            raise RuntimeError(
                "new_images中出现重复文件名：\n"
                f"{image_path.name}\n"
                f"{result[image_path.name]}\n"
                f"{image_path}"
            )

        result[
            image_path.name
        ] = image_path

    return result


# ============================================================
# 读取CVAT V2 checked ZIP
# ============================================================

def load_v2_checked(
    v1_names: set[str],
) -> tuple[
    list[DataItem],
    list[str],
]:

    if not V2_CHECKED_ZIP.exists():

        raise FileNotFoundError(
            "找不到V2 checked ZIP：\n"
            f"{V2_CHECKED_ZIP}"
        )

    image_index = (
        build_v2_image_index()
    )

    with zipfile.ZipFile(
        V2_CHECKED_ZIP,
        "r",
    ) as archive:

        names = archive.namelist()

        if "train.txt" not in names:

            raise RuntimeError(
                "V2 ZIP中没有train.txt"
            )

        train_text = archive.read(
            "train.txt"
        ).decode(
            "utf-8-sig"
        )

        train_entries = [
            line.strip()
            for line
            in train_text.splitlines()
            if line.strip()
        ]

        # ----------------------------------------------------
        # Label索引
        # ----------------------------------------------------

        label_entries = {}

        for name in names:

            path = Path(name)

            if (
                name.startswith(
                    "labels/train/"
                )
                and
                path.suffix.lower()
                ==
                ".txt"
            ):

                stem = path.stem

                if stem in label_entries:

                    raise RuntimeError(
                        "ZIP内出现重复Label stem："
                        f"{stem}"
                    )

                label_entries[
                    stem
                ] = name

        checked_items = []

        ignored_legacy_entries = []

        seen_names = set()

        # ----------------------------------------------------
        # train.txt逐项处理
        # ----------------------------------------------------

        for entry in train_entries:

            image_name = Path(
                entry
            ).name

            if image_name in seen_names:

                raise RuntimeError(
                    "train.txt存在重复图片："
                    f"{image_name}"
                )

            seen_names.add(
                image_name
            )

            # =================================================
            # CVAT中混入的V1旧图片
            #
            # 例如：
            # top_000130.jpg
            #
            # 不加入V2新增数据。
            # =================================================

            if not image_name.startswith(
                "top_v2_"
            ):

                if image_name not in v1_names:

                    raise RuntimeError(
                        "V2 ZIP中发现未知的非V2图片：\n"
                        f"{image_name}"
                    )

                ignored_legacy_entries.append(
                    image_name
                )

                continue

            # =================================================
            # V2图片必须能在new_images找到
            # =================================================

            if image_name not in image_index:

                raise FileNotFoundError(
                    "CVAT中的V2图片在new_images中找不到：\n"
                    f"{image_name}"
                )

            image_path = image_index[
                image_name
            ]

            validate_image(
                image_path
            )

            (
                scene,
                color,
                shape,
                relative_path,
            ) = parse_v2_metadata(
                image_path
            )

            stem = Path(
                image_name
            ).stem

            # =================================================
            # 正常Label
            # =================================================

            if stem in label_entries:

                label_text = archive.read(
                    label_entries[
                        stem
                    ]
                ).decode(
                    "utf-8-sig"
                )

            # =================================================
            # 没有Label
            #
            # 只有empty场景允许。
            #
            # 这里正好处理你确认的5张空托盘。
            # =================================================

            else:

                if scene != "empty":

                    raise RuntimeError(
                        "非empty图片没有CVAT Label：\n"
                        f"{relative_path}\n"
                        "这通常说明漏标。"
                    )

                label_text = ""

                print(
                    "[EMPTY LABEL CREATED] "
                    f"{relative_path}"
                )

            instance_count = (
                validate_label_text(
                    label_text,
                    image_name,
                )
            )

            # =================================================
            # empty场景应该没有目标
            # =================================================

            if (
                scene == "empty"
                and
                instance_count != 0
            ):

                raise RuntimeError(
                    "empty场景却包含目标Label：\n"
                    f"{relative_path}\n"
                    f"instances={instance_count}"
                )

            # =================================================
            # 非empty理论上应该至少一个实例
            # =================================================

            if (
                scene != "empty"
                and
                instance_count == 0
            ):

                raise RuntimeError(
                    "非empty图片Label为空：\n"
                    f"{relative_path}"
                )

            checked_items.append(
                DataItem(
                    name=image_name,
                    image_path=image_path,
                    label_text=label_text,
                    source_version="v2_new",
                    original_split="unsplit",
                    scene=scene,
                    color=color,
                    shape=shape,
                    source_relative_path=relative_path,
                    instance_count=instance_count,
                )
            )

    # ========================================================
    # 严格检查：
    #
    # new_images里的所有top_v2图片
    # 必须全部出现在CVAT train.txt中。
    # ========================================================

    all_new_names = {
        name
        for name in image_index
        if name.startswith(
            "top_v2_"
        )
    }

    checked_names = {
        item.name
        for item in checked_items
    }

    missing_from_cvat = (
        all_new_names
        -
        checked_names
    )

    if missing_from_cvat:

        print()

        print(
            "以下new_images图片没有进入本次CVAT："
        )

        for name in sorted(
            missing_from_cvat
        ):

            print(
                f"  {name}"
            )

        raise RuntimeError(
            "存在尚未标注的V2图片，"
            "停止构建。"
        )

    return (
        checked_items,
        ignored_legacy_entries,
    )


# ============================================================
# 精确重复检测
#
# 在V1与V2之间检查SHA256。
#
# 如果V2只是把V1图片改了个名字重新加入，
# 这里会直接拦截。
# ============================================================

def check_exact_duplicates(
    items: list[DataItem],
) -> None:

    digest_map = defaultdict(
        list
    )

    print()

    print(
        "Calculating SHA256..."
    )

    for index, item in enumerate(
        items,
        start=1,
    ):

        digest = calculate_sha256(
            item.image_path
        )

        item.sha256 = digest

        digest_map[
            digest
        ].append(
            item
        )

        if (
            index % 50 == 0
            or
            index == len(items)
        ):

            print(
                f"  {index}/"
                f"{len(items)}"
            )

    duplicates = {

        digest: duplicated_items

        for (
            digest,
            duplicated_items
        )
        in digest_map.items()

        if len(
            duplicated_items
        ) > 1
    }

    if not duplicates:

        print(
            "SHA256 duplicate check: PASSED"
        )

        return

    print()

    print(
        "============================================================"
    )

    print(
        "发现完全重复图片"
    )

    print(
        "============================================================"
    )

    for digest, duplicated_items in (
        duplicates.items()
    ):

        print()

        print(
            f"SHA256: {digest}"
        )

        for item in duplicated_items:

            print(
                f"  [{item.source_version}] "
                f"{item.image_path}"
            )

    raise RuntimeError(
        "检测到完全重复图像。"
        "为避免数据泄漏，停止构建。"
    )


# ============================================================
# 按scene对V2新增数据做确定性划分
# ============================================================

def split_v2_items(
    items: list[DataItem],
) -> dict[
    str,
    list[DataItem],
]:

    groups = defaultdict(
        list
    )

    for item in items:

        groups[
            item.scene
        ].append(
            item
        )

    result = {
        "train": [],
        "val": [],
        "test": [],
    }

    rng = random.Random(
        SEED
    )

    print()

    print(
        "V2 NEW SPLIT"
    )

    print(
        "============================================================"
    )

    for scene in sorted(
        groups
    ):

        group = list(
            groups[
                scene
            ]
        )

        group.sort(
            key=lambda item:
            item.name
        )

        rng.shuffle(
            group
        )

        n = len(
            group
        )

        # ====================================================
        # 很少的数据不强制切val/test
        # ====================================================

        if n < 5:

            n_val = 0
            n_test = 0

            print(
                f"[WARNING] scene={scene} "
                f"只有{n}张，全部进入train。"
            )

        elif n < 10:

            n_val = 1
            n_test = 1

        else:

            n_val = max(
                1,
                round(
                    n
                    *
                    VAL_RATIO
                ),
            )

            n_test = max(
                1,
                round(
                    n
                    *
                    TEST_RATIO
                ),
            )

        # ====================================================
        # 防止val + test占完
        # ====================================================

        while (
            n_val
            +
            n_test
            >=
            n
        ):

            if n_test >= n_val:

                n_test -= 1

            else:

                n_val -= 1

        n_train = (
            n
            -
            n_val
            -
            n_test
        )

        train_items = (
            group[
                :n_train
            ]
        )

        val_items = (
            group[
                n_train:
                n_train + n_val
            ]
        )

        test_items = (
            group[
                n_train + n_val:
            ]
        )

        result[
            "train"
        ].extend(
            train_items
        )

        result[
            "val"
        ].extend(
            val_items
        )

        result[
            "test"
        ].extend(
            test_items
        )

        print(
            f"{scene:<12} "
            f"total={n:<4} "
            f"train={len(train_items):<4} "
            f"val={len(val_items):<4} "
            f"test={len(test_items):<4}"
        )

    # ========================================================
    # 每个split内部保持稳定顺序
    # ========================================================

    for split in result:

        result[
            split
        ].sort(
            key=lambda item:
            item.name
        )

    return result


# ============================================================
# 复制一个数据项
# ============================================================

def copy_item(
    item: DataItem,
    split: str,
) -> None:

    image_dir = (
        OUTPUT_ROOT
        / "images"
        / split
    )

    label_dir = (
        OUTPUT_ROOT
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

    image_destination = (
        image_dir
        /
        item.name
    )

    label_destination = (
        label_dir
        /
        f"{Path(item.name).stem}.txt"
    )

    if image_destination.exists():

        raise RuntimeError(
            "输出发生图片重名：\n"
            f"{image_destination}"
        )

    if label_destination.exists():

        raise RuntimeError(
            "输出发生Label重名：\n"
            f"{label_destination}"
        )

    shutil.copy2(
        item.image_path,
        image_destination,
    )

    label_destination.write_text(
        item.label_text,
        encoding="utf-8",
    )


# ============================================================
# 写data.yaml
# ============================================================

def write_yaml_files() -> None:

    dataset_path = (
        OUTPUT_ROOT
        .resolve()
        .as_posix()
    )

    data_yaml = (
        f'path: "{dataset_path}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )

    (
        OUTPUT_ROOT
        /
        "data.yaml"
    ).write_text(
        data_yaml,
        encoding="utf-8",
    )

    # ========================================================
    # 原V1 test专用
    # ========================================================

    legacy_yaml = (
        f'path: "{dataset_path}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/legacy_test\n"
        "\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )

    (
        OUTPUT_ROOT
        /
        "data_legacy_test.yaml"
    ).write_text(
        legacy_yaml,
        encoding="utf-8",
    )


# ============================================================
# Manifest
# ============================================================

def write_manifest(
    rows: list[dict],
) -> None:

    path = (
        OUTPUT_ROOT
        /
        "split_manifest.csv"
    )

    fieldnames = [
        "image",
        "split",
        "source_version",
        "original_split",
        "scene",
        "color",
        "shape",
        "instance_count",
        "sha256",
        "source_relative_path",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
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
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有prepared/v2",
    )

    args = parser.parse_args()

    # ========================================================
    # 输入检查
    # ========================================================

    if not V1_ROOT.exists():

        raise FileNotFoundError(
            "prepared/v1不存在：\n"
            f"{V1_ROOT}"
        )

    if not V2_IMAGE_ROOT.exists():

        raise FileNotFoundError(
            "new_images不存在：\n"
            f"{V2_IMAGE_ROOT}"
        )

    if not V2_CHECKED_ZIP.exists():

        raise FileNotFoundError(
            "top_seg_v2_checked.zip不存在：\n"
            f"{V2_CHECKED_ZIP}"
        )

    # ========================================================
    # 输出保护
    # ========================================================

    if OUTPUT_ROOT.exists():

        if not args.force:

            raise RuntimeError(
                "prepared/v2已经存在。\n"
                "如果确定重新生成，请增加 --force"
            )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    # ========================================================
    # 加载V1
    # ========================================================

    print(
        "============================================================"
    )

    print(
        "LOAD V1"
    )

    print(
        "============================================================"
    )

    v1_train = load_v1_split(
        "train"
    )

    v1_val = load_v1_split(
        "val"
    )

    v1_test = load_v1_split(
        "test"
    )

    print(
        f"V1 train : {len(v1_train)}"
    )

    print(
        f"V1 val   : {len(v1_val)}"
    )

    print(
        f"V1 test  : {len(v1_test)}"
    )

    # ========================================================
    # V1名字
    # ========================================================

    v1_all = (
        v1_train
        +
        v1_val
        +
        v1_test
    )

    v1_names = {
        item.name
        for item in v1_all
    }

    if len(v1_names) != len(v1_all):

        raise RuntimeError(
            "V1内部存在重复文件名。"
        )

    # ========================================================
    # 加载V2 checked
    # ========================================================

    print()

    print(
        "============================================================"
    )

    print(
        "LOAD V2 CHECKED"
    )

    print(
        "============================================================"
    )

    (
        v2_items,
        ignored_legacy_entries,
    ) = load_v2_checked(
        v1_names
    )

    print()

    print(
        f"V2 new images : {len(v2_items)}"
    )

    print(
        f"Ignored legacy CVAT entries : "
        f"{len(ignored_legacy_entries)}"
    )

    if ignored_legacy_entries:

        for name in sorted(
            ignored_legacy_entries
        ):

            print(
                f"  [IGNORED V1] {name}"
            )

    # ========================================================
    # 全部源数据SHA256检查
    # ========================================================

    check_exact_duplicates(
        v1_all
        +
        v2_items
    )

    # ========================================================
    # V2新增数据划分
    # ========================================================

    v2_split = split_v2_items(
        v2_items
    )

    # ========================================================
    # 正式split设计
    #
    # train:
    #   V1 train
    #   +
    #   V2 new train
    #
    # val:
    #   V1 val
    #   +
    #   V2 new val
    #
    # test:
    #   V2 new test ONLY
    #
    # legacy_test:
    #   V1 test ONLY
    # ========================================================

    final_splits = {

        "train":
            v1_train
            +
            v2_split[
                "train"
            ],

        "val":
            v1_val
            +
            v2_split[
                "val"
            ],

        "test":
            v2_split[
                "test"
            ],

        "legacy_test":
            v1_test,
    }

    # ========================================================
    # 输出目录
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest_rows = []

    # ========================================================
    # Copy
    # ========================================================

    print()

    print(
        "============================================================"
    )

    print(
        "BUILD PREPARED/V2"
    )

    print(
        "============================================================"
    )

    for split in (
        "train",
        "val",
        "test",
        "legacy_test",
    ):

        items = final_splits[
            split
        ]

        print()

        print(
            f"{split}: {len(items)}"
        )

        for item in items:

            copy_item(
                item,
                split,
            )

            manifest_rows.append(
                {
                    "image":
                        item.name,

                    "split":
                        split,

                    "source_version":
                        item.source_version,

                    "original_split":
                        item.original_split,

                    "scene":
                        item.scene,

                    "color":
                        item.color,

                    "shape":
                        item.shape,

                    "instance_count":
                        item.instance_count,

                    "sha256":
                        item.sha256,

                    "source_relative_path":
                        item.source_relative_path,
                }
            )

    # ========================================================
    # YAML
    # ========================================================

    write_yaml_files()

    # ========================================================
    # Manifest
    # ========================================================

    write_manifest(
        manifest_rows
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "seed":
            SEED,

        "v2_new_split_ratio": {
            "train":
                1.0
                -
                VAL_RATIO
                -
                TEST_RATIO,

            "val":
                VAL_RATIO,

            "test":
                TEST_RATIO,
        },

        "v1": {
            "train":
                len(
                    v1_train
                ),

            "val":
                len(
                    v1_val
                ),

            "legacy_test":
                len(
                    v1_test
                ),
        },

        "v2_new": {
            "total":
                len(
                    v2_items
                ),

            "train":
                len(
                    v2_split[
                        "train"
                    ]
                ),

            "val":
                len(
                    v2_split[
                        "val"
                    ]
                ),

            "test":
                len(
                    v2_split[
                        "test"
                    ]
                ),
        },

        "final": {

            split:
                len(
                    items
                )

            for (
                split,
                items
            )
            in final_splits.items()
        },

        "ignored_v1_entries_from_v2_cvat":
            ignored_legacy_entries,

        "v2_scene_counts":
            dict(
                Counter(
                    item.scene
                    for item
                    in v2_items
                )
            ),

        "v2_total_instances":
            sum(
                item.instance_count
                for item
                in v2_items
            ),
    }

    (
        OUTPUT_ROOT
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

    # ========================================================
    # 最后重新检查输出数量
    # ========================================================

    print()

    print(
        "============================================================"
    )

    print(
        "FINAL CHECK"
    )

    print(
        "============================================================"
    )

    for split in (
        "train",
        "val",
        "test",
        "legacy_test",
    ):

        image_count = len(
            find_images(
                OUTPUT_ROOT
                /
                "images"
                /
                split
            )
        )

        label_count = len(
            list(
                (
                    OUTPUT_ROOT
                    /
                    "labels"
                    /
                    split
                ).glob(
                    "*.txt"
                )
            )
        )

        if image_count != label_count:

            raise RuntimeError(
                f"{split}图片/Label数量不一致："
                f"{image_count} != {label_count}"
            )

        print(
            f"{split:<12}: "
            f"images={image_count:<4} "
            f"labels={label_count:<4}"
        )

    print()

    print(
        "============================================================"
    )

    print(
        "PREPARED V2 GENERATED SUCCESSFULLY"
    )

    print(
        "============================================================"
    )

    print(
        OUTPUT_ROOT
    )

    print()

    print(
        "Training YAML:"
    )

    print(
        OUTPUT_ROOT
        /
        "data.yaml"
    )

    print()

    print(
        "Legacy V1 Test YAML:"
    )

    print(
        OUTPUT_ROOT
        /
        "data_legacy_test.yaml"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":

    main()