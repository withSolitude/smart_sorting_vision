from __future__ import annotations

"""
侧面相机 YOLO-Cls 数据集合并脚本
================================

侧面分类数据集格式：

data/datasets/side_cls/<dataset_name>/
├── train/
│   └── <shape>/
│       └── <color>/
│           └── *.jpg
├── val/
│   └── <shape>/
│       └── <color>/
│           └── *.jpg
└── test/
    └── <shape>/
        └── <color>/
            └── *.jpg

说明：
1. 第一层类别目录是 shape，YOLO-Cls 会把 shape 作为分类类别。
2. color 只是 shape 下的子目录，用来保留采集颜色信息，不作为分类类别。
3. 本脚本不会重新随机划分 train/val/test，只合并原有 split。
4. 会按 SHA256 去除完全重复图片，避免同一张图重复进入合并数据集。
5. 如果同一张图片出现在不同 split，优先保留 test > val > train。
6. 如果同一张图片被放在不同 shape 类别下，会直接报错。

推荐放置：
    ~/smart_sorting_vision/scripts/last_use/merge_side_cls_datasets.py

推荐运行：
    cd ~/smart_sorting_vision
    source .venv/bin/activate
    python -m scripts.last_use.merge_side_cls_datasets
"""

import csv
import hashlib
import shutil

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# 1. 用户配置
# =============================================================================

# 例如：
#   v2 + school_test -> school_mixed
#
# 需要合并哪些数据集，就把名字写在这里。
SOURCE_DATASETS = [
    "school",
    "v2",
]

# 类别过滤规则：
# 以 SOURCE_DATASETS[0]（第一个数据集）中的 shape 类别为准。
# 后续数据集中不存在于第一个数据集的 shape 会自动忽略。
# 例如：
# v2: [圆柱, 正方体]
# school: [圆柱, 圆锥]
# 合并结果只包含: [圆柱, 正方体]
FILTER_BY_FIRST_DATASET_CLASSES = True

# 合并以后生成的新数据集名称
OUTPUT_DATASET = "school_test_new"

# 运行前是否删除原来的输出数据集
CLEAR_OUTPUT = True

# 支持的图片格式
IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# 数据集 split
SPLITS = (
    "train",
    "val",
    "test",
)

# 如果同一张完全一样的图片同时出现在多个 split，
# 优先保留 test，其次 val，最后 train。
#
# 这样可以防止：
#
# train/a.jpg
# val/a.jpg
#
# 完全相同图片造成明显的数据泄露。
SPLIT_PRIORITY = {
    "test": 0,
    "val": 1,
    "train": 2,
}


# =============================================================================
# 2. 路径
# =============================================================================

# 当前脚本推荐放在：
#
# smart_sorting_vision/
# └── scripts/
#     └── last_use/
#         └── merge_side_cls_datasets.py
#
# 所以 parents[2] 就是工程根目录。
ROOT = Path(__file__).resolve().parents[3]


SIDE_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "side_cls"
)


OUTPUT_ROOT = (
    SIDE_ROOT
    / OUTPUT_DATASET
)


# 合并明细
SUMMARY_CSV = (
    OUTPUT_ROOT
    / "merge_summary.csv"
)


# 每个类别数量统计
CLASS_COUNTS_CSV = (
    OUTPUT_ROOT
    / "class_counts.csv"
)


# 标签冲突记录
CONFLICT_CSV = (
    OUTPUT_ROOT
    / "label_conflicts.csv"
)


# =============================================================================
# 3. 图片记录结构
# =============================================================================

@dataclass(frozen=True)
class ImageRecord:

    # 图片来自哪个数据集
    source_dataset: str

    # train / val / test
    split: str

    # shape 类别
    shape: str

    # shape 后面的相对路径
    #
    # 例如：
    #
    # train/正方体/red/001.jpg
    #
    # relative_after_shape =
    #
    # red/001.jpg
    relative_after_shape: Path

    # 原始图片路径
    source_path: Path

    # 图片 SHA256
    sha256: str


    @property
    def color(self) -> str:
        """
        标准结构：

            split/shape/color/image.jpg

        例如：

            train/正方体/red/001.jpg

        此时：

            color = red

        如果历史数据没有 color 子目录，例如：

            train/正方体/001.jpg

        则：

            color = _no_color
        """

        parts = self.relative_after_shape.parts

        if len(parts) >= 2:
            return parts[0]

        return "_no_color"


# =============================================================================
# 4. 计算文件 SHA256
# =============================================================================

def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


# =============================================================================
# 5. 输出文件名
# =============================================================================

def safe_filename(
    source_dataset: str,
    original_name: str,
) -> str:
    """
    为了防止两个数据集中存在相同文件名：

        v2/xxx/001.jpg
        school_test/xxx/001.jpg

    输出文件自动加来源数据集前缀：

        v2__001.jpg
        school_test__001.jpg
    """

    source_tag = (
        source_dataset
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    return (
        f"{source_tag}__"
        f"{original_name}"
    )


# =============================================================================
# 6. 扫描一个侧面分类数据集
# =============================================================================

def find_dataset_images(
    dataset_name: str,
) -> list[ImageRecord]:

    dataset_root = (
        SIDE_ROOT
        / dataset_name
    )


    if not dataset_root.is_dir():

        raise FileNotFoundError(
            f"找不到侧面数据集：{dataset_root}"
        )


    records: list[ImageRecord] = []


    # -------------------------------------------------------------------------
    # 分别扫描：
    #
    # train
    # val
    # test
    # -------------------------------------------------------------------------

    for split in SPLITS:

        split_root = (
            dataset_root
            / split
        )


        if not split_root.is_dir():

            print(
                f"[WARN] "
                f"{dataset_name}: "
                f"不存在 "
                f"{split_root.relative_to(ROOT)}，"
                f"跳过"
            )

            continue


        # ---------------------------------------------------------------------
        # 侧面 YOLO-Cls 格式：
        #
        # train/
        # ├── 正方体/
        # ├── 三棱柱/
        # ├── 五棱柱/
        # └── ...
        #
        # 因此 split 下一级文件夹就是 shape 分类类别。
        # ---------------------------------------------------------------------

        shape_dirs = sorted(
            p
            for p in split_root.iterdir()
            if p.is_dir()
        )


        for shape_dir in shape_dirs:

            shape = shape_dir.name


            # -----------------------------------------------------------------
            # shape 下面允许继续有 color 子目录。
            #
            # 例如：
            #
            # train/
            # └── 五棱柱/
            #     ├── red/
            #     ├── blue/
            #     ├── green/
            #     └── ...
            #
            # 所以这里使用 rglob。
            # -----------------------------------------------------------------

            for image_path in sorted(
                shape_dir.rglob("*")
            ):

                if not image_path.is_file():
                    continue


                if (
                    image_path.suffix.lower()
                    not in IMAGE_SUFFIXES
                ):
                    continue


                relative_after_shape = (
                    image_path
                    .relative_to(
                        shape_dir
                    )
                )


                records.append(

                    ImageRecord(

                        source_dataset=
                            dataset_name,

                        split=
                            split,

                        shape=
                            shape,

                        relative_after_shape=
                            relative_after_shape,

                        source_path=
                            image_path,

                        sha256=
                            sha256_file(
                                image_path
                            ),
                    )

                )


    return records



# =============================================================================
# 6.5 按第一个数据集类别过滤后续数据集
# =============================================================================

def filter_records_by_first_dataset_classes(
    records: list[ImageRecord],
) -> list[ImageRecord]:
    """
    只保留第一个 SOURCE_DATASETS 数据集中存在的 shape。

    规则：
    - SOURCE_DATASETS[0] 是类别白名单来源。
    - 后面的数据集只提供已有类别的数据增强。
    - 新增类别不会加入合并结果。

    例如：
        v2:
            圆柱
            正方体

        school:
            圆柱
            圆锥

    最终：
        圆柱 + 正方体

    圆锥会被跳过。
    """

    if not FILTER_BY_FIRST_DATASET_CLASSES:
        return records

    if not SOURCE_DATASETS:
        return records

    first_dataset = SOURCE_DATASETS[0]

    allowed_shapes = {
        r.shape
        for r in records
        if r.source_dataset == first_dataset
    }

    if not allowed_shapes:
        raise RuntimeError(
            f"第一个数据集 {first_dataset} 没有找到任何shape类别"
        )

    filtered = []
    skipped = Counter()

    for r in records:
        if r.shape in allowed_shapes:
            filtered.append(r)
        else:
            skipped[r.shape] += 1

    print()
    print("[CLASS FILTER]")
    print("base dataset :", first_dataset)
    print(
        "allowed shapes:",
        ", ".join(sorted(allowed_shapes))
    )

    for shape, count in sorted(skipped.items()):
        print(
            f"[SKIP] shape={shape} "
            f"images={count}"
        )

    print(
        f"[CLASS FILTER] "
        f"{len(records)} -> {len(filtered)} images"
    )

    return filtered


# =============================================================================
# 7. 完全重复图片去重
# =============================================================================

def resolve_records(
    records: list[ImageRecord],
) -> tuple[
    list[ImageRecord],
    list[dict],
]:
    """
    完全重复图片处理规则：

    1. 同一个 SHA256 出现在不同 shape：
       说明同一张图片被标成了不同类别。
       这是严重标签问题，记录冲突并停止合并。

    2. 同一个 SHA256 出现在相同 shape：
       只保留一个。

    3. 如果重复图片跨 split：

           test
           >
           val
           >
           train

       避免完全相同图片同时进入训练和验证/测试数据。

    4. 同一个 split 内如果重复：
       按 SOURCE_DATASETS 中的顺序优先。
    """


    by_hash: dict[
        str,
        list[ImageRecord],
    ] = defaultdict(
        list
    )


    for record in records:

        by_hash[
            record.sha256
        ].append(
            record
        )


    source_order = {

        name: index

        for index, name
        in enumerate(
            SOURCE_DATASETS
        )

    }


    selected: list[
        ImageRecord
    ] = []


    conflicts: list[
        dict
    ] = []


    # -------------------------------------------------------------------------
    # 按 hash 分组检查
    # -------------------------------------------------------------------------

    for sha256, group in by_hash.items():


        # 这一张图片出现过哪些 shape 标签
        shapes = {

            record.shape

            for record in group

        }


        # ---------------------------------------------------------------------
        # 同一张图被标成不同 shape
        # ---------------------------------------------------------------------

        if len(shapes) > 1:


            for record in group:

                conflicts.append(

                    {
                        "sha256":
                            sha256,

                        "source_dataset":
                            record.source_dataset,

                        "split":
                            record.split,

                        "shape":
                            record.shape,

                        "path":
                            str(
                                record
                                .source_path
                                .relative_to(
                                    ROOT
                                )
                            ),
                    }

                )


            continue


        # ---------------------------------------------------------------------
        # 相同 shape 的完全重复图片
        #
        # 按：
        #
        # test > val > train
        #
        # 再按照 SOURCE_DATASETS 顺序。
        # ---------------------------------------------------------------------

        group_sorted = sorted(

            group,

            key=lambda record: (

                SPLIT_PRIORITY[
                    record.split
                ],

                source_order.get(
                    record.source_dataset,
                    999,
                ),

                str(
                    record.source_path
                ),
            ),
        )


        # 只保留优先级最高的一个
        selected.append(
            group_sorted[0]
        )


    return (
        selected,
        conflicts,
    )


# =============================================================================
# 8. 复制图片
# =============================================================================

def copy_record(
    record: ImageRecord,
) -> Path:
    """
    保留原来的侧面数据集结构。

    标准结构：

        train/
          正方体/
            red/
              xxx.jpg

    也兼容老数据：

        train/
          正方体/
            xxx.jpg
    """


    relative_parent = (
        record
        .relative_after_shape
        .parent
    )


    output_dir = (

        OUTPUT_ROOT

        / record.split

        / record.shape

        / relative_parent

    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_name = safe_filename(

        record.source_dataset,

        record.source_path.name,

    )


    output_path = (

        output_dir

        / output_name

    )


    # -------------------------------------------------------------------------
    # 正常情况下，加上来源数据集前缀后已经不会重名。
    #
    # 如果仍然重名，再追加 hash。
    # -------------------------------------------------------------------------

    if output_path.exists():


        short_hash = (
            record.sha256[:10]
        )


        output_path = (

            output_dir

            / (

                f"{output_path.stem}"

                f"__{short_hash}"

                f"{output_path.suffix}"

            )

        )


    shutil.copy2(

        record.source_path,

        output_path,

    )


    return output_path


# =============================================================================
# 9. 写标签冲突文件
# =============================================================================

def write_conflicts(
    conflicts: list[dict],
) -> None:


    if not conflicts:
        return


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    with CONFLICT_CSV.open(

        "w",

        newline="",

        encoding="utf-8-sig",

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=[

                "sha256",

                "source_dataset",

                "split",

                "shape",

                "path",

            ],

        )


        writer.writeheader()


        writer.writerows(
            conflicts
        )


# =============================================================================
# 10. 主程序
# =============================================================================

def main() -> None:


    print(
        "=" * 78
    )


    print(
        "SIDE CLS DATASET MERGER"
    )


    print(
        "=" * 78
    )


    print(
        "dataset root :",
        SIDE_ROOT,
    )


    print(
        "sources      :",
        ", ".join(
            SOURCE_DATASETS
        ),
    )


    print(
        "output       :",
        OUTPUT_ROOT,
    )


    print()


    # -------------------------------------------------------------------------
    # 防止把输出数据集自己再次作为输入
    # -------------------------------------------------------------------------

    if (
        OUTPUT_DATASET
        in SOURCE_DATASETS
    ):

        raise ValueError(

            "OUTPUT_DATASET "
            "不能同时出现在 "
            "SOURCE_DATASETS 中"

        )


    # =========================================================================
    # 扫描所有来源数据集
    # =========================================================================

    all_records: list[
        ImageRecord
    ] = []


    for dataset_name in (
        SOURCE_DATASETS
    ):


        records = (
            find_dataset_images(
                dataset_name
            )
        )


        all_records.extend(
            records
        )


        print(

            f"[SCAN] "
            f"{dataset_name}: "
            f"{len(records)} images"

        )


    if not all_records:

        raise RuntimeError(
            "没有找到任何侧面分类图片"
        )


    # -------------------------------------------------------------------------
    # 类别过滤：
    # 只允许第一个数据集中的shape进入合并。
    # -------------------------------------------------------------------------
    all_records = filter_records_by_first_dataset_classes(
        all_records
    )


    print()


    print(

        "[SCAN] total before dedup:",

        len(
            all_records
        ),

    )


    # =========================================================================
    # 去重 + 标签冲突检查
    # =========================================================================

    selected_records, conflicts = (

        resolve_records(
            all_records
        )

    )


    # -------------------------------------------------------------------------
    # 重复数量
    # -------------------------------------------------------------------------

    duplicate_count = (

        len(
            all_records
        )

        -

        len(
            selected_records
        )

        -

        len(
            conflicts
        )

    )


    print(

        "[DEDUP] selected:",

        len(
            selected_records
        ),

    )


    print(

        "[DEDUP] exact duplicates removed:",

        max(
            0,
            duplicate_count,
        ),

    )


    # =========================================================================
    # 如果发现标签冲突
    # =========================================================================

    if conflicts:


        if (

            CLEAR_OUTPUT

            and OUTPUT_ROOT.exists()

        ):

            shutil.rmtree(
                OUTPUT_ROOT
            )


        write_conflicts(
            conflicts
        )


        print()


        print(
            "[ERROR] "
            "发现同一张图片被标成不同 shape。"
        )


        print(

            "[ERROR] "
            "冲突明细：",

            CONFLICT_CSV,

        )


        raise RuntimeError(
            "存在 shape 标签冲突，停止合并"
        )


    # =========================================================================
    # 清空旧输出
    # =========================================================================

    if (

        CLEAR_OUTPUT

        and OUTPUT_ROOT.exists()

    ):


        print(

            "[CLEAR]",

            OUTPUT_ROOT,

        )


        shutil.rmtree(
            OUTPUT_ROOT
        )


    OUTPUT_ROOT.mkdir(

        parents=True,

        exist_ok=True,

    )


    # =========================================================================
    # 开始复制图片
    # =========================================================================

    summary_rows: list[
        dict
    ] = []


    count_by_split_shape = Counter()


    # -------------------------------------------------------------------------
    # 排序主要是让输出过程可重复
    # -------------------------------------------------------------------------

    selected_sorted = sorted(

        selected_records,

        key=lambda record: (

            SPLIT_PRIORITY[
                record.split
            ],

            record.shape,

            record.source_dataset,

            str(
                record.relative_after_shape
            ),

        ),

    )


    for index, record in enumerate(

        selected_sorted,

        start=1,

    ):


        output_path = (
            copy_record(
                record
            )
        )


        count_by_split_shape[
            (
                record.split,
                record.shape,
            )
        ] += 1


        summary_rows.append(

            {

                "source_dataset":
                    record.source_dataset,


                "source_split":
                    record.split,


                "shape":
                    record.shape,


                "color":
                    record.color,


                "sha256":
                    record.sha256,


                "source_path":
                    str(
                        record
                        .source_path
                        .relative_to(
                            ROOT
                        )
                    ),


                "output_path":
                    str(
                        output_path
                        .relative_to(
                            ROOT
                        )
                    ),

            }

        )


        # ---------------------------------------------------------------------
        # 每100张打印一次进度
        # ---------------------------------------------------------------------

        if (

            index == 1

            or index % 100 == 0

            or index == len(
                selected_sorted
            )

        ):


            print(

                f"[COPY] "

                f"{index}/"

                f"{len(selected_sorted)}"

            )


    # =========================================================================
    # 写 merge_summary.csv
    # =========================================================================

    with SUMMARY_CSV.open(

        "w",

        newline="",

        encoding="utf-8-sig",

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=[

                "source_dataset",

                "source_split",

                "shape",

                "color",

                "sha256",

                "source_path",

                "output_path",

            ],

        )


        writer.writeheader()


        writer.writerows(
            summary_rows
        )


    # =========================================================================
    # 写 class_counts.csv
    # =========================================================================

    with CLASS_COUNTS_CSV.open(

        "w",

        newline="",

        encoding="utf-8-sig",

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=[

                "split",

                "shape",

                "count",

            ],

        )


        writer.writeheader()


        for (
            split,
            shape,
        ), count in sorted(

            count_by_split_shape.items(),

            key=lambda item: (

                SPLIT_PRIORITY[
                    item[0][0]
                ],

                item[0][1],

            ),

        ):


            writer.writerow(

                {

                    "split":
                        split,


                    "shape":
                        shape,


                    "count":
                        count,

                }

            )


    # =========================================================================
    # 输出统计
    # =========================================================================

    print()


    print(
        "=" * 78
    )


    print(
        "MERGE FINISHED"
    )


    print(
        "=" * 78
    )


    total_counter = Counter(

        record.split

        for record in selected_records

    )


    # -------------------------------------------------------------------------
    # split 总数
    # -------------------------------------------------------------------------

    for split in SPLITS:

        print(

            f"{split:5s}: "

            f"{total_counter[split]}"

        )


    print()


    print(
        "class counts:"
    )


    # -------------------------------------------------------------------------
    # 获取所有 shape
    # -------------------------------------------------------------------------

    shapes = sorted(

        {

            record.shape

            for record in selected_records

        }

    )


    # -------------------------------------------------------------------------
    # 打印每个 shape 的 train / val / test 数量
    # -------------------------------------------------------------------------

    for shape in shapes:


        train_count = (

            count_by_split_shape[
                (
                    "train",
                    shape,
                )
            ]

        )


        val_count = (

            count_by_split_shape[
                (
                    "val",
                    shape,
                )
            ]

        )


        test_count = (

            count_by_split_shape[
                (
                    "test",
                    shape,
                )
            ]

        )


        print(

            f"  {shape}: "

            f"train={train_count}, "

            f"val={val_count}, "

            f"test={test_count}"

        )


    print()


    print(

        "output:",

        OUTPUT_ROOT,

    )


    print(

        "summary:",

        SUMMARY_CSV,

    )


    print(

        "counts :",

        CLASS_COUNTS_CSV,

    )


# =============================================================================
# 11. 入口
# =============================================================================

if __name__ == "__main__":

    main()
