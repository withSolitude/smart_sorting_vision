from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

from vision.cameras.top_camera import TopCamera


# ============================================================
# 项目根目录
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# 数据集目录
# ============================================================

DATASET_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "raw"
)

IMAGE_DIR = (
    DATASET_ROOT
    / "images"
)

SESSION_DIR = (
    DATASET_ROOT
    / "sessions"
)

OBJECTS_FILE = (
    DATASET_ROOT
    / "objects.csv"
)

LAYOUTS_FILE = (
    DATASET_ROOT
    / "layouts.csv"
)

IMAGES_FILE = (
    DATASET_ROOT
    / "images.csv"
)


# ============================================================
# 图像保存质量
# ============================================================

JPEG_QUALITY = 95


# ============================================================
# 同一个Layout建议保存图片数量
#
# 超过这个数量不会禁止保存，
# 但会警告。
#
# 推荐：
# 一个独立布局保存1~2张即可。
# ============================================================

MAX_IMAGES_PER_LAYOUT_WARNING = 2


# ============================================================
# 场景类型
#
# edge 不属于 scene_type，
# 单独使用 edge=yes/no。
# ============================================================

SCENE_TYPES = (
    "single",
    "sparse",
    "mixed",
    "crowded",
    "stacked",
    "empty",
)


# ============================================================
# 遮挡程度
# ============================================================
OCCLUSION_LEVELS = (
    "none",
    "low",
    "medium",
    "high",
)


# ============================================================
# 托盘区域
#
# LT = 左上
# CT = 上中
# RT = 右上
#
# LC = 左中
# CC = 中央
# RC = 右中
#
# LB = 左下
# CB = 下中
# RB = 右下
#
# mixed = 分布在多个区域
# ============================================================

ZONES = (
    "LT",
    "CT",
    "RT",
    "LC",
    "CC",
    "RC",
    "LB",
    "CB",
    "RB",
    "mixed",
)


# ============================================================
# CSV字段
#
# 注意：
# 已经彻底删除 size。
# ============================================================

OBJECT_FIELDS = (
    "object_id",
    "name",
    "color",
    "shape",
    "notes",
)

LAYOUT_FIELDS = (
    "layout_id",
    "session_id",
    "timestamp",
    "scene_type",
    "object_ids",
    "object_count",
    "occlusion",
    "edge",
    "zone",
    "notes",
)

IMAGE_FIELDS = (
    "image",
    "session_id",
    "layout_id",
    "timestamp",
    "width",
    "height",
)


# ============================================================
# 创建CSV文件
# ============================================================

def ensure_csv_file(
    path: Path,
    fieldnames: tuple[str, ...],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 如果不存在，创建
    # --------------------------------------------------------

    if not path.exists():

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

        return

    # --------------------------------------------------------
    # 如果已经存在，则检查表头是否与当前代码一致
    #
    # 防止旧版 objects.csv 里面还有 size，
    # 然后继续追加导致数据错列。
    # --------------------------------------------------------

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.reader(
            file
        )

        try:

            header = next(
                reader
            )

        except StopIteration:

            header = []

    expected_header = list(
        fieldnames
    )

    if (
        header
        != expected_header
    ):

        raise RuntimeError(
            "\nCSV字段与当前程序不一致：\n"
            f"{path}\n"
            "\n"
            f"当前文件字段：\n"
            f"{header}\n"
            "\n"
            f"程序要求字段：\n"
            f"{expected_header}\n"
            "\n"
            "如果这些只是之前的测试数据，"
            "请删除旧CSV后重新运行。\n"
            "如果已经有正式数据，不要直接删除，"
            "需要先迁移CSV。"
        )


# ============================================================
# 初始化数据集
# ============================================================

def initialize_dataset() -> None:

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SESSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ensure_csv_file(
        OBJECTS_FILE,
        OBJECT_FIELDS,
    )

    ensure_csv_file(
        LAYOUTS_FILE,
        LAYOUT_FIELDS,
    )

    ensure_csv_file(
        IMAGES_FILE,
        IMAGE_FIELDS,
    )


# ============================================================
# 读取CSV
# ============================================================

def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:

    if not path.exists():

        return []

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
# CSV追加一行
# ============================================================

def append_csv_row(
    path: Path,
    fieldnames: tuple[str, ...],
    row: dict,
) -> None:

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writerow(
            {
                key: row.get(
                    key,
                    "",
                )
                for key
                in fieldnames
            }
        )


# ============================================================
# Session ID
#
# 每次启动一次采集程序，
# 自动建立一个新的session。
# ============================================================

def generate_session_id() -> str:

    return (
        datetime.now()
        .strftime(
            "S%Y%m%d_%H%M%S"
        )
    )


# ============================================================
# 获取下一个图片编号
#
# top_000000.jpg
# top_000001.jpg
# ...
#
# 程序重新启动后不会覆盖旧图片。
# ============================================================

def get_next_image_index() -> int:

    max_index = -1

    for path in IMAGE_DIR.glob(
        "top_*.jpg"
    ):

        try:

            index = int(
                path.stem
                .split("_")[-1]
            )

        except ValueError:

            continue

        max_index = max(
            max_index,
            index,
        )

    return (
        max_index
        +
        1
    )


# ============================================================
# 获取下一个Object ID
#
# OBJ0001
# OBJ0002
# ...
# ============================================================

def get_next_object_id() -> str:

    rows = read_csv_rows(
        OBJECTS_FILE
    )

    max_index = 0

    for row in rows:

        object_id = (
            row.get(
                "object_id",
                "",
            )
            .strip()
            .upper()
        )

        if not object_id.startswith(
            "OBJ"
        ):

            continue

        try:

            index = int(
                object_id[3:]
            )

        except ValueError:

            continue

        max_index = max(
            max_index,
            index,
        )

    return (
        f"OBJ{max_index + 1:04d}"
    )


# ============================================================
# 获取下一个Layout ID
#
# L000001
# L000002
# ...
# ============================================================

def get_next_layout_id() -> str:

    rows = read_csv_rows(
        LAYOUTS_FILE
    )

    max_index = 0

    for row in rows:

        layout_id = (
            row.get(
                "layout_id",
                "",
            )
            .strip()
            .upper()
        )

        if not layout_id.startswith(
            "L"
        ):

            continue

        try:

            index = int(
                layout_id[1:]
            )

        except ValueError:

            continue

        max_index = max(
            max_index,
            index,
        )

    return (
        f"L{max_index + 1:06d}"
    )


# ============================================================
# 加载所有登记物体
#
# 返回：
#
# {
#     "OBJ0001": {...},
#     "OBJ0002": {...}
# }
# ============================================================

def load_object_catalog() -> dict[str, dict[str, str]]:

    rows = read_csv_rows(
        OBJECTS_FILE
    )

    catalog = {}

    for row in rows:

        object_id = (
            row.get(
                "object_id",
                "",
            )
            .strip()
            .upper()
        )

        if not object_id:

            continue

        catalog[
            object_id
        ] = row

    return catalog


# ============================================================
# 显示物体列表
# ============================================================

def print_objects() -> None:

    catalog = (
        load_object_catalog()
    )

    print()

    print(
        "=============================================================="
    )

    print(
        "OBJECT CATALOG"
    )

    print(
        "=============================================================="
    )

    if not catalog:

        print(
            "还没有登记物体。"
        )

        print(
            "按 R 登记。"
        )

        print(
            "=============================================================="
        )

        return

    print(
        f"{'ID':<10}"
        f"{'NAME':<22}"
        f"{'COLOR':<14}"
        f"{'SHAPE':<16}"
        f"NOTES"
    )

    print(
        "--------------------------------------------------------------"
    )

    for object_id in sorted(
        catalog.keys()
    ):

        item = catalog[
            object_id
        ]

        print(
            f"{object_id:<10}"
            f"{item.get('name', ''):<22}"
            f"{item.get('color', ''):<14}"
            f"{item.get('shape', ''):<16}"
            f"{item.get('notes', '')}"
        )

    print(
        "=============================================================="
    )


# ============================================================
# 登记新物体
#
# 这里只登记：
#
# object_id
# name
# color
# shape
# notes
#
# 不登记size。
# ============================================================

def register_object() -> None:

    print()

    print(
        "========================================"
    )

    print(
        "REGISTER NEW OBJECT"
    )

    print(
        "========================================"
    )

    object_id = (
        get_next_object_id()
    )

    print(
        f"Object ID: {object_id}"
    )

    print()

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name = input(
        "物体名称，例如 blue_cube_1: "
    ).strip()

    # --------------------------------------------------------
    # COLOR
    #
    # 建议统一使用英文小写：
    #
    # white
    # black
    # gray
    # red
    # orange
    # yellow
    # green
    # blue
    # purple
    # pink
    # brown
    # --------------------------------------------------------

    color = input(
        "颜色，例如 blue / red / white / black: "
    ).strip().lower()

    # --------------------------------------------------------
    # SHAPE
    #
    # 建议统一：
    #
    # cube
    # cuboid
    # sphere
    # cylinder
    # cone
    # gear
    # prism
    # polyhedron
    # irregular
    # --------------------------------------------------------

    shape = input(
        "形状，例如 cube / sphere / cylinder / gear: "
    ).strip().lower()

    notes = input(
        "备注，可直接回车跳过: "
    ).strip()

    # ========================================================
    # 基本检查
    # ========================================================

    if not color:

        print()

        print(
            "[CANCEL] color不能为空。"
        )

        return

    if not shape:

        print()

        print(
            "[CANCEL] shape不能为空。"
        )

        return

    if not name:

        name = (
            object_id.lower()
        )

    # ========================================================
    # 检查是否已经存在完全同名物体
    # ========================================================

    catalog = (
        load_object_catalog()
    )

    for existing in catalog.values():

        if (
            existing.get(
                "name",
                "",
            )
            .strip()
            .lower()
            ==
            name.lower()
        ):

            print()

            print(
                "[WARNING]"
            )

            print(
                f"已经存在同名物体：{name}"
            )

            answer = input(
                "仍然登记新的独立物体？ [y/N]: "
            ).strip().lower()

            if answer != "y":

                print(
                    "[CANCEL]"
                )

                return

            break

    # ========================================================
    # 保存
    # ========================================================

    row = {

        "object_id":
            object_id,

        "name":
            name,

        "color":
            color,

        "shape":
            shape,

        "notes":
            notes,
    }

    append_csv_row(
        OBJECTS_FILE,
        OBJECT_FIELDS,
        row,
    )

    print()

    print(
        "----------------------------------------"
    )

    print(
        f"[REGISTERED] {object_id}"
    )

    print(
        f"name  = {name}"
    )

    print(
        f"color = {color}"
    )

    print(
        f"shape = {shape}"
    )

    print(
        "----------------------------------------"
    )


# ============================================================
# 让用户从选项中选择
# ============================================================

def ask_choice(
    title: str,
    choices: tuple[str, ...],
    default: str | None = None,
) -> str:

    while True:

        print()

        print(
            title
        )

        for index, choice in enumerate(
            choices,
            start=1,
        ):

            print(
                f"  {index}. {choice}"
            )

        if default is None:

            prompt = (
                "请输入编号: "
            )

        else:

            prompt = (
                f"请输入编号，"
                f"直接回车默认 {default}: "
            )

        text = input(
            prompt
        ).strip()

        if (
            not text
            and
            default is not None
        ):

            return default

        try:

            index = int(
                text
            )

        except ValueError:

            print(
                "输入无效，请重新输入。"
            )

            continue

        if (
            1
            <= index
            <= len(choices)
        ):

            return (
                choices[
                    index - 1
                ]
            )

        print(
            "编号超出范围，请重新输入。"
        )


# ============================================================
# 创建一个新的独立布局
#
# 注意：
#
# 这里先不立即写入 layouts.csv。
#
# 只有当该layout第一次真正保存图片后，
# 才写入layouts.csv。
#
# 这样可以避免：
# 创建了layout但没有拍图，
# 最终CSV里产生大量空记录。
# ============================================================

def create_layout(
    session_id: str,
) -> dict | None:

    catalog = (
        load_object_catalog()
    )

    layout_id = (
        get_next_layout_id()
    )

    print()

    print(
        "========================================"
    )

    print(
        "CREATE NEW LAYOUT"
    )

    print(
        "========================================"
    )

    print(
        f"Layout ID: {layout_id}"
    )

    # ========================================================
    # Scene Type
    # ========================================================

    scene_type = (
        ask_choice(
            "场景类型：",
            SCENE_TYPES,
            default="mixed",
        )
    )

    # ========================================================
    # EMPTY
    # ========================================================

    if (
        scene_type
        ==
        "empty"
    ):

        selected_object_ids = []

        object_count = 0

        occlusion = "none"

        edge = "no"

        zone = "mixed"

    # ========================================================
    # 非空场景
    # ========================================================

    else:

        if not catalog:

            print()

            print(
                "当前没有登记任何物体。"
            )

            print(
                "请先按 R 登记物体。"
            )

            return None

        print_objects()

        print()

        print(
            "请输入当前托盘中所有实际物体的Object ID。"
        )

        print(
            "多个ID使用英文逗号分隔。"
        )

        print()

        print(
            "例如："
        )

        print(
            "OBJ0001,OBJ0003,OBJ0007"
        )

        while True:

            text = input(
                "Object IDs: "
            ).strip()

            selected_object_ids = [

                item.strip().upper()

                for item
                in text.split(",")

                if item.strip()
            ]

            # ------------------------------------------------
            # 非空场景必须有物体
            # ------------------------------------------------

            if not selected_object_ids:

                print(
                    "非空场景至少需要1个物体。"
                )

                continue

            # ------------------------------------------------
            # 禁止同一个真实物体重复登记
            # ------------------------------------------------

            if (
                len(
                    selected_object_ids
                )
                !=
                len(
                    set(
                        selected_object_ids
                    )
                )
            ):

                print(
                    "同一个 object_id "
                    "不能在一个布局中重复出现。"
                )

                continue

            # ------------------------------------------------
            # 检查ID是否存在
            # ------------------------------------------------

            unknown_ids = [

                object_id

                for object_id
                in selected_object_ids

                if (
                    object_id
                    not in catalog
                )
            ]

            if unknown_ids:

                print(
                    "以下Object ID不存在："
                )

                print(
                    unknown_ids
                )

                continue

            break

        object_count = (
            len(
                selected_object_ids
            )
        )

        # ----------------------------------------------------
        # single应该严格只有1个物体
        # ----------------------------------------------------

        if (
            scene_type
            ==
            "single"
            and
            object_count
            !=
            1
        ):

            print()

            print(
                "[WARNING]"
            )

            print(
                "scene_type=single "
                "通常应当只有1个物体。"
            )

            print(
                f"当前输入了 {object_count} 个物体。"
            )

            answer = input(
                "仍然继续？ [y/N]: "
            ).strip().lower()

            if answer != "y":

                print(
                    "[CANCEL]"
                )

                return None

        # ====================================================
        # 遮挡程度
        # ====================================================

        occlusion = (
            ask_choice(
                "遮挡程度：",
                OCCLUSION_LEVELS,
                default="none",
            )
        )

        # ====================================================
        # 是否有靠边物体
        # ====================================================

        edge = (
            ask_choice(
                "是否存在靠托盘边缘物体：",
                (
                    "no",
                    "yes",
                ),
                default="no",
            )
        )

        # ====================================================
        # 主要区域
        # ====================================================

        zone = (
            ask_choice(
                "主要位置区域：",
                ZONES,
                default="mixed",
            )
        )

    # ========================================================
    # Notes
    # ========================================================

    notes = input(
        "布局备注，可直接回车跳过: "
    ).strip()

    layout = {

        "layout_id":
            layout_id,

        "session_id":
            session_id,

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "scene_type":
            scene_type,

        "object_ids":
            "|".join(
                selected_object_ids
            ),

        "object_count":
            object_count,

        "occlusion":
            occlusion,

        "edge":
            edge,

        "zone":
            zone,

        "notes":
            notes,

        # -----------------------------------------------
        # 内部状态，不写入CSV
        # -----------------------------------------------

        "_persisted":
            False,
    }

    print()

    print(
        "========================================"
    )

    print(
        "CURRENT LAYOUT"
    )

    print(
        "========================================"
    )

    print(
        f"Layout     : {layout_id}"
    )

    print(
        f"Scene      : {scene_type}"
    )

    print(
        f"Objects    : {object_count}"
    )

    print(
        f"Object IDs : "
        f"{layout['object_ids'] or 'NONE'}"
    )

    print(
        f"Occlusion  : {occlusion}"
    )

    print(
        f"Edge       : {edge}"
    )

    print(
        f"Zone       : {zone}"
    )

    print(
        "========================================"
    )

    print()

    print(
        "现在按 S 保存当前布局图片。"
    )

    return layout


# ============================================================
# 将Layout正式写入CSV
# ============================================================

def persist_layout_if_needed(
    layout: dict,
) -> None:

    if layout.get(
        "_persisted",
        False,
    ):

        return

    append_csv_row(
        LAYOUTS_FILE,
        LAYOUT_FIELDS,
        layout,
    )

    layout[
        "_persisted"
    ] = True


# ============================================================
# 保存Session相机配置
# ============================================================

def save_session_config(
    session_id: str,
    camera: TopCamera,
) -> None:

    path = (
        SESSION_DIR
        /
        f"{session_id}.json"
    )

    config = {

        "session_id":
            session_id,

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "camera":
            camera.get_stream_info(),

        "fixed_controls":
            camera.get_fixed_controls(),

        "safe_roi": {

            "x1":
                TopCamera.SAFE_ROI_X1,

            "y1":
                TopCamera.SAFE_ROI_Y1,

            "x2":
                TopCamera.SAFE_ROI_X2,

            "y2":
                TopCamera.SAFE_ROI_Y2,

            "width":
                TopCamera.SAFE_ROI_WIDTH,

            "height":
                TopCamera.SAFE_ROI_HEIGHT,
        },

        "calibration_file":
            str(
                TopCamera.CALIBRATION_FILE
            ),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=4,
        )


# ============================================================
# 某Layout已经有多少张图片
# ============================================================

def count_layout_images(
    layout_id: str,
) -> int:

    rows = read_csv_rows(
        IMAGES_FILE
    )

    count = 0

    for row in rows:

        if (
            row.get(
                "layout_id",
                ""
            )
            ==
            layout_id
        ):

            count += 1

    return count


# ============================================================
# 保存图片
# ============================================================

def save_image(
    frame,
    image_index: int,
    session_id: str,
    layout: dict | None,
) -> bool:

    if layout is None:

        print()

        print(
            "[ERROR]"
        )

        print(
            "当前没有Layout。"
        )

        print(
            "请先按 N 创建一个新的独立布局。"
        )

        return False

    # ========================================================
    # 图像尺寸检查
    # ========================================================

    expected_shape = (

        TopCamera.SAFE_ROI_HEIGHT,

        TopCamera.SAFE_ROI_WIDTH,

        3,
    )

    if (
        frame.shape
        !=
        expected_shape
    ):

        raise RuntimeError(
            "训练图片尺寸异常：\n"
            f"expected = {expected_shape}\n"
            f"actual   = {frame.shape}"
        )

    layout_id = (
        layout[
            "layout_id"
        ]
    )

    # ========================================================
    # 一个布局图片不要太多
    # ========================================================

    layout_image_count = (
        count_layout_images(
            layout_id
        )
    )

    if (
        layout_image_count
        >=
        MAX_IMAGES_PER_LAYOUT_WARNING
    ):

        print()

        print(
            "========================================"
        )

        print(
            "[WARNING]"
        )

        print(
            f"{layout_id} 已经保存 "
            f"{layout_image_count} 张图片。"
        )

        print()

        print(
            "同一个布局通常只建议保存1~2张。"
        )

        print(
            "继续大量拍同一布局不能明显增加数据多样性。"
        )

        print(
            "更推荐重新摆放，然后按 N 新建Layout。"
        )

        print(
            "========================================"
        )

        answer = input(
            "仍然保存？ [y/N]: "
        ).strip().lower()

        if answer != "y":

            print(
                "[CANCEL SAVE]"
            )

            return False

    # ========================================================
    # 文件名
    # ========================================================

    image_name = (
        f"top_{image_index:06d}.jpg"
    )

    image_path = (
        IMAGE_DIR
        /
        image_name
    )

    if image_path.exists():

        raise RuntimeError(
            "图片文件已经存在：\n"
            f"{image_path}"
        )

    # ========================================================
    # 保存原始训练Frame
    #
    # 注意：
    #
    # 保存的是 camera.read() 返回的 frame，
    # 不包含窗口上显示的任何文字。
    # ========================================================

    success = cv2.imwrite(
        str(
            image_path
        ),
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    if not success:

        print()

        print(
            "[ERROR]"
        )

        print(
            "图片保存失败："
        )

        print(
            image_path
        )

        return False

    # ========================================================
    # 第一次真正保存当前Layout时，
    # 将Layout写入CSV。
    # ========================================================

    persist_layout_if_needed(
        layout
    )

    # ========================================================
    # 记录图片
    # ========================================================

    row = {

        "image":
            image_name,

        "session_id":
            session_id,

        "layout_id":
            layout_id,

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "width":
            int(
                frame.shape[1]
            ),

        "height":
            int(
                frame.shape[0]
            ),
    }

    append_csv_row(
        IMAGES_FILE,
        IMAGE_FIELDS,
        row,
    )

    print()

    print(
        "----------------------------------------"
    )

    print(
        f"[SAVE] {image_name}"
    )

    print(
        f"Layout : {layout_id}"
    )

    print(
        f"Image  : {layout_image_count + 1}"
    )

    print(
        "----------------------------------------"
    )

    return True


# ============================================================
# 打印当前Layout
# ============================================================

def print_current_layout(
    layout: dict | None,
) -> None:

    print()

    print(
        "========================================"
    )

    print(
        "CURRENT LAYOUT"
    )

    print(
        "========================================"
    )

    if layout is None:

        print(
            "NONE"
        )

        print(
            "请按 N 创建新布局。"
        )

        print(
            "========================================"
        )

        return

    print(
        f"Layout     : {layout['layout_id']}"
    )

    print(
        f"Scene      : {layout['scene_type']}"
    )

    print(
        f"Objects    : {layout['object_count']}"
    )

    print(
        f"Object IDs : "
        f"{layout['object_ids'] or 'NONE'}"
    )

    print(
        f"Occlusion  : {layout['occlusion']}"
    )

    print(
        f"Edge       : {layout['edge']}"
    )

    print(
        f"Zone       : {layout['zone']}"
    )

    print(
        f"Saved      : "
        f"{count_layout_images(layout['layout_id'])}"
    )

    print(
        "========================================"
    )


# ============================================================
# 数据覆盖统计
#
# 核心原则：
#
# 统计“独立布局数量”，
# 而不是只统计照片数量。
# ============================================================

def print_statistics() -> None:

    objects = (
        load_object_catalog()
    )

    layouts = (
        read_csv_rows(
            LAYOUTS_FILE
        )
    )

    images = (
        read_csv_rows(
            IMAGES_FILE
        )
    )

    # ========================================================
    # 每个属性出现在哪些layout
    # ========================================================

    object_layouts = defaultdict(
        set
    )

    color_layouts = defaultdict(
        set
    )

    shape_layouts = defaultdict(
        set
    )

    combination_layouts = defaultdict(
        set
    )

    scene_layouts = defaultdict(
        set
    )

    occlusion_layouts = defaultdict(
        set
    )

    edge_layouts = defaultdict(
        set
    )

    zone_layouts = defaultdict(
        set
    )

    # ========================================================
    # 分析Layouts
    # ========================================================

    for layout in layouts:

        layout_id = (
            layout.get(
                "layout_id",
                ""
            )
        )

        if not layout_id:

            continue

        scene_type = (
            layout.get(
                "scene_type",
                ""
            )
        )

        occlusion = (
            layout.get(
                "occlusion",
                ""
            )
        )

        edge = (
            layout.get(
                "edge",
                ""
            )
        )

        zone = (
            layout.get(
                "zone",
                ""
            )
        )

        scene_layouts[
            scene_type
        ].add(
            layout_id
        )

        occlusion_layouts[
            occlusion
        ].add(
            layout_id
        )

        edge_layouts[
            edge
        ].add(
            layout_id
        )

        zone_layouts[
            zone
        ].add(
            layout_id
        )

        object_ids = [

            item.strip().upper()

            for item
            in layout.get(
                "object_ids",
                ""
            ).split("|")

            if item.strip()
        ]

        for object_id in object_ids:

            if object_id not in objects:

                continue

            item = (
                objects[
                    object_id
                ]
            )

            color = (
                item.get(
                    "color",
                    ""
                )
            )

            shape = (
                item.get(
                    "shape",
                    ""
                )
            )

            object_layouts[
                object_id
            ].add(
                layout_id
            )

            color_layouts[
                color
            ].add(
                layout_id
            )

            shape_layouts[
                shape
            ].add(
                layout_id
            )

            combination_layouts[
                (
                    color,
                    shape,
                )
            ].add(
                layout_id
            )

    # ========================================================
    # 总体
    # ========================================================

    print()

    print(
        "=============================================================="
    )

    print(
        "DATASET COVERAGE STATISTICS"
    )

    print(
        "=============================================================="
    )

    print(
        f"Registered objects : {len(objects)}"
    )

    print(
        f"Independent layouts: {len(layouts)}"
    )

    print(
        f"Images             : {len(images)}"
    )

    if layouts:

        image_per_layout = (
            len(images)
            /
            len(layouts)
        )

        print(
            f"Images / layout    : "
            f"{image_per_layout:.2f}"
        )

    # ========================================================
    # Object Coverage
    # ========================================================

    print()

    print(
        "--------------------------------------------------------------"
    )

    print(
        "OBJECT COVERAGE"
    )

    print(
        "--------------------------------------------------------------"
    )

    if not objects:

        print(
            "No objects"
        )

    else:

        object_statistics = []

        for (
            object_id,
            item,
        ) in objects.items():

            count = len(
                object_layouts[
                    object_id
                ]
            )

            object_statistics.append(
                (
                    count,
                    object_id,
                    item,
                )
            )

        # ----------------------------------------------------
        # 最少的排前面
        # ----------------------------------------------------

        object_statistics.sort(
            key=lambda value:
                (
                    value[0],
                    value[1],
                )
        )

        for (
            count,
            object_id,
            item,
        ) in object_statistics:

            print(
                f"{object_id:<10}"
                f"{item.get('color', ''):<14}"
                f"{item.get('shape', ''):<16}"
                f"{count:4d} layouts"
            )

    # ========================================================
    # Color
    # ========================================================

    print()

    print(
        "--------------------------------------------------------------"
    )

    print(
        "COLOR COVERAGE"
    )

    print(
        "--------------------------------------------------------------"
    )

    all_colors = sorted(
        {
            item.get(
                "color",
                ""
            )
            for item
            in objects.values()
            if item.get(
                "color",
                ""
            )
        }
    )

    color_stats = [

        (
            len(
                color_layouts[
                    color
                ]
            ),
            color,
        )

        for color
        in all_colors
    ]

    color_stats.sort()

    for count, color in color_stats:

        print(
            f"{color:<20}"
            f"{count:4d} layouts"
        )

    # ========================================================
    # Shape
    # ========================================================

    print()

    print(
        "--------------------------------------------------------------"
    )

    print(
        "SHAPE COVERAGE"
    )

    print(
        "--------------------------------------------------------------"
    )

    all_shapes = sorted(
        {
            item.get(
                "shape",
                ""
            )
            for item
            in objects.values()
            if item.get(
                "shape",
                ""
            )
        }
    )

    shape_stats = [

        (
            len(
                shape_layouts[
                    shape
                ]
            ),
            shape,
        )

        for shape
        in all_shapes
    ]

    shape_stats.sort()

    for count, shape in shape_stats:

        print(
            f"{shape:<20}"
            f"{count:4d} layouts"
        )

    # ========================================================
    # Color x Shape
    # ========================================================

    print()

    print(
        "--------------------------------------------------------------"
    )

    print(
        "COLOR x SHAPE COVERAGE"
    )

    print(
        "--------------------------------------------------------------"
    )

    # --------------------------------------------------------
    # 只统计现实中确实登记过的颜色×形状组合。
    # 不人为要求不存在的组合。
    # --------------------------------------------------------

    registered_combinations = set()

    for item in objects.values():

        color = (
            item.get(
                "color",
                ""
            )
        )

        shape = (
            item.get(
                "shape",
                ""
            )
        )

        if (
            color
            and
            shape
        ):

            registered_combinations.add(
                (
                    color,
                    shape,
                )
            )

    combination_stats = [

        (
            len(
                combination_layouts[
                    combination
                ]
            ),
            combination,
        )

        for combination
        in registered_combinations
    ]

    combination_stats.sort()

    for (
        count,
        combination,
    ) in combination_stats:

        color, shape = (
            combination
        )

        print(
            f"{color:<14}"
            f"{shape:<18}"
            f"{count:4d} layouts"
        )

    # ========================================================
    # Scene
    # ========================================================

    print()

    print(
        "--------------------------------------------------------------"
    )

    print(
        "SCENE COVERAGE"
    )

    print(
        "--------------------------------------------------------------"
    )

    for scene_type in SCENE_TYPES:

        print(
            f"{scene_type:<20}"
            f"{len(scene_layouts[scene_type]):4d} layouts"
        )

    # ========================================================
    # Occlusion
    # ========================================================

    print()

    print(
        "--------------------------------------------------------------"
    )

    print(
        "OCCLUSION COVERAGE"
    )

    print(
        "--------------------------------------------------------------"
    )

    for level in OCCLUSION_LEVELS:

        print(
            f"{level:<20}"
            f"{len(occlusion_layouts[level]):4d} layouts"
        )

    # ========================================================
    # Edge
    # ========================================================

    print()

    print(
        "--------------------------------------------------------------"
    )

    print(
        "EDGE COVERAGE"
    )

    print(
        "--------------------------------------------------------------"
    )

    print(
        f"{'no':<20}"
        f"{len(edge_layouts['no']):4d} layouts"
    )

    print(
        f"{'yes':<20}"
        f"{len(edge_layouts['yes']):4d} layouts"
    )

    # ========================================================
    # Zone
    # ========================================================

    print()

    print(
        "--------------------------------------------------------------"
    )

    print(
        "ZONE COVERAGE"
    )

    print(
        "--------------------------------------------------------------"
    )

    for zone in ZONES:

        print(
            f"{zone:<20}"
            f"{len(zone_layouts[zone]):4d} layouts"
        )

    print(
        "=============================================================="
    )


# ============================================================
# 打印操作帮助
# ============================================================

def print_help() -> None:

    print()

    print(
        "========================================"
    )

    print(
        "KEYBOARD"
    )

    print(
        "========================================"
    )

    print(
        "N       : 新建独立Layout"
    )

    print(
        "S/SPACE : 保存当前图片"
    )

    print(
        "R       : 登记新的真实物体"
    )

    print(
        "O       : 查看所有登记物体"
    )

    print(
        "C       : 查看当前Layout"
    )

    print(
        "T       : 查看数据覆盖统计"
    )

    print(
        "H       : 显示帮助"
    )

    print(
        "Q/ESC   : 退出"
    )

    print(
        "========================================"
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:

    # ========================================================
    # 初始化目录及CSV
    # ========================================================

    initialize_dataset()

    session_id = (
        generate_session_id()
    )

    image_index = (
        get_next_image_index()
    )

    current_layout = None

    saved_this_session = 0

    # ========================================================
    # Startup
    # ========================================================

    print()

    print(
        "=============================================================="
    )

    print(
        "TOP INSTANCE SEGMENTATION DATASET CAPTURE"
    )

    print(
        "=============================================================="
    )

    print(
        f"Session:"
        f"\n{session_id}"
    )

    print()

    print(
        f"Dataset:"
        f"\n{DATASET_ROOT}"
    )

    print()

    print(
        f"Next image:"
        f"\ntop_{image_index:06d}.jpg"
    )

    print(
        "=============================================================="
    )

    print_help()

    # ========================================================
    # Camera
    # ========================================================

    camera = (
        TopCamera()
    )

    previous_time = (
        time.perf_counter()
    )

    smooth_fps = 0.0

    try:

        camera.open()

        # ----------------------------------------------------
        # 保存本次采集对应相机配置
        # ----------------------------------------------------

        save_session_config(
            session_id,
            camera,
        )

        print()

        print(
            "[CAMERA READY]"
        )

        print(
            camera.get_stream_info()
        )

        # ====================================================
        # 主循环
        # ====================================================

        while True:

            # ------------------------------------------------
            # 正式训练数据只能使用TopCamera.read()
            #
            # 内部已经：
            #
            # 原始1920x1080
            #       ↓
            # 去畸变
            #       ↓
            # SAFE ROI
            #       ↓
            # 870x840
            # ------------------------------------------------

            frame = (
                camera.read()
            )

            # =================================================
            # FPS
            # =================================================

            now = (
                time.perf_counter()
            )

            dt = (
                now
                -
                previous_time
            )

            previous_time = (
                now
            )

            if dt > 0:

                current_fps = (
                    1.0
                    /
                    dt
                )

                if (
                    smooth_fps
                    <=
                    0.0
                ):

                    smooth_fps = (
                        current_fps
                    )

                else:

                    smooth_fps = (
                        0.90
                        *
                        smooth_fps
                        +
                        0.10
                        *
                        current_fps
                    )

            # =================================================
            # 显示副本
            #
            # 这些文字绝不会保存进训练图。
            # =================================================

            display = (
                frame.copy()
            )

            if current_layout:

                layout_name = (
                    current_layout[
                        "layout_id"
                    ]
                )

                scene_type = (
                    current_layout[
                        "scene_type"
                    ]
                )

                object_count = (
                    current_layout[
                        "object_count"
                    ]
                )

                layout_saved_count = (
                    count_layout_images(
                        layout_name
                    )
                )

            else:

                layout_name = "NONE"

                scene_type = "NONE"

                object_count = 0

                layout_saved_count = 0

            # ------------------------------------------------
            # Session
            # ------------------------------------------------

            cv2.putText(
                display,
                (
                    f"Session: "
                    f"{session_id}"
                ),
                (
                    15,
                    28,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # Layout
            # ------------------------------------------------

            cv2.putText(
                display,
                (
                    f"Layout: "
                    f"{layout_name}"
                ),
                (
                    15,
                    58,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (
                    0,
                    255,
                    0,
                )
                if current_layout
                else
                (
                    0,
                    0,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # Scene
            # ------------------------------------------------

            cv2.putText(
                display,
                (
                    f"Scene: "
                    f"{scene_type}   "
                    f"Objects: "
                    f"{object_count}"
                ),
                (
                    15,
                    88,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # Layout保存数量
            # ------------------------------------------------

            cv2.putText(
                display,
                (
                    f"Layout images: "
                    f"{layout_saved_count}"
                ),
                (
                    15,
                    116,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # Next image / FPS
            # ------------------------------------------------

            cv2.putText(
                display,
                (
                    f"Next: "
                    f"top_{image_index:06d}.jpg   "
                    f"FPS: "
                    f"{smooth_fps:.1f}"
                ),
                (
                    15,
                    144,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # Bottom help
            # ------------------------------------------------

            cv2.putText(
                display,
                (
                    "N:new layout  "
                    "S:save  "
                    "R:register  "
                    "O:objects  "
                    "T:stats"
                ),
                (
                    15,
                    display.shape[0]
                    -
                    18,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Top Dataset Capture",
                display,
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # =================================================
            # Quit
            # =================================================

            if (
                key == ord("q")
                or
                key == ord("Q")
                or
                key == 27
            ):

                break

            # =================================================
            # N = 新Layout
            # =================================================

            elif (
                key == ord("n")
                or
                key == ord("N")
            ):

                # --------------------------------------------
                # 当前有未保存Layout
                # --------------------------------------------

                if (
                    current_layout is not None
                    and
                    not current_layout.get(
                        "_persisted",
                        False,
                    )
                ):

                    print()

                    answer = input(
                        "当前Layout还没有保存任何图片，"
                        "确定放弃并新建？ [y/N]: "
                    ).strip().lower()

                    if answer != "y":

                        print(
                            "[KEEP CURRENT LAYOUT]"
                        )

                        continue

                new_layout = (
                    create_layout(
                        session_id
                    )
                )

                if new_layout is not None:

                    current_layout = (
                        new_layout
                    )

            # =================================================
            # S / Space = 保存
            # =================================================

            elif (
                key == ord("s")
                or
                key == ord("S")
                or
                key == 32
            ):

                success = (
                    save_image(
                        frame,
                        image_index,
                        session_id,
                        current_layout,
                    )
                )

                if success:

                    image_index += 1

                    saved_this_session += 1

            # =================================================
            # R = 登记物体
            # =================================================

            elif (
                key == ord("r")
                or
                key == ord("R")
            ):

                register_object()

            # =================================================
            # O = 物体列表
            # =================================================

            elif (
                key == ord("o")
                or
                key == ord("O")
            ):

                print_objects()

            # =================================================
            # C = 当前Layout
            # =================================================

            elif (
                key == ord("c")
                or
                key == ord("C")
            ):

                print_current_layout(
                    current_layout
                )

            # =================================================
            # T = Statistics
            # =================================================

            elif (
                key == ord("t")
                or
                key == ord("T")
            ):

                print_statistics()

            # =================================================
            # H = Help
            # =================================================

            elif (
                key == ord("h")
                or
                key == ord("H")
            ):

                print_help()

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print()

        print(
            "=============================================================="
        )

        print(
            "CAPTURE FINISHED"
        )

        print(
            "=============================================================="
        )

        print(
            f"Session:"
            f"\n{session_id}"
        )

        print()

        print(
            f"Saved this session:"
            f"\n{saved_this_session}"
        )

        print()

        print(
            f"Next image:"
            f"\ntop_{image_index:06d}.jpg"
        )

        print()

        print_statistics()


if __name__ == "__main__":

    main()