from __future__ import annotations

"""
通用侧面相机分类数据集采集脚本。

数据集目录示例：

data/datasets/side_cls/
└── v1/
    ├── train/
    │   ├── 球/
    │   │   ├── black/
    │   │   ├── blue/
    │   │   ├── green/
    │   │   └── ...
    │   ├── 三棱柱/
    │   ├── 三棱锥/
    │   ├── 四棱台/
    │   ├── 四棱锥/
    │   ├── 五棱柱/
    │   ├── 圆台/
    │   ├── 圆柱/
    │   ├── 圆锥/
    │   └── 正方体/
    ├── val/
    └── test/

以后新建 v2 / v3 时，只需要修改：

    DATASET_VERSION = "v1"

例如：

    DATASET_VERSION = "v2"

脚本会自动扫描该版本目录中的：
    split -> shape -> color

不需要在代码里写死形状和颜色。

采集链与正式侧面识别保持一致：

    runtime_config.yaml
        ↓
    SideCamera.configure()
        ↓
    SideCamera.read()
        ↓
    当前机械 ROI
        ↓
    SideObjectExtractor + 固定空背景
        ↓
    物体 crop
        ↓
    保存 image_000001.jpg ...

运行：

    cd ~/smart_sorting_vision
    source .venv/bin/activate
    python scripts/dataset/side_cls/capture_side_cls_dataset.py

窗口按键：
    SPACE / S : 保存当前物体 crop
    H         : 打印帮助
    Q / ESC   : 退出
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


# ============================================================================
# 以后通常只改这里
# ============================================================================

DATASET_VERSION = "school"


# ============================================================================
# 一般不需要修改
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    ROOT
    / "data"
    / "datasets"
    / "side_cls"
    / DATASET_VERSION
)

RUNTIME_CONFIG = (
    ROOT
    / "config"
    / "runtime_config.yaml"
)

JPEG_QUALITY = 95
FILE_DIGITS = 6

MAIN_WINDOW = "Side Dataset Capture"
CROP_WINDOW = "Side Object Crop"

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# ============================================================================
# 工程导入
# ============================================================================

def _ensure_project_import_path() -> None:
    root_text = str(ROOT)

    if root_text not in sys.path:
        sys.path.insert(0, root_text)


# ============================================================================
# 配置读取
# ============================================================================

def _load_runtime_config() -> dict:
    if not RUNTIME_CONFIG.exists():
        raise FileNotFoundError(
            f"找不到运行配置：{RUNTIME_CONFIG}"
        )

    with RUNTIME_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f) or {}

    return cfg


def _resolve_project_path(
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


# ============================================================================
# 目录菜单
# ============================================================================

def _child_directories(
    parent: Path,
) -> list[Path]:
    if not parent.exists():
        return []

    return sorted(
        [
            path
            for path in parent.iterdir()
            if path.is_dir()
        ],
        key=lambda p: p.name,
    )


def _choose_directory(
    parent: Path,
    title: str,
    *,
    allow_back: bool = False,
) -> Path | None:
    directories = _child_directories(
        parent
    )

    if not directories:
        raise RuntimeError(
            f"目录下没有可选子目录：{parent}"
        )

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print()

    for index, directory in enumerate(
        directories,
        start=1,
    ):
        print(
            f"  [{index:>2}] "
            f"{directory.name}"
        )

    print()

    if allow_back:
        print("  [ 0] 返回")
        print()

    while True:
        value = input(
            "请输入数字序号：\n> "
        ).strip()

        try:
            index = int(value)

        except ValueError:
            print(
                "[ERROR] 请输入数字序号。"
            )
            continue

        if allow_back and index == 0:
            return None

        if 1 <= index <= len(
            directories
        ):
            return directories[
                index - 1
            ]

        if allow_back:
            print(
                f"[ERROR] 请输入 "
                f"0~{len(directories)}。"
            )
        else:
            print(
                f"[ERROR] 请输入 "
                f"1~{len(directories)}。"
            )


def _choose_save_directory(
) -> tuple[
    Path,
    str,
    str,
    str,
]:
    """
    数字菜单：
        split
        -> shape
        -> color

    目录内容完全从 DATASET_ROOT 自动扫描。
    """

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            "\n数据集版本目录不存在：\n"
            f"  {DATASET_ROOT}\n\n"
            "请先按原有格式创建目录。"
        )

    while True:
        split_dir = _choose_directory(
            DATASET_ROOT,
            (
                f"当前数据集：side_cls/"
                f"{DATASET_VERSION}\n"
                "选择 train / val / test"
            ),
            allow_back=False,
        )

        assert split_dir is not None

        shape_dir = _choose_directory(
            split_dir,
            (
                f"当前 split："
                f"{split_dir.name}\n"
                "选择形状"
            ),
            allow_back=True,
        )

        if shape_dir is None:
            continue

        color_dirs = _child_directories(
            shape_dir
        )

        # 保留兼容性：
        # 如果某个版本以后不再使用 color 子目录，
        # 就直接保存到 shape 目录。
        if not color_dirs:
            save_dir = shape_dir
            color_name = ""
        else:
            color_dir = _choose_directory(
                shape_dir,
                (
                    f"当前形状："
                    f"{shape_dir.name}\n"
                    "选择颜色"
                ),
                allow_back=True,
            )

            if color_dir is None:
                continue

            save_dir = color_dir
            color_name = color_dir.name

        print()
        print("=" * 72)
        print("采集目标确认")
        print("=" * 72)
        print(
            f"Dataset : {DATASET_VERSION}"
        )
        print(
            f"Split   : {split_dir.name}"
        )
        print(
            f"Shape   : {shape_dir.name}"
        )
        print(
            f"Color   : "
            f"{color_name if color_name else '(none)'}"
        )
        print(
            f"Save    : {save_dir}"
        )
        print("=" * 72)
        print()

        return (
            save_dir,
            split_dir.name,
            shape_dir.name,
            color_name,
        )


# ============================================================================
# 图片编号
# ============================================================================

def _next_image_index(
    save_dir: Path,
) -> int:
    """
    与旧侧面数据集保持一致：

        每个 shape/color 文件夹独立编号。

    例如：

        train/圆柱/red/image_000001.jpg
        train/圆柱/red/image_000002.jpg

        train/圆柱/blue/image_000001.jpg

    如果该目录已经删空，就会重新从 000001 开始。
    """

    maximum_index = 0

    if not save_dir.exists():
        return 1

    for path in save_dir.iterdir():
        if (
            not path.is_file()
            or path.suffix.lower()
            not in IMAGE_SUFFIXES
        ):
            continue

        stem = path.stem

        if not stem.startswith(
            "image_"
        ):
            continue

        number_text = stem[
            len("image_"):
        ]

        try:
            index = int(
                number_text
            )

        except ValueError:
            continue

        maximum_index = max(
            maximum_index,
            index,
        )

    return maximum_index + 1


# ============================================================================
# 中文路径安全保存
# ============================================================================

def _imwrite_unicode(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    if not ok:
        raise RuntimeError(
            f"JPEG 编码失败：{path}"
        )

    encoded.tofile(
        str(path)
    )


# ============================================================================
# 预览叠字
# ============================================================================

def _draw_status(
    frame: np.ndarray,
    *,
    split: str,
    shape: str,
    color: str,
    next_index: int,
    saved_count: int,
    found: bool,
    reason: str,
) -> np.ndarray:
    display = frame.copy()

    lines = [
        (
            f"{DATASET_VERSION} | "
            f"{split} | "
            f"{shape} | "
            f"{color if color else '-'}"
        ),
        (
            f"Object={'YES' if found else 'NO'} "
            f"reason={reason}"
        ),
        (
            f"Next=image_"
            f"{next_index:0{FILE_DIGITS}d}.jpg "
            f"saved={saved_count}"
        ),
        "SPACE/S=save  H=help  Q=quit",
    ]

    y = 28

    for line in lines:
        cv2.putText(
            display,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                255,
                0,
            )
            if found
            else (
                0,
                200,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        y += 28

    return display


def _print_help(
    *,
    save_dir: Path,
    split: str,
    shape: str,
    color: str,
    next_index: int,
    background_path: Path,
) -> None:
    print()
    print("=" * 72)
    print("SIDE CLASSIFICATION DATASET CAPTURE")
    print("=" * 72)
    print(
        f"Dataset    : {DATASET_VERSION}"
    )
    print(
        f"Split      : {split}"
    )
    print(
        f"Shape      : {shape}"
    )
    print(
        f"Color      : "
        f"{color if color else '(none)'}"
    )
    print(
        f"Save dir   : {save_dir}"
    )
    print(
        f"Background : {background_path}"
    )
    print(
        f"Next image : "
        f"image_{next_index:0{FILE_DIGITS}d}.jpg"
    )
    print()
    print(
        "SPACE / S : 保存当前检测到的物体 crop"
    )
    print(
        "H         : 打印本帮助"
    )
    print(
        "Q / ESC   : 退出"
    )
    print("=" * 72)
    print()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    (
        save_dir,
        split_name,
        shape_name,
        color_name,
    ) = _choose_save_directory()

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_index = _next_image_index(
        save_dir
    )

    saved_count = 0

    # ------------------------------------------------------------------------
    # 使用正式运行配置
    # ------------------------------------------------------------------------

    cfg = _load_runtime_config()

    try:
        side_camera_cfg = (
            cfg["camera"]["side"]
        )

    except KeyError as exc:
        raise RuntimeError(
            "runtime_config.yaml "
            "缺少 camera.side"
        ) from exc

    side_vision_cfg = (
        cfg
        .get(
            "vision",
            {},
        )
        .get(
            "side",
            {},
        )
        or {}
    )

    background_value = (
        cfg
        .get(
            "calibration",
            {},
        )
        .get(
            "side_background"
        )
    )

    if not background_value:
        raise RuntimeError(
            "runtime_config.yaml "
            "缺少 calibration.side_background"
        )

    background_path = (
        _resolve_project_path(
            background_value
        )
    )

    if not background_path.exists():
        raise FileNotFoundError(
            "\n侧面固定背景不存在：\n"
            f"  {background_path}\n\n"
            "请先采集与当前侧面 ROI "
            "完全对应的空背景。"
        )

    _ensure_project_import_path()

    from vision.cameras.side_camera import (
        SideCamera,
    )

    from vision.processors.side_object_extractor import (
        SideObjectExtractor,
    )

    # 关键：
    # 采集脚本使用正式 runtime_config 中的侧面相机参数，
    # 不再依赖 side_camera.py 里的旧默认值。
    SideCamera.configure(
        side_camera_cfg
    )

    padding_ratio = float(
        side_vision_cfg.get(
            "padding_ratio",
            0.15,
        )
    )

    extractor = SideObjectExtractor(
        padding_ratio=padding_ratio
    )

    background = extractor.load_background(
        background_path
    )

    camera = SideCamera()

    _print_help(
        save_dir=save_dir,
        split=split_name,
        shape=shape_name,
        color=color_name,
        next_index=image_index,
        background_path=background_path,
    )

    try:
        camera.open()

        print(
            "[OK] Side camera opened: "
            f"{camera.device}"
        )

        print(
            "[CONFIG] ROI = "
            f"[{SideCamera.ROI_X1}, "
            f"{SideCamera.ROI_Y1}, "
            f"{SideCamera.ROI_X2}, "
            f"{SideCamera.ROI_Y2}]"
        )

        print(
            "[CONFIG] ROI size = "
            f"{SideCamera.ROI_WIDTH}x"
            f"{SideCamera.ROI_HEIGHT}"
        )

        print(
            "[CONFIG] padding_ratio = "
            f"{padding_ratio}"
        )

        print(
            "[BACKGROUND] "
            f"{background_path}"
        )

        # 再额外丢弃少量帧，
        # 保证采集界面开始时图像已经稳定。
        current_frame = None

        for _ in range(10):
            current_frame = (
                camera.read()
            )

        if current_frame is None:
            raise RuntimeError(
                "无法读取侧面相机画面"
            )

        # 这个检查很重要：
        # 修改 ROI 后必须重新采空背景。
        if (
            current_frame.shape
            != background.shape
        ):
            raise RuntimeError(
                "\n当前侧面 ROI 与固定背景尺寸不一致："
                f"\n  current    = "
                f"{current_frame.shape}"
                f"\n  background = "
                f"{background.shape}"
                "\n\n修改 SideCamera ROI 后，"
                "必须重新采集侧面固定背景。"
            )

        while True:
            mechanical_roi = (
                camera.read()
            )

            result = extractor.extract(
                mechanical_roi
            )

            crop = None

            if (
                result.found
                and result.crop is not None
                and result.crop.size > 0
            ):
                crop = result.crop

            display = _draw_status(
                mechanical_roi,
                split=split_name,
                shape=shape_name,
                color=color_name,
                next_index=image_index,
                saved_count=saved_count,
                found=bool(
                    result.found
                ),
                reason=str(
                    result.reason
                ),
            )

            # 将实际 crop 区域画在机械 ROI 上。
            crop_bbox = getattr(
                result,
                "crop_bbox",
                None,
            )

            if crop_bbox is not None:
                x1, y1, x2, y2 = (
                    map(
                        int,
                        crop_bbox,
                    )
                )

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (
                        max(
                            x1,
                            x2 - 1,
                        ),
                        max(
                            y1,
                            y2 - 1,
                        ),
                    ),
                    (
                        0,
                        255,
                        0,
                    ),
                    2,
                )

            cv2.imshow(
                MAIN_WINDOW,
                display,
            )

            if crop is not None:
                cv2.imshow(
                    CROP_WINDOW,
                    crop,
                )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # --------------------------------------------------------------
            # Quit
            # --------------------------------------------------------------

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            # --------------------------------------------------------------
            # Help
            # --------------------------------------------------------------

            if key in (
                ord("h"),
                ord("H"),
            ):
                _print_help(
                    save_dir=save_dir,
                    split=split_name,
                    shape=shape_name,
                    color=color_name,
                    next_index=image_index,
                    background_path=background_path,
                )

                continue

            # --------------------------------------------------------------
            # Save
            # --------------------------------------------------------------

            if key in (
                32,
                ord("s"),
                ord("S"),
            ):
                if crop is None:
                    print(
                        "[SAVE] 当前没有可靠检测到物体，"
                        "未保存。"
                    )

                    continue

                filename = (
                    f"image_"
                    f"{image_index:0{FILE_DIGITS}d}"
                    f".jpg"
                )

                save_path = (
                    save_dir / filename
                )

                if save_path.exists():
                    print(
                        "[WARN] 文件已经存在，"
                        "自动寻找下一个编号："
                        f"{save_path}"
                    )

                    image_index = (
                        _next_image_index(
                            save_dir
                        )
                    )

                    continue

                _imwrite_unicode(
                    save_path,
                    crop,
                )

                h, w = crop.shape[:2]

                print(
                    "[SAVE] "
                    f"{save_path.relative_to(ROOT)} "
                    f"crop={w}x{h}"
                )

                image_index += 1
                saved_count += 1

    finally:
        try:
            camera.release()

        except Exception:
            pass

        cv2.destroyAllWindows()

    print()
    print("=" * 72)
    print(
        f"[DONE] 本次保存 "
        f"{saved_count} 张。"
    )
    print(
        "[NEXT] 当前目录下次将从 "
        f"image_{_next_image_index(save_dir):0{FILE_DIGITS}d}.jpg "
        "继续。"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
