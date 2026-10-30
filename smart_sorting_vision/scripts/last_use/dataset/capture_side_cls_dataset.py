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
    直接保存整张 ROI（模型负责找物体和 mask）
        ↓
    保存 image_000001.jpg ...

运行：

    cd ~/smart_sorting_vision
    source .venv/bin/activate
    python scripts/dataset/side_cls/capture_side_cls_dataset.py

窗口按键：
    SPACE / S : 保存当前完整侧面 ROI
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

DATASET_VERSION = "v3"


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
) -> np.ndarray:
    display = frame.copy()

    lines = [
        (
            f"{DATASET_VERSION} | "
            f"{split} | "
            f"{shape} | "
            f"{color if color else '-'}"
        ),
        "Input=full side ROI (no background crop)",
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
            (0, 255, 0),
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
        f"Next image : "
        f"image_{next_index:0{FILE_DIGITS}d}.jpg"
    )
    print()
    print(
        "SPACE / S : 保存当前完整侧面 ROI"
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
    # 先按实际保存目录逐级选择类别。
    save_dir, split_name, shape_name, color_name = _choose_save_directory()
    save_dir.mkdir(parents=True, exist_ok=True)
    image_index = _next_image_index(save_dir)
    saved_count = 0

    # 只读取正式侧面相机配置；不加载固定背景，也不做 crop。
    cfg = _load_runtime_config()
    try:
        side_camera_cfg = cfg["camera"]["side"]
    except KeyError as exc:
        raise RuntimeError("runtime_config.yaml 缺少 camera.side") from exc

    _ensure_project_import_path()
    from vision.cameras.side_camera import SideCamera

    SideCamera.configure(side_camera_cfg)
    camera = SideCamera()

    _print_help(
        save_dir=save_dir,
        split=split_name,
        shape=shape_name,
        color=color_name,
        next_index=image_index,
    )

    try:
        camera.open()
        print(f"[OK] Side camera opened: {camera.device}")
        print(
            "[CONFIG] ROI = "
            f"[{SideCamera.ROI_X1}, {SideCamera.ROI_Y1}, "
            f"{SideCamera.ROI_X2}, {SideCamera.ROI_Y2}]"
        )
        print(
            "[CONFIG] ROI size = "
            f"{SideCamera.ROI_WIDTH}x{SideCamera.ROI_HEIGHT}"
        )
        print("[INFO] 保存内容：SideCamera.read() 返回的完整侧面 ROI。")
        print("[INFO] 后续由 YOLO-seg 在整张 ROI 上找物体，再使用 mask 做颜色/二维码识别。")

        for _ in range(10):
            camera.read()

        while True:
            frame = camera.read()
            if frame is None or frame.size == 0:
                print("[WARN] 侧面相机读取到空帧。")
                continue

            display = _draw_status(
                frame,
                split=split_name,
                shape=shape_name,
                color=color_name,
                next_index=image_index,
                saved_count=saved_count,
            )
            cv2.imshow(MAIN_WINDOW, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("h"), ord("H")):
                _print_help(
                    save_dir=save_dir,
                    split=split_name,
                    shape=shape_name,
                    color=color_name,
                    next_index=image_index,
                )
                continue

            if key not in (32, ord("s"), ord("S")):
                continue

            save_path = save_dir / f"image_{image_index:0{FILE_DIGITS}d}.jpg"
            if save_path.exists():
                image_index = _next_image_index(save_dir)
                save_path = save_dir / f"image_{image_index:0{FILE_DIGITS}d}.jpg"

            _imwrite_unicode(save_path, frame)
            height, width = frame.shape[:2]
            print(
                "[SAVE] "
                f"{save_path.relative_to(ROOT)} "
                f"full_roi={width}x{height}"
            )
            image_index += 1
            saved_count += 1
    finally:
        try:
            camera.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    print("\n" + "=" * 72)
    print(f"[DONE] 本次共保存 {saved_count} 张完整侧面 ROI。")
    print("=" * 72)


if __name__ == "__main__":
    main()
