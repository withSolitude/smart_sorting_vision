from __future__ import annotations

"""
通用顶部相机数据集采集脚本。

目录：
    smart_sorting_vision/data/datasets/top_seg/<版本>/

例如：
    data/datasets/top_seg/v1/
    data/datasets/top_seg/v2/

以后更换数据集版本，只需要修改：

    DATASET_VERSION = "v1"

为：

    DATASET_VERSION = "v2"

不需要修改任何目录代码。

运行：
    cd ~/smart_sorting_vision
    source .venv/bin/activate
    python scripts/dataset/top_seg/capture_top_seg_dataset.py

运行后：
1. 数字选择 train / val / test
2. 按层级数字选择采集目录
3. 打开顶部相机

相机窗口：
    SPACE / S : 保存当前图片
    H         : 打印帮助
    Q / ESC   : 退出

图片编号会自动扫描整个当前版本数据集：
- 如果没有任何图片，从 000001 开始
- 如果已有 000001~000120，则继续从 000121 开始
- 删除全部旧图片后，会自动重新从 000001 开始
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


# ============================================================================
# 以后新建 v2 / v3 时，通常只改这里
# ============================================================================

DATASET_VERSION = "v1"

# ============================================================================
# 一般不需要修改
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]

TOP_SEG_ROOT = ROOT / "data" / "datasets" / "top_seg"
DATASET_ROOT = TOP_SEG_ROOT / DATASET_VERSION

RUNTIME_CONFIG = ROOT / "config" / "runtime_config.yaml"

JPEG_QUALITY = 95
FILE_DIGITS = 6

WINDOW_NAME = "Top Dataset Capture"

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

CAPTURE_LOG = DATASET_ROOT / "capture_log.csv"


# ============================================================================
# Project / Camera
# ============================================================================

def _ensure_project_import_path() -> None:
    root_text = str(ROOT)

    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _read_top_camera_config() -> tuple[dict, Path]:
    """
    从正式 runtime_config.yaml 读取顶部相机参数。

    这样数据采集与正式程序保持相同：
        相机参数
        -> 去畸变
        -> SAFE ROI
        -> 保存
    """

    if not RUNTIME_CONFIG.exists():
        raise FileNotFoundError(
            f"找不到运行配置：{RUNTIME_CONFIG}"
        )

    with RUNTIME_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f) or {}

    try:
        top_cfg = cfg["camera"]["top"]
        calibration_value = cfg["calibration"]["top_intrinsics"]

    except KeyError as exc:
        raise RuntimeError(
            "runtime_config.yaml 缺少 "
            "camera.top 或 calibration.top_intrinsics"
        ) from exc

    calibration_path = Path(
        calibration_value
    ).expanduser()

    if not calibration_path.is_absolute():
        calibration_path = ROOT / calibration_path

    return (
        dict(top_cfg),
        calibration_path.resolve(),
    )


# ============================================================================
# 数据集目录扫描
# ============================================================================

def _direct_child_folders(
    parent: Path,
) -> list[Path]:
    """只返回当前目录的直接子目录，不递归展开更深层目录。"""

    if not parent.exists():
        return []

    folders = [
        path
        for path in parent.iterdir()
        if path.is_dir()
    ]

    folders.sort(
        key=lambda path: path.name.casefold()
    )

    return folders


# ============================================================================
# 数字菜单
# ============================================================================

def _select_split() -> str:
    """
    数字选择 train / val / test。
    """

    options = [
        "train",
        "val",
        "test",
    ]

    print()
    print("=" * 72)
    print(
        f" 当前数据集：top_seg/{DATASET_VERSION}"
    )
    print("=" * 72)
    print()
    print("请选择数据集区域：")
    print()

    for i, name in enumerate(
        options,
        start=1,
    ):
        split_root = (
            DATASET_ROOT / name
        )

        folders = _direct_child_folders(
            split_root
        )

        print(
            f"  [{i}] "
            f"{name:<5} "
            f"({len(folders)} 个一级目录)"
        )

    print()
    print("  [0] 退出")
    print()

    while True:

        text = input(
            "请输入数字序号：\n> "
        ).strip()

        try:
            number = int(text)

        except ValueError:
            print(
                "[ERROR] 请输入数字。"
            )
            continue

        if number == 0:
            raise KeyboardInterrupt

        if 1 <= number <= len(options):

            split = options[
                number - 1
            ]

            split_root = (
                DATASET_ROOT / split
            )

            if not split_root.exists():
                print(
                    f"[ERROR] 目录不存在："
                    f"{split_root}"
                )
                continue

            return split

        print(
            f"[ERROR] 请输入 "
            f"0~{len(options)}。"
        )


def _select_capture_folder(
    split: str,
) -> tuple[str, Path]:
    """
    按层级选择当前 split 下的采集目录。

    例如目录结构为：

        train/single/yellow/cube
        train/single/white/cylinder
        train/stacked/yellow

    菜单会依次显示：

        train -> single / stacked
        single -> yellow / white
        yellow -> cube

    每次只显示当前目录的直接子目录，
    不会把所有深层目录一次性列出来。
    """

    split_root = (
        DATASET_ROOT / split
    )

    if not split_root.exists():
        raise RuntimeError(
            f"没有找到数据集目录："
            f"{split_root}\n"
            f"请先建立数据集文件夹结构。"
        )

    # current_root 表示当前所在层级；每次只读取这一层的直接子目录。
    current_root = split_root
    while True:
        folders = _direct_child_folders(
            current_root
        )

        # 没有子目录时，当前目录就是最终图片保存目录。
        if not folders:
            if current_root == split_root:
                raise RuntimeError(
                    f"没有找到可采集目录："
                    f"{split_root}\n"
                    f"请先建立数据集文件夹结构。"
                )

            relative = current_root.relative_to(
                split_root
            ).as_posix()

            print()
            print(
                "[SELECT] "
                f"{split}/{relative}"
            )

            return relative, current_root

        relative_current = current_root.relative_to(
            split_root
        ).as_posix()
        display_current = (
            "根目录"
            if relative_current == "."
            else relative_current
        )

        print()
        print("=" * 72)
        print(
            f" 数据集："
            f"{DATASET_VERSION}/{split}"
        )
        print(
            f" 当前层级：{display_current}"
        )
        print("=" * 72)
        print()
        print("请选择下一级目录：")
        print()

        for index, folder in enumerate(
            folders,
            start=1,
        ):
            print(
                f"  [{index:>3}] "
                f"{folder.name}"
            )

        print()
        print("  [  0] 返回上一级")
        print()

        text = input(
            "请输入数字序号：\n> "
        ).strip()

        try:
            number = int(text)

        except ValueError:
            print(
                "[ERROR] 请输入数字。"
            )
            continue

        if number == 0:
            if current_root == split_root:
                return "", Path()

            current_root = current_root.parent
            continue

        if 1 <= number <= len(folders):
            current_root = folders[
                number - 1
            ]
            continue

        print(
            f"[ERROR] 请输入 "
            f"0~{len(folders)}。"
        )


def _interactive_select_target(
) -> tuple[str, str, Path]:
    """
    完整的数字菜单流程：

        选择 split
        ↓
        按层级进入采集目录
    """

    while True:

        split = _select_split()

        folder, target = (
            _select_capture_folder(
                split
            )
        )

        if folder:
            return (
                split,
                folder,
                target,
            )

        print(
            "[INFO] 返回上一级。"
        )


# ============================================================================
# 图片编号
# ============================================================================

def _find_next_global_index() -> int:
    """
    整个当前数据集版本使用一个全局编号。

    例如：

        v1/train/contact/top_v1_000001.jpg
        v1/train/edge/top_v1_000002.jpg
        v1/val/contact/top_v1_000003.jpg

    如果整个 v1 一张图片都没有：
        返回 1

    如果当前最大编号为 69：
        返回 70

    如果已经把旧图全部删除：
        自动重新返回 1
    """

    max_index = 0

    number_pattern = re.compile(
        r"_(\d+)$"
    )

    if not DATASET_ROOT.exists():
        return 1

    for path in DATASET_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in IMAGE_SUFFIXES
        ):
            continue

        match = number_pattern.search(
            path.stem
        )

        if not match:
            continue

        try:
            number = int(
                match.group(1)
            )

        except ValueError:
            continue

        max_index = max(
            max_index,
            number,
        )

    return max_index + 1


# ============================================================================
# 保存图片
# ============================================================================

def _imwrite_unicode(
    path: Path,
    frame: np.ndarray,
) -> None:
    """
    支持中文目录名的 JPEG 保存。
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
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
# 采集日志
# ============================================================================

def _append_capture_log(
    image_path: Path,
    split: str,
    folder: str,
    width: int,
    height: int,
) -> None:

    CAPTURE_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_file = (
        not CAPTURE_LOG.exists()
    )

    with CAPTURE_LOG.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "dataset_version",
                "split",
                "folder",
                "image",
                "width",
                "height",
            ],
        )

        if new_file:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp":
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),

                "dataset_version":
                    DATASET_VERSION,

                "split":
                    split,

                "folder":
                    folder,

                "image":
                    image_path.relative_to(
                        DATASET_ROOT
                    ).as_posix(),

                "width":
                    width,

                "height":
                    height,
            }
        )


# ============================================================================
# 预览画面
# ============================================================================

def _overlay_status(
    frame: np.ndarray,
    *,
    split: str,
    folder: str,
    next_index: int,
    saved_count: int,
) -> np.ndarray:

    canvas = frame.copy()

    h, w = canvas.shape[:2]

    overlay = canvas.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, 118),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        canvas,
        0.45,
        0.0,
        canvas,
    )

    lines = [
        (
            f"dataset={DATASET_VERSION}  "
            f"split={split}"
        ),

        f"folder={folder}",

        (
            f"next="
            f"{next_index:0{FILE_DIGITS}d}  "
            f"saved={saved_count}  "
            f"SPACE/S=save  Q=quit"
        ),
    ]

    for i, text in enumerate(
        lines
    ):

        cv2.putText(
            canvas,
            text,
            (
                16,
                30 + i * 34,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            (
                0.72
                if i < 2
                else 0.60
            ),
            (
                255,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

    return canvas


# ============================================================================
# 帮助
# ============================================================================

def _print_help(
    target: Path,
    split: str,
    folder: str,
    next_index: int,
) -> None:

    print()
    print("=" * 72)
    print(" TOP SEG DATASET CAPTURE")
    print("=" * 72)

    print(
        f"Project root    : "
        f"{ROOT}"
    )

    print(
        f"Dataset version : "
        f"{DATASET_VERSION}"
    )

    print(
        f"Dataset root    : "
        f"{DATASET_ROOT}"
    )

    print(
        f"Split           : "
        f"{split}"
    )

    print(
        f"Folder          : "
        f"{folder}"
    )

    print(
        f"Save to         : "
        f"{target}"
    )

    print(
        f"Next image      : "
        f"top_{DATASET_VERSION}_"
        f"{next_index:0{FILE_DIGITS}d}.jpg"
    )

    print()

    print("窗口按键：")
    print(
        "  SPACE / S : 保存当前帧"
    )
    print(
        "  H         : 打印本帮助"
    )
    print(
        "  Q / ESC   : 结束本次采集"
    )

    print("=" * 72)
    print()


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            "\n"
            f"数据集版本目录不存在：\n"
            f"  {DATASET_ROOT}\n\n"
            f"请先按原来的目录格式创建 "
            f"{DATASET_VERSION}。"
        )

    # ------------------------------------------------------------------------
    # 数字菜单选择
    # ------------------------------------------------------------------------

    try:
        split, folder, target = (
            _interactive_select_target()
        )

    except KeyboardInterrupt:
        print()
        print("[EXIT] 已退出。")
        return

    # ------------------------------------------------------------------------
    # 自动检测图片编号
    # ------------------------------------------------------------------------

    next_index = (
        _find_next_global_index()
    )

    saved_count = 0

    _print_help(
        target,
        split,
        folder,
        next_index,
    )

    # ------------------------------------------------------------------------
    # 加载正式顶部相机
    # ------------------------------------------------------------------------

    _ensure_project_import_path()

    from vision.cameras.top_camera import (
        TopCamera,
    )

    top_cfg, calibration_path = (
        _read_top_camera_config()
    )

    TopCamera.configure(
        top_cfg,
        calibration_file=calibration_path,
    )

    camera = TopCamera()

    # ------------------------------------------------------------------------
    # Camera loop
    # ------------------------------------------------------------------------

    try:

        camera.open()

        print(
            f"[OK] Top camera opened: "
            f"{TopCamera.DEVICE}"
        )

        print(
            "[INFO] 保存图像使用 "
            "TopCamera.read() 输出。"
        )

        print(
            "[INFO] 即："
            "去畸变 + SAFE ROI。"
        )

        print(
            f"[INFO] 下一张编号："
            f"{next_index:0{FILE_DIGITS}d}"
        )

        print()

        while True:

            frame = camera.read()

            if (
                frame is None
                or frame.size == 0
            ):
                print(
                    "[WARN] "
                    "顶部相机读取到空帧。"
                )
                continue

            h, w = frame.shape[:2]

            display = _overlay_status(
                frame,
                split=split,
                folder=folder,
                next_index=next_index,
                saved_count=saved_count,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
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
                    target,
                    split,
                    folder,
                    next_index,
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

                filename = (
                    f"top_"
                    f"{DATASET_VERSION}_"
                    f"{next_index:0{FILE_DIGITS}d}"
                    f".jpg"
                )

                save_path = (
                    target / filename
                )

                # 理论上正常不会出现；
                # 防止意外覆盖。
                if save_path.exists():

                    print(
                        "[WARN] "
                        f"文件已经存在："
                        f"{save_path}"
                    )

                    next_index += 1

                    continue

                _imwrite_unicode(
                    save_path,
                    frame,
                )

                _append_capture_log(
                    save_path,
                    split,
                    folder,
                    w,
                    h,
                )

                print(
                    "[SAVE] "
                    f"{save_path.relative_to(DATASET_ROOT).as_posix()} "
                    f"({w}x{h})"
                )

                next_index += 1
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
        f"[DONE] 本次共保存 "
        f"{saved_count} 张图片。"
    )

    print(
        f"[NEXT] 下次继续采集将自动从 "
        f"{_find_next_global_index():0{FILE_DIGITS}d} "
        f"开始。"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
