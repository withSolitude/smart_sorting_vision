"""采集侧面形状分类数据。

形状模型现在直接看完整侧面机械 ROI，因此每张图片都保存完整 ROI，不能再
保存背景差分得到的 crop。空托盘请保存到 ``no_object`` 目录，作为正式流程
的“无物体”类别。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.cameras.side_camera import SideCamera
from runtime.app_config import load_runtime_config


DATASET_ROOT = ROOT / "data" / "datasets" / "side_cls" / "v1"


def choose_split() -> Path:
    """选择 train、val 或 test 数据集目录。"""
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    splits = [DATASET_ROOT / name for name in ("train", "val", "test")]
    for path in splits:
        path.mkdir(exist_ok=True)

    print("\n选择数据集：")
    for index, path in enumerate(splits, 1):
        print(f"{index}. {path.name}")
    while True:
        try:
            index = int(input("请输入序号：").strip())
        except ValueError:
            print("请输入数字。")
            continue
        if 1 <= index <= len(splits):
            return splits[index - 1]
        print("序号超出范围。")


def choose_save_directory() -> Path:
    """输入一个一级类别目录名，保证 YOLO 分类目录结构清晰。"""
    split_dir = choose_split()
    print("\n输入类别目录名，例如：三棱柱、正方体、no_object")
    label = input("类别目录名：").strip()
    if not label or label in (".", "..") or "/" in label or "\\" in label:
        raise ValueError("类别目录名不能为空，也不能包含路径分隔符")
    save_dir = split_dir / label
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def next_image_index(save_dir: Path) -> int:
    maximum = 0
    for path in save_dir.glob("image_*.jpg"):
        try:
            maximum = max(maximum, int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return maximum + 1


def main() -> None:
    # 采集脚本也使用正式 YAML 的相机设备、分辨率、ROI 和曝光参数，
    # 避免保存出的训练图尺寸与正式程序不一致。
    config_path = ROOT / "config" / "runtime_config.yaml"
    config = load_runtime_config(ROOT, config_path)
    SideCamera.configure(config["camera"]["side"])

    save_dir = choose_save_directory()
    image_index = next_image_index(save_dir)
    camera = SideCamera()

    try:
        camera.open()
        frame = None
        for _ in range(20):
            frame = camera.read()
        if frame is None:
            raise RuntimeError("无法读取侧面相机画面")

        print("\n" + "=" * 64)
        print("SIDE SHAPE DATASET CAPTURE")
        print("保存对象：完整侧面 ROI，不做背景差分、不保存 crop")
        print(f"类别目录：{save_dir}")
        print("S：保存当前完整 ROI    Q / ESC：退出")
        print("空托盘采集时请把类别目录填写为 no_object")
        print("=" * 64)

        while True:
            frame = camera.read()
            if frame is None:
                continue

            display = frame.copy()
            cv2.putText(
                display,
                f"label={save_dir.name}  next={image_index:06d}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "S=save full ROI   Q=quit",
                (16, display.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Side Shape Dataset Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key not in (ord("s"), ord("S")):
                continue

            save_path = save_dir / f"image_{image_index:06d}.jpg"
            if not cv2.imwrite(
                str(save_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                print(f"[SAVE ERROR] {save_path}")
                continue
            print(f"[SAVE] {save_path.relative_to(ROOT)}")
            image_index += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nShape dataset capture finished.")


if __name__ == "__main__":
    main()
