# from ultralytics import YOLO

# MODEL = "outputs/training/top_seg/top_seg_v1_aug/weights/best.pt"

# model = YOLO(MODEL)

# model.export(
#     format="engine",
#     imgsz=832,
#     quantize=16,
#     batch=1,
#     dynamic=False,
#     simplify=False,
#     device=0,
# )


# from ultralytics import YOLO

# model = YOLO(
#     "outputs/training/side_cls/school_test/weights/best.pt"
# )

# model.export(
#     format="engine",
#     imgsz=224,          
#     quantize=16,
#     batch=1,
#     dynamic=False,
#     simplify=False,
#     device=0,
# )


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
侧面相机 YOLO-Cls -> TensorRT 导出脚本
=====================================

推荐放置位置：
    ~/smart_sorting_vision/scripts/last_use/export_tensorrt.py

推荐运行：
    cd ~/smart_sorting_vision
    source .venv/bin/activate
    python -m scripts.last_use.export_tensorrt

默认模型：
    outputs/training/side_cls/school_test/weights/best.pt

默认导出：
    FP16 TensorRT
    imgsz=224
    batch=1
    device=0

说明：
1. 旧 school_test checkpoint 保存过训练时自定义 Transform 类引用。
2. 因此 FixedRelativeROI、LetterboxSquare、ShapeSafeAppearanceAugment
   必须在 YOLO(best.pt) 加载前定义。
3. TensorRT engine 建议直接在最终部署的 Jetson 上生成。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


# =============================================================================
# 0. 工程根目录
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# 1. 旧 checkpoint 兼容类
#
# 注意：
# 必须放在 from ultralytics import YOLO 后真正加载 checkpoint 之前。
#
# 旧训练脚本如果以 python xxx.py 运行，pickle 中可能保存为：
#     __main__.FixedRelativeROI
#     __main__.LetterboxSquare
#     __main__.ShapeSafeAppearanceAugment
#
# 本脚本使用 python -m scripts.last_use.export_tensorrt 运行时，
# 当前模块同样作为 __main__ 执行，因此可以正确解析这些类。
# =============================================================================


class FixedRelativeROI:
    """
    与旧分类训练时的 FixedRelativeROI 保持 pickle 兼容。

    enabled=False 时不做裁剪。
    roi 使用归一化坐标：
        (left, top, right, bottom)
    """

    def __init__(
        self,
        enabled: bool = False,
        roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ) -> None:
        self.enabled = bool(enabled)

        self.left = float(roi[0])
        self.top = float(roi[1])
        self.right = float(roi[2])
        self.bottom = float(roi[3])

        if self.enabled:
            if not (
                0.0 <= self.left < self.right <= 1.0
                and 0.0 <= self.top < self.bottom <= 1.0
            ):
                raise ValueError(
                    "FixedRelativeROI roi 必须满足 "
                    "0<=left<right<=1 且 0<=top<bottom<=1"
                )

    def __call__(self, img):
        if not self.enabled:
            return img

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        width, height = img.size

        x1 = int(round(self.left * width))
        y1 = int(round(self.top * height))
        x2 = int(round(self.right * width))
        y2 = int(round(self.bottom * height))

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        return img.crop((x1, y1, x2, y2))


class LetterboxSquare:
    """
    与旧训练时的等比缩放 + 方形填充 Transform 保持兼容。
    """

    def __init__(
        self,
        size: int = 224,
        fill: tuple[int, int, int] = (114, 114, 114),
        margin: float = 0.02,
    ) -> None:
        self.size = int(size)
        self.fill = tuple(int(v) for v in fill)
        self.margin = float(margin)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img = img.convert("RGB")

        width, height = img.size

        if width <= 0 or height <= 0:
            raise ValueError("LetterboxSquare 收到空图像")

        usable = max(
            1,
            int(round(self.size * (1.0 - 2.0 * self.margin))),
        )

        scale = min(
            usable / float(width),
            usable / float(height),
        )

        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))

        img = img.resize(
            (new_w, new_h),
            Image.Resampling.BILINEAR,
        )

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            self.fill,
        )

        x = (self.size - new_w) // 2
        y = (self.size - new_h) // 2

        canvas.paste(img, (x, y))

        return canvas


class ShapeSafeAppearanceAugment:
    """
    与旧训练 checkpoint 中的 ShapeSafeAppearanceAugment 保持兼容。

    TensorRT 导出阶段本身不会依赖随机训练增强。
    这里保留该类的主要目的，是让 pickle 能恢复旧 checkpoint。
    """

    def __init__(
        self,
        p: float = 0.80,
        num_ops: int = 1,
    ) -> None:
        self.p = float(p)
        self.num_ops = max(1, int(num_ops))

    def __call__(self, img):
        if random.random() > self.p:
            return img

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        operations = [
            lambda x: ImageOps.autocontrast(x),
            lambda x: ImageOps.equalize(x),
            lambda x: ImageEnhance.Sharpness(x).enhance(
                random.uniform(0.7, 1.4)
            ),
        ]

        for op in random.sample(
            operations,
            k=min(self.num_ops, len(operations)),
        ):
            img = op(img)

        return img


# =============================================================================
# 2. 在兼容类定义完成后再导入 Ultralytics
# =============================================================================

from ultralytics import YOLO  # noqa: E402


# =============================================================================
# 3. 默认导出参数
# =============================================================================

DEFAULT_MODEL = (
    ROOT
    / "outputs"
    / "training"
    / "side_cls"
    / "school_test_new-3"
    / "weights"
    / "best.pt"
)

DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH = 1
DEFAULT_DEVICE = 0


# =============================================================================
# 4. 命令行参数
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export side YOLO classification model to TensorRT."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help=(
            "输入 .pt 模型路径。"
            "可以使用工程根目录相对路径或绝对路径。"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="TensorRT 输入尺寸，侧面分类模型默认 224。",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="TensorRT batch size，比赛部署默认 1。",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_DEVICE,
        help="CUDA device，Jetson 默认 0。",
    )

    parser.add_argument(
        "--fp32",
        action="store_true",
        help="使用 FP32 导出；默认使用 FP16。",
    )

    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="导出动态 shape。比赛固定 224 输入通常不需要。",
    )

    parser.add_argument(
        "--simplify",
        action="store_true",
        help="启用 ONNX simplify；默认关闭以减少 Jetson 导出兼容问题。",
    )

    return parser.parse_args()


def resolve_model_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


# =============================================================================
# 5. 主程序
# =============================================================================

def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model)

    use_fp16 = not args.fp32

    print("=" * 78)
    print("SIDE YOLO-CLS -> TENSORRT")
    print("=" * 78)

    print(f"Project root : {ROOT}")
    print(f"Model        : {model_path}")
    print(f"imgsz        : {args.imgsz}")
    print(f"batch        : {args.batch}")
    print(f"device       : {args.device}")
    print(f"FP16         : {use_fp16}")
    print(f"dynamic      : {args.dynamic}")
    print(f"simplify     : {args.simplify}")

    print("=" * 78)

    if not model_path.is_file():
        raise FileNotFoundError(
            "找不到侧面分类模型：\n"
            f"{model_path}\n\n"
            "请检查模型路径，或者使用：\n"
            "  python -m scripts.last_use.export_tensorrt "
            "--model <你的best.pt路径>"
        )

    if model_path.suffix.lower() != ".pt":
        raise ValueError(
            "输入模型必须是 .pt 文件：\n"
            f"{model_path}"
        )

    # -------------------------------------------------------------------------
    # 加载 PyTorch 模型
    # -------------------------------------------------------------------------

    print()
    print("[1/2] Loading PyTorch checkpoint...")

    model = YOLO(
        str(model_path)
    )

    print("[OK] best.pt loaded successfully.")

    # -------------------------------------------------------------------------
    # 导出 TensorRT
    # -------------------------------------------------------------------------

    print()
    print("[2/2] Exporting TensorRT engine on Jetson...")

    result = model.export(
        format="engine",

        imgsz=int(args.imgsz),
        batch=int(args.batch),

        device=int(args.device),

        # 比赛侧面分类输入固定为 224，因此默认 static shape。
        dynamic=bool(args.dynamic),

        # Jetson 部署默认 FP16。
        half=bool(use_fp16),

        # 默认关闭，避免增加额外导出依赖/兼容风险。
        simplify=bool(args.simplify),
    )

    print()
    print("=" * 78)
    print("EXPORT FINISHED")
    print("=" * 78)

    print(f"Ultralytics result: {result}")

    result_path = Path(str(result))

    if result_path.is_file():
        print(f"[OK] TensorRT engine: {result_path}")
        print(
            f"[OK] Size: "
            f"{result_path.stat().st_size / 1024.0 / 1024.0:.2f} MB"
        )
    else:
        # Ultralytics 某些版本返回字符串或其它对象，
        # 同时 best.engine 一般位于 best.pt 同目录。
        expected = model_path.with_suffix(".engine")

        if expected.is_file():
            print(f"[OK] TensorRT engine: {expected}")
            print(
                f"[OK] Size: "
                f"{expected.stat().st_size / 1024.0 / 1024.0:.2f} MB"
            )
        else:
            print(
                "[WARN] 导出命令已结束，但脚本没有在预期位置找到 .engine。"
            )
            print(
                "请检查上方 Ultralytics 输出中的实际保存路径。"
            )

    print()
    print("正式配置如果要使用新 engine，请确认 runtime_config.yaml 中：")
    print()
    print("model:")
    print("  side:")
    print("    path: <刚生成的 best.engine 路径>")
    print("    imgsz: 224")
    print("    device: 0")


if __name__ == "__main__":
    main()
