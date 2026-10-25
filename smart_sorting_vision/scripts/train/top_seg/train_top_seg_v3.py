from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# 工程根目录
#
# scripts/train/train_top_seg_v3.py
# -> parents[2] = 工程根目录
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# prepared/v3
#
# 当前数据策略：
#
# train
#   = prepared/v2 train
#   + 全部 V3 人工检查数据
#
# val
#   = prepared/v2 val
#
# test
#   = prepared/v2 原 test
#
# legacy_test
#   = 原 V1 legacy test
# ============================================================

DEFAULT_DATA_PATH = (
    ROOT
    / "data"
    / "datasets"
    / "top_seg"
    / "prepared"
    / "v3"
    / "data.yaml"
)


# ============================================================
# 初始化权重
#
# V3 是在当前正式 V2 模型基础上继续训练，
# 不从 yolo26n-seg.pt 重新开始。
# ============================================================

DEFAULT_MODEL_PATH = (
    ROOT
    / "weights"
    / "top_seg"
    / "v2"
    / "best.pt"
)


# ============================================================
# 训练输出
# ============================================================

DEFAULT_PROJECT_DIR = (
    ROOT
    / "outputs"
    / "training"
    / "top_seg"
    / "v3"
)


# ============================================================
# 默认训练参数
# ============================================================

DEFAULT_IMGSZ = 832

DEFAULT_EPOCHS = 120

DEFAULT_PATIENCE = 30

DEFAULT_WORKERS = 4

DEFAULT_SEED = 2026


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train top-camera YOLO segmentation V3 model."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="prepared/v3/data.yaml",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="初始化模型，默认 weights/top_seg/v2/best.pt",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="v3_interim_aug",
        help="本次训练输出目录名称",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "默认自动选择。"
            "有CUDA时使用 0，否则 cpu。"
            "例如：--device 0"
        ),
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help=(
            "默认 -1，让 Ultralytics 自动选择 batch size。"
            "也可指定固定值，例如 --batch 8"
        ),
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help=(
            "允许输出目录已存在。"
            "默认不允许，避免覆盖历史实验。"
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ========================================================
    # 路径
    # ========================================================

    data_path = (
        args.data
        .resolve()
    )

    model_path = (
        args.model
        .resolve()
    )

    project_dir = (
        DEFAULT_PROJECT_DIR
        .resolve()
    )

    # ========================================================
    # 基础检查
    # ========================================================

    if not data_path.exists():

        raise FileNotFoundError(
            "找不到 V3 data.yaml：\n"
            f"{data_path}\n\n"
            "请先运行：\n"
            "python -m scripts.dataset.prepare_top_seg_v3_dataset"
        )

    if not model_path.exists():

        raise FileNotFoundError(
            "找不到 V2 初始化模型：\n"
            f"{model_path}"
        )

    # ========================================================
    # Device
    # ========================================================

    if args.device is None:

        if torch.cuda.is_available():

            device = 0

        else:

            device = "cpu"

    else:

        # "0" -> 0
        # "cpu" -> "cpu"
        try:

            device = int(
                args.device
            )

        except ValueError:

            device = (
                args.device
            )

    # ========================================================
    # 环境信息
    # ========================================================

    print(
        "=" * 72
    )

    print(
        "TOP SEGMENTATION V3 TRAINING"
    )

    print(
        "=" * 72
    )

    print(
        f"Data       : {data_path}"
    )

    print(
        f"Init model : {model_path}"
    )

    print(
        f"Project    : {project_dir}"
    )

    print(
        f"Run name   : {args.name}"
    )

    print(
        f"Device     : {device}"
    )

    print(
        f"CUDA       : {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"imgsz      : {args.imgsz}"
    )

    print(
        f"epochs     : {args.epochs}"
    )

    print(
        f"patience   : {args.patience}"
    )

    print(
        f"batch      : {args.batch}"
    )

    print(
        f"workers    : {args.workers}"
    )

    print(
        f"seed       : {args.seed}"
    )

    print()

    print(
        "Dataset policy:"
    )

    print(
        "  train = V2 train + ALL V3 checked data"
    )

    print(
        "  val   = V2 val"
    )

    print(
        "  test  = V2 original test"
    )

    print()

    print(
        "Augmentation policy:"
    )

    print(
        "  继续沿用已经验证有效的 V2 augmentation。"
    )

    print(
        "  不使用 Mosaic / MixUp / CutMix。"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # 加载 V2 best.pt
    # ========================================================

    model = YOLO(
        str(
            model_path
        )
    )

    # ========================================================
    # 开始训练
    #
    # 关键原则：
    #
    # 1. V3 数据变化较大，但模型结构不变；
    # 2. 从 V2 best.pt 继续训练；
    # 3. 增强参数继续使用 V2 中已经验证过的版本；
    # 4. 不在本轮额外修改损失、网络结构和优化器，
    #    避免同时引入太多变量。
    # ========================================================

    results = model.train(

        # ----------------------------------------------------
        # Dataset / input
        # ----------------------------------------------------

        data=str(
            data_path
        ),

        imgsz=
            args.imgsz,

        # ----------------------------------------------------
        # Training length
        # ----------------------------------------------------

        epochs=
            args.epochs,

        patience=
            args.patience,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        seed=
            args.seed,

        deterministic=
            True,

        # ----------------------------------------------------
        # Hardware
        # ----------------------------------------------------

        device=
            device,

        batch=
            args.batch,

        workers=
            args.workers,

        amp=
            True,

        # ----------------------------------------------------
        # Optimizer
        #
        # 与 V2 保持一致。
        # ----------------------------------------------------

        optimizer=
            "auto",

        # ----------------------------------------------------
        # Segmentation
        # ----------------------------------------------------

        overlap_mask=
            True,

        mask_ratio=
            4,

        # ----------------------------------------------------
        # V2 已验证有效的颜色增强
        # ----------------------------------------------------

        hsv_h=
            0.008,

        hsv_s=
            0.20,

        hsv_v=
            0.18,

        # ----------------------------------------------------
        # 轻微几何增强
        #
        # 顶部相机固定，所以不要做过强旋转/平移。
        # ----------------------------------------------------

        degrees=
            3.0,

        translate=
            0.03,

        scale=
            0.10,

        shear=
            0.0,

        perspective=
            0.0,

        # ----------------------------------------------------
        # 不做翻转
        #
        # 机械坐标系统和托盘方向固定。
        # ----------------------------------------------------

        flipud=
            0.0,

        fliplr=
            0.0,

        # ----------------------------------------------------
        # 不使用会明显改变真实场景结构的增强
        # ----------------------------------------------------

        mosaic=
            0.0,

        mixup=
            0.0,

        cutmix=
            0.0,

        # ----------------------------------------------------
        # 保留 V2 中有效的 Copy-Paste
        #
        # 这对多实例 / 接触场景有一定帮助。
        # ----------------------------------------------------

        copy_paste=
            0.15,

        copy_paste_mode=
            "mixup",

        # ----------------------------------------------------
        # 不做 BGR 通道交换
        # ----------------------------------------------------

        bgr=
            0.0,

        # ----------------------------------------------------
        # Logging / output
        # ----------------------------------------------------

        project=str(
            project_dir
        ),

        name=
            args.name,

        exist_ok=
            args.exist_ok,

        plots=
            True,

        verbose=
            True,

        save=
            True,

        save_period=
            -1,
    )

    # ========================================================
    # 输出位置
    # ========================================================

    run_dir = (
        project_dir
        / args.name
    )

    best_model = (
        run_dir
        / "weights"
        / "best.pt"
    )

    last_model = (
        run_dir
        / "weights"
        / "last.pt"
    )

    print()
    print(
        "=" * 72
    )

    print(
        "V3 TRAINING FINISHED"
    )

    print(
        "=" * 72
    )

    print(
        f"Run directory : "
        f"{run_dir}"
    )

    print(
        f"Best model    : "
        f"{best_model}"
    )

    print(
        f"Last model    : "
        f"{last_model}"
    )

    print(
        "=" * 72
    )

    return results


if __name__ == "__main__":

    # Windows 下使用 workers > 0 时必须保留 main guard。
    main()
