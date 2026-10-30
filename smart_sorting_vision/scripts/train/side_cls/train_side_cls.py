from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[3]


DATASET_DIR = (
    ROOT
    / "data"
    / "datasets"
    / "side_cls"
    / "v1"
)


OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "training"
    / "side_cls"
)


def main():

    print(f"ROOT        = {ROOT}")
    print(f"DATASET_DIR = {DATASET_DIR}")
    print(f"OUTPUT_DIR  = {OUTPUT_DIR}")

    # 数据集必须包含与正式配置一致的 no_object 类别，例如：
    # data/datasets/side_cls/v1/train/no_object/*.jpg
    # train/val/test 都建议采集不同曝光、轻微位置变化的空托盘图。
    for split in ("train", "val", "test"):
        no_object_dir = DATASET_DIR / split / "no_object"
        if not no_object_dir.is_dir():
            print(f"[WARN] 缺少空背景类别：{no_object_dir}")

    # ============================================================
    # 1. 加载 YOLO 分类预训练模型
    # ============================================================

    model = YOLO(
        "yolo26n-cls.pt"
    )

    # ============================================================
    # 2. 开始训练
    # ============================================================

    model.train(
        data=str(DATASET_DIR),

        epochs=100,
        imgsz=224,
        batch=32,
        device=0,
        workers=4,

        project=str(OUTPUT_DIR),
        name="v2_yolo26n_cls",

        # --------------------------------------------------------
        # 数据增强
        # --------------------------------------------------------

        fliplr=0.5,
        flipud=0.0,

        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        translate=0.10,
        scale=0.30,

        degrees=8.0,

        shear=0.0,
        perspective=0.0,

        auto_augment="randaugment",

        erasing=0.15,
    )


if __name__ == "__main__":
    main()
