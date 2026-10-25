from pathlib import Path
import shutil

from ultralytics import YOLO


# ============================================================
# 工程根目录
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 路径
# ============================================================

# 官方预训练模型
PRETRAINED_MODEL = (
    ROOT
    / "weights"
    / "pretrained"
    / "yolo26n-seg.pt"
)

# 数据集配置
DATA_YAML = (
    ROOT
    / "datasets"
    / "top"
    / "yolo"
    / "top.yaml"
)

# Ultralytics训练输出目录
PROJECT_DIR = (
    ROOT
    / "outputs"
    / "runs"
    / "top"
)

# 本次训练名称
RUN_NAME = "first_test"

# 最终工程模型保存目录
FINAL_WEIGHT_DIR = (
    ROOT
    / "weights"
    / "top"
)

FINAL_WEIGHT = (
    FINAL_WEIGHT_DIR
    / "top_best.pt"
)


def main():

    print("==================================")
    print("顶部YOLO26实例分割模型训练")
    print("==================================")

    # ========================================================
    # 1. 检查文件
    # ========================================================

    if not PRETRAINED_MODEL.exists():

        raise FileNotFoundError(
            f"找不到预训练模型：\n"
            f"{PRETRAINED_MODEL}"
        )

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"找不到数据集配置：\n"
            f"{DATA_YAML}"
        )

    print(
        f"预训练模型：{PRETRAINED_MODEL}"
    )

    print(
        f"数据集配置：{DATA_YAML}"
    )

    # ========================================================
    # 2. 加载YOLO26-Seg预训练模型
    # ========================================================

    print()
    print(
        "[1/3] 正在加载YOLO26n-Seg..."
    )

    model = YOLO(
        str(PRETRAINED_MODEL)
    )

    print(
        "[1/3] 模型加载完成"
    )

    # ========================================================
    # 3. 开始训练
    # ========================================================

    print()
    print(
        "[2/3] 开始训练..."
    )

    results = model.train(

        # ------------------------------
        # 数据集
        # ------------------------------

        data=str(DATA_YAML),

        # ------------------------------
        # 第一轮只跑100轮
        # ------------------------------

        epochs=100,

        # ------------------------------
        # YOLO输入尺寸
        # ------------------------------

        imgsz=640,

        # ------------------------------
        # NUC目前使用CPU
        # ------------------------------

        device="cpu",

        # ------------------------------
        # 小数据集先用较小batch
        # ------------------------------

        batch=4,

        # ------------------------------
        # 数据加载线程
        # ------------------------------

        workers=2,

        # ------------------------------
        # 保存位置
        # ------------------------------

        project=str(PROJECT_DIR),

        name=RUN_NAME,

        # 不自动生成first_test2等目录
        exist_ok=True,

        # ------------------------------
        # 固定随机种子
        # ------------------------------

        seed=42,

        deterministic=True,

        # ------------------------------
        # 保存训练图表
        # ------------------------------

        plots=True,

        # ------------------------------
        # 打印训练信息
        # ------------------------------

        verbose=True,
    )

    print()
    print(
        "[2/3] 训练结束"
    )

    # ========================================================
    # 4. 找到best.pt
    # ========================================================

    best_model = (
        PROJECT_DIR
        / RUN_NAME
        / "weights"
        / "best.pt"
    )

    if not best_model.exists():

        raise FileNotFoundError(
            f"训练结束但找不到best.pt：\n"
            f"{best_model}"
        )

    # ========================================================
    # 5. 复制到工程正式权重目录
    # ========================================================

    FINAL_WEIGHT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        best_model,
        FINAL_WEIGHT,
    )

    print()
    print(
        "[3/3] 最佳模型已复制"
    )

    print(
        f"原始best.pt：\n"
        f"{best_model}"
    )

    print(
        f"工程模型：\n"
        f"{FINAL_WEIGHT}"
    )

    print()
    print("==================================")
    print("顶部模型训练流程完成")
    print("==================================")


if __name__ == "__main__":
    main()