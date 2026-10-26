"""训练顶部 YOLO 分割模型；命令行参数原样传给统一训练脚本。"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    run("scripts/train/top_seg/train_top_seg_v3.py")
