"""把人工标注的顶部分割数据整理为 YOLO train/val/test。"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    run("scripts/dataset/top_seg/prepare_top_seg_dataset.py")
