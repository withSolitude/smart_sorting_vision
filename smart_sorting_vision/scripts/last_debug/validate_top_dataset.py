"""检查顶部 YOLO 分割数据集；统一从 last_debug 入口运行。"""
try:
    from ._delegate import run
except ImportError:  # 允许直接执行此文件
    from scripts.last_debug._delegate import run


if __name__ == "__main__":
    run("scripts/dataset/top_seg/validate_top_seg_dataset.py")
